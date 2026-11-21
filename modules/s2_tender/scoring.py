from dataclasses import dataclass
from decimal import Decimal, localcontext
from uuid import UUID

from core.contracts import round_money
from core.profile_loader import S2Profile
from modules.s2_tender.contracts import (
    SCORE,
    Criterion,
    QuoteIdentity,
    S2EvaluateTenderRequest,
    S2EvaluateTenderResponse,
    S2ParsedQuote,
    S2QuoteItem,
    S2ScorePenalty,
    S2SupplierGateSnapshot,
    S2TenderParticipant,
)
from modules.s2_tender.matcher import match_items

UPSTREAM_UNCONFIRMED = "S2_UPSTREAM_UNCONFIRMED"


@dataclass(frozen=True)
class EligibleQuote:
    identity: QuoteIdentity
    total: Decimal
    delivery: int
    deferment: int
    rating: Decimal


def rejection(
    quote: S2ParsedQuote,
    gate: S2SupplierGateSnapshot | None,
    verified_quote: bool,
) -> str | None:
    if gate is None or gate.s1_item_id != quote.s1_item_id or not gate.stage_result_published:
        return "S2_S1_UNCONFIRMED"
    if gate.kyc_status == "REJECTED":
        return "S2_S1_REJECTED"
    if not verified_quote or quote.need_human or not quote.arithmetic_ok:
        return "S2_QUOTE_INVALID"
    return None


def ratio(best: Decimal, value: Decimal) -> Decimal:
    return Decimal(1) if value == best else best / value


def rank_quotes(
    quotes: list[EligibleQuote], profile: S2Profile, required_days: int
) -> list[S2TenderParticipant]:
    if not quotes:
        return []
    cheapest = min(quote.total for quote in quotes)
    fastest = Decimal(min(quote.delivery for quote in quotes))
    weights = profile.weights
    ranked = []
    for quote in quotes:
        criteria: list[tuple[Criterion, Decimal, Decimal]] = [
            ("PRICE", Decimal(weights.price), ratio(cheapest, quote.total)),
            ("DELIVERY", Decimal(weights.delivery), ratio(fastest, Decimal(quote.delivery))),
            ("S1_RELIABILITY", Decimal(weights.s1_reliability), quote.rating / 100),
            (
                "DEFERMENT",
                Decimal(weights.deferment),
                (
                    min(Decimal(1), Decimal(quote.deferment) / required_days)
                    if required_days
                    else Decimal(1)
                ),
            ),
        ]
        score = round_money(
            sum((weight * factor for _, weight, factor in criteria), Decimal(0)), SCORE
        )
        penalties = []
        for criterion, weight, factor in criteria:
            points = round_money(weight * (1 - factor), SCORE)
            if points > 0:
                penalties.append(S2ScorePenalty(criterion=criterion, points=points))
        ranked.append(
            (
                (-score, quote.total, quote.delivery, quote.identity.supplier_inn),
                quote,
                score,
                penalties,
            )
        )

    return [
        S2TenderParticipant(
            **quote.identity.model_dump(),
            rank=rank,
            score=score,
            total_rub=quote.total,
            delivery_days=quote.delivery,
            deferment_days=quote.deferment,
            rejection_reasons=[],
            penalties=penalties,
        )
        for rank, (_, quote, score, penalties) in enumerate(
            sorted(ranked, key=lambda row: row[0]), 1
        )
    ]


def cell(value: object) -> str:
    if value is None:
        return "—"
    # Numeric entities keep Markdown, raw HTML and table separators inert.
    collapsed = " ".join(str(value).split())
    return "".join(
        f"&#{ord(char)};" if char in '&<>"\\|[]()*_!~#{}' or ord(char) == 96 else char
        for char in collapsed
    )


def row(*values: object) -> str:
    return "| " + " | ".join(map(cell, values)) + " |"


def matrix(
    request: S2EvaluateTenderRequest,
    participants: list[S2TenderParticipant],
    gates: dict[str, S2SupplierGateSnapshot],
    matches: dict[UUID, dict[str, S2QuoteItem]],
    reasons: list[str],
) -> str:
    lines = [
        "# Свод КП",
        "",
        f"Бюджет: {request.target_budget_rub} RUB",
        "",
        "| Позиция | Наименование | Ед. | Количество | Целевая цена с НДС, RUB |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        row(item.item_code, item.name, item.unit, item.quantity, item.target_unit_price_rub)
        for item in request.items
    )
    lines.extend(
        [
            "",
            (
                "| ИНН | Поставщик | Ранг | Балл | Итог с НДС, RUB | Срок, дн. | Отсрочка, дн. |"
                " Рейтинг S1 | Причины отклонения |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for participant in participants:
        gate = gates.get(participant.supplier_inn)
        rating = (
            gate.reliability_score if gate and gate.s1_item_id == participant.s1_item_id else None
        )
        lines.append(
            row(
                participant.supplier_inn,
                participant.supplier_name,
                participant.rank,
                participant.score,
                participant.total_rub,
                participant.delivery_days,
                participant.deferment_days,
                rating,
                ", ".join(participant.rejection_reasons) or None,
            )
        )
    lines.extend(
        [
            "",
            "| ИНН | Позиция | Строка КП | Цена с НДС, RUB | Сумма с НДС, RUB |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for participant in participants:
        for code, line in sorted(matches.get(participant.quote_id, {}).items()):
            lines.append(
                row(
                    participant.supplier_inn,
                    code,
                    line.line_no,
                    line.unit_price_rub,
                    line.total_rub,
                )
            )
    if reasons:
        lines.extend(["", "Проверка: " + ", ".join(map(cell, reasons))])
    return "\n".join(lines)


def evaluate(
    request: S2EvaluateTenderRequest,
    profile: S2Profile,
    *,
    upstream_confirmed: bool,
    verified_quote_ids: set[UUID],
) -> tuple[S2EvaluateTenderResponse, list[str]]:
    gates = {gate.inn: gate for gate in request.supplier_gate_snapshots}
    eligible: list[EligibleQuote] = []
    rejected: list[S2TenderParticipant] = []
    matches: dict[UUID, dict[str, S2QuoteItem]] = {}
    with localcontext() as context:
        context.prec = 50
        budget_limit = round_money(
            request.target_budget_rub * (1 + Decimal(profile.max_budget_overrun_pct) / 100)
        )
        for quote in request.quotes:
            gate = gates.get(quote.supplier_inn)
            reason = rejection(quote, gate, quote.quote_id in verified_quote_ids)
            identity = QuoteIdentity.model_validate(
                quote.model_dump(include=set(QuoteIdentity.model_fields))
            )
            total, delivery = None, None
            if reason is None:
                assert gate is not None
                matched = match_items(request.items, quote.items)
                if matched is None:
                    reason = "S2_ITEM_MATCH_FAILED"
                elif (
                    quote.quoted_total_rub is None
                    or quote.deferment_days is None
                    or any(
                        line.delivery_days is None or line.total_rub is None
                        for line in matched.values()
                    )
                ):
                    reason = "S2_QUOTE_INVALID"
                else:
                    matches[quote.quote_id] = matched
                    total = sum(
                        (line.total_rub or Decimal(0) for line in matched.values()), Decimal(0)
                    )
                    delivery = max(line.delivery_days or 0 for line in matched.values())
                    if (
                        profile.credit_limit_policy == "REQUIRE_FOR_DEFERRED"
                        and quote.deferment_days > 0
                        and total > gate.approved_credit_limit_rub
                    ):
                        reason = "S2_CREDIT_LIMIT_EXCEEDED"
                    elif total > budget_limit:
                        reason = "S2_BUDGET_EXCEEDED"
                    else:
                        eligible.append(
                            EligibleQuote(
                                identity,
                                total,
                                delivery,
                                quote.deferment_days,
                                gate.reliability_score,
                            )
                        )
            if reason is not None:
                rejected.append(
                    S2TenderParticipant(
                        **identity.model_dump(),
                        rank=None,
                        score=None,
                        total_rub=total,
                        delivery_days=delivery,
                        deferment_days=quote.deferment_days,
                        rejection_reasons=[reason],
                        penalties=[],
                    )
                )
        ranked = rank_quotes(eligible, profile, request.required_deferment_days)
        reasons = []
        if len(ranked) < profile.min_eligible_quotes:
            reasons.append("S2_QUORUM_NOT_MET")
        if not upstream_confirmed:
            reasons.append(UPSTREAM_UNCONFIRMED)
        winner = ranked[0] if not reasons else None
        total = winner.total_rub if winner else None
        savings = round_money(request.target_budget_rub - total) if total is not None else None

    participants = ranked + sorted(rejected, key=lambda participant: participant.supplier_inn)
    return S2EvaluateTenderResponse(
        winner_inn=winner.supplier_inn if winner else None,
        winner_s1_item_id=winner.s1_item_id if winner else None,
        winner_total_rub=total,
        savings_rub=savings,
        participants=participants,
        comparison_matrix_markdown=matrix(request, participants, gates, matches, reasons),
    ), reasons
