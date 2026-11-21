from decimal import Decimal, localcontext

from core.contracts import round_money
from core.profile_loader import S3Profile
from modules.s3_claims.contracts import (
    S3EvaluateClaimsRequest,
    S3EvaluateClaimsResponse,
    S3ItemDiscrepancy,
)
from modules.s3_claims.matcher import Match, match_package

ZERO = Decimal("0.00")


def cell(value: object) -> str:
    return "".join(
        char if char.isalnum() or char == " " else f"&#{ord(char)};"
        for char in " ".join(str(value).split())
    )


def report(
    payload: S3EvaluateClaimsRequest,
    rows: list[S3ItemDiscrepancy],
    reasons: set[str],
    totals: str = "",
) -> str:
    lines = ["# Акт сверки поставки", f"Пакет {payload.package_id}"]
    if payload.award is not None:
        award = payload.award
        lines.extend(
            [
                f"S1 №{award.s1_item_id}, S2 №{award.s2_item_id}, ИНН {award.winner_inn}",
                f"Поставщик {cell(award.winner_name)}",
                (
                    f"Договорный срок {award.promised_delivery_days} дн., "
                    f"отсрочка {award.promised_deferment_days} дн., "
                    f"лимит {award.approved_credit_limit_rub} ₽"
                ),
            ]
        )
    for doc in payload.documents:
        lines.append(
            f"- {doc.doc_type} №{cell(doc.doc_number)} от {doc.doc_date}, "
            f"предъявление на {doc.actual_delivery_days} день"
        )
    if rows:
        lines.extend(
            [
                "",
                "| Код | Ед. | Заказ | Поставка | Брак | Принято | Недостача | Излишек | Откл. цены |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            values = (
                cell(row.item_code),
                cell(row.unit),
                row.awarded_quantity,
                row.delivered_quantity,
                row.defect_quantity,
                row.accepted_quantity,
                row.shortage_quantity,
                row.excess_quantity,
                row.price_variance_rub,
            )
            lines.append("| " + " | ".join(map(str, values)) + " |")
    if totals:
        lines.extend(["", totals])
    if reasons:
        lines.extend(
            ["", "Основания проверки", *[f"- {cell(reason)}" for reason in sorted(reasons)]]
        )
    return "\n".join(lines)


def calculate_line(
    match: Match, quantity_tolerance: Decimal, price_tolerance: Decimal
) -> tuple[dict[str, object], Decimal, Decimal, bool, set[str]]:
    item, delivery, act = match.item, match.delivery, match.defect
    delivered = (delivery.quantity if delivery else None) or ZERO
    defect = (act.defect_quantity if act else None) or ZERO
    price = delivery.unit_price_rub if delivery else None
    shortage = max(ZERO, item.quantity - delivered)
    excess = max(ZERO, delivered - item.quantity)
    claim_defect = min(defect, item.quantity, delivered)
    accepted = min(item.quantity, delivered) - claim_defect
    delta = price - item.unit_price_rub if price is not None else ZERO
    qty_out = shortage * 100 > item.quantity * quantity_tolerance
    price_out = accepted > 0 and abs(delta) * 100 > item.unit_price_rub * price_tolerance
    variance = round_money(accepted * delta)
    reasons = set()
    if excess:
        reasons.add("S3_OVERDELIVERY")
    if price_out and delta < 0:
        reasons.add("S3_PRICE_DECREASE_REVIEW")
    return (
        {
            "item_code": item.item_code,
            "unit": item.unit,
            "awarded_quantity": item.quantity,
            "delivered_quantity": delivered,
            "defect_quantity": defect,
            "accepted_quantity": accepted,
            "shortage_quantity": shortage,
            "excess_quantity": excess,
            "awarded_unit_price_rub": item.unit_price_rub,
            "actual_unit_price_rub": price,
            "accepted_amount_rub": round_money(accepted * item.unit_price_rub),
            "shortage_or_defect_rub": round_money((shortage + claim_defect) * item.unit_price_rub),
            "price_variance_rub": variance,
            "quantity_outside_tolerance": qty_out,
            "price_outside_tolerance": price_out,
        },
        round_money(((shortage if qty_out else ZERO) + claim_defect) * item.unit_price_rub),
        max(ZERO, variance) if price_out else ZERO,
        bool(claim_defect or qty_out or (price_out and delta > 0)),
        reasons,
    )


def blocked(
    payload: S3EvaluateClaimsRequest, reasons: set[str]
) -> tuple[S3EvaluateClaimsResponse, list[str]]:
    return S3EvaluateClaimsResponse(
        package_id=payload.package_id,
        outcome="ESCALATED",
        accepted_amount_rub=None,
        shortage_or_defect_rub=None,
        price_variance_rub=None,
        delay_days=None,
        penalty_rub=None,
        claim_amount_rub=None,
        discrepancies=[],
        reconciliation_report_markdown=report(payload, [], reasons),
    ), sorted(reasons)


def evaluate(
    payload: S3EvaluateClaimsRequest,
    profile: S3Profile,
    *,
    gate_reasons: list[str] | None = None,
    penalty_confirmed: bool = False,
) -> tuple[S3EvaluateClaimsResponse, list[str]]:
    if gate_reasons:
        return blocked(payload, set(gate_reasons))
    matches, reasons = match_package(payload)
    if reasons:
        return blocked(payload, reasons)
    assert payload.award is not None
    actual_days = next(
        doc.actual_delivery_days for doc in payload.documents if doc.doc_type != "DEFECT_ACT"
    )
    delay = max(0, actual_days - payload.award.promised_delivery_days)
    if delay and not penalty_confirmed:
        return blocked(payload, {"S3_PENALTY_TERMS_UNCONFIRMED"})

    with localcontext() as ctx:
        ctx.prec = 50
        penalty = round_money(
            payload.award.awarded_total_rub
            * min(delay * Decimal(profile.daily_penalty_pct), Decimal(profile.max_penalty_pct))
            / 100
        )
        rows = []
        claim, required = penalty, False
        for match in matches:
            values, loss, overcharge, needs_claim, flags = calculate_line(
                match, Decimal(profile.quantity_tolerance_pct), Decimal(profile.price_tolerance_pct)
            )
            variance = values["price_variance_rub"]
            assert isinstance(variance, Decimal)
            if abs(variance) >= 10**18:
                return blocked(payload, {"S3_UNSUPPORTED_PRECISION"})
            rows.append(S3ItemDiscrepancy.model_validate(values))
            claim += loss + overcharge
            required |= needs_claim
            reasons.update(flags)
        accepted = sum((row.accepted_amount_rub for row in rows), ZERO)
        shortage = sum((row.shortage_or_defect_rub for row in rows), ZERO)
        variance = sum((row.price_variance_rub for row in rows), ZERO)
        if any(abs(amount) >= 10**18 for amount in (accepted, shortage, variance, claim)):
            return blocked(payload, {"S3_UNSUPPORTED_PRECISION"})
        if claim >= Decimal(profile.manual_claim_threshold_rub):
            reasons.add("S3_CLAIM_THRESHOLD")
        totals = (
            f"Принято {accepted} ₽. Недопоставка и брак {shortage} ₽. "
            f"Отклонение цены {variance} ₽. Просрочка {delay} дн. Пеня {penalty} ₽.\n\n"
            f"## Проект претензии\nСумма требования {claim} ₽."
        )
        return S3EvaluateClaimsResponse(
            package_id=payload.package_id,
            outcome="ESCALATED"
            if reasons
            else "CLAIM_REQUIRED"
            if required or claim
            else "ACCEPTED",
            accepted_amount_rub=accepted,
            shortage_or_defect_rub=shortage,
            price_variance_rub=variance,
            delay_days=delay,
            penalty_rub=penalty,
            claim_amount_rub=claim,
            discrepancies=rows,
            reconciliation_report_markdown=report(payload, rows, reasons, totals),
        ), sorted(reasons)
