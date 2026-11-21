import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from html import escape

from core.contracts import round_money
from core.profile_loader import RISKS, Risk, S1Profile
from modules.s1_kyc.contracts import PaymentTerms, S1EvaluateRequest, S1EvaluateResponse

ZERO = Decimal("0.00")


def evaluate(
    request: S1EvaluateRequest,
    profile: S1Profile,
    *,
    now: datetime | None = None,
) -> S1EvaluateResponse:
    snapshot = request.registry
    now = now if now is not None else datetime.now(UTC)
    age = now - snapshot.observed_at
    if age < timedelta(0):
        raise ValueError("Registry snapshot is from the future")
    facts = snapshot.model_dump(mode="json", exclude={"inn", "ogrn", "observed_at"})
    if len(snapshot.inn) == 12:
        del facts["charter_capital_rub"]
    missing = sorted(key for key, value in facts.items() if value is None)
    stale = age > timedelta(hours=profile.registry.max_age_hours)
    scoring = profile.scoring
    stops = sorted(
        code
        for code, rule in scoring.stop_factors.items()
        if not stale
        and facts[rule.field] is not None
        and ((facts[rule.field] == rule.value) == (rule.operator == "eq"))
    )
    reasons = set(stops)
    if stale:
        reasons.add("REGISTRY_STALE")
    if missing:
        reasons.add("REGISTRY_INCOMPLETE")
    reason = "REGISTRY_STALE" if stale else "REGISTRY_INCOMPLETE" if missing else None
    score, risk = None, None
    limit, days = ZERO, 0
    applied = []
    if reason is None:
        for rule in scoring.penalties:
            value = Decimal(str(facts[rule.field]))
            threshold = Decimal(rule.threshold)
            if (rule.operator == "lt" and value < threshold) or (
                rule.operator == "gt" and value > threshold
            ):
                applied.append(rule)
        penalty = sum((Decimal(rule.points) for rule in applied), ZERO)
        score = round_money(max(ZERO, Decimal(100) - penalty))
        risk = _risk(score, profile) if not stops else "CRITICAL"
        if risk in scoring.manual_review_risk_levels:
            reasons.add("RISK_REVIEW")
        if not reasons:
            band = scoring.risk_bands[risk]
            assert request.parsed.requested_limit_rub is not None
            limit = round_money(
                min(request.parsed.requested_limit_rub, Decimal(band.max_credit_limit_rub))
            )
            days = band.deferment_days if limit > 0 else 0
    facts_lines = [
        f"- {key}: {value if value is not None else 'не подтверждено'}"
        for key, value in facts.items()
    ]
    penalty_lines = [
        f"- {rule.field} {rule.operator} {rule.threshold}: −{rule.points}" for rule in applied
    ]
    name = escape(" ".join((request.parsed.legal_name or "").split()))
    name = re.sub(r"([\\`*_[\]()!])", r"\\\1", name)
    orders = [
        "# Досье S1",
        "",
        f"- Наименование: {name}",
        f"- ИНН: {snapshot.inn}",
        f"- ОГРН / ОГРНИП: {snapshot.ogrn}",
        f"- Снимок: {snapshot.observed_at.isoformat()}",
        "",
        "## Факты реестров",
        *facts_lines,
        "",
        "## Оценка",
        f"- Балл: {score if score is not None else 'не рассчитан'}",
        f"- Риск: {risk or 'не определён'}",
        *penalty_lines,
        f"- Стоп-факторы: {', '.join(stops) or 'нет подтверждённых'}",
        f"- Причины эскалации: {', '.join(sorted(reasons)) or 'нет'}",
        "",
        "## Рекомендация",
        f"- Лимит: {limit:.2f} руб.",
        f"- Условия: {'отсрочка, дней: ' + str(days) if days else 'предоплата'}",
    ]
    return S1EvaluateResponse(
        reliability_score=score,
        risk_level=risk,
        recommended_credit_limit_rub=limit,
        payment_terms=PaymentTerms(kind="DEFERRED" if days else "PREPAYMENT", deferment_days=days),
        stop_factors=stops,
        dossier_markdown="\n".join(orders),
        need_human=bool(reasons),
        escalation_reasons=sorted(reasons),
        cost_rub=Decimal("0.00000000"),
        mode="fallback" if reason else "deterministic",
        fallback_reason=reason,
    )


def _risk(score: Decimal, profile: S1Profile) -> Risk:
    return next(
        risk
        for risk in reversed(RISKS)
        if score >= Decimal(profile.scoring.risk_bands[risk].min_score)
    )
