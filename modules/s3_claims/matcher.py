from dataclasses import dataclass
from unicodedata import normalize

from modules.s3_claims.contracts import (
    S3AwardedItem,
    S3DeliveryLine,
    S3EvaluateClaimsRequest,
)


@dataclass(frozen=True)
class Match:
    item: S3AwardedItem
    delivery: S3DeliveryLine | None
    defect: S3DeliveryLine | None


def normalized(value: str) -> str:
    return " ".join(normalize("NFKC", value).casefold().split())


def match_lines(
    items: list[S3AwardedItem], lines: list[S3DeliveryLine]
) -> dict[str, S3DeliveryLine] | None:
    matched: dict[str, S3DeliveryLine] = {}
    for line in lines:
        if line.unit is None:
            return None
        candidates = [
            item
            for item in items
            if normalized(item.unit) == normalized(line.unit)
            and (
                item.item_code == line.item_code
                if line.item_code is not None
                else line.name is not None and normalized(item.name) == normalized(line.name)
            )
        ]
        if len(candidates) != 1 or candidates[0].item_code in matched:
            return None
        matched[candidates[0].item_code] = line
    return matched


def match_package(payload: S3EvaluateClaimsRequest) -> tuple[list[Match], set[str]]:
    award = payload.award
    if award is None:
        return [], {"S3_AWARD_UNCONFIRMED"}
    delivery = [doc for doc in payload.documents if doc.doc_type != "DEFECT_ACT"]
    defects = [doc for doc in payload.documents if doc.doc_type == "DEFECT_ACT"]
    if len(delivery) != 1 or len(defects) > 1:
        return [], {"S3_PACKAGE_UNSUPPORTED"}
    primary = delivery[0]
    act = defects[0] if defects else None
    reasons = {reason for doc in payload.documents for reason in doc.escalation_reasons}
    if any(not doc.arithmetic_ok for doc in payload.documents) and not reasons:
        reasons.add("S3_ARITHMETIC_MISMATCH")
    if any(doc.supplier_inn != award.winner_inn for doc in payload.documents):
        reasons.add("S3_DOCUMENT_SUPPLIER_MISMATCH")
    if act and act.actual_delivery_days != primary.actual_delivery_days:
        reasons.add("S3_PACKAGE_UNSUPPORTED")
    if act and act.referenced_delivery_document_id != primary.document_id:
        reasons.add("S3_DEFECT_CONFLICT")
    actual = match_lines(award.awarded_items, primary.lines)
    defective = match_lines(award.awarded_items, act.lines) if act else {}
    if actual is None or defective is None:
        return [], reasons | {"S3_ITEM_MATCH_FAILED"}
    for code, line in defective.items():
        received = actual.get(code)
        if (
            received is None
            or received.quantity is None
            or line.quantity is None
            or line.quantity > received.quantity
            or line.defect_quantity is None
            or line.defect_quantity > line.quantity
            or line.unit_price_rub != received.unit_price_rub
        ):
            reasons.add("S3_DEFECT_CONFLICT")
    if reasons:
        return [], reasons
    return [
        Match(item, actual.get(item.item_code), defective.get(item.item_code))
        for item in award.awarded_items
    ], set()
