from collections import defaultdict
from unicodedata import normalize

from modules.s2_tender.contracts import S2QuoteItem, S2TenderItem


def normalized(value: str) -> str:
    return " ".join(normalize("NFKC", value).casefold().split())


def match_items(
    items: list[S2TenderItem], lines: list[S2QuoteItem]
) -> dict[str, S2QuoteItem] | None:
    by_code = {item.item_code: item for item in items}
    by_name: dict[tuple[str, str], list[S2TenderItem]] = defaultdict(list)
    for tender_item in items:
        by_name[normalized(tender_item.name), normalized(tender_item.unit)].append(tender_item)

    matches: dict[str, S2QuoteItem] = {}
    for line in lines:
        if line.name is None or line.unit is None:
            return None
        unit = normalized(line.unit)
        if line.item_code is not None:
            item = by_code.get(line.item_code)
        else:
            candidates = by_name[normalized(line.name), unit]
            item = candidates[0] if len(candidates) == 1 else None
        if (
            item is None
            or item.item_code in matches
            or normalized(item.unit) != unit
            or item.quantity != line.quantity
        ):
            return None
        matches[item.item_code] = line
    return matches if len(matches) == len(items) else None
