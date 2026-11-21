import asyncio
import base64
import binascii
import json
import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from io import BytesIO
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, StrictBool, StrictStr, ValidationError
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from core.contracts import Mode, Nonempty, StrictDTO, round_money
from core.llm_gateway import CycleBudget, LLMGateway
from core.profile_loader import ExtractionProfile
from modules.s2_tender.contracts import Days, S2ParsedQuote, S2QuoteItem, S2QuoteSource

INCOMPLETE = "S2_QUOTE_INCOMPLETE"
MISMATCH = "S2_ARITHMETIC_MISMATCH"
PRECISION = "S2_UNSUPPORTED_PRECISION"


class Prompt(StrictDTO):
    version: Nonempty
    variant: Nonempty
    template: Nonempty


class ExtractedLine(StrictDTO):
    item_code: Nonempty | None = None
    name: Nonempty | None = None
    unit: Nonempty | None = None
    quantity: StrictStr | None = None
    unit_price: StrictStr | None = None
    unit_price_includes_vat: StrictBool | None = None
    vat_treatment: Literal["TAXABLE", "NO_VAT"] | None = None
    vat_rate_pct: StrictStr | None = None
    vat_amount: StrictStr | None = None
    line_total: StrictStr | None = None
    line_total_includes_vat: StrictBool | None = None
    delivery_days: Days | None = None


class ExtractedQuote(StrictDTO):
    currency: Nonempty | None = None
    quoted_total: StrictStr | None = None
    deferment_days: Days | None = None
    items: list[ExtractedLine] = Field(default_factory=list)


@dataclass(frozen=True)
class ParseResult:
    quote: S2ParsedQuote
    mode: Mode = "deterministic"
    fallback_reason: str | None = None
    cost_rub: Decimal | None = Decimal(0)
    tokens_in: int | None = 0
    tokens_out: int | None = 0
    prompt_version: str | None = None


class SourceLimit(ValueError):
    pass


def source_text(source: S2QuoteSource, profile: ExtractionProfile) -> str:
    if source.text is not None:
        if len(source.text) > profile.max_text_chars:
            raise SourceLimit
        return source.text
    assert source.pdf_base64 is not None
    encoded = source.pdf_base64
    if len(encoded) // 4 * 3 - (len(encoded) - len(encoded.rstrip("="))) > profile.max_pdf_bytes:
        raise SourceLimit
    data = base64.b64decode(encoded, validate=True)
    if not data.startswith(b"%PDF-"):
        return ""
    with BytesIO(data) as stream:
        reader = PdfReader(stream)
        if reader.is_encrypted:
            return ""
        length = 0
        pages: list[str] = []
        for page in reader.pages:
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > profile.max_pdf_bytes:
                raise SourceLimit
            part = page.extract_text()
            length += len(part) + bool(pages)
            if length > profile.max_text_chars:
                raise SourceLimit
            pages.append(part)
    return "\n".join(pages)


def number(raw: str | None, scale: int, reasons: set[str]) -> Decimal | None:
    if raw is None:
        return None
    if not re.fullmatch(r"[0-9]{1,18}(?:[.,][0-9]+)?", raw):
        reasons.add(INCOMPLETE)
        return None
    value = Decimal(raw.replace(",", "."))
    if len(re.split(r"[.,]", raw)[-1]) > scale and ("." in raw or "," in raw):
        reasons.add(PRECISION)
        return None
    return round_money(value, Decimal(1).scaleb(-scale))


def gross(
    amount: Decimal | None, includes_vat: bool | None, rate: Decimal | None, reasons: set[str]
) -> Decimal | None:
    if amount is None or rate is None or (includes_vat is None and rate):
        return None
    result = round_money(amount * (1 + rate / 100)) if includes_vat is False else amount
    if result >= 10**18:
        reasons.add(PRECISION)
        return None
    return result


def normalize_line(index: int, line: ExtractedLine, reasons: set[str]) -> S2QuoteItem:
    qty = number(line.quantity, 3, reasons)
    rate = number(line.vat_rate_pct, 2, reasons)
    vat = number(line.vat_amount, 2, reasons)
    vat_exempt = line.vat_treatment == "NO_VAT"
    if vat_exempt:
        if rate is not None or (vat is not None and vat != 0):
            reasons.add(MISMATCH)
        vat = vat if vat is not None else Decimal("0.00")
    elif line.vat_treatment != "TAXABLE" or (rate is not None and rate > 100):
        rate = None
    effective_rate = Decimal(0) if vat_exempt else rate
    price = gross(
        number(line.unit_price, 2, reasons), line.unit_price_includes_vat, effective_rate, reasons
    )
    total = gross(
        number(line.line_total, 2, reasons), line.line_total_includes_vat, effective_rate, reasons
    )
    arithmetic = False
    if qty and price is not None and total is not None and vat is not None:
        assert effective_rate is not None
        expected_vat = round_money(total * effective_rate / (100 + effective_rate))
        arithmetic = (
            round_money(qty * price) == total
            and vat == expected_vat
            and (not vat_exempt or rate is None)
        )
        if not arithmetic:
            reasons.add(MISMATCH)
    if not qty or None in (line.name, line.unit, price, vat, total, line.delivery_days):
        reasons.add(INCOMPLETE)
    return S2QuoteItem(
        line_no=index,
        item_code=line.item_code,
        name=line.name,
        unit=line.unit,
        quantity=qty or None,
        unit_price_rub=price,
        vat_treatment=line.vat_treatment,
        vat_rate_pct=None if vat_exempt else rate,
        vat_amount_rub=vat,
        total_rub=total,
        delivery_days=line.delivery_days,
        arithmetic_ok=arithmetic,
    )


def normalize(source: S2QuoteSource, raw: ExtractedQuote) -> S2ParsedQuote:
    reasons: set[str] = set()
    with localcontext() as context:
        context.prec = 50
        lines = [normalize_line(index, line, reasons) for index, line in enumerate(raw.items, 1)]
        total = number(raw.quoted_total, 2, reasons)
        arithmetic = bool(lines) and total is not None and all(line.arithmetic_ok for line in lines)
        if arithmetic and sum(line.total_rub or Decimal(0) for line in lines) != total:
            arithmetic = False
            reasons.add(MISMATCH)
    if not lines or total is None or raw.deferment_days is None:
        reasons.add(INCOMPLETE)
    if raw.currency != "RUB":
        reasons.add("S2_CURRENCY_UNSUPPORTED")
    return S2ParsedQuote(
        **source.model_dump(include=set(S2ParsedQuote.model_fields)),
        items=lines,
        quoted_total_rub=total,
        deferment_days=raw.deferment_days,
        arithmetic_ok=arithmetic,
        need_human=bool(reasons),
        escalation_reasons=sorted(reasons),
    )


def unreadable(source: S2QuoteSource, reason: str) -> S2ParsedQuote:
    return S2ParsedQuote(
        **source.model_dump(include=set(S2ParsedQuote.model_fields)),
        items=[],
        quoted_total_rub=None,
        deferment_days=None,
        arithmetic_ok=False,
        need_human=True,
        escalation_reasons=[reason],
    )


class QuoteParser:
    def __init__(self, gateway: LLMGateway) -> None:
        self.gateway = gateway
        path = Path(__file__).with_name("prompts") / "extract_quote.yaml"
        self.prompt = Prompt.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    async def parse(
        self, source: S2QuoteSource, profile: ExtractionProfile, budget: CycleBudget
    ) -> ParseResult:
        try:
            text = (await asyncio.to_thread(source_text, source, profile)).strip()
        except SourceLimit:
            return ParseResult(unreadable(source, "S2_SOURCE_LIMIT"))
        except (
            PyPdfError,
            ValueError,
            TypeError,
            KeyError,
            OSError,
            binascii.Error,
            RecursionError,
        ):
            text = ""
        if not text:
            return ParseResult(unreadable(source, "S2_SOURCE_UNREADABLE"))
        if text.startswith("{"):
            try:
                raw = ExtractedQuote.model_validate_json(text)
            except ValidationError:
                return ParseResult(unreadable(source, INCOMPLETE))
            return ParseResult(normalize(source, raw))

        result = await self.gateway.chat_json(
            "FAST_LLM",
            self.prompt.template + "\n" + json.dumps(text, ensure_ascii=False),
            ExtractedQuote,
            budget,
            max_output_tokens=4096,
        )
        reason = (
            "S2_PARSE_BUDGET_EXCEEDED"
            if result.fallback_reason == "budget_exceeded"
            else "S2_PARSE_LLM_UNAVAILABLE"
        )
        quote = (
            normalize(source, result.value)
            if result.value is not None
            else unreadable(source, reason)
        )
        return ParseResult(
            quote,
            result.mode,
            result.fallback_reason,
            result.cost_rub,
            result.tokens_in,
            result.tokens_out,
            f"{self.prompt.version}/{self.prompt.variant}",
        )
