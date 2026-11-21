import asyncio
import base64
import json
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
from core.requisites import valid_inn
from modules.s3_claims.contracts import (
    Q2,
    Q3,
    Cost,
    S3DeliveryDocumentSource,
    S3DeliveryLine,
    S3ParsedDeliveryDocument,
    fixed,
)

INCOMPLETE = "S3_DOCUMENT_INCOMPLETE"


class Prompt(StrictDTO):
    version: Nonempty
    variant: Nonempty
    system: Nonempty
    user: Nonempty


class ExtractedLine(StrictDTO):
    item_code: Nonempty | None = None
    name: Nonempty | None = None
    unit: Nonempty | None = None
    quantity: StrictStr | None = None
    defect_quantity: StrictStr | None = None
    unit_price_rub: StrictStr | None = None
    line_total_rub: StrictStr | None = None
    vat_treatment: Literal["GROSS", "NO_VAT", "TAXABLE"] | None = None
    vat_rate_pct: StrictStr | None = None
    unit_price_includes_vat: StrictBool | None = None
    line_total_includes_vat: StrictBool | None = None


class ExtractedDocument(StrictDTO):
    doc_number: Nonempty | None = None
    doc_date: StrictStr | None = None
    currency: Nonempty | None = None
    supplier_inn: StrictStr | None = None
    reference_number: Nonempty | None = None
    reference_date: StrictStr | None = None
    lines: list[ExtractedLine] = Field(default_factory=list)


class ParseRecord(StrictDTO):
    tokens_in: int | None = 0
    tokens_out: int | None = 0
    document: S3ParsedDeliveryDocument
    mode: Mode = "deterministic"
    fallback_reason: Nonempty | None = None
    cost_rub: Cost | None = Decimal("0.00000000")
    prompt_version: Nonempty | None = None
    prompt_variant: Nonempty | None = None


class SourceLimit(ValueError):
    pass


def source_text(source: S3DeliveryDocumentSource, limits: ExtractionProfile, pages: int) -> str:
    if source.text is not None:
        if len(source.text) > limits.max_text_chars:
            raise SourceLimit
        return source.text
    assert source.pdf_base64 is not None
    data = base64.b64decode(source.pdf_base64, validate=True)
    if len(data) > limits.max_pdf_bytes:
        raise SourceLimit
    if not data.startswith(b"%PDF-"):
        return ""
    with BytesIO(data) as stream:
        reader = PdfReader(stream)
        if reader.is_encrypted:
            return ""
        if len(reader.pages) > pages:
            raise SourceLimit
        parts: list[str] = []
        size = 0
        for page in reader.pages:
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > limits.max_pdf_bytes:
                raise SourceLimit
            part = page.extract_text()
            size += len(part) + bool(parts)
            if size > limits.max_text_chars:
                raise SourceLimit
            parts.append(part)
        return "\n".join(parts)


def number(value: str | None, unit: Decimal, reasons: set[str]) -> Decimal | None:
    if value is None:
        reasons.add(INCOMPLETE)
        return None
    try:
        return fixed(value.replace(",", "."), unit)
    except ValueError:
        reasons.add("S3_UNSUPPORTED_PRECISION")
        return None


def normalize_line(index: int, raw: ExtractedLine, defect: bool) -> S3DeliveryLine:
    reasons: set[str] = set()
    quantity = number(raw.quantity, Q3, reasons)
    defective = number(raw.defect_quantity, Q3, reasons) if defect else None
    price = number(raw.unit_price_rub, Q2, reasons)
    total = number(raw.line_total_rub, Q2, reasons)
    if (
        raw.unit is None
        or (raw.item_code is None and raw.name is None)
        or raw.vat_treatment is None
    ):
        reasons.add(INCOMPLETE)
    tax_confirmed = raw.vat_treatment in ("GROSS", "NO_VAT")
    if raw.vat_treatment == "TAXABLE":
        rate = number(raw.vat_rate_pct, Q2, reasons)
        if (
            rate is None
            or rate > 100
            or price is None
            or total is None
            or raw.unit_price_includes_vat is None
            or raw.line_total_includes_vat is None
        ):
            reasons.add(INCOMPLETE)
            price, total = None, None
        else:
            tax_confirmed = True
            factor = 1 + rate / 100
            price = price if raw.unit_price_includes_vat else round_money(price * factor)
            total = total if raw.line_total_includes_vat else round_money(total * factor)
            if price >= 10**18 or total >= 10**18:
                reasons.add("S3_UNSUPPORTED_PRECISION")
                price, total = None, None
    if not defect and raw.defect_quantity is not None:
        reasons.add(INCOMPLETE)
    if defective is not None and quantity is not None and defective > quantity:
        reasons.add("S3_DEFECT_CONFLICT")
    arithmetic = False
    if quantity is not None and price is not None and total is not None and tax_confirmed:
        with localcontext() as ctx:
            ctx.prec = 50
            arithmetic = round_money(quantity * price) == total
        if not arithmetic:
            reasons.add("S3_ARITHMETIC_MISMATCH")
    return S3DeliveryLine(
        line_no=index,
        item_code=raw.item_code,
        name=raw.name,
        unit=raw.unit,
        quantity=quantity,
        defect_quantity=defective,
        unit_price_rub=price,
        line_total_rub=total,
        arithmetic_ok=arithmetic,
        escalation_reasons=sorted(reasons),
    )


def normalize(
    source: S3DeliveryDocumentSource,
    raw: ExtractedDocument | None,
    sources: list[S3DeliveryDocumentSource],
    reason: str | None = None,
) -> S3ParsedDeliveryDocument:
    reasons = {reason} if reason else set()
    lines, inn, reference = [], None, None
    if raw is not None:
        defect = source.doc_type == "DEFECT_ACT"
        lines = [normalize_line(i, line, defect) for i, line in enumerate(raw.lines, 1)]
        inn = raw.supplier_inn if raw.supplier_inn and valid_inn(raw.supplier_inn) else None
        if raw.doc_number != source.doc_number or raw.doc_date != str(source.doc_date):
            reasons.add(INCOMPLETE)
        if not lines or inn is None:
            reasons.add(INCOMPLETE)
        if raw.currency != "RUB":
            reasons.add("S3_CURRENCY_UNSUPPORTED")
        if defect:
            candidates = [
                doc.document_id
                for doc in sources
                if doc.doc_type != "DEFECT_ACT"
                and doc.doc_number == raw.reference_number
                and str(doc.doc_date) == raw.reference_date
            ]
            if len(candidates) == 1:
                reference = candidates[0]
            else:
                reasons.add("S3_DEFECT_CONFLICT")
        for line in lines:
            reasons.update(line.escalation_reasons)
    return S3ParsedDeliveryDocument(
        **source.model_dump(
            include={"document_id", "doc_type", "doc_number", "doc_date", "actual_delivery_days"}
        ),
        supplier_inn=inn,
        referenced_delivery_document_id=reference,
        lines=lines,
        arithmetic_ok=bool(lines) and all(line.arithmetic_ok for line in lines),
        need_human=bool(reasons),
        escalation_reasons=sorted(reasons),
    )


class AcceptanceParser:
    def __init__(self, gateway: LLMGateway) -> None:
        self.gateway = gateway
        path = Path(__file__).with_name("prompts") / "extract_acceptance.yaml"
        self.prompt = Prompt.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    async def parse(
        self,
        source: S3DeliveryDocumentSource,
        sources: list[S3DeliveryDocumentSource],
        limits: ExtractionProfile,
        pages: int,
        budget: CycleBudget,
    ) -> ParseRecord:
        reason = "S3_SOURCE_UNREADABLE"
        try:
            text = (await asyncio.to_thread(source_text, source, limits, pages)).strip()
        except SourceLimit:
            text, reason = "", "S3_SOURCE_LIMIT"
        except (PyPdfError, ValueError, TypeError, KeyError, OSError, RecursionError):
            text = ""
        if not text:
            return ParseRecord(document=normalize(source, None, sources, reason))
        if text.startswith("{"):
            try:
                raw = ExtractedDocument.model_validate_json(text)
            except ValidationError:
                return ParseRecord(document=normalize(source, None, sources, INCOMPLETE))
            return ParseRecord(document=normalize(source, raw, sources))
        result = await self.gateway.chat_json(
            "FAST_LLM",
            self.prompt.system
            + "\n"
            + self.prompt.user
            + "\n"
            + json.dumps(text, ensure_ascii=False),
            ExtractedDocument,
            budget,
            max_output_tokens=4096,
            system_prompt=self.prompt.system,
        )
        reason = (
            "S3_PARSE_BUDGET_EXCEEDED"
            if result.fallback_reason == "budget_exceeded"
            else ("S3_PARSE_LLM_UNAVAILABLE")
        )
        return ParseRecord(
            document=normalize(
                source, result.value, sources, reason if result.value is None else None
            ),
            mode=result.mode,
            fallback_reason=result.fallback_reason,
            cost_rub=result.cost_rub,
            prompt_version=self.prompt.version,
            prompt_variant=self.prompt.variant,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
        )
