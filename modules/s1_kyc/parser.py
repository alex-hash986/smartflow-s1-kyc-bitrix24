import asyncio
import base64
import re
from dataclasses import dataclass
from decimal import Decimal
from io import BytesIO
from pathlib import Path

import yaml
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from core.contracts import Mode, Nonempty, StrictDTO
from core.llm_gateway import CycleBudget, LLMGateway
from core.profile_loader import ExtractionProfile
from modules.s1_kyc.contracts import ExtractedFields, S1ParseRequest, S1ParseResponse
from modules.s1_kyc.requisites import valid_inn, valid_kpp, valid_ogrn, valid_ogrnip

LABELS = {
    "inn": r"ИНН|INN",
    "ogrn": r"ОГРНИП|ОГРН|OGRNIP|OGRN",
    "kpp": r"КПП|KPP",
    "legal_name": r"(?:полное\s+)?наименование(?:\s+организации)?|legal name",
    "requested_limit_rub": r"(?:запрашиваемый\s+|кредитный\s+)?лимит|requested limit",
}
AMOUNT = r"[0-9]+(?:[ \u00a0\u202f][0-9]{3})*(?:[,.][0-9]{1,2})?"
ZERO = Decimal(0)


class Prompt(StrictDTO):
    version: Nonempty
    variant: Nonempty
    instruction: Nonempty


@dataclass(frozen=True)
class ParseResult:
    payload: S1ParseResponse
    mode: Mode = "deterministic"
    fallback_reason: str | None = None
    cost_rub: Decimal | None = ZERO
    tokens_in: int | None = 0
    tokens_out: int | None = 0
    prompt_version: str | None = None


class SourceLimit(ValueError):
    pass


def _source(request: S1ParseRequest, profile: ExtractionProfile) -> str:
    if request.text is not None:
        if len(request.text) > profile.max_text_chars:
            raise SourceLimit
        return request.text
    assert request.pdf_base64 is not None
    encoded = request.pdf_base64
    if len(encoded) // 4 * 3 - (len(encoded) - len(encoded.rstrip("="))) > profile.max_pdf_bytes:
        raise SourceLimit
    data = base64.b64decode(encoded, validate=True)
    if not data.startswith(b"%PDF-"):
        return ""
    with BytesIO(data) as stream:
        reader = PdfReader(stream)
        if reader.is_encrypted:
            return ""
        size = 0
        pages: list[str] = []
        for page in reader.pages:
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > profile.max_pdf_bytes:
                raise SourceLimit
            text = page.extract_text()
            size += len(text) + bool(pages)
            if size > profile.max_text_chars:
                raise SourceLimit
            pages.append(text)
    return "\n".join(pages)


def _amount(raw: str) -> Decimal | None:
    match = re.fullmatch(rf"({AMOUNT})(?:\s*(?:₽|руб\.?|RUB))?", raw, re.IGNORECASE)
    if match is None:
        return None
    value = Decimal(re.sub(r"\s+", "", match[1]).replace(",", "."))
    return value if value > 0 else None


def _candidates(text: str) -> tuple[dict[str, object], set[str]]:
    values: dict[str, object] = dict.fromkeys(LABELS)
    reasons: set[str] = set()
    for field, labels in LABELS.items():
        # Stop at the next field label, a newline, or a semicolon.
        boundary = "|".join(LABELS.values())
        pattern = rf"(?<!\w)(?:{labels})[ \t]*[:=][ \t]*(.*?)(?=\s+(?:{boundary})\s*[:=]|[;\r\n]|$)"
        matches = {value.strip() for value in re.findall(pattern, text, re.IGNORECASE)}
        if len(matches) > 1:
            reasons.add("PARSE_AMBIGUOUS_FIELD")
        elif matches:
            raw = matches.pop()
            value = _amount(raw) if field == "requested_limit_rub" else raw or None
            if value is None and raw:
                reasons.add("PARSE_INVALID_FIELD")
            values[field] = value
    return values, reasons


def _checked(values: dict[str, object], reasons: set[str]) -> S1ParseResponse:
    inn, ogrn, kpp = values["inn"], values["ogrn"], values["kpp"]
    entrepreneur = (isinstance(inn, str) and len(inn) == 12) or (
        isinstance(ogrn, str) and len(ogrn) == 15
    )
    mismatched = (
        isinstance(inn, str)
        and isinstance(ogrn, str)
        and ((len(inn), len(ogrn)) not in ((10, 13), (12, 15)))
    )
    invalid = {
        "inn": isinstance(inn, str) and (not valid_inn(inn) or mismatched),
        "ogrn": isinstance(ogrn, str)
        and (not (valid_ogrn(ogrn) or valid_ogrnip(ogrn)) or mismatched),
        "kpp": isinstance(kpp, str) and (not valid_kpp(kpp) or entrepreneur),
    }
    if any(invalid.values()):
        reasons.add("PARSE_INVALID_FIELD")
    fields = {key: None if invalid.get(key) else value for key, value in values.items()}
    if any(value is None for key, value in fields.items() if key != "kpp" or not entrepreneur):
        reasons.add("PARSE_MISSING_FIELD")
    return S1ParseResponse.model_validate(
        {
            **fields,
            "need_human": bool(reasons),
            "escalation_reasons": sorted(reasons),
        }
    )


def _grounded(value: object, source: str) -> bool:
    if isinstance(value, Decimal):
        return any(
            _amount(match[0]) == value
            for match in re.finditer(rf"(?<![\w.,-]){AMOUNT}(?![\w.,])", source)
        )
    if isinstance(value, str):
        return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", source, re.IGNORECASE) is not None
    return False


class RequisitesParser:
    def __init__(self, gateway: LLMGateway) -> None:
        self.gateway = gateway
        path = Path(__file__).parent / "prompts" / "extract_requisites.yaml"
        self.prompt = Prompt.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    async def parse(
        self,
        request: S1ParseRequest,
        profile: ExtractionProfile,
        budget: CycleBudget,
    ) -> ParseResult:
        try:
            source = await asyncio.to_thread(_source, request, profile)
        except SourceLimit:
            return ParseResult(_checked(dict.fromkeys(LABELS), {"PARSE_SOURCE_LIMIT"}))
        except (PyPdfError, ValueError, RecursionError, OverflowError):
            source = ""
        if not source.strip():
            return ParseResult(_checked(dict.fromkeys(LABELS), {"PARSE_SOURCE_UNREADABLE"}))
        candidates, issues = _candidates(source)
        initial = _checked(candidates, issues.copy())
        if not initial.need_human or issues or "PARSE_INVALID_FIELD" in initial.escalation_reasons:
            return ParseResult(initial)
        raw = await self.gateway.chat_json(
            "FAST_LLM", self.prompt.instruction + "\n\n" + source, ExtractedFields, budget
        )
        reason = raw.fallback_reason
        mode = raw.mode
        if reason == "budget_exceeded" or (
            raw.cost_rub is not None
            and raw.cost_rub > min(budget.limit, self.gateway.settings.max_cycle_cost_rub)
        ):
            mode, reason = "fallback", "budget_exceeded"
            issues.add("PARSE_BUDGET_EXCEEDED")
        elif raw.value is not None:
            for field, value in raw.value.model_dump().items():
                if candidates[field] is None and _grounded(value, source):
                    candidates[field] = value
        elif reason == "input_limit":
            issues.add("PARSE_SOURCE_LIMIT")
        else:
            issues.add("PARSE_LLM_UNAVAILABLE")
        return ParseResult(
            _checked(candidates, issues),
            mode,
            reason,
            raw.cost_rub,
            raw.tokens_in,
            raw.tokens_out,
            f"{self.prompt.version}/{self.prompt.variant}",
        )
