import re
from collections.abc import Sequence
from decimal import Decimal
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AfterValidator, BeforeValidator, Field, StrictBool, StrictStr, model_validator

from core.contracts import Nonempty, PositiveID, StrictDTO, round_money
from core.requisites import valid_inn, valid_ogrn, valid_ogrnip

QTY = Decimal("0.001")
SCORE = Decimal("0.0001")
Days = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
ReasonCodes = Annotated[list[Nonempty], Field(strict=True)]
Criterion = Literal["PRICE", "DELIVERY", "S1_RELIABILITY", "DEFERMENT"]
KYCStatus = Literal["APPROVED", "REJECTED"]


def decimal_input(value: object) -> Decimal:
    if isinstance(value, Decimal) and value.is_finite():
        return value
    if isinstance(value, str) and re.fullmatch(r"-?[0-9]{1,18}(?:\.[0-9]{1,4})?", value.strip()):
        return Decimal(value.strip())
    raise ValueError("Expected a finite decimal string")


def nonnegative_input(value: object) -> Decimal:
    amount = decimal_input(value)
    if amount.is_signed():
        raise ValueError("Unexpected sign for nonnegative value")
    return amount


def quantized(value: Decimal, unit: Decimal) -> Decimal:
    exponent = value.as_tuple().exponent
    scale = unit.as_tuple().exponent
    assert isinstance(scale, int)
    if not isinstance(exponent, int) or exponent < scale or value.copy_abs() >= 10**18:
        raise ValueError("Decimal outside supported precision")
    return round_money(value, unit)


def money(value: Decimal) -> Decimal:
    return quantized(value, Decimal("0.01"))


def quantity(value: Decimal) -> Decimal:
    return quantized(value, QTY)


def score(value: Decimal) -> Decimal:
    return quantized(value, SCORE)


def inn(value: str) -> str:
    if not valid_inn(value):
        raise ValueError("Invalid INN")
    return value


Money = Annotated[Decimal, BeforeValidator(nonnegative_input), Field(ge=0), AfterValidator(money)]
Savings = Annotated[Decimal, BeforeValidator(decimal_input), AfterValidator(money)]
Quantity = Annotated[
    Decimal, BeforeValidator(nonnegative_input), Field(gt=0), AfterValidator(quantity)
]
Percent = Annotated[Money, Field(le=100)]
Score = Annotated[
    Decimal, BeforeValidator(nonnegative_input), Field(ge=0, le=100), AfterValidator(score)
]
INN = Annotated[Nonempty, AfterValidator(inn)]


def require_unique(values: Sequence[str | int | UUID], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {label}")


class Escalation(StrictDTO):
    need_human: StrictBool
    escalation_reasons: ReasonCodes

    @model_validator(mode="after")
    def check_reasons(self) -> Self:
        if self.need_human != bool(self.escalation_reasons) or self.escalation_reasons != sorted(
            set(self.escalation_reasons)
        ):
            raise ValueError("Expected consistent, unique and sorted escalation reasons")
        return self


class S2Context(StrictDTO):
    tenant_id: Nonempty
    request_id: UUID
    profile_slug: Nonempty
    profile_revision: Nonempty


class S2TenderItem(StrictDTO):
    item_code: Nonempty
    name: Nonempty
    unit: Nonempty
    quantity: Quantity
    target_unit_price_rub: Money | None


class S2SupplierGateSnapshot(StrictDTO):
    s1_item_id: PositiveID
    inn: INN
    ogrn: Nonempty
    kyc_status: KYCStatus
    reliability_score: Percent
    approved_credit_limit_rub: Money
    stage_result_published: StrictBool

    @model_validator(mode="after")
    def check_identity_and_limit(self) -> Self:
        valid = valid_ogrn if len(self.inn) == 10 else valid_ogrnip
        if not valid(self.ogrn) or (
            self.kyc_status == "REJECTED" and self.approved_credit_limit_rub != 0
        ):
            raise ValueError("Invalid OGRN or credit limit")
        return self


class QuoteIdentity(StrictDTO):
    quote_id: UUID
    supplier_inn: INN
    supplier_name: Nonempty
    s1_item_id: PositiveID


class S2QuoteSource(QuoteIdentity):
    source_type: Literal["TEXT", "PDF"]
    text: StrictStr | None
    pdf_base64: StrictStr | None

    @model_validator(mode="after")
    def check_source(self) -> Self:
        if self.source_type == "TEXT":
            if self.text is None or self.pdf_base64 is not None:
                raise ValueError("Expected only text")
        elif (
            self.text is not None
            or self.pdf_base64 is None
            or not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", self.pdf_base64)
            or len(self.pdf_base64) % 4
        ):
            raise ValueError("Expected only base64 PDF")
        return self


class S2ParseQuotesRequest(StrictDTO):
    quotes: Annotated[list[S2QuoteSource], Field(strict=True, min_length=1)]

    @model_validator(mode="after")
    def check_ids(self) -> Self:
        require_unique([quote.quote_id for quote in self.quotes], "quote ID")
        return self


class S2QuoteItem(StrictDTO):
    line_no: PositiveID
    item_code: Nonempty | None
    name: Nonempty | None
    unit: Nonempty | None
    quantity: Quantity | None
    unit_price_rub: Money | None
    vat_treatment: Literal["TAXABLE", "NO_VAT"] | None
    vat_rate_pct: Percent | None
    vat_amount_rub: Money | None
    total_rub: Money | None
    delivery_days: Days | None
    arithmetic_ok: StrictBool


class S2ParsedQuote(QuoteIdentity, Escalation):
    items: Annotated[list[S2QuoteItem], Field(strict=True)]
    quoted_total_rub: Money | None
    deferment_days: Days | None
    arithmetic_ok: StrictBool

    @model_validator(mode="after")
    def check_lines(self) -> Self:
        require_unique([line.line_no for line in self.items], "line number")
        if not self.arithmetic_ok and not self.need_human:
            raise ValueError("Unverified arithmetic requires review")
        return self


class S2ParseQuotesResponse(Escalation):
    quotes: Annotated[list[S2ParsedQuote], Field(strict=True, min_length=1)]

    @model_validator(mode="after")
    def check_batch(self) -> Self:
        require_unique([quote.quote_id for quote in self.quotes], "quote ID")
        if self.escalation_reasons != sorted(
            {reason for quote in self.quotes for reason in quote.escalation_reasons}
        ):
            raise ValueError("Batch reasons must match quote reasons")
        return self


class S2ParseCall(S2Context):
    payload: S2ParseQuotesRequest


class S2TenderSpec(StrictDTO):
    items: Annotated[list[S2TenderItem], Field(strict=True, min_length=1)]
    target_budget_rub: Annotated[Money, Field(gt=0)]
    required_deferment_days: Days

    @model_validator(mode="after")
    def check_item_codes(self) -> Self:
        require_unique([item.item_code for item in self.items], "item code")
        return self


class S2EvaluateTenderRequest(S2TenderSpec):
    supplier_gate_snapshots: Annotated[list[S2SupplierGateSnapshot], Field(strict=True)]
    quotes: Annotated[list[S2ParsedQuote], Field(strict=True)]

    @model_validator(mode="after")
    def check_keys(self) -> Self:
        require_unique([gate.inn for gate in self.supplier_gate_snapshots], "snapshot INN")
        require_unique([gate.s1_item_id for gate in self.supplier_gate_snapshots], "snapshot ID")
        require_unique([quote.quote_id for quote in self.quotes], "quote ID")
        require_unique([quote.supplier_inn for quote in self.quotes], "quote INN")
        return self


class S2ScorePenalty(StrictDTO):
    criterion: Criterion
    points: Annotated[Score, Field(gt=0)]


class S2TenderParticipant(QuoteIdentity):
    rank: PositiveID | None
    score: Score | None
    total_rub: Money | None
    delivery_days: Days | None
    deferment_days: Days | None
    rejection_reasons: ReasonCodes
    penalties: Annotated[list[S2ScorePenalty], Field(strict=True)]

    @model_validator(mode="after")
    def check_rank(self) -> Self:
        if self.rejection_reasons != sorted(set(self.rejection_reasons)):
            raise ValueError("Rejection reasons must be sorted and unique")
        if self.rejection_reasons:
            if self.rank is not None or self.score is not None or self.penalties:
                raise ValueError("Rejected participant cannot be ranked")
        elif None in (
            self.rank,
            self.score,
            self.total_rub,
            self.delivery_days,
            self.deferment_days,
        ):
            raise ValueError("Ranked participant requires complete metrics")
        require_unique([penalty.criterion for penalty in self.penalties], "penalty criterion")
        return self


class S2EvaluateTenderResponse(StrictDTO):
    winner_inn: INN | None
    winner_s1_item_id: PositiveID | None
    winner_total_rub: Money | None
    savings_rub: Savings | None
    participants: Annotated[list[S2TenderParticipant], Field(strict=True)]
    comparison_matrix_markdown: Nonempty

    @model_validator(mode="after")
    def check_winner(self) -> Self:
        winner = (self.winner_inn, self.winner_s1_item_id, self.winner_total_rub)
        if all(value is None for value in (*winner, self.savings_rub)):
            return self
        first = self.participants[0] if self.participants else None
        if (
            first is None
            or first.rank != 1
            or self.savings_rub is None
            or winner != (first.supplier_inn, first.s1_item_id, first.total_rub)
        ):
            raise ValueError("Winner must match first-ranked participant")
        return self
