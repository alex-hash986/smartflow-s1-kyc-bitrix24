import base64
import re
from datetime import date
from decimal import Decimal, localcontext
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BeforeValidator, Field, PlainSerializer, StrictBool, model_validator

from core.contracts import Mode, Nonempty, PositiveID, StrictDTO, round_money
from core.requisites import valid_inn

Q2 = Decimal("0.01")
Q3 = Decimal("0.001")
Q8 = Decimal("0.00000001")
Days = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
DocumentType = Literal["UPD", "WAYBILL", "DEFECT_ACT"]


def fixed(value: object, unit: Decimal, *, signed: bool = False) -> Decimal:
    exponent = unit.as_tuple().exponent
    assert isinstance(exponent, int)
    scale = -exponent
    if isinstance(value, str):
        pattern = rf"-?[0-9]{{1,18}}(?:\.[0-9]{{1,{scale}}})?"
        if not re.fullmatch(pattern, value.strip()):
            raise ValueError("Invalid decimal string")
        value = Decimal(value.strip())
    if (
        not isinstance(value, Decimal)
        or not value.is_finite()
        or value.copy_abs() >= 10**18
        or not isinstance(value.as_tuple().exponent, int)
        or (not signed and value.is_signed())
    ):
        raise ValueError("Decimal outside supported precision or range")
    precision = value.as_tuple().exponent
    assert isinstance(precision, int)
    if precision < -scale:
        raise ValueError("Excess fractional digits")
    return round_money(value, unit)


def as_money(value: object) -> Decimal:
    return fixed(value, Q2)


def as_quantity(value: object) -> Decimal:
    return fixed(value, Q3)


def as_signed_money(value: object) -> Decimal:
    return fixed(value, Q2, signed=True)


def as_cost(value: object) -> Decimal:
    return fixed(value, Q8)


def as_date(value: object) -> date:
    if type(value) is date:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value.strip()):
        return date.fromisoformat(value.strip())
    raise ValueError("Expected YYYY-MM-DD")


def as_inn(value: object) -> str:
    if isinstance(value, str) and valid_inn(value.strip()):
        return value.strip()
    raise ValueError("Invalid INN")


Money = Annotated[Decimal, BeforeValidator(as_money)]
Quantity = Annotated[Decimal, BeforeValidator(as_quantity)]
SignedMoney = Annotated[Decimal, BeforeValidator(as_signed_money)]
Cost = Annotated[
    Decimal, BeforeValidator(as_cost), PlainSerializer(lambda v: format(v, ".8f"), when_used="json")
]
DocDate = Annotated[date, BeforeValidator(as_date)]
INN = Annotated[str, BeforeValidator(as_inn)]


class Escalation(StrictDTO):
    need_human: StrictBool
    escalation_reasons: list[Nonempty]

    @model_validator(mode="after")
    def check_reasons(self) -> Self:
        if self.need_human != bool(self.escalation_reasons) or self.escalation_reasons != sorted(
            set(self.escalation_reasons)
        ):
            raise ValueError("Expected consistent, unique and sorted escalation reasons")
        return self


class S3Context(StrictDTO):
    tenant_id: Nonempty
    request_id: UUID
    profile_slug: Nonempty
    profile_revision: Nonempty


class S3AwardedItem(StrictDTO):
    item_code: Nonempty
    name: Nonempty
    unit: Nonempty
    quantity: Annotated[Quantity, Field(gt=0)]
    unit_price_rub: Money
    line_total_rub: Money

    @model_validator(mode="after")
    def check_amount(self) -> Self:
        with localcontext() as ctx:
            ctx.prec = 50
            if round_money(self.quantity * self.unit_price_rub) != self.line_total_rub:
                raise ValueError("Awarded line amount mismatch")
        return self


class S3TenderAwardSnapshot(StrictDTO):
    s2_item_id: PositiveID
    s1_item_id: PositiveID
    tender_status: Literal["AWARDED"]
    winner_inn: INN
    winner_name: Nonempty
    awarded_total_rub: Money
    promised_delivery_days: Days
    promised_deferment_days: Days
    approved_credit_limit_rub: Money
    awarded_items: Annotated[list[S3AwardedItem], Field(min_length=1)]

    @model_validator(mode="after")
    def check_items(self) -> Self:
        codes = {item.item_code for item in self.awarded_items}
        with localcontext() as ctx:
            ctx.prec = 50
            total = sum(item.line_total_rub for item in self.awarded_items)
        if len(codes) != len(self.awarded_items) or total != self.awarded_total_rub:
            raise ValueError("Invalid awarded items or total")
        return self


class DocumentIdentity(StrictDTO):
    document_id: UUID
    doc_type: DocumentType
    doc_number: Nonempty
    doc_date: DocDate
    actual_delivery_days: Days


class S3DeliveryDocumentSource(DocumentIdentity):
    source_type: Literal["TEXT", "PDF"]
    text: Nonempty | None
    pdf_base64: Nonempty | None

    @model_validator(mode="after")
    def check_source(self) -> Self:
        if self.source_type == "TEXT":
            if self.text is None or self.pdf_base64 is not None:
                raise ValueError("Expected only TEXT")
        else:
            if self.text is not None or self.pdf_base64 is None:
                raise ValueError("Expected only PDF")
            base64.b64decode(self.pdf_base64, validate=True)
        return self


class S3IntakeRequest(S3Context):
    delivery_key: Nonempty
    s2_item_id: PositiveID
    documents: Annotated[list[S3DeliveryDocumentSource], Field(min_length=1)]

    @model_validator(mode="after")
    def check_ids(self) -> Self:
        if len({doc.document_id for doc in self.documents}) != len(self.documents):
            raise ValueError("Repeated document ID")
        return self


class S3IntakeReceipt(S3Context):
    package_id: UUID
    s2_item_id: PositiveID


class S3TenderGateRequest(S3Context):
    package_id: UUID


class S3TenderGateResponse(S3TenderGateRequest, Escalation):
    award: S3TenderAwardSnapshot | None

    @model_validator(mode="after")
    def check_award(self) -> Self:
        if (self.award is None) != self.need_human:
            raise ValueError("Invalid gate outcome")
        return self


class S3ParseDocsRequest(StrictDTO):
    package_id: UUID


class S3ParseCall(S3Context):
    payload: S3ParseDocsRequest


class S3DeliveryLine(StrictDTO):
    line_no: PositiveID
    item_code: Nonempty | None
    name: Nonempty | None
    unit: Nonempty | None
    quantity: Quantity | None
    defect_quantity: Quantity | None
    unit_price_rub: Money | None
    line_total_rub: Money | None
    arithmetic_ok: StrictBool
    escalation_reasons: list[Nonempty]

    @model_validator(mode="after")
    def check_line(self) -> Self:
        expected = False
        if self.quantity is not None and self.unit_price_rub is not None:
            with localcontext() as ctx:
                ctx.prec = 50
                expected = round_money(self.quantity * self.unit_price_rub) == self.line_total_rub
        if self.arithmetic_ok and not expected:
            raise ValueError("Line amount mismatch")
        if self.escalation_reasons != sorted(set(self.escalation_reasons)):
            raise ValueError("Invalid line reasons")
        return self


class S3ParsedDeliveryDocument(DocumentIdentity, Escalation):
    supplier_inn: INN | None
    referenced_delivery_document_id: UUID | None
    lines: list[S3DeliveryLine]
    arithmetic_ok: StrictBool

    @model_validator(mode="after")
    def check_lines(self) -> Self:
        if len({line.line_no for line in self.lines}) != len(self.lines):
            raise ValueError("Repeated line number")
        if self.arithmetic_ok != (bool(self.lines) and all(x.arithmetic_ok for x in self.lines)):
            raise ValueError("Invalid document arithmetic")
        if not {reason for line in self.lines for reason in line.escalation_reasons} <= set(
            self.escalation_reasons
        ):
            raise ValueError("Missing line reasons")
        if self.doc_type != "DEFECT_ACT" and (
            self.referenced_delivery_document_id is not None
            or any(line.defect_quantity is not None for line in self.lines)
        ):
            raise ValueError("Unexpected defect data on delivery document")
        return self


class S3ParseDocsResponse(Escalation):
    package_id: UUID
    documents: list[S3ParsedDeliveryDocument]

    @model_validator(mode="after")
    def check_documents(self) -> Self:
        reasons = {reason for doc in self.documents for reason in doc.escalation_reasons}
        if self.escalation_reasons != sorted(reasons):
            raise ValueError("Invalid package reasons")
        if len({doc.document_id for doc in self.documents}) != len(self.documents):
            raise ValueError("Repeated document")
        return self


class S3ParseReply(S3Context):
    payload: S3ParseDocsResponse
    mode: Mode
    fallback_reason: Nonempty | None
    cost_rub: Cost | None
    prompt_version: Nonempty | None
    prompt_variant: Nonempty | None


class S3EvaluateClaimsRequest(StrictDTO):
    package_id: UUID
    award: S3TenderAwardSnapshot | None
    documents: list[S3ParsedDeliveryDocument]


class S3ItemDiscrepancy(StrictDTO):
    item_code: Nonempty
    unit: Nonempty
    awarded_quantity: Quantity
    delivered_quantity: Quantity
    defect_quantity: Quantity
    accepted_quantity: Quantity
    shortage_quantity: Quantity
    excess_quantity: Quantity
    awarded_unit_price_rub: Money
    actual_unit_price_rub: Money | None
    accepted_amount_rub: Money
    shortage_or_defect_rub: Money
    price_variance_rub: SignedMoney
    quantity_outside_tolerance: StrictBool
    price_outside_tolerance: StrictBool


class S3EvaluateClaimsResponse(StrictDTO):
    package_id: UUID
    outcome: Literal["ACCEPTED", "CLAIM_REQUIRED", "ESCALATED"]
    accepted_amount_rub: Money | None
    shortage_or_defect_rub: Money | None
    price_variance_rub: SignedMoney | None
    delay_days: Days | None
    penalty_rub: Money | None
    claim_amount_rub: Money | None
    discrepancies: list[S3ItemDiscrepancy]
    reconciliation_report_markdown: Nonempty

    @model_validator(mode="after")
    def check_amounts(self) -> Self:
        values = (
            self.accepted_amount_rub,
            self.shortage_or_defect_rub,
            self.price_variance_rub,
            self.delay_days,
            self.penalty_rub,
            self.claim_amount_rub,
        )
        if any(value is None for value in values) and (
            any(value is not None for value in values)
            or self.outcome != "ESCALATED"
            or self.discrepancies
        ):
            raise ValueError("Incomplete calculation")
        return self


class S3PublicationRequest(S3TenderGateRequest):
    attempt: PositiveID
