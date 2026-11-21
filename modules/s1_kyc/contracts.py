import base64
import binascii
import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BeforeValidator,
    Field,
    StrictBool,
    StrictStr,
    field_serializer,
    model_validator,
)

from core.contracts import (
    Nonempty,
    PositiveID,
    StrictDTO,
    WorkerRequestEnvelope,
    WorkerResponseEnvelope,
    round_money,
)
from core.profile_loader import Risk
from modules.s1_kyc.requisites import valid_inn, valid_kpp, valid_ogrn, valid_ogrnip


def wire_money(value: object) -> Decimal:
    if isinstance(value, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]{1,2})?", value.strip()):
        value = Decimal(value.strip())
    if not isinstance(value, Decimal) or not value.is_finite() or value.is_signed():
        raise ValueError("Expected nonnegative decimal string")
    exponent = value.as_tuple().exponent
    if isinstance(exponent, int) and exponent < -2:
        raise ValueError("Expected at most two fractional digits")
    return round_money(value)


Money = Annotated[Decimal, BeforeValidator(wire_money)]
PositiveMoney = Annotated[Money, Field(gt=0)]
Score = Annotated[Money, Field(le=100)]
EntityStatus = Literal["ACTIVE", "IN_LIQUIDATION", "TERMINATED"]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
Codes = Annotated[list[Nonempty], Field(strict=True)]


class S1Context(StrictDTO):
    tenant_id: Nonempty
    request_id: UUID
    profile_slug: Nonempty
    profile_revision: Nonempty


class S1ParseRequest(StrictDTO):
    source_type: Literal["TEXT", "PDF"]
    text: StrictStr | None
    pdf_base64: Nonempty | None

    @model_validator(mode="after")
    def check_source(self) -> Self:
        if (self.source_type == "TEXT") != (self.text is not None) or (
            self.source_type == "PDF"
        ) != (self.pdf_base64 is not None):
            raise ValueError("Provide exactly one source matching source_type")
        if self.pdf_base64 is not None:
            encoded = self.pdf_base64
            if re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", encoded) is None or len(encoded) % 4:
                raise ValueError("Invalid base64 PDF")
            tail = encoded[-4:]
            try:
                if base64.b64encode(base64.b64decode(tail, validate=True)).decode() != tail:
                    raise ValueError("Noncanonical base64 PDF")
            except binascii.Error as exc:
                raise ValueError("Invalid base64 PDF") from exc
        return self


def _valid_pair(inn: str, ogrn: str) -> bool:
    return (len(inn), len(ogrn)) in ((10, 13), (12, 15))


class ExtractedFields(StrictDTO):
    inn: Nonempty | None
    ogrn: Nonempty | None
    kpp: Nonempty | None
    legal_name: Nonempty | None
    requested_limit_rub: PositiveMoney | None


def _codes(values: list[str]) -> list[str]:
    if values != sorted(set(values)):
        raise ValueError("Expected unique sorted reason codes")
    return values


class Escalation(StrictDTO):
    need_human: StrictBool
    escalation_reasons: Annotated[Codes, AfterValidator(_codes)]

    @model_validator(mode="after")
    def check_escalation(self) -> Self:
        if self.need_human != bool(self.escalation_reasons):
            raise ValueError("Escalation requires reason codes")
        return self


class S1ParseResponse(ExtractedFields, Escalation):
    @model_validator(mode="after")
    def check_fields(self) -> Self:
        if self.inn is not None and not valid_inn(self.inn):
            raise ValueError("Invalid INN")
        if self.ogrn is not None and not (valid_ogrn(self.ogrn) or valid_ogrnip(self.ogrn)):
            raise ValueError("Invalid OGRN or OGRNIP")
        if self.inn and self.ogrn and not _valid_pair(self.inn, self.ogrn):
            raise ValueError("INN and registration number types differ")
        entrepreneur = len(self.inn or "") == 12 or len(self.ogrn or "") == 15
        if self.kpp is not None and (entrepreneur or not valid_kpp(self.kpp)):
            raise ValueError("Invalid KPP for counterparty")
        required = (self.inn, self.ogrn, self.legal_name, self.requested_limit_rub)
        if not self.need_human and (None in required or (not entrepreneur and self.kpp is None)):
            raise ValueError("Incomplete requisites require review")
        return self


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


ObservedAt = Annotated[
    AwareDatetime, BeforeValidator(WorkerRequestEnvelope.require_rfc3339), AfterValidator(_utc)
]


class S1RegistrySnapshot(StrictDTO):
    inn: Nonempty
    ogrn: Nonempty
    legal_entity_status: EntityStatus | None
    registration_age_months: NonnegativeInt | None
    charter_capital_rub: Money | None
    tax_debt_rub: Money | None
    defendant_claims_rub: Money | None
    fns_address_unreliable: StrictBool | None
    bankruptcy: StrictBool | None
    observed_at: ObservedAt

    @model_validator(mode="after")
    def check_identity(self) -> Self:
        if (
            not valid_inn(self.inn)
            or not (valid_ogrn(self.ogrn) or valid_ogrnip(self.ogrn))
            or not _valid_pair(self.inn, self.ogrn)
        ):
            raise ValueError("Invalid registry identity")
        if len(self.inn) == 12 and self.charter_capital_rub is not None:
            raise ValueError("Charter capital does not apply to entrepreneurs")
        return self


class S1EvaluateRequest(StrictDTO):
    parsed: S1ParseResponse
    registry: S1RegistrySnapshot

    @model_validator(mode="after")
    def check_registry(self) -> Self:
        if self.parsed.need_human or (self.parsed.inn, self.parsed.ogrn) != (
            self.registry.inn,
            self.registry.ogrn,
        ):
            raise ValueError("Evaluation requires confirmed requisites matching the registry")
        return self


class PaymentTerms(StrictDTO):
    kind: Literal["PREPAYMENT", "DEFERRED"]
    deferment_days: NonnegativeInt


class S1Evaluation(StrictDTO):
    reliability_score: Score | None
    risk_level: Risk | None
    recommended_credit_limit_rub: Money
    payment_terms: PaymentTerms
    stop_factors: Annotated[Codes, AfterValidator(_codes)]
    dossier_markdown: Nonempty

    @model_validator(mode="after")
    def check_terms(self) -> Self:
        if (self.risk_level is None) != (self.reliability_score is None):
            raise ValueError("Score and risk must be present together")
        deferred = self.payment_terms.kind == "DEFERRED"
        if deferred != (self.payment_terms.deferment_days > 0) or deferred != (
            self.recommended_credit_limit_rub > 0
        ):
            raise ValueError("Inconsistent credit limit and payment terms")
        return self


class S1EvaluateResponse(S1Evaluation, Escalation):
    cost_rub: Annotated[Decimal | None, BeforeValidator(WorkerResponseEnvelope.validate_money)]
    mode: Literal["deterministic", "fallback"]
    fallback_reason: Nonempty | None

    @field_serializer("cost_rub", when_used="json")
    def serialize_cost(self, value: Decimal | None) -> str | None:
        return None if value is None else format(value, ".8f")

    @model_validator(mode="after")
    def check_result(self) -> Self:
        if (self.mode == "fallback") != (self.fallback_reason is not None):
            raise ValueError("Fallback requires a reason code")
        if self.need_human and self.recommended_credit_limit_rub != 0:
            raise ValueError("Escalated results cannot recommend credit")
        if (self.stop_factors or self.risk_level is None) and not self.need_human:
            raise ValueError("Unscored results and stop factors require escalation")
        return self


class S1ParseCall(S1Context):
    payload: S1ParseRequest


class S1EvaluateCall(S1Context):
    received_at: ObservedAt
    attempt: PositiveID
    payload: S1EvaluateRequest
