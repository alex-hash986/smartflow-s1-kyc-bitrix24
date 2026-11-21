import re
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    field_serializer,
    field_validator,
    model_validator,
)

Stage = Literal["S1", "S2", "S3"]
Mode = Literal["llm", "deterministic", "fallback"]
RunOrigin = Literal["mock", "benchmark", "pilot"]
Outcome = Literal["running", "waiting_human", "completed", "failed", "reconcile"]
PositiveID = Annotated[int, Field(strict=True, gt=0, le=2**63 - 1)]
Nonempty = Annotated[str, Field(strict=True, min_length=1)]
JSONObject = dict[str, JsonValue]
COST_UNIT = Decimal("0.00000001")
MAX_COST_ATOMS = 2**63 - 1


def round_money(value: Decimal, unit: Decimal = Decimal("0.01")) -> Decimal:
    if not value.is_finite():
        raise ValueError("Amount must be finite")
    with localcontext() as context:
        context.prec = max(context.prec, value.adjusted() - unit.adjusted() + 2)
        return value.quantize(unit, rounding=ROUND_HALF_UP)


def cost_atoms(value: Decimal | None) -> int | None:
    if value is None:
        return None
    if not value.is_finite() or value < 0 or value >= Decimal(2**63) * COST_UNIT:
        raise ValueError("Cost outside SQLite range")
    atoms = int(round_money(value, COST_UNIT) / COST_UNIT)
    if atoms > MAX_COST_ATOMS:
        raise ValueError("Cost outside SQLite range")
    return atoms


class StrictDTO(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ItemRef(StrictDTO):
    entity_type_id: PositiveID
    item_id: PositiveID


class WorkerIdentity(StrictDTO):
    schema_version: Literal["1"]
    tenant_id: Nonempty
    request_id: UUID
    profile_slug: Nonempty
    profile_revision: Nonempty
    stage: Stage
    attempt: PositiveID


class WorkerRequestEnvelope(WorkerIdentity):
    received_at: AwareDatetime
    upstream_item: ItemRef | None
    payload: Annotated[JSONObject, Field(strict=True)]

    @field_validator("received_at", mode="before")
    @classmethod
    def require_rfc3339(cls, value: object) -> object:
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str) or not re.fullmatch(
            r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)", value
        ):
            raise ValueError("Expected RFC 3339 with timezone")
        return value

    @model_validator(mode="after")
    def validate_upstream(self) -> Self:
        if (self.stage == "S1") != (self.upstream_item is None):
            raise ValueError("S1 has no upstream item; S2 and S3 require one")
        self.received_at = self.received_at.astimezone(UTC)
        return self


class WorkerResponseEnvelope(WorkerIdentity):
    payload: Annotated[JSONObject, Field(strict=True)] | None
    need_human: StrictBool
    escalation_reasons: Annotated[list[Nonempty], Field(strict=True)]
    cost_rub: Decimal | None
    mode: Mode
    fallback_reason: Nonempty | None

    @field_validator("cost_rub", mode="before")
    @classmethod
    def validate_money(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and re.fullmatch(r"[0-9]+\.[0-9]{8}", value):
            value = Decimal(value)
        if not isinstance(value, Decimal):
            raise ValueError("Expected a decimal string with eight fractional digits")  # noqa: TRY004
        cost_atoms(value)
        return round_money(value, COST_UNIT)

    @field_serializer("cost_rub", when_used="json")
    def serialize_money(self, value: Decimal | None) -> str | None:
        return None if value is None else format(value, ".8f")

    @model_validator(mode="after")
    def check_result(self) -> Self:
        if self.need_human != bool(self.escalation_reasons):
            raise ValueError("Escalation requires reason codes")
        if self.payload is None and not self.need_human:
            raise ValueError("A non-escalated result requires payload")
        if (self.mode == "fallback") != (self.fallback_reason is not None):
            raise ValueError("Fallback requires a reason code")
        return self
