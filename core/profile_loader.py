import re
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo

import yaml
from pydantic import Field, StrictBool, field_validator, model_validator

from core.contracts import Nonempty, PositiveID, Stage, StrictDTO

Risk = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RISKS: tuple[Risk, ...] = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
DecimalString = Annotated[str, Field(strict=True, pattern=r"^[0-9]+(?:\.[0-9]{1,2})?$")]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]


class S1StageIDs(StrictDTO):
    initial: Nonempty
    review: Nonempty
    approved: Nonempty
    rejected: Nonempty

    @model_validator(mode="after")
    def unique(self) -> Self:
        if len(set(self.model_dump().values())) != 4:
            raise ValueError("S1 stage IDs must be unique")
        return self


class StageProfile(StrictDTO):
    stage_ids: S1StageIDs | None = None
    entity_type_id: PositiveID | None
    category_id: NonnegativeInt | None
    stage_id: Nonempty | None
    external_key_field: Nonempty | None
    upstream_field: Nonempty | None = None
    responsible_id: PositiveID | None


class ExtractionProfile(StrictDTO):
    max_text_chars: PositiveID
    max_pdf_bytes: PositiveID


class RegistryProfile(StrictDTO):
    max_age_hours: Annotated[int, Field(strict=True, gt=0, le=87600)]


class Penalty(StrictDTO):
    field: Literal["registration_age_months", "tax_debt_rub", "defendant_claims_rub"]
    operator: Literal["lt", "gt"]
    threshold: NonnegativeInt | DecimalString
    points: DecimalString

    @model_validator(mode="after")
    def check_rule(self) -> Self:
        if self.field == "registration_age_months":
            valid = isinstance(self.threshold, int) and self.operator == "lt"
        else:
            valid = isinstance(self.threshold, str) and self.operator == "gt"
        if not valid or not 0 < Decimal(self.points) <= 100:
            raise ValueError("Invalid penalty")
        return self


class StopFactor(StrictDTO):
    field: Literal["legal_entity_status", "fns_address_unreliable", "bankruptcy"]
    operator: Literal["eq", "ne"]
    value: StrictBool | Literal["ACTIVE", "IN_LIQUIDATION", "TERMINATED"]

    @model_validator(mode="after")
    def check_rule(self) -> Self:
        if (self.field == "legal_entity_status") != isinstance(self.value, str):
            raise ValueError("Stop factor value does not match field")
        return self


class RiskBand(StrictDTO):
    min_score: DecimalString
    max_credit_limit_rub: DecimalString
    deferment_days: NonnegativeInt

    @model_validator(mode="after")
    def check_terms(self) -> Self:
        if (Decimal(self.max_credit_limit_rub) > 0) != (self.deferment_days > 0):
            raise ValueError("Credit and deferment must be configured together")
        return self


class ScoringProfile(StrictDTO):
    penalties: list[Penalty]
    stop_factors: dict[Nonempty, StopFactor]
    risk_bands: dict[Risk, RiskBand]
    manual_review_risk_levels: list[Risk]

    @model_validator(mode="after")
    def check_bands(self) -> Self:
        if self.risk_bands.keys() != set(RISKS):
            raise ValueError("All four risk bands are required")
        scores = [Decimal(self.risk_bands[risk].min_score) for risk in RISKS]
        if scores[0] != 0 or scores[-1] > 100 or scores != sorted(set(scores)):
            raise ValueError("Risk band scores must increase from zero through at most 100")
        if len(self.manual_review_risk_levels) != len(set(self.manual_review_risk_levels)):
            raise ValueError("Manual review levels must be unique")
        rules = {(rule.field, rule.operator, Decimal(rule.threshold)) for rule in self.penalties}
        if len(rules) != len(self.penalties):
            raise ValueError("Penalty rules must be unique")
        return self


FieldKey = Literal["INN", "OGRN", "KYC_STATUS", "RELIABILITY_SCORE", "APPROVED_CREDIT_LIMIT_RUB"]
KycStatus = Literal["APPROVED", "REJECTED"]
FIELD_KEYS = {"INN", "OGRN", "KYC_STATUS", "RELIABILITY_SCORE", "APPROVED_CREDIT_LIMIT_RUB"}


class S1BitrixProfile(StrictDTO):
    tasks_enabled: StrictBool = True
    fields: dict[FieldKey, Nonempty | None]
    kyc_status_values: dict[KycStatus, PositiveID | None]

    @model_validator(mode="after")
    def check_keys(self) -> Self:
        if self.fields.keys() != FIELD_KEYS or self.kyc_status_values.keys() != {
            "APPROVED",
            "REJECTED",
        }:
            raise ValueError("Incomplete S1 Bitrix mapping")
        for values in (list(self.fields.values()), list(self.kyc_status_values.values())):
            bound = [value for value in values if value is not None]
            if len(bound) != len(set(bound)):
                raise ValueError("Bitrix mappings must be unique")
        return self


class S1Profile(StrictDTO):
    extraction: ExtractionProfile
    registry: RegistryProfile
    scoring: ScoringProfile
    bitrix24: S1BitrixProfile


S1_REASONS = frozenset(
    {
        "PARSE_MISSING_FIELD",
        "PARSE_AMBIGUOUS_FIELD",
        "PARSE_INVALID_FIELD",
        "PARSE_SOURCE_UNREADABLE",
        "PARSE_SOURCE_LIMIT",
        "PARSE_BUDGET_EXCEEDED",
        "PARSE_LLM_UNAVAILABLE",
        "REGISTRY_INCOMPLETE",
        "REGISTRY_STALE",
        "RISK_REVIEW",
    }
)


class S2Weights(StrictDTO):
    price: DecimalString
    delivery: DecimalString
    s1_reliability: DecimalString
    deferment: DecimalString

    @model_validator(mode="after")
    def check_total(self) -> Self:
        if sum(map(Decimal, self.model_dump().values())) != Decimal("100.00"):
            raise ValueError("S2 weights must sum to 100.00")
        return self


class S2StageIDs(StrictDTO):
    initial: Nonempty | None = None
    awarded: Nonempty | None
    review: Nonempty | None
    cancelled: Nonempty | None


class S2BitrixFields(StrictDTO):
    TENDER_STATUS: Nonempty | None
    WINNER_INN: Nonempty | None
    WINNER_S1_ITEM_ID: Nonempty | None
    TOTAL_AMOUNT_RUB: Nonempty | None
    SAVINGS_RUB: Nonempty | None
    EXTERNAL_KEY: Nonempty | None


class S2StatusValues(StrictDTO):
    AWARDED: PositiveID | None
    REVIEW_REQUIRED: PositiveID | None
    CANCELLED: PositiveID | None


class S2BitrixProfile(StrictDTO):
    type_id: PositiveID | None = None
    entity_type_id: PositiveID | None
    category_id: NonnegativeInt | None
    responsible_id: PositiveID | None
    stage_ids: S2StageIDs
    fields: S2BitrixFields
    status_values: S2StatusValues

    @model_validator(mode="after")
    def check_distinct_mappings(self) -> Self:
        for mapping in (self.stage_ids, self.fields, self.status_values):
            values = [value for value in mapping.model_dump().values() if value is not None]
            if len(set(values)) != len(values):
                raise ValueError("S2 Bitrix mappings must be distinct within each group")
        return self


class S2Profile(StrictDTO):
    extraction: ExtractionProfile
    min_eligible_quotes: Annotated[int, Field(strict=True, ge=2)]
    weights: S2Weights
    max_budget_overrun_pct: DecimalString
    credit_limit_policy: Literal["REQUIRE_FOR_DEFERRED", "IGNORE"]
    bitrix24: S2BitrixProfile

    @field_validator("max_budget_overrun_pct")
    @classmethod
    def check_budget_percentage(cls, value: str) -> str:
        if Decimal(value) > Decimal("100.00"):
            raise ValueError("S2 budget overrun must be at most 100.00 percent")
        return value


class Profile(StrictDTO):
    slug: Nonempty
    revision: Nonempty
    tenant_id: Nonempty | None
    portal_id: Nonempty | None
    timezone: Nonempty
    stages: dict[Stage, StageProfile]
    escalation_reasons: list[Nonempty] = Field(default_factory=list)
    s1_kyc: S1Profile | None = None
    s2_tender: S2Profile | None = None
    template: StrictBool = False

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (KeyError, ValueError) as exc:
            raise ValueError("Unknown IANA timezone") from exc
        return value

    @model_validator(mode="after")
    def check_bindings(self) -> Self:
        if self.s1_kyc is not None:
            required = S1_REASONS | self.s1_kyc.scoring.stop_factors.keys()
            if "S1" not in self.stages or not required <= set(self.escalation_reasons):
                raise ValueError("S1 stage and all escalation codes are required")
        if self.template:
            if self.s1_kyc is None:
                raise ValueError("Template mode requires an S1 section")
            return self
        if self.tenant_id is None or self.portal_id is None:
            raise ValueError("Tenant and portal bindings are required")
        for stage in self.stages.values():
            if stage.stage_ids is not None and stage.stage_ids.initial != stage.stage_id:
                raise ValueError("S1 initial stage must match the configured stage ID")
            if None in stage.model_dump(exclude={"upstream_field", "stage_ids"}).values():
                raise ValueError("Stage bindings are required")
        if self.s1_kyc is not None:
            mapping = self.s1_kyc.bitrix24
            if self.stages["S1"].external_key_field in mapping.fields.values():
                raise ValueError("S1 external key must have its own field")
            if None in mapping.fields.values() or None in mapping.kyc_status_values.values():
                raise ValueError("S1 Bitrix mappings are required")
        return self


def load_profile(root: Path, slug: str, revision: str | None = None) -> Profile:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", slug):
        raise ValueError("Invalid profile slug")
    base = root.resolve()
    path = (base / slug / "profile.yaml").resolve()
    if not path.is_relative_to(base):
        raise ValueError("Profile path escapes its root")
    with path.open(encoding="utf-8") as stream:
        profile = Profile.model_validate(yaml.safe_load(stream))
    if revision is not None and profile.revision != revision:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", revision):
            raise ValueError("Invalid profile revision")
        archived = (base / slug / f"profile.v{revision}.yaml").resolve()
        if not archived.is_relative_to(base):
            raise ValueError("Profile archive escapes its root")
        with archived.open(encoding="utf-8") as stream:
            profile = Profile.model_validate(yaml.safe_load(stream))
        if profile.revision != revision:
            raise ValueError("Archived profile revision does not match")
    if profile.slug != slug:
        raise ValueError("Profile slug does not match directory")
    return profile
