"""Continue a registered S1 cycle and read portal evidence for S2."""

from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import ValidationError

from core.b24_client import BitrixError
from core.contracts import (
    ItemRef,
    JSONObject,
    Nonempty,
    PositiveID,
    StrictDTO,
    WorkerResponseEnvelope,
)
from core.database import digest
from core.idempotency import Conflict, Journal, PublicationDecision, PublicationKey, Receipt
from core.profile_loader import Profile, load_profile
from core.publisher import Publisher
from modules.s2_tender.contracts import (
    INN,
    S2Context,
    S2ParseQuotesRequest,
    S2SupplierGateSnapshot,
    S2TenderSpec,
)
from modules.s2_tender.module import profile_for, published_gate
from modules.s2_tender.publication import S2PublicationRequest


class TenderIntake(S2ParseQuotesRequest, S2TenderSpec):
    profile_slug: Nonempty
    delivery_key: Nonempty
    upstream_s1_item_id: PositiveID


class TenderReceipt(Receipt):
    tenant_id: Nonempty
    attempt: PositiveID
    upstream_item: ItemRef


class SupplierRef(StrictDTO):
    supplier_inn: INN
    s1_item_id: PositiveID


class S2GateRequest(S2Context):
    upstream_item: ItemRef
    suppliers: list[SupplierRef]


class S2GateReply(StrictDTO):
    supplier_gate_snapshots: list[S2SupplierGateSnapshot]


def register_tender(
    journal: Journal, profiles_path: Path, portal_id: str, call: TenderIntake
) -> TenderReceipt:
    try:
        profile = load_profile(profiles_path, call.profile_slug)
    except (ValueError, OSError) as exc:
        raise Conflict("Tender profile unavailable") from exc
    stage = profile.stages.get("S1")
    if (
        profile.template
        or profile.portal_id != portal_id
        or profile.tenant_id is None
        or stage is None
        or stage.entity_type_id is None
        or profile.s2_tender is None
        or profile.s2_tender.bitrix24.entity_type_id is None
    ):
        raise Conflict("Tender requires a bound S2 portal profile")
    with journal.db.transaction() as conn:
        rows = conn.execute(
            """SELECT p.*,d.profile_slug,d.profile_revision,d.received_at
               FROM crm_publications AS p JOIN idempotency_records AS d USING (tenant_id,request_id)
               WHERE p.tenant_id=? AND p.portal_id=? AND p.entity_type_id=? AND p.item_id=?
               AND p.stage='S1' AND p.purpose='stage_result' AND p.status='published' LIMIT 2""",
            (profile.tenant_id, portal_id, stage.entity_type_id, call.upstream_s1_item_id),
        ).fetchall()
        if len(rows) != 1:
            raise Conflict("No unique published S1 anchor")
        anchor = rows[0]
        gate = published_gate(profile, anchor)
        if gate is None or gate.kyc_status != "APPROVED":
            raise Conflict("S1 anchor is not approved")
        if (anchor["profile_slug"], anchor["profile_revision"]) != (profile.slug, profile.revision):
            raise Conflict("The anchored S1 cycle uses a different profile revision")
        key = profile.tenant_id, anchor["request_id"]
        fingerprint = digest(call.model_dump(mode="json"))
        previous = conn.execute(
            """SELECT request_id,delivery_key,input_sha256 FROM s2_intakes
               WHERE tenant_id=? AND (request_id=? OR delivery_key=?)""",
            (*key, call.delivery_key),
        ).fetchall()
        if previous and [tuple(row) for row in previous] != [
            (anchor["request_id"], call.delivery_key, fingerprint)
        ]:
            raise Conflict("Tender intake differs from the registered delivery")
        if not previous:
            if conn.execute(
                """SELECT 1 FROM transaction_stages WHERE tenant_id=? AND request_id=? AND stage='S2'""",
                key,
            ).fetchone():
                raise Conflict("S2 already started outside this tender intake")
            conn.execute(
                """INSERT INTO s2_intakes (tenant_id,request_id,delivery_key,input_sha256)
                   VALUES (?,?,?,?)""",
                (*key, call.delivery_key, fingerprint),
            )
    return TenderReceipt(
        tenant_id=profile.tenant_id,
        request_id=UUID(anchor["request_id"]),
        profile_slug=profile.slug,
        profile_revision=profile.revision,
        received_at=datetime.fromisoformat(anchor["received_at"]),
        attempt=1,
        upstream_item=ItemRef(
            entity_type_id=stage.entity_type_id, item_id=call.upstream_s1_item_id
        ),
    )


def live_gate(profile: Profile, item_id: int, item: JSONObject) -> S2SupplierGateSnapshot | None:
    assert profile.s1_kyc is not None
    mapping = profile.s1_kyc.bitrix24
    stage = profile.stages["S1"]
    if (
        str(item.get("id")) != str(item_id)
        or stage.stage_ids is None
        or str(item.get("categoryId")) != str(stage.category_id)
    ):
        return None
    values = {name: item.get(code) for name, code in mapping.fields.items() if code}
    status = next(
        (
            name
            for name, value in mapping.kyc_status_values.items()
            if value is not None and str(value) == str(values.get("KYC_STATUS"))
        ),
        None,
    )
    if status is None or item.get("stageId") != (
        stage.stage_ids.approved if status == "APPROVED" else stage.stage_ids.rejected
    ):
        return None
    try:
        return S2SupplierGateSnapshot.model_validate(
            {
                "s1_item_id": item_id,
                "inn": values.get("INN"),
                "ogrn": values.get("OGRN"),
                "kyc_status": status,
                "stage_result_published": True,
                "reliability_score": values.get("RELIABILITY_SCORE"),
                "approved_credit_limit_rub": values.get("APPROVED_CREDIT_LIMIT_RUB"),
            }
        )
    except ValidationError:
        return None


async def read_s1_gates(
    publisher: Publisher, profiles_path: Path, call: S2GateRequest
) -> S2GateReply:
    profile = profile_for(profiles_path, call)
    stage = profile.stages.get("S1")
    if (
        profile.s1_kyc is None
        or profile.portal_id != publisher.portal_id
        or stage is None
        or stage.entity_type_id != call.upstream_item.entity_type_id
    ):
        raise Conflict("S1 process does not belong to this portal")
    entity_type = call.upstream_item.entity_type_id
    anchor_id = call.upstream_item.item_id
    expected = {supplier.s1_item_id: supplier.supplier_inn for supplier in call.suppliers}
    if len(expected) != len(call.suppliers) or len(set(expected.values())) != len(expected):
        raise Conflict("Each supplier must have one distinct S1 card")
    confirmed = {}
    with publisher.journal.db.connect() as conn:
        registered = conn.execute(
            """SELECT 1 FROM s2_intakes WHERE tenant_id=? AND request_id=?""",
            (call.tenant_id, str(call.request_id)),
        ).fetchone()
        if registered is None:
            raise Conflict("S2 tender intake is not registered")
        for item_id in expected.keys() | {anchor_id}:
            rows = conn.execute(
                """SELECT request_id,item_id,command_json FROM crm_publications
                   WHERE tenant_id=? AND portal_id=? AND stage='S1' AND purpose='stage_result'
                   AND status='published' AND entity_type_id=? AND item_id=? LIMIT 2""",
                (call.tenant_id, publisher.portal_id, entity_type, item_id),
            ).fetchall()
            if len(rows) != 1 or (
                item_id == anchor_id and rows[0]["request_id"] != str(call.request_id)
            ):
                continue
            gate = published_gate(profile, rows[0])
            if gate is not None:
                confirmed[item_id] = gate

    matches = []
    for item_id in sorted(confirmed, key=lambda item_id: (item_id != anchor_id, item_id)):
        try:
            item = await publisher.client.item_get(entity_type, item_id)
        except BitrixError as error:
            if error.code != "ERROR_NOT_FOUND":
                raise
            item = {}
        gate = live_gate(profile, item_id, item)
        if gate != confirmed[item_id]:
            if item_id == anchor_id:
                raise Conflict("Current S1 anchor differs from its published result")
            continue
        if gate is not None and item_id in expected and gate.inn == expected[item_id]:
            matches.append(gate)
    if anchor_id not in confirmed or confirmed[anchor_id].kyc_status != "APPROVED":
        raise Conflict("S1 anchor is not confirmed")
    return S2GateReply(supplier_gate_snapshots=matches)


async def confirm_tender_decision(
    publisher: Publisher, profiles_path: Path, call: S2PublicationRequest
) -> None:
    profile = profile_for(profiles_path, call)
    review_key = PublicationKey(
        tenant_id=call.tenant_id, request_id=call.request_id, stage="S2", purpose="human_review"
    )
    state = publisher.state(review_key)
    assert profile.s2_tender is not None
    mapping = profile.s2_tender.bitrix24
    if (
        state.status != "published"
        or state.item_id is None
        or state.command.stage_attempt != call.attempt
        or state.command.entity_type_id != mapping.entity_type_id
        or profile.portal_id != publisher.portal_id
        or mapping.fields.EXTERNAL_KEY is None
        or mapping.fields.TENDER_STATUS is None
    ):
        raise Conflict("Decision does not match a published S2 review")
    item = await publisher.client.item_get(state.command.entity_type_id, state.item_id)
    if (
        str(item.get("id")) != str(state.item_id)
        or str(item.get("categoryId")) != str(mapping.category_id)
        or str(item.get("updatedBy")) != str(mapping.responsible_id)
        or item.get(mapping.fields.EXTERNAL_KEY) != state.command.card_external_key
    ):
        raise Conflict("Portal decision ownership could not be verified")
    status = str(item.get(mapping.fields.TENDER_STATUS))
    stage = item.get("stageId")
    decision: Literal["approve", "cancel"]
    if stage == mapping.stage_ids.awarded and status == str(mapping.status_values.AWARDED):
        with publisher.journal.db.connect() as conn:
            row = conn.execute(
                """SELECT response_json FROM transaction_stages WHERE tenant_id=? AND request_id=?
                   AND stage='S2' AND attempt=? AND status='completed'""",
                (call.tenant_id, str(call.request_id), call.attempt),
            ).fetchone()
        if (
            row is None
            or WorkerResponseEnvelope.model_validate_json(row["response_json"]).need_human
        ):
            raise Conflict("The evaluated attempt does not have an eligible winner")
        decision = "approve"
    elif stage == mapping.stage_ids.cancelled and status == str(mapping.status_values.CANCELLED):
        decision = "cancel"
    else:
        raise Conflict("No matching final stage and decision status on the portal card")
    publisher.journal.record_decision(
        PublicationDecision(
            tenant_id=call.tenant_id,
            request_id=call.request_id,
            stage="S2",
            attempt=call.attempt,
            decision=decision,
            evidence_ref=f"bitrix:{publisher.portal_id}:{state.item_id}:{call.attempt}:{decision}",
        )
    )
