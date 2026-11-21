import sqlite3
from typing import Literal

from core.contracts import JSONObject, Nonempty, PositiveID, StrictDTO, WorkerResponseEnvelope
from core.database import digest, utc_now
from core.idempotency import Conflict, PublicationCommand, PublicationKey
from core.publisher import Publisher
from modules.s3_claims.contracts import (
    S3EvaluateClaimsResponse,
    S3PublicationRequest,
    S3TenderGateRequest,
    S3TenderGateResponse,
)
from modules.s3_claims.module import S3Worker, context, cycle, package


class S3DecisionNotice(StrictDTO):
    tenant_id: Nonempty
    profile_slug: Nonempty
    profile_revision: Nonempty
    item_id: PositiveID


def resolve_decision_notice(
    worker: S3Worker,
    publisher: Publisher,
    notice: S3DecisionNotice,
) -> S3PublicationRequest:
    with worker.db.connect() as conn:
        rows = conn.execute(
            """SELECT p.request_id,a.package_id,a.attempt
               FROM crm_publications AS p JOIN s3_attempts AS a
               ON (a.tenant_id,a.request_id,a.attempt)=(p.tenant_id,p.request_id,p.stage_attempt)
               JOIN s3_packages AS g
               ON (g.tenant_id,g.request_id,g.package_id)=(a.tenant_id,a.request_id,a.package_id)
               JOIN idempotency_records AS d ON (d.tenant_id,d.request_id)=(p.tenant_id,p.request_id)
               WHERE p.tenant_id=? AND p.portal_id=? AND p.item_id=? AND g.active=1
               AND d.profile_slug=? AND d.profile_revision=?
               AND p.stage='S3' AND p.purpose='human_review:' || a.attempt
               AND p.status='published'
               AND a.attempt=(SELECT MAX(attempt) FROM transaction_stages
                   WHERE tenant_id=p.tenant_id AND request_id=p.request_id AND stage='S3')
               LIMIT 2""",
            (
                notice.tenant_id,
                publisher.portal_id,
                notice.item_id,
                notice.profile_slug,
                notice.profile_revision,
            ),
        ).fetchall()
    if len(rows) != 1:
        raise Conflict("S3 event lacks a unique current published review")
    call = S3PublicationRequest.model_validate(
        {
            **notice.model_dump(exclude={"item_id"}),
            **dict(rows[0]),
        }
    )
    worker.profile(call)
    return call


def result_row(conn: sqlite3.Connection, call: S3PublicationRequest) -> sqlite3.Row:
    active = package(conn, call, call.package_id)
    row = conn.execute(
        """SELECT s.response_json,a.gate_json,d.decision,d.evidence_ref
           FROM transaction_stages AS s JOIN s3_attempts AS a USING (tenant_id,request_id,stage,attempt)
           LEFT JOIN s3_decisions AS d USING (tenant_id,request_id,package_id,attempt)
           WHERE s.tenant_id=? AND s.request_id=? AND s.stage='S3' AND a.package_id=?
           AND s.attempt=? AND s.status='completed' AND s.attempt=(
               SELECT MAX(attempt) FROM transaction_stages
               WHERE tenant_id=s.tenant_id AND request_id=s.request_id AND stage='S3')""",
        (*cycle(call), active["package_id"], call.attempt),
    ).fetchone()
    if row is None:
        raise Conflict("S3 publication requires the current completed attempt")
    response = WorkerResponseEnvelope.model_validate_json(row["response_json"])
    if context(call).model_dump() != response.model_dump(
        include=set(type(context(call)).model_fields)
    ):
        raise Conflict("S3 publication does not match the cycle profile")
    assert isinstance(row, sqlite3.Row)
    return row


def review_key(call: S3PublicationRequest) -> PublicationKey:
    return PublicationKey(
        tenant_id=call.tenant_id,
        request_id=call.request_id,
        stage="S3",
        purpose="human_review",
        review_attempt=call.attempt,
    )


async def confirm_decision(
    worker: S3Worker,
    publisher: Publisher,
    call: S3PublicationRequest,
) -> None:
    profile = worker.profile(call)
    assert profile.s3_claims is not None
    mapping = profile.s3_claims.bitrix24
    if profile.portal_id != publisher.portal_id:
        raise Conflict("S3 decision belongs to another portal")
    with worker.db.connect() as conn:
        row = result_row(conn, call)
        if (
            row["decision"] is not None
            and conn.execute(
                """SELECT 1 FROM crm_publications WHERE tenant_id=? AND request_id=? AND stage='S3'
               AND purpose='stage_result'""",
                cycle(call),
            ).fetchone()
        ):
            return
    state = publisher.state(review_key(call))
    if state.status != "published" or state.item_id is None:
        raise Conflict("S3 review has not been published")
    fields = mapping.fields
    assert fields.ACCEPTANCE_STATUS is not None
    assert fields.EXTERNAL_KEY is not None
    item = await publisher.client.item_get(state.command.entity_type_id, state.item_id)
    expected: JSONObject = {
        **state.command.fields,
        fields.EXTERNAL_KEY: state.command.card_external_key,
        "id": state.item_id,
        "updatedBy": mapping.responsible_id,
    }
    expected.pop("stageId")
    expected.pop(fields.ACCEPTANCE_STATUS)
    if not Publisher._matches(item, expected):
        raise Conflict("S3 decision does not match the review card, revision or authorised author")
    decision: Literal["ACCEPTED", "CLAIM_REQUIRED"]
    status = str(item.get(fields.ACCEPTANCE_STATUS))
    stage = item.get("stageId")
    if status == str(mapping.status_values.ACCEPTED) and stage == mapping.stage_ids.accepted:
        decision = "ACCEPTED"
    elif (
        status == str(mapping.status_values.CLAIM_REQUIRED)
        and stage == mapping.stage_ids.claim_required
    ):
        decision = "CLAIM_REQUIRED"
    else:
        raise Conflict("No confirmed S3 decision on the review card")
    response = WorkerResponseEnvelope.model_validate_json(row["response_json"])
    result = S3EvaluateClaimsResponse.model_validate(response.payload)
    if result.claim_amount_rub is None:
        raise Conflict("S3 requires a complete calculation before confirmation")
    grounds = any(
        line.defect_quantity > 0
        or line.quantity_outside_tolerance
        or (
            line.price_outside_tolerance
            and line.actual_unit_price_rub is not None
            and line.actual_unit_price_rub > line.awarded_unit_price_rub
        )
        for line in result.discrepancies
    )
    if (result.claim_amount_rub > 0 or grounds) != (decision == "CLAIM_REQUIRED"):
        raise Conflict("S3 decision conflicts with the calculated claim")
    evidence = (
        f"bitrix:{publisher.portal_id}:{state.item_id}:{call.package_id}:{call.attempt}:{decision}"
    )
    with worker.db.transaction() as conn:
        row = result_row(conn, call)
        if row["decision"] is not None:
            if (row["decision"], row["evidence_ref"]) != (decision, evidence):
                raise Conflict("S3 decision is immutable")
            return
        conn.execute(
            "INSERT INTO s3_decisions VALUES (?,?,?,?,?,?,?)",
            (*cycle(call), str(call.package_id), call.attempt, decision, evidence, utc_now()),
        )


async def prepare(
    worker: S3Worker,
    publisher: Publisher,
    call: S3PublicationRequest,
) -> PublicationKey:
    profile = worker.profile(call)
    assert profile.s3_claims is not None
    mapping = profile.s3_claims.bitrix24
    entity_id, external_field = mapping.entity_type_id, mapping.fields.EXTERNAL_KEY
    assert entity_id is not None and external_field is not None
    if profile.portal_id != publisher.portal_id:
        raise Conflict("S3 publication belongs to another portal")
    with worker.db.connect() as conn:
        row = result_row(conn, call)
        package_row = package(conn, call, call.package_id)
    decision = row["decision"]
    gate = S3TenderGateResponse.model_validate_json(row["gate_json"])
    key = (
        review_key(call)
        if decision is None
        else PublicationKey(
            tenant_id=call.tenant_id,
            request_id=call.request_id,
            stage="S3",
            purpose="stage_result",
        )
    )
    if decision is not None:
        try:
            state = publisher.state(key)
        except Conflict:
            live = await worker.gate(
                S3TenderGateRequest(**call.model_dump(exclude={"attempt"})),
                publisher,
                save=False,
            )
            if live.award is None or live.award != gate.award:
                raise Conflict("S2 award changed before final S3 preparation")
        else:
            if state.command.stage_attempt == call.attempt:
                return key
            raise Conflict("Final S3 command belongs to another attempt")
    response = WorkerResponseEnvelope.model_validate_json(row["response_json"])
    result = S3EvaluateClaimsResponse.model_validate(response.payload)
    card_key = digest([*cycle(call), "S3", "stage_result"])
    if decision == "ACCEPTED":
        status, stage = mapping.status_values.ACCEPTED, mapping.stage_ids.accepted
    elif decision == "CLAIM_REQUIRED":
        status, stage = mapping.status_values.CLAIM_REQUIRED, mapping.stage_ids.claim_required
    else:
        status, stage = mapping.status_values.REVIEW_REQUIRED, mapping.stage_ids.review
    logical: JSONObject = {
        "ACCEPTANCE_STATUS": status,
        "S2_ITEM_ID": package_row["s2_item_id"],
        "SUPPLIER_INN": gate.award.winner_inn if gate.award else None,
        "ACCEPTED_AMOUNT_RUB": (
            str(result.accepted_amount_rub) if result.accepted_amount_rub is not None else None
        ),
        "CLAIM_AMOUNT_RUB": str(result.claim_amount_rub)
        if result.claim_amount_rub is not None
        else None,
        "EXTERNAL_KEY": card_key,
    }
    fields: JSONObject = {
        "title": f"Приёмка · {call.package_id} · {call.attempt}",
        "categoryId": mapping.category_id,
        "stageId": stage,
        "assignedById": mapping.responsible_id,
    }
    fields.update({code: logical[name] for name, code in mapping.fields.model_dump().items()})
    publisher.journal.approve(
        PublicationCommand(
            key=key,
            stage_attempt=call.attempt,
            portal_id=publisher.portal_id,
            approval_ref=row["evidence_ref"]
            or f"policy:s3-review:{call.package_id}:{call.attempt}",
            entity_type_id=entity_id,
            task_required=False,
            fields=fields,
            external_key_field=external_field,
            card_external_key=card_key,
            timeline_comment=result.reconciliation_report_markdown,
        ),
        human_approved=decision is not None,
        allow_review_card=decision is None,
    )
    return key
