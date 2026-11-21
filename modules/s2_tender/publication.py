from pathlib import Path

from core.contracts import JSONObject, PositiveID, WorkerResponseEnvelope
from core.database import digest
from core.idempotency import Conflict, Journal, PublicationCommand, PublicationKey, Purpose
from modules.s2_tender.contracts import S2Context, S2EvaluateTenderResponse
from modules.s2_tender.module import profile_for


class S2PublicationRequest(S2Context):
    attempt: PositiveID


def prepare_s2_publication(
    journal: Journal, profiles_path: Path, portal_id: str, call: S2PublicationRequest
) -> PublicationKey:
    profile = profile_for(profiles_path, call)
    assert profile.s2_tender is not None
    mapping = profile.s2_tender.bitrix24
    if (
        profile.template
        or profile.portal_id != portal_id
        or mapping.entity_type_id is None
        or mapping.category_id is None
        or mapping.responsible_id is None
        or mapping.fields.EXTERNAL_KEY is None
        or any(
            value is None
            for group in (mapping.stage_ids, mapping.fields, mapping.status_values)
            for value in group.model_dump(exclude={"initial"}).values()
        )
    ):
        raise Conflict("S2 publication requires a fully bound portal profile")

    with journal.db.connect() as conn:
        row = conn.execute(
            """SELECT s.response_json,t.run_origin,d.decision,d.evidence_ref,i.request_id AS intake
               FROM transaction_stages AS s
               JOIN transaction_telemetry AS t USING (tenant_id,request_id)
               LEFT JOIN publication_decisions AS d USING (tenant_id,request_id,stage,attempt)
               LEFT JOIN s2_intakes AS i USING (tenant_id,request_id)
               WHERE s.tenant_id=? AND s.request_id=? AND s.stage='S2'
               AND s.attempt=? AND s.status='completed'""",
            (call.tenant_id, str(call.request_id), call.attempt),
        ).fetchone()
    if row is None:
        raise Conflict("Publication needs a completed S2 evaluation")
    response = WorkerResponseEnvelope.model_validate_json(row["response_json"])
    if (
        response.profile_slug != call.profile_slug
        or response.profile_revision != call.profile_revision
    ):
        raise Conflict("Publication profile does not match the saved result")
    result = S2EvaluateTenderResponse.model_validate(response.payload)
    cancelled = row["decision"] == "cancel"
    intake_needs_approval = row["intake"] is not None and row["decision"] is None
    review = (response.need_human or intake_needs_approval) and not cancelled
    if not (review or cancelled) and (
        result.winner_inn is None or (row["run_origin"] != "mock" and row["decision"] != "approve")
    ):
        raise Conflict("S2 award requires an authorised decision")
    awarded = not (cancelled or review)
    if cancelled:
        stage_id = mapping.stage_ids.cancelled
        status_id = mapping.status_values.CANCELLED
    elif review:
        stage_id = mapping.stage_ids.review
        status_id = mapping.status_values.REVIEW_REQUIRED
    else:
        stage_id = mapping.stage_ids.awarded
        status_id = mapping.status_values.AWARDED
    purpose: Purpose = "human_review" if review else "stage_result"
    key = PublicationKey(
        tenant_id=call.tenant_id, request_id=call.request_id, stage="S2", purpose=purpose
    )
    card_key = digest([call.tenant_id, str(call.request_id), "S2", "stage_result"])
    logical_fields: JSONObject = {
        "TENDER_STATUS": status_id,
        "WINNER_INN": result.winner_inn if awarded else None,
        "WINNER_S1_ITEM_ID": result.winner_s1_item_id if awarded else None,
        "TOTAL_AMOUNT_RUB": str(result.winner_total_rub) if awarded else None,
        "SAVINGS_RUB": str(result.savings_rub) if awarded else None,
        "EXTERNAL_KEY": card_key,
    }
    fields: JSONObject = {
        "title": f"Тендер / Свод КП · {call.request_id}",
        "categoryId": mapping.category_id,
        "stageId": stage_id,
        "assignedById": mapping.responsible_id,
    }
    fields.update(
        {code: logical_fields[name] for name, code in mapping.fields.model_dump().items()}
    )
    journal.approve(
        PublicationCommand(
            key=key,
            stage_attempt=call.attempt,
            portal_id=portal_id,
            approval_ref=(
                row["evidence_ref"]
                if not review and row["decision"] is not None
                else f"policy:s2-{'review' if review else 'mock'}:{profile.slug}:{profile.revision}"
            ),
            entity_type_id=mapping.entity_type_id,
            task_required=False,
            fields=fields,
            external_key_field=mapping.fields.EXTERNAL_KEY,
            card_external_key=card_key,
            closes_cycle=True if cancelled else None,
            timeline_comment=result.comparison_matrix_markdown,
        ),
        allow_review_card=review,
        human_approved=cancelled or row["decision"] == "approve",
    )
    return key
