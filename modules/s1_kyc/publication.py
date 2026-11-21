from decimal import Decimal
from pathlib import Path

from core.contracts import JSONObject, PositiveID, WorkerResponseEnvelope
from core.database import digest, utc_now
from core.idempotency import (
    Conflict,
    Journal,
    PublicationCommand,
    PublicationKey,
    Purpose,
    TaskCommand,
)
from core.profile_loader import load_profile
from modules.s1_kyc.contracts import S1Context
from modules.s1_kyc.module import S1ParseReply


class S1PublicationRequest(S1Context):
    attempt: PositiveID


def _completed_attempt(
    journal: Journal,
    call: S1PublicationRequest,
) -> tuple[S1ParseReply, WorkerResponseEnvelope]:
    key = (call.tenant_id, str(call.request_id))
    with journal.db.transaction() as conn:
        row = conn.execute(
            """SELECT d.profile_slug,d.profile_revision,t.run_origin,e.response_json
               FROM idempotency_records AS d
               JOIN transaction_telemetry AS t USING (tenant_id,request_id)
               JOIN s1_extractions AS e USING (tenant_id,request_id)
               WHERE d.tenant_id=? AND d.request_id=?""",
            key,
        ).fetchone()
        if (
            row is None
            or row["response_json"] is None
            or row["run_origin"] != "mock"
            or (row["profile_slug"], row["profile_revision"])
            != (
                call.profile_slug,
                call.profile_revision,
            )
        ):
            raise Conflict("S1 mock publication requires a completed, registered extraction")
        parsed = S1ParseReply.model_validate_json(row["response_json"])
        stage = conn.execute(
            """SELECT status,response_json FROM transaction_stages
               WHERE tenant_id=? AND request_id=? AND stage='S1' AND attempt=?""",
            (*key, call.attempt),
        ).fetchone()
        if stage is not None and stage["status"] == "completed":
            return parsed, WorkerResponseEnvelope.model_validate_json(stage["response_json"])
        if stage is not None or not parsed.payload.need_human or call.attempt != 1:
            raise Conflict("Publication needs a completed S1 evaluation")
        response = WorkerResponseEnvelope(
            **call.model_dump(),
            schema_version="1",
            stage="S1",
            payload=None,
            need_human=True,
            escalation_reasons=parsed.payload.escalation_reasons,
            cost_rub=Decimal(0),
            mode="deterministic",
            fallback_reason=None,
        )
        now = utc_now()
        conn.execute(
            """INSERT INTO transaction_stages
               (tenant_id,request_id,stage,attempt,status,input_sha256,response_json,
                started_at,finished_at,mode,tokens_in,tokens_out,cost_rub_e8)
               VALUES (?,?,'S1',1,'completed',?,?,?,?,'deterministic',0,0,0)""",
            (*key, digest(parsed.model_dump(mode="json")), response.model_dump_json(), now, now),
        )
        return parsed, response


def prepare_s1_publication(
    journal: Journal,
    profiles_path: Path,
    portal_id: str,
    call: S1PublicationRequest,
) -> PublicationKey:
    try:
        profile = load_profile(profiles_path, call.profile_slug)
    except (ValueError, OSError) as exc:
        raise Conflict("S1 publication profile unavailable") from exc
    stage = profile.stages.get("S1")
    if (
        profile.template
        or profile.revision != call.profile_revision
        or profile.tenant_id != call.tenant_id
        or profile.portal_id != portal_id
        or profile.s1_kyc is None
        or stage is None
        or stage.stage_ids is None
        or stage.entity_type_id is None
        or stage.category_id is None
        or stage.external_key_field is None
        or stage.responsible_id is None
    ):
        raise Conflict("S1 publication requires a bound portal profile")
    parsed, response = _completed_attempt(journal, call)
    mapping = profile.s1_kyc.bitrix24
    marked_for_review = response.need_human
    purpose: Purpose = "human_review" if marked_for_review else "stage_result"
    selected_stage = stage.stage_ids.review if marked_for_review else stage.stage_ids.approved
    fields: JSONObject = {
        "title": (
            f"Проверка контрагента · {parsed.payload.inn}"
            if parsed.payload.inn
            else "Проверка контрагента · ручная проверка"
        ),
        "categoryId": stage.category_id,
        "stageId": selected_stage,
        "assignedById": stage.responsible_id,
    }
    values: JSONObject = {"INN": parsed.payload.inn, "OGRN": parsed.payload.ogrn}
    if response.payload is not None:
        values["RELIABILITY_SCORE"] = response.payload["reliability_score"]
    if not marked_for_review:
        if response.payload is None or (
            parsed.payload.need_human or response.payload["risk_level"] not in ("LOW", "MEDIUM")
        ):
            raise Conflict("Mock policy cannot approve this S1 result")
        values.update(
            KYC_STATUS=mapping.kyc_status_values["APPROVED"],
            APPROVED_CREDIT_LIMIT_RUB=response.payload["recommended_credit_limit_rub"],
        )
        if any(value is None for value in values.values()):
            raise Conflict("Final result requires all five card fields")
    for name, code in mapping.fields.items():
        if code is not None and values.get(name) is not None:
            fields[code] = values[name]
    dossier = response.payload.get("dossier_markdown") if response.payload is not None else None
    if not isinstance(dossier, str):
        dossier = "# Ручная проверка S1\n\n" + "\n".join(
            f"- {reason}" for reason in response.escalation_reasons
        )
    key = PublicationKey(
        tenant_id=call.tenant_id,
        request_id=call.request_id,
        stage="S1",
        purpose=purpose,
    )
    task = (
        TaskCommand(title="Проверить контрагента", responsible_id=stage.responsible_id)
        if marked_for_review and mapping.tasks_enabled
        else None
    )
    journal.approve(
        PublicationCommand(
            key=key,
            stage_attempt=call.attempt,
            portal_id=portal_id,
            approval_ref=f"policy:s1-mock:{profile.slug}:{profile.revision}",
            entity_type_id=stage.entity_type_id,
            task_required=task is not None,
            task=task,
            fields=fields,
            external_key_field=stage.external_key_field,
            timeline_comment=dossier,
            inn_field=mapping.fields["INN"],
        ),
        allow_review_card=True,
    )
    return key
