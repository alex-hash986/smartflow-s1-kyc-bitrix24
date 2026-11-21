import sqlite3
from datetime import UTC, datetime
from typing import Literal, Self
from uuid import UUID, uuid4

from pydantic import JsonValue, SerializerFunctionWrapHandler, model_serializer, model_validator

from core.contracts import (
    JSONObject,
    Nonempty,
    PositiveID,
    RunOrigin,
    Stage,
    StrictDTO,
    WorkerResponseEnvelope,
)
from core.database import Database, canonical_json, digest, utc_now

Purpose = Literal["stage_result", "human_review"]
Step = Literal["item", "task", "comment"]


class Conflict(ValueError):
    """A request conflicts with durable state."""


class Delivery(StrictDTO):
    tenant_id: Nonempty
    source: Nonempty
    delivery_key: Nonempty
    profile_slug: Nonempty
    profile_revision: Nonempty
    channel: Nonempty
    is_after_hours: bool
    run_origin: RunOrigin
    payload: JSONObject


class Receipt(StrictDTO):
    request_id: UUID
    profile_slug: Nonempty
    profile_revision: Nonempty
    received_at: datetime


class PublicationKey(StrictDTO):
    tenant_id: Nonempty
    request_id: UUID
    stage: Stage
    purpose: Purpose
    review_attempt: PositiveID | None = None

    @model_validator(mode="after")
    def check_review_attempt(self) -> Self:
        if (self.stage == "S3" and self.purpose == "human_review") != (
            self.review_attempt is not None
        ):
            raise ValueError("review_attempt is required only for S3 human_review")
        return self

    @model_serializer(mode="wrap")
    def serialize_key(self, handler: SerializerFunctionWrapHandler) -> dict[str, JsonValue]:
        fields: dict[str, JsonValue] = handler(self)
        if self.review_attempt is None:
            fields.pop("review_attempt", None)
        return fields


class TaskCommand(StrictDTO):
    title: Nonempty
    responsible_id: PositiveID
    description: str = ""


class PublicationDecision(StrictDTO):
    tenant_id: Nonempty
    request_id: UUID
    stage: Stage
    attempt: PositiveID
    decision: Literal["approve", "cancel"]
    evidence_ref: Nonempty


class PublicationCommand(StrictDTO):
    key: PublicationKey
    stage_attempt: PositiveID
    portal_id: Nonempty
    approval_ref: Nonempty
    entity_type_id: PositiveID
    task_required: bool
    item_id: PositiveID | None = None
    fields: JSONObject
    external_key_field: Nonempty
    task: TaskCommand | None = None
    timeline_comment: Nonempty | None = None
    inn_field: Nonempty | None = None
    card_external_key: Nonempty | None = None
    closes_cycle: Literal[True] | None = None


class Journal:
    def __init__(self, db: Database) -> None:
        self.db = db

    def register(self, delivery: Delivery) -> Receipt:
        fingerprint = digest(delivery.payload)
        key = (delivery.tenant_id, delivery.source, delivery.delivery_key)
        with self.db.transaction() as conn:
            row = conn.execute(
                """SELECT * FROM idempotency_records
                   WHERE tenant_id=? AND source=? AND delivery_key=?""",
                key,
            ).fetchone()
            if row is not None:
                if row["input_sha256"] != fingerprint:
                    raise Conflict("Delivery key reused with different payload")
                return Receipt(
                    request_id=UUID(row["request_id"]),
                    profile_slug=row["profile_slug"],
                    profile_revision=row["profile_revision"],
                    received_at=datetime.fromisoformat(row["received_at"]),
                )
            receipt = Receipt(
                request_id=uuid4(),
                profile_slug=delivery.profile_slug,
                profile_revision=delivery.profile_revision,
                received_at=datetime.now(UTC),
            )
            when = receipt.received_at.isoformat()
            conn.execute(
                "INSERT INTO idempotency_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    *key,
                    fingerprint,
                    canonical_json(delivery.payload),
                    str(receipt.request_id),
                    receipt.profile_slug,
                    receipt.profile_revision,
                    when,
                ),
            )
            conn.execute(
                """INSERT INTO transaction_telemetry
                   (tenant_id,request_id,channel,is_after_hours,run_origin,outcome,updated_at)
                   VALUES (?,?,?,?,?,'running',?)""",
                (
                    delivery.tenant_id,
                    str(receipt.request_id),
                    delivery.channel,
                    delivery.is_after_hours,
                    delivery.run_origin,
                    when,
                ),
            )
            return receipt

    @staticmethod
    def _key(key: PublicationKey) -> tuple[str, str, Stage, str]:
        purpose = (
            key.purpose if key.review_attempt is None else f"human_review:{key.review_attempt}"
        )
        return key.tenant_id, str(key.request_id), key.stage, purpose

    def record_decision(self, decision: PublicationDecision) -> None:
        """Persist an already verified external decision from a trusted adapter."""
        key = decision.tenant_id, str(decision.request_id), decision.stage
        with self.db.transaction() as conn:
            prior = conn.execute(
                """SELECT decision,evidence_ref FROM publication_decisions
                   WHERE tenant_id=? AND request_id=? AND stage=? AND attempt=?""",
                (*key, decision.attempt),
            ).fetchone()
            if prior is not None:
                if tuple(prior) != (decision.decision, decision.evidence_ref):
                    raise Conflict("Publication decision is immutable")
                return
            latest = conn.execute(
                """SELECT attempt,status FROM transaction_stages
                   WHERE tenant_id=? AND request_id=? AND stage=? ORDER BY attempt DESC LIMIT 1""",
                key,
            ).fetchone()
            final = conn.execute(
                """SELECT 1 FROM crm_publications WHERE tenant_id=? AND request_id=?
                   AND stage=? AND purpose='stage_result'""",
                key,
            ).fetchone()
            if latest is None or tuple(latest) != (decision.attempt, "completed") or final:
                raise Conflict("Decision requires the latest completed attempt before publication")
            conn.execute(
                """INSERT INTO publication_decisions VALUES (?,?,?,?,?,?,?)""",
                (*key, decision.attempt, decision.decision, decision.evidence_ref, utc_now()),
            )

    def approve(
        self,
        command: PublicationCommand,
        *,
        human_approved: bool = False,
        allow_review_card: bool = False,
    ) -> str:
        key = self._key(command.key)
        marker = digest(list(key))
        if command.task_required != (command.task is not None):
            raise Conflict("Task requirement and command must agree")
        if command.closes_cycle and (command.card_external_key is None or key[3] != "stage_result"):
            raise Conflict("Closing a cycle requires an S2 final publication")
        optional = {"timeline_comment", "inn_field", "card_external_key", "closes_cycle"}
        omitted = {key for key in optional if getattr(command, key) is None}
        serialized = canonical_json(command.model_dump(mode="json", exclude=omitted))
        with self.db.transaction() as conn:
            prior = conn.execute(
                """SELECT command_json FROM crm_publications
                   WHERE tenant_id=? AND request_id=? AND stage=? AND purpose=?""",
                key,
            ).fetchone()
            if prior is not None:
                if prior["command_json"] != serialized:
                    raise Conflict("Publication command is immutable")
                return marker
            if (
                command.key.stage == "S3"
                and command.card_external_key is None
                and conn.execute(
                    """SELECT 1 FROM s3_attempts WHERE tenant_id=? AND request_id=? AND attempt=?""",
                    (*key[:2], command.stage_attempt),
                ).fetchone()
                is not None
            ):
                raise Conflict(
                    "S3 claims publication requires a shared card key and versioned approval"
                )
            if command.card_external_key is not None:
                if key[2] not in ("S2", "S3") or command.card_external_key != digest(
                    [*key[:3], "stage_result"]
                ):
                    raise Conflict("Invalid shared card key")
                latest = conn.execute(
                    """SELECT MAX(attempt) FROM transaction_stages
                       WHERE tenant_id=? AND request_id=? AND stage=?""",
                    key[:3],
                ).fetchone()[0]
                if latest != command.stage_attempt:
                    raise Conflict("Publication requires the latest evaluation")
                if key[3] == "stage_result":
                    decision = conn.execute(
                        """SELECT decision,evidence_ref FROM publication_decisions
                           WHERE tenant_id=? AND request_id=? AND stage=? AND attempt=?""",
                        (*key[:3], command.stage_attempt),
                    ).fetchone()
                    if decision is not None and (
                        command.approval_ref != decision["evidence_ref"]
                        or bool(command.closes_cycle) != (decision["decision"] == "cancel")
                    ):
                        raise Conflict("Publication does not match the recorded decision")
                    if decision is None and command.closes_cycle:
                        raise Conflict("Cancellation requires a recorded decision")
                if key[2] == "S3":
                    self._verify_s3_approval(conn, command)
                counterpart = conn.execute(
                    """SELECT status,portal_id,entity_type_id FROM crm_publications
                       WHERE tenant_id=? AND request_id=? AND stage=? AND purpose<>?""",
                    key,
                ).fetchone()
                if (
                    key[2] == "S2"
                    and counterpart is not None
                    and (
                        key[3] == "human_review"
                        or counterpart["status"] != "published"
                        or counterpart["portal_id"] != command.portal_id
                        or counterpart["entity_type_id"] != command.entity_type_id
                    )
                ):
                    raise Conflict("The other S2 publication must be resolved first")
            row = conn.execute(
                """SELECT response_json FROM transaction_stages
                   WHERE tenant_id=? AND request_id=? AND stage=? AND attempt=?
                   AND status='completed'""",
                (*key[:3], command.stage_attempt),
            ).fetchone()
            if row is None:
                raise Conflict("Approval requires a completed attempt")
            response = WorkerResponseEnvelope.model_validate_json(row["response_json"])
            if key[3] == "stage_result" and response.need_human and not human_approved:
                raise Conflict("This revision requires a human decision")
            if command.key.purpose == "human_review" and (
                (not response.need_human and not (command.card_external_key and allow_review_card))
                or (
                    (not command.task_required or command.item_id is None)
                    and not (allow_review_card and command.timeline_comment)
                )
            ):
                raise Conflict("Human review requires a task on a configured card")
            now = utc_now()
            conn.execute(
                """INSERT INTO crm_publications
                   (tenant_id,request_id,stage,purpose,stage_attempt,portal_id,external_key,status,
                    approved_at,approval_ref,command_json,entity_type_id,item_id,task_required,
                    updated_at) VALUES (?,?,?,?,?,?,?,'approved',?,?,?,?,?,?,?)""",
                (
                    *key,
                    command.stage_attempt,
                    command.portal_id,
                    marker,
                    now,
                    command.approval_ref,
                    serialized,
                    command.entity_type_id,
                    command.item_id,
                    command.task_required,
                    now,
                ),
            )
            if command.timeline_comment is not None:
                conn.execute(
                    "INSERT INTO crm_publication_comments (tenant_id,request_id,stage,purpose) VALUES (?,?,?,?)",
                    key,
                )
        return marker

    def _verify_s3_approval(self, conn: sqlite3.Connection, command: PublicationCommand) -> None:
        key = self._key(command.key)
        row = conn.execute(
            """SELECT d.evidence_ref FROM s3_attempts AS a JOIN s3_packages AS p
               USING (tenant_id,request_id,package_id)
               LEFT JOIN s3_decisions AS d USING (tenant_id,request_id,package_id,attempt)
               WHERE a.tenant_id=? AND a.request_id=? AND a.attempt=? AND p.active=1""",
            (*key[:2], command.stage_attempt),
        ).fetchone()
        if row is None:
            raise Conflict("S3 publication requires the active package")
        if command.key.purpose == "human_review":
            if command.key.review_attempt != command.stage_attempt:
                raise Conflict("S3 queue must match its attempt")
            return
        if row["evidence_ref"] is None or row["evidence_ref"] != command.approval_ref:
            raise Conflict("S3 final publication requires the recorded decision")
        review = conn.execute(
            """SELECT item_id FROM crm_publications WHERE tenant_id=? AND request_id=?
               AND stage='S3' AND purpose=? AND status='published'""",
            (*key[:2], f"human_review:{command.stage_attempt}"),
        ).fetchone()
        if review is None:
            raise Conflict("S3 decision requires a confirmed review card")

    def claim(self, key: PublicationKey) -> str | None:
        token = str(uuid4())
        with self.db.transaction() as conn:
            if key.review_attempt is not None:
                active = conn.execute(
                    """SELECT 1 FROM s3_attempts AS a JOIN s3_packages AS p
                       USING (tenant_id,request_id,package_id)
                       WHERE a.tenant_id=? AND a.request_id=? AND a.attempt=? AND p.active=1
                       AND a.attempt=(SELECT MAX(attempt) FROM transaction_stages
                           WHERE tenant_id=a.tenant_id AND request_id=a.request_id AND stage='S3')""",
                    (*self._key(key)[:2], key.review_attempt),
                ).fetchone()
                if active is None:
                    prior = conn.execute(
                        """SELECT status FROM crm_publications WHERE tenant_id=? AND request_id=?
                           AND stage=? AND purpose=?""",
                        self._key(key),
                    ).fetchone()
                    if prior is None or prior["status"] != "published":
                        raise Conflict("S3 review attempt was superseded")
                    return None
            result = conn.execute(
                """UPDATE crm_publications SET status='publishing', owner_token=?, updated_at=?
                   WHERE tenant_id=? AND request_id=? AND stage=? AND purpose=?
                   AND status='approved' AND pending_step IS NULL""",
                (token, utc_now(), *self._key(key)),
            )
            return token if result.rowcount == 1 else None

    def begin_step(self, key: PublicationKey, token: str, step: Step) -> bool:
        if step == "comment":
            with self.db.transaction() as conn:
                result = conn.execute(
                    """UPDATE crm_publication_comments SET pending=1
                       WHERE (tenant_id,request_id,stage,purpose)=(?,?,?,?)
                       AND pending=0 AND comment_id IS NULL AND EXISTS (
                           SELECT 1 FROM crm_publications
                           WHERE (tenant_id,request_id,stage,purpose)=(?,?,?,?)
                           AND status='publishing' AND owner_token=? AND item_id IS NOT NULL
                           AND pending_step IS NULL)""",
                    (*self._key(key), *self._key(key), token),
                )
                return result.rowcount == 1
        with self.db.transaction() as conn:
            result = conn.execute(
                """UPDATE crm_publications SET pending_step=?, updated_at=?
                   WHERE tenant_id=? AND request_id=? AND stage=? AND purpose=?
                   AND status='publishing' AND owner_token=? AND pending_step IS NULL
                   AND ((?='item' AND item_id IS NULL)
                     OR (?='task' AND item_id IS NOT NULL AND task_required=1 AND task_id IS NULL))""",
                (step, utc_now(), *self._key(key), token, step, step),
            )
            return result.rowcount == 1

    def record_step(self, key: PublicationKey, token: str, step: Step, external_id: int) -> bool:
        return self._confirm_step(key, token, step, external_id, "publishing")

    def _confirm_step(
        self,
        key: PublicationKey,
        token: str,
        step: Step,
        external_id: int,
        state: Literal["publishing", "reconcile"],
    ) -> bool:
        if type(external_id) is not int or not 0 < external_id < 2**63:
            raise ValueError("Invalid external ID")
        if step == "comment":
            with self.db.transaction() as conn:
                result = conn.execute(
                    """UPDATE crm_publication_comments SET pending=0,comment_id=?
                       WHERE (tenant_id,request_id,stage,purpose)=(?,?,?,?) AND pending=1
                       AND EXISTS (SELECT 1 FROM crm_publications
                           WHERE (tenant_id,request_id,stage,purpose)=(?,?,?,?)
                           AND status=? AND owner_token=?)""",
                    (external_id, *self._key(key), *self._key(key), state, token),
                )
                return result.rowcount == 1
        column = "item_id" if step == "item" else "task_id"
        with self.db.transaction() as conn:
            result = conn.execute(
                f"""UPDATE crm_publications SET {column}=?, pending_step=NULL, updated_at=?
                    WHERE tenant_id=? AND request_id=? AND stage=? AND purpose=?
                    AND status=? AND owner_token=? AND pending_step=?""",
                (external_id, utc_now(), *self._key(key), state, token, step),
            )
            return result.rowcount == 1

    def finish(self, key: PublicationKey, token: str) -> bool:
        with self.db.transaction() as conn:
            now = utc_now()
            result = conn.execute(
                """UPDATE crm_publications SET status='published', owner_token=NULL, updated_at=?
                   WHERE tenant_id=? AND request_id=? AND stage=? AND purpose=?
                   AND status='publishing' AND owner_token=? AND pending_step IS NULL
                   AND item_id IS NOT NULL AND (task_required=0 OR task_id IS NOT NULL)
                   AND NOT EXISTS (SELECT 1 FROM crm_publication_comments AS c
                       WHERE (c.tenant_id,c.request_id,c.stage,c.purpose)=
                             (crm_publications.tenant_id,crm_publications.request_id,
                              crm_publications.stage,crm_publications.purpose)
                       AND c.comment_id IS NULL)""",
                (now, *self._key(key), token),
            )
            if result.rowcount == 1 and key.purpose == "stage_result":
                cycle_key = self._key(key)[:2]
                elapsed = None
                command_json = conn.execute(
                    """SELECT command_json FROM crm_publications
                       WHERE tenant_id=? AND request_id=? AND stage=? AND purpose=?""",
                    self._key(key),
                ).fetchone()[0]
                command = PublicationCommand.model_validate_json(command_json)
                completed = key.stage == "S3" or bool(command.closes_cycle)
                if completed:
                    received_at = conn.execute(
                        """SELECT received_at FROM idempotency_records
                           WHERE tenant_id=? AND request_id=?""",
                        cycle_key,
                    ).fetchone()[0]
                    elapsed = max(
                        0.0,
                        (
                            datetime.fromisoformat(now) - datetime.fromisoformat(received_at)
                        ).total_seconds(),
                    )
                conn.execute(
                    """UPDATE transaction_telemetry SET outcome=?,processing_time_sec=?,
                       updated_at=? WHERE tenant_id=? AND request_id=?""",
                    ("completed" if completed else "running", elapsed, now, *cycle_key),
                )
            elif result.rowcount == 1 and key.stage in ("S2", "S3"):
                conn.execute(
                    """UPDATE transaction_telemetry SET outcome='waiting_human',need_human=1,
                       updated_at=? WHERE tenant_id=? AND request_id=?""",
                    (now, *self._key(key)[:2]),
                )
            return result.rowcount == 1

    def reconcile(self, key: PublicationKey, token: str) -> bool:
        with self.db.transaction() as conn:
            now = utc_now()
            result = conn.execute(
                """UPDATE crm_publications SET status='reconcile', updated_at=?
                   WHERE tenant_id=? AND request_id=? AND stage=? AND purpose=?
                   AND status='publishing' AND owner_token=?""",
                (now, *self._key(key), token),
            )
            if result.rowcount == 1:
                conn.execute(
                    """UPDATE transaction_telemetry SET outcome='reconcile', updated_at=?
                       WHERE tenant_id=? AND request_id=?""",
                    (now, *self._key(key)[:2]),
                )
            return result.rowcount == 1

    def resolve(
        self,
        key: PublicationKey,
        token: str,
        decision_ref: str,
        *,
        previous_stopped: bool,
        confirmed_absent: bool = False,
    ) -> bool:
        if not previous_stopped or not decision_ref.strip():
            raise ValueError("Reconciliation requires a recorded decision and stopped owner")
        with self.db.transaction() as conn:
            comment = conn.execute(
                """SELECT pending FROM crm_publication_comments
                   WHERE (tenant_id,request_id,stage,purpose)=(?,?,?,?)""",
                self._key(key),
            ).fetchone()
            if comment is not None and comment["pending"] and not confirmed_absent:
                return False
            changed = (
                conn.execute(
                    """UPDATE crm_publications SET status='approved', owner_token=NULL,
                   pending_step=NULL, approval_ref=?, updated_at=?
                   WHERE tenant_id=? AND request_id=? AND stage=? AND purpose=?
                   AND status='reconcile' AND owner_token=?
                   AND (pending_step IS NULL OR ?)""",
                    (decision_ref, utc_now(), *self._key(key), token, confirmed_absent),
                ).rowcount
                == 1
            )

            if changed and comment is not None and comment["pending"]:
                conn.execute(
                    """UPDATE crm_publication_comments SET pending=0
                       WHERE (tenant_id,request_id,stage,purpose)=(?,?,?,?)""",
                    self._key(key),
                )
            return changed

    def confirm_reconciled_step(
        self,
        key: PublicationKey,
        token: str,
        step: Step,
        external_id: int,
        *,
        previous_stopped: bool,
    ) -> bool:
        if not previous_stopped:
            raise ValueError("The previous executor may still be active")
        return self._confirm_step(key, token, step, external_id, "reconcile")
