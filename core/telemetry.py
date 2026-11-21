import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from core.contracts import (
    Mode,
    Outcome,
    Stage,
    WorkerRequestEnvelope,
    WorkerResponseEnvelope,
    cost_atoms,
)
from core.database import Database, digest, utc_now
from core.idempotency import Conflict

PREVIOUS_STAGE: dict[Stage, Stage] = {"S2": "S1", "S3": "S2"}


@dataclass(frozen=True)
class Attempt:
    status: Literal["reserved", "running", "completed"]
    response: WorkerResponseEnvelope | None = None


def identity(envelope: WorkerRequestEnvelope | WorkerResponseEnvelope) -> tuple[str, str, Stage]:
    return envelope.tenant_id, str(envelope.request_id), envelope.stage


class Telemetry:
    def __init__(self, db: Database) -> None:
        self.db = db

    def reserve(self, request: WorkerRequestEnvelope) -> Attempt:
        key = identity(request)
        fingerprint = digest(request.model_dump(mode="json"))
        with self.db.transaction() as conn:
            delivery = conn.execute(
                """SELECT profile_slug,profile_revision,received_at FROM idempotency_records
                   WHERE tenant_id=? AND request_id=?""",
                key[:2],
            ).fetchone()
            if delivery is None or (
                delivery["profile_slug"] != request.profile_slug
                or delivery["profile_revision"] != request.profile_revision
                or datetime.fromisoformat(delivery["received_at"]) != request.received_at
            ):
                raise Conflict("Unregistered request or different delivery context")
            existing = conn.execute(
                """SELECT * FROM transaction_stages WHERE tenant_id=? AND request_id=?
                   AND stage=? AND attempt=?""",
                (*key, request.attempt),
            ).fetchone()
            if existing is not None:
                if existing["input_sha256"] != fingerprint:
                    raise Conflict("Attempt payload has changed")
                if existing["status"] == "completed":
                    response = WorkerResponseEnvelope.model_validate_json(existing["response_json"])
                    return Attempt("completed", response)
                if existing["status"] == "running":
                    return Attempt("running")
                raise Conflict("Attempt is closed")
            if request.upstream_item is not None:
                upstream = conn.execute(
                    """SELECT 1 FROM crm_publications WHERE tenant_id=? AND request_id=?
                       AND stage=? AND purpose='stage_result' AND status='published'
                       AND entity_type_id=? AND item_id=?""",
                    (
                        *key[:2],
                        PREVIOUS_STAGE[request.stage],
                        request.upstream_item.entity_type_id,
                        request.upstream_item.item_id,
                    ),
                ).fetchone()
                if upstream is None:
                    raise Conflict("Upstream publication is unconfirmed")
            if (
                request.stage == "S1"
                and conn.execute(
                    """SELECT 1 FROM s1_extractions WHERE tenant_id=? AND request_id=?
                   AND response_json IS NULL""",
                    key[:2],
                ).fetchone()
                is not None
            ):
                raise Conflict("S1 extraction is pending")
            previous = conn.execute(
                """SELECT attempt,status FROM transaction_stages WHERE tenant_id=?
                   AND request_id=? AND stage=? ORDER BY attempt DESC LIMIT 1""",
                key,
            ).fetchone()
            expected = 1 if previous is None else previous["attempt"] + 1
            if request.attempt != expected or (
                previous is not None and previous["status"] in ("running", "completed")
            ):
                raise Conflict("Attempt number is unavailable")
            conn.execute(
                """INSERT INTO transaction_stages
                   (tenant_id,request_id,stage,attempt,status,input_sha256,started_at)
                   VALUES (?,?,?,?,'running',?,?)""",
                (*key, request.attempt, fingerprint, utc_now()),
            )
        return Attempt("reserved")

    def complete(
        self,
        request: WorkerRequestEnvelope,
        response: WorkerResponseEnvelope,
        *,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        prompt_versions: list[str] | None = None,
    ) -> None:
        key = identity(request)
        if identity(response) != key or any(
            getattr(request, field) != getattr(response, field)
            for field in ("schema_version", "attempt", "profile_slug", "profile_revision")
        ):
            raise Conflict("Worker response correlation mismatch")
        now = utc_now()
        with self.db.transaction() as conn:
            result = conn.execute(
                """UPDATE transaction_stages SET status='completed',response_json=?,finished_at=?,
                   mode=?,fallback_reason=?,tokens_in=?,tokens_out=?,cost_rub_e8=?,
                   prompt_versions_json=? WHERE tenant_id=? AND request_id=? AND stage=?
                   AND attempt=? AND status='running' AND input_sha256=?""",
                (
                    response.model_dump_json(),
                    now,
                    response.mode,
                    response.fallback_reason,
                    tokens_in,
                    tokens_out,
                    cost_atoms(response.cost_rub),
                    None if prompt_versions is None else json.dumps(prompt_versions),
                    *key,
                    request.attempt,
                    digest(request.model_dump(mode="json")),
                ),
            )
            if result.rowcount != 1:
                raise Conflict("Attempt is closed or was not reserved")
            self._rollup(
                conn,
                key[:2],
                "waiting_human" if response.need_human else "running",
                response.need_human,
            )

    def fail(
        self,
        request: WorkerRequestEnvelope,
        error_code: str,
        *,
        interrupted: bool = False,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cost_rub: Decimal | None = None,
    ) -> bool:
        key = identity(request)
        with self.db.transaction() as conn:
            result = conn.execute(
                """UPDATE transaction_stages SET status=?,error_code=?,finished_at=?,
                   tokens_in=?,tokens_out=?,cost_rub_e8=? WHERE tenant_id=? AND request_id=?
                   AND stage=? AND attempt=? AND status='running' AND input_sha256=?""",
                (
                    "interrupted" if interrupted else "failed",
                    error_code,
                    utc_now(),
                    tokens_in,
                    tokens_out,
                    cost_atoms(cost_rub),
                    *key,
                    request.attempt,
                    digest(request.model_dump(mode="json")),
                ),
            )
            if result.rowcount:
                self._rollup(conn, key[:2], "running", None)
            return result.rowcount == 1

    @classmethod
    def refresh_extraction(
        cls,
        conn: sqlite3.Connection,
        key: tuple[str, str],
        need_human: bool,
    ) -> None:
        cls._rollup(conn, key, "waiting_human" if need_human else "running", need_human)

    @staticmethod
    def _rollup(
        conn: sqlite3.Connection,
        key: tuple[str, str],
        outcome: Outcome,
        need_human: bool | None,
    ) -> None:
        rows = conn.execute(
            """SELECT tokens_in,tokens_out,cost_rub_e8,mode,fallback_reason,finished_at,
                      stage,attempt FROM transaction_stages WHERE tenant_id=? AND request_id=?
               UNION ALL
               SELECT tokens_in,tokens_out,cost_rub_e8,mode,fallback_reason,finished_at,
                      'S1',0 FROM s1_extractions WHERE tenant_id=? AND request_id=?
               ORDER BY finished_at, stage, attempt""",
            (*key, *key),
        ).fetchall()
        totals = []
        for field in ("tokens_in", "tokens_out", "cost_rub_e8"):
            values: list[int | None] = [row[field] for row in rows]
            total = sum(value for value in values if value is not None)
            if total >= 2**63:
                raise OverflowError(f"{field} exceeds SQLite range")
            totals.append(None if None in values else total)
        modes: list[Mode | None] = [row["mode"] for row in rows]
        reason = next((row["fallback_reason"] for row in rows if row["mode"] == "fallback"), None)
        mode = (
            "fallback"
            if reason
            else "llm"
            if "llm" in modes
            else next(
                (value for value in modes if value is not None),
                None,
            )
        )
        conn.execute(
            """UPDATE transaction_telemetry SET tokens_in=?,tokens_out=?,cost_rub_e8=?,mode=?,
               fallback_reason=?,outcome=?,need_human=COALESCE(?,need_human),updated_at=?
               WHERE tenant_id=? AND request_id=?""",
            (*totals, mode, reason, outcome, need_human, utc_now(), *key),
        )

    def remaining_budget(self, tenant_id: str, request_id: UUID, limit: Decimal) -> Decimal:
        with self.db.connect() as conn:
            rows = conn.execute(
                """SELECT cost_rub_e8 FROM transaction_stages
                   WHERE tenant_id=? AND request_id=? AND status<>'running'
                   UNION ALL
                   SELECT cost_rub_e8 FROM s1_extractions
                   WHERE tenant_id=? AND request_id=?""",
                (tenant_id, str(request_id), tenant_id, str(request_id)),
            ).fetchall()
        if any(row[0] is None for row in rows):
            return Decimal(0)
        subtotal = sum(row[0] for row in rows)
        return max(Decimal(0), limit - Decimal(subtotal).scaleb(-8))

    def record_cycle(
        self,
        tenant_id: str,
        request_id: UUID,
        *,
        outcome: Outcome,
        processing_time_sec: float | None,
        saved_minutes: float | None,
        ab_variant: str | None,
    ) -> bool:
        with self.db.transaction() as conn:
            return (
                conn.execute(
                    """UPDATE transaction_telemetry SET outcome=?,processing_time_sec=?,
                   saved_minutes=?,ab_variant=?,updated_at=? WHERE tenant_id=? AND request_id=?""",
                    (
                        outcome,
                        processing_time_sec,
                        saved_minutes,
                        ab_variant,
                        utc_now(),
                        tenant_id,
                        str(request_id),
                    ),
                ).rowcount
                == 1
            )
