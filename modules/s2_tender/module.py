import asyncio
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import ValidationError

from core.contracts import WorkerRequestEnvelope, WorkerResponseEnvelope
from core.database import Database
from core.idempotency import Conflict, PublicationCommand
from core.interfaces import BaseSmartWorker, WorkerResult
from core.llm_gateway import CycleBudget, LLMGateway
from core.profile_loader import Profile, load_profile
from core.telemetry import Telemetry
from modules.s2_tender.contracts import (
    S2Context,
    S2EvaluateTenderRequest,
    S2EvaluateTenderResponse,
    S2ParseCall,
    S2SupplierGateSnapshot,
)
from modules.s2_tender.extractions import ExtractionJournal, S2ParseReply, make_reply
from modules.s2_tender.parser import ParseResult, QuoteParser
from modules.s2_tender.scoring import UPSTREAM_UNCONFIRMED, evaluate


def profile_for(profiles_path: Path, context: S2Context) -> Profile:
    try:
        profile = load_profile(profiles_path, context.profile_slug, context.profile_revision)
    except (OSError, ValueError) as exc:
        raise Conflict("S2 profile unavailable") from exc
    if (
        profile.s2_tender is None
        or profile.revision != context.profile_revision
        or (profile.tenant_id is not None and profile.tenant_id != context.tenant_id)
    ):
        raise Conflict("S2 profile does not match delivery context")
    return profile


def published_gate(profile: Profile, row: sqlite3.Row) -> S2SupplierGateSnapshot | None:
    assert profile.s1_kyc is not None
    mapping = profile.s1_kyc.bitrix24
    command = PublicationCommand.model_validate_json(row["command_json"])
    if command.portal_id != profile.portal_id:
        return None
    values = {name: command.fields.get(code) for name, code in mapping.fields.items() if code}
    status_id = values.get("KYC_STATUS")
    status = next(
        (
            name
            for name, value in mapping.kyc_status_values.items()
            if (type(status_id) is int and status_id == value)
        ),
        None,
    )
    if status is None:
        return None
    stage = profile.stages["S1"].stage_ids
    if stage is None or command.fields.get("stageId") != (
        stage.approved if status == "APPROVED" else stage.rejected
    ):
        return None
    try:
        return S2SupplierGateSnapshot.model_validate(
            {
                "s1_item_id": row["item_id"],
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


class S2Worker(BaseSmartWorker):
    request_payload = S2EvaluateTenderRequest
    response_payload = S2EvaluateTenderResponse

    def __init__(self, profiles_path: Path, gateway: LLMGateway, db: Database) -> None:
        self.profiles_path = profiles_path
        self.db = db
        self.parser = QuoteParser(gateway)
        self.extractions = ExtractionJournal(db)
        self.telemetry = Telemetry(db)

    async def parse(self, call: S2ParseCall, limit: Decimal) -> S2ParseReply | Literal["running"]:
        profile = await asyncio.to_thread(profile_for, self.profiles_path, call)
        assert profile.s2_tender is not None
        batch = await asyncio.to_thread(self.extractions.reserve, call)
        if batch == "running":
            return batch
        if batch.pending:
            results: dict[UUID, ParseResult] = {}
            try:
                remaining = await asyncio.to_thread(
                    self.telemetry.remaining_budget,
                    call.tenant_id,
                    call.request_id,
                    limit,
                    ignore_pending_s2=True,
                )
                budget = CycleBudget(remaining)
                for source in batch.pending:
                    results[source.quote_id] = await self.parser.parse(
                        source, profile.s2_tender.extraction, budget
                    )
            except (Exception, asyncio.CancelledError):
                # Leave unfinished reservations for reconciliation if spend cannot be verified.
                if results:
                    await asyncio.shield(
                        asyncio.to_thread(self.extractions.complete, call, results)
                    )
                raise
            batch.records.update(await asyncio.to_thread(self.extractions.complete, call, results))
        return make_reply(call, [batch.records[source.quote_id] for source in call.payload.quotes])

    def validate_request(self, request: WorkerRequestEnvelope) -> None:
        self.request_payload.model_validate(request.payload)
        profile_for(self.profiles_path, self.context(request))

    @staticmethod
    def context(request: WorkerRequestEnvelope) -> S2Context:
        return S2Context.model_validate(request.model_dump(include=set(S2Context.model_fields)))

    def compute(self, request: WorkerRequestEnvelope) -> WorkerResult:
        profile = profile_for(self.profiles_path, self.context(request))
        assert profile.s2_tender is not None and request.upstream_item is not None
        payload = self.request_payload.model_validate(request.payload)
        key = (request.tenant_id, str(request.request_id))
        stage = profile.stages.get("S1")
        verified_gates = []
        upstream_confirmed = False
        with self.db.connect() as conn:
            conn.execute("BEGIN")
            verified_quotes = {
                quote.quote_id
                for quote in payload.quotes
                if self.extractions.verified(conn, key, quote)
            }
            if profile.s1_kyc is not None and profile.portal_id is not None and stage is not None:
                snapshots = {gate.s1_item_id: gate for gate in payload.supplier_gate_snapshots}
                for item_id in sorted(snapshots.keys() | {request.upstream_item.item_id}):
                    rows = conn.execute(
                        """SELECT request_id,item_id,command_json FROM crm_publications
                           WHERE tenant_id=? AND portal_id=? AND stage='S1' AND purpose='stage_result'
                           AND status='published' AND entity_type_id=? AND item_id=? LIMIT 2""",
                        (request.tenant_id, profile.portal_id, stage.entity_type_id, item_id),
                    ).fetchall()
                    if len(rows) != 1 or (gate := published_gate(profile, rows[0])) is None:
                        continue
                    if snapshots.get(item_id) == gate:
                        verified_gates.append(gate)
                    if (
                        item_id == request.upstream_item.item_id
                        and request.upstream_item.entity_type_id == stage.entity_type_id
                        and rows[0]["request_id"] == key[1]
                        and gate.kyc_status == "APPROVED"
                    ):
                        upstream_confirmed = True
        payload = payload.model_copy(update={"supplier_gate_snapshots": verified_gates})
        result, reasons = evaluate(
            payload,
            profile.s2_tender,
            upstream_confirmed=upstream_confirmed,
            verified_quote_ids=verified_quotes,
        )
        return WorkerResult(
            WorkerResponseEnvelope(
                **request.model_dump(exclude={"payload", "received_at", "upstream_item"}),
                payload=result.model_dump(mode="json"),
                need_human=bool(reasons),
                escalation_reasons=reasons,
                cost_rub=Decimal(0),
                mode="deterministic" if upstream_confirmed else "fallback",
                fallback_reason=None if upstream_confirmed else UPSTREAM_UNCONFIRMED,
            ),
            tokens_in=0,
            tokens_out=0,
        )

    async def run(self, request: WorkerRequestEnvelope, budget: CycleBudget) -> WorkerResult:
        return await asyncio.to_thread(self.compute, request)
