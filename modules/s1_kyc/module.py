import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from core.contracts import (
    JSONObject,
    Mode,
    WorkerRequestEnvelope,
    WorkerResponseEnvelope,
    cost_atoms,
)
from core.database import Database, digest, utc_now
from core.idempotency import Conflict
from core.interfaces import BaseSmartWorker, WorkerResult
from core.llm_gateway import CycleBudget, LLMGateway
from core.profile_loader import S1Profile, load_profile
from core.telemetry import Telemetry
from modules.s1_kyc.contracts import (
    S1Context,
    S1EvaluateRequest,
    S1Evaluation,
    S1ParseCall,
    S1ParseResponse,
)
from modules.s1_kyc.parser import ParseResult, RequisitesParser
from modules.s1_kyc.scoring import evaluate


class S1ParseReply(S1Context):
    payload: S1ParseResponse
    mode: Mode
    fallback_reason: str | None
    cost_rub: str | None
    prompt_version: str | None


class ParseJournal:
    def __init__(self, db: Database) -> None:
        self.db = db

    def reserve(self, call: S1ParseCall) -> S1ParseReply | Literal["running"] | None:
        key = (call.tenant_id, str(call.request_id))
        fingerprint = digest(call.payload.model_dump(mode="json"))
        with self.db.transaction() as conn:
            delivery = conn.execute(
                """SELECT profile_slug,profile_revision FROM idempotency_records
                   WHERE tenant_id=? AND request_id=?""",
                key,
            ).fetchone()
            if delivery is None or tuple(delivery) != (call.profile_slug, call.profile_revision):
                raise Conflict("Unregistered request or different delivery context")
            prior = conn.execute(
                """SELECT input_sha256,response_json FROM s1_extractions
                   WHERE tenant_id=? AND request_id=?""",
                key,
            ).fetchone()
            if prior is not None:
                if prior["input_sha256"] != fingerprint:
                    raise Conflict("Extraction source has changed")
                if prior["response_json"] is None:
                    return "running"
                return S1ParseReply.model_validate_json(prior["response_json"])
            if (
                conn.execute(
                    """SELECT 1 FROM s2_extractions WHERE tenant_id=? AND request_id=?
                   AND response_json IS NULL""",
                    key,
                ).fetchone()
                is not None
            ):
                return "running"
            if (
                conn.execute(
                    "SELECT 1 FROM transaction_stages WHERE tenant_id=? AND request_id=?",
                    key,
                ).fetchone()
                is not None
            ):
                raise Conflict("Worker attempt already started")
            conn.execute(
                """INSERT INTO s1_extractions (tenant_id,request_id,input_sha256,started_at)
                   VALUES (?,?,?,?)""",
                (*key, fingerprint, utc_now()),
            )
        return None

    def complete(self, reply: S1ParseReply, result: ParseResult) -> None:
        key = (reply.tenant_id, str(reply.request_id))
        with self.db.transaction() as conn:
            updated = conn.execute(
                """UPDATE s1_extractions SET response_json=?,finished_at=?,mode=?,
                   fallback_reason=?,cost_rub_e8=?,tokens_in=?,tokens_out=?
                   WHERE tenant_id=? AND request_id=? AND response_json IS NULL""",
                (
                    reply.model_dump_json(),
                    utc_now(),
                    result.mode,
                    result.fallback_reason,
                    cost_atoms(result.cost_rub),
                    result.tokens_in,
                    result.tokens_out,
                    *key,
                ),
            )
            if updated.rowcount != 1:
                raise Conflict("Extraction already completed")
            Telemetry.refresh_extraction(conn, key, reply.payload.need_human)


class S1Worker(BaseSmartWorker):
    request_payload = S1EvaluateRequest
    response_payload = S1Evaluation

    def __init__(self, profiles_path: Path, gateway: LLMGateway, db: Database) -> None:
        self.profiles_path = profiles_path
        self.parser = RequisitesParser(gateway)
        self.journal = ParseJournal(db)

    def profile(self, context: S1Context) -> S1Profile:
        try:
            profile = load_profile(
                self.profiles_path, context.profile_slug, context.profile_revision
            )
        except (OSError, ValueError) as exc:
            raise Conflict("S1 profile unavailable") from exc
        if (
            profile.s1_kyc is None
            or profile.revision != context.profile_revision
            or (profile.tenant_id is not None and profile.tenant_id != context.tenant_id)
        ):
            raise Conflict("S1 profile does not match delivery context")
        return profile.s1_kyc

    async def parse(
        self,
        call: S1ParseCall,
        budget: CycleBudget,
    ) -> S1ParseReply | Literal["running"]:
        profile = await asyncio.to_thread(self.profile, call)
        prior = await asyncio.to_thread(self.journal.reserve, call)
        if prior is not None:
            return prior
        try:
            result = await self.parser.parse(call.payload, profile.extraction, budget)
        except (Exception, asyncio.CancelledError):
            # An interrupted provider call can have unknown cost. Never retry it implicitly.
            result = ParseResult(
                S1ParseResponse(
                    inn=None,
                    ogrn=None,
                    kpp=None,
                    legal_name=None,
                    requested_limit_rub=None,
                    need_human=True,
                    escalation_reasons=["PARSE_LLM_UNAVAILABLE"],
                ),
                "fallback",
                "extraction_interrupted",
                None,
                None,
                None,
            )
            await asyncio.shield(self._save(call, result))
            raise
        return await asyncio.shield(self._save(call, result))

    async def _save(self, call: S1ParseCall, result: ParseResult) -> S1ParseReply:
        atoms = cost_atoms(result.cost_rub)
        reply = S1ParseReply(
            **call.model_dump(exclude={"payload"}),
            payload=result.payload,
            mode=result.mode,
            fallback_reason=result.fallback_reason,
            cost_rub=(None if atoms is None else f"{atoms // 10**8}.{atoms % 10**8:08d}"),
            prompt_version=result.prompt_version,
        )
        await asyncio.to_thread(self.journal.complete, reply, result)
        return reply

    def validate_request(self, request: WorkerRequestEnvelope) -> None:
        payload = self.request_payload.model_validate(request.payload)
        if payload.registry.observed_at > datetime.now(UTC):
            raise ValueError("Registry snapshot is from the future")
        self.profile(
            S1Context.model_validate(request.model_dump(include=set(S1Context.model_fields)))
        )

    async def run(self, request: WorkerRequestEnvelope, budget: CycleBudget) -> WorkerResult:
        payload = self.request_payload.model_validate(request.payload)
        context = S1Context.model_validate(request.model_dump(include=set(S1Context.model_fields)))
        result = evaluate(payload, self.profile(context))
        metadata = {"need_human", "escalation_reasons", "cost_rub", "mode", "fallback_reason"}
        data: JSONObject = result.model_dump(mode="json", exclude=metadata)
        response = WorkerResponseEnvelope(
            **request.model_dump(exclude={"payload", "received_at", "upstream_item"}),
            payload=data,
            **result.model_dump(include=metadata),
        )
        return WorkerResult(response, tokens_in=0, tokens_out=0)
