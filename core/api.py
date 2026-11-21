import asyncio
import secrets
import sqlite3
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from core.b24_client import BitrixClient, BitrixError, PortalLimiter
from core.config import Settings
from core.contracts import Stage, WorkerRequestEnvelope, WorkerResponseEnvelope
from core.database import Database
from core.idempotency import Conflict, Delivery, Journal, PublicationKey, Receipt
from core.interfaces import BaseSmartWorker
from core.llm_gateway import CycleBudget, LLMGateway
from core.publisher import Publisher
from core.telemetry import Telemetry
from modules.s1_kyc.contracts import S1EvaluateCall, S1ParseCall
from modules.s1_kyc.module import S1ParseReply, S1Worker
from modules.s1_kyc.publication import S1PublicationRequest, prepare_s1_publication
from modules.s2_tender.contracts import S2ParseCall
from modules.s2_tender.extractions import S2ParseReply
from modules.s2_tender.intake import (
    S2GateReply,
    S2GateRequest,
    TenderIntake,
    TenderReceipt,
    confirm_tender_decision,
    read_s1_gates,
    register_tender,
)
from modules.s2_tender.module import S2Worker
from modules.s2_tender.publication import S2PublicationRequest, prepare_s2_publication
from modules.s3_claims.contracts import (
    S3IntakeReceipt,
    S3IntakeRequest,
    S3ParseCall,
    S3ParseReply,
    S3PublicationRequest,
    S3TenderGateRequest,
    S3TenderGateResponse,
)
from modules.s3_claims.module import S3Worker, context
from modules.s3_claims.publication import (
    S3DecisionNotice,
    confirm_decision,
    prepare,
    resolve_decision_notice,
)


def create_app(
    settings: Settings,
    workers: Mapping[Stage, BaseSmartWorker] | None = None,
    *,
    publisher: Publisher | None = None,
    gateway: LLMGateway | None = None,
) -> FastAPI:
    db = Database(settings.sqlite_path)
    journal, telemetry = Journal(db), Telemetry(db)
    registered = dict(workers or {})
    llm = gateway if gateway is not None else LLMGateway(settings)
    s1 = S1Worker(settings.profiles_path, llm, db)
    registered.setdefault("S1", s1)
    s2 = S2Worker(settings.profiles_path, llm, db)
    s3 = S3Worker(settings.profiles_path, llm, db)
    if workers is None:
        registered["S2"] = s2
        registered["S3"] = s3

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await asyncio.to_thread(db.initialize)
        try:
            yield
        finally:
            if gateway is None:
                await llm.aclose()
            if publisher is not None:
                await publisher.client.aclose()

    app = FastAPI(lifespan=lifespan)

    def authenticate(
        token: Annotated[str | None, Header(alias="X-Smartflow-Token")] = None,
    ) -> None:
        expected = settings.worker_token
        if (
            expected is None
            or not token
            or not secrets.compare_digest(
                token.encode(),
                expected.get_secret_value().encode(),
            )
        ):
            raise HTTPException(401, "Invalid service credentials")

    @app.exception_handler(Conflict)
    async def conflict(request: Request, exc: Conflict) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(sqlite3.OperationalError)
    async def unavailable(request: Request, exc: sqlite3.OperationalError) -> JSONResponse:
        return JSONResponse({"detail": "Journal temporarily unavailable"}, status_code=503)

    @app.get("/health")
    def health() -> dict[str, str]:
        with db.connect() as conn:
            if conn.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                raise HTTPException(503, "WAL not available")
            conn.execute("SELECT 1 FROM idempotency_records LIMIT 1")
        return {"status": "ok"}

    @app.post("/v1/deliveries", dependencies=[Depends(authenticate)])
    def register(delivery: Delivery) -> Receipt:
        return journal.register(delivery)

    @app.post(
        "/v1/workers/{stage}/run",
        response_model=WorkerResponseEnvelope,
        dependencies=[Depends(authenticate)],
    )
    async def run(
        stage: Stage, envelope: WorkerRequestEnvelope
    ) -> WorkerResponseEnvelope | JSONResponse:
        if stage != envelope.stage:
            raise HTTPException(422, "Stage and route differ")
        worker = registered.get(stage)
        if worker is None:
            raise HTTPException(503, "No worker registered for this stage")
        try:
            if isinstance(worker, (S1Worker, S2Worker)):
                await asyncio.to_thread(worker.validate_request, envelope)
            else:
                worker.request_payload.model_validate(envelope.payload)
                if isinstance(worker, S3Worker):
                    await asyncio.to_thread(worker.profile, context(envelope))
        except ValueError as exc:
            if isinstance(exc, Conflict):
                raise
            raise HTTPException(422, "Invalid worker input payload") from None
        attempt = await asyncio.to_thread(
            telemetry.reserve,
            envelope,
            verify_s2_in_worker=isinstance(worker, S2Worker),
            pin_attempt=partial(worker.pin_attempt, envelope)
            if isinstance(worker, S3Worker)
            else None,
        )
        if attempt.response is not None:
            return attempt.response
        if attempt.status == "running":
            return JSONResponse(
                {"status": "running"}, status_code=202, headers={"Retry-After": "1"}
            )
        try:
            budget = CycleBudget(
                await asyncio.to_thread(
                    telemetry.remaining_budget,
                    envelope.tenant_id,
                    envelope.request_id,
                    settings.max_cycle_cost_rub,
                )
            )
            result = await worker.run(envelope, budget)
            if result.envelope.payload is not None:
                worker.response_payload.model_validate(result.envelope.payload)
            await asyncio.to_thread(
                telemetry.complete,
                envelope,
                result.envelope,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                prompt_versions=list(result.prompt_versions),
            )
        except asyncio.CancelledError:
            await asyncio.shield(
                asyncio.to_thread(
                    telemetry.fail,
                    envelope,
                    "worker_interrupted",
                    interrupted=True,
                )
            )
            raise
        except Exception:  # noqa: BLE001 - isolate failures from registered plugins
            await asyncio.to_thread(telemetry.fail, envelope, "worker_failed")
            raise HTTPException(503, "Worker failed") from None
        return result.envelope

    @app.post(
        "/s1/parse",
        response_model=S1ParseReply,
        dependencies=[Depends(authenticate)],
    )
    async def parse_s1(call: S1ParseCall) -> S1ParseReply | JSONResponse:
        limit = await asyncio.to_thread(
            telemetry.remaining_budget,
            call.tenant_id,
            call.request_id,
            settings.max_cycle_cost_rub,
        )
        reply = await asyncio.shield(s1.parse(call, CycleBudget(limit)))
        if reply == "running":
            return JSONResponse(
                {"status": "running"},
                status_code=202,
                headers={"Retry-After": "1"},
            )
        return reply

    @app.post(
        "/s1/evaluate",
        response_model=WorkerResponseEnvelope,
        dependencies=[Depends(authenticate)],
    )
    async def evaluate_s1(call: S1EvaluateCall) -> WorkerResponseEnvelope | JSONResponse:
        envelope = WorkerRequestEnvelope(
            **call.model_dump(exclude={"payload"}),
            schema_version="1",
            stage="S1",
            upstream_item=None,
            payload=call.payload.model_dump(mode="json"),
        )
        return await run("S1", envelope)

    @app.post("/s1/publication/prepare", dependencies=[Depends(authenticate)])
    def prepare_s1(call: S1PublicationRequest) -> PublicationKey:
        if publisher is None:
            raise HTTPException(503, "No publication dispatcher configured")
        return prepare_s1_publication(journal, settings.profiles_path, publisher.portal_id, call)

    @app.post("/s2/intake", dependencies=[Depends(authenticate)])
    def intake_s2(call: TenderIntake) -> TenderReceipt:
        if publisher is None:
            raise HTTPException(503, "No publication dispatcher configured")
        return register_tender(journal, settings.profiles_path, publisher.portal_id, call)

    @app.post("/s2/supplier-gates", dependencies=[Depends(authenticate)])
    async def gates_s2(call: S2GateRequest) -> S2GateReply:
        if publisher is None:
            raise HTTPException(503, "No publication dispatcher configured")
        try:
            return await read_s1_gates(publisher, settings.profiles_path, call)
        except BitrixError:
            raise HTTPException(503, "S1 portal read unavailable") from None

    @app.post("/s2/publication/decision", dependencies=[Depends(authenticate)], status_code=204)
    async def decision_s2(call: S2PublicationRequest) -> None:
        if publisher is None:
            raise HTTPException(503, "No publication dispatcher configured")
        try:
            await confirm_tender_decision(publisher, settings.profiles_path, call)
        except BitrixError:
            raise HTTPException(503, "S2 decision portal read unavailable") from None

    @app.post("/s2/parse", response_model=S2ParseReply, dependencies=[Depends(authenticate)])
    async def parse_s2(call: S2ParseCall) -> S2ParseReply | JSONResponse:
        reply = await asyncio.shield(s2.parse(call, settings.max_cycle_cost_rub))
        if reply == "running":
            return JSONResponse(
                {"status": "running"}, status_code=202, headers={"Retry-After": "1"}
            )
        return reply

    @app.post(
        "/s2/evaluate", response_model=WorkerResponseEnvelope, dependencies=[Depends(authenticate)]
    )
    async def evaluate_s2(envelope: WorkerRequestEnvelope) -> WorkerResponseEnvelope | JSONResponse:
        return await run("S2", envelope)

    @app.post("/s2/publication/prepare", dependencies=[Depends(authenticate)])
    def prepare_s2(call: S2PublicationRequest) -> PublicationKey:
        if publisher is None:
            raise HTTPException(503, "No publication dispatcher configured")
        return prepare_s2_publication(journal, settings.profiles_path, publisher.portal_id, call)

    @app.post("/s3/intake", dependencies=[Depends(authenticate)])
    def intake_s3(call: S3IntakeRequest) -> S3IntakeReceipt:
        return s3.intake(call)

    @app.post("/s3/tender-gate", dependencies=[Depends(authenticate)])
    async def gate_s3(call: S3TenderGateRequest) -> S3TenderGateResponse:
        if publisher is None:
            raise HTTPException(503, "No publication dispatcher configured")
        try:
            return await s3.gate(call, publisher)
        except BitrixError:
            raise HTTPException(503, "S2 portal read unavailable") from None

    @app.post("/s3/parse", response_model=S3ParseReply, dependencies=[Depends(authenticate)])
    async def parse_s3(call: S3ParseCall) -> S3ParseReply | JSONResponse:
        reply = await asyncio.shield(s3.parse(call, settings.max_cycle_cost_rub))
        if reply == "running":
            return JSONResponse(
                {"status": "running"}, status_code=202, headers={"Retry-After": "1"}
            )
        return reply

    @app.post(
        "/s3/evaluate", response_model=WorkerResponseEnvelope, dependencies=[Depends(authenticate)]
    )
    async def evaluate_s3(envelope: WorkerRequestEnvelope) -> WorkerResponseEnvelope | JSONResponse:
        return await run("S3", envelope)

    @app.post("/s3/publication/resolve", dependencies=[Depends(authenticate)])
    def resolve_s3(call: S3DecisionNotice) -> S3PublicationRequest:
        if publisher is None:
            raise HTTPException(503, "No publication dispatcher configured")
        return resolve_decision_notice(s3, publisher, call)

    @app.post("/s3/publication/decision", dependencies=[Depends(authenticate)], status_code=204)
    async def decision_s3(call: S3PublicationRequest) -> None:
        if publisher is None:
            raise HTTPException(503, "No publication dispatcher configured")
        try:
            await confirm_decision(s3, publisher, call)
        except BitrixError:
            raise HTTPException(503, "S3 decision portal read unavailable") from None

    @app.post("/s3/publication/prepare", dependencies=[Depends(authenticate)])
    async def prepare_s3(call: S3PublicationRequest) -> PublicationKey:
        if publisher is None:
            raise HTTPException(503, "No publication dispatcher configured")
        try:
            return await prepare(s3, publisher, call)
        except BitrixError:
            raise HTTPException(503, "S2 portal read unavailable") from None

    @app.post("/v1/publications/dispatch", dependencies=[Depends(authenticate)])
    async def dispatch(key: PublicationKey) -> dict[str, str | int | None]:
        if publisher is None:
            raise HTTPException(503, "No publication dispatcher configured")
        try:
            state = await publisher.dispatch(key)
        except BitrixError:
            raise HTTPException(502, "Bitrix24 write requires reconciliation") from None
        return {
            "status": state.status,
            "item_id": state.item_id,
            "task_id": state.task_id,
            **({"comment_id": state.comment_id} if state.command.timeline_comment else {}),
        }

    return app


def app_factory() -> FastAPI:
    settings = Settings()
    if settings.bitrix24_webhook_url is None:
        return create_app(settings)
    url = settings.bitrix24_webhook_url.get_secret_value()
    client = BitrixClient(url, PortalLimiter(settings.bitrix24_rate_limit_rps))
    return create_app(
        settings,
        publisher=Publisher(Journal(Database(settings.sqlite_path)), client, client.portal_id),
    )
