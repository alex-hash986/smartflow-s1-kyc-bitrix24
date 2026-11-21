import asyncio

import httpx
import pytest
from pydantic import SecretStr, StrictBool

from core.api import create_app
from core.config import Settings
from core.contracts import (
    ItemRef,
    Nonempty,
    Stage,
    StrictDTO,
    WorkerRequestEnvelope,
    WorkerResponseEnvelope,
)
from core.database import Database
from core.idempotency import Conflict, Journal, PublicationCommand, PublicationKey
from core.interfaces import BaseSmartWorker, WorkerResult
from core.llm_gateway import CycleBudget
from core.telemetry import Telemetry


class WorkerInput(StrictDTO):
    text: Nonempty


class WorkerOutput(StrictDTO):
    answer: StrictBool


class PausedWorker(BaseSmartWorker):
    request_payload = WorkerInput
    response_payload = WorkerOutput

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.resume = asyncio.Event()

    async def run(self, request: WorkerRequestEnvelope, budget: CycleBudget) -> WorkerResult:
        self.started.set()
        await self.resume.wait()
        raise RuntimeError("simulated worker failure")


@pytest.mark.asyncio
async def test_running_replay_and_failure_record(
    db: Database,
    envelope: WorkerRequestEnvelope,
) -> None:
    worker = PausedWorker()
    settings = Settings(sqlite_path=db.path, worker_token=SecretStr("test"))
    app = create_app(settings, {"S1": worker})
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://kernel.invalid",
            headers={"X-Smartflow-Token": "test"},
        ) as client,
    ):
        body = envelope.model_dump(mode="json")
        route = "/v1/workers/S1/run"
        assert (await client.post("/v1/workers/S2/run", json=body)).status_code == 422
        assert (
            await client.post(
                "/v1/workers/S2/run",
                json=body
                | {
                    "stage": "S2",
                    "upstream_item": {"entity_type_id": 1, "item_id": 1},
                },
            )
        ).status_code == 503
        pending = asyncio.create_task(client.post(route, json=body))
        try:
            await asyncio.wait_for(worker.started.wait(), timeout=5)
            replay = await client.post(route, json=body)
            assert replay.status_code == 202 and replay.headers["retry-after"] == "1"
            assert (
                await client.post(
                    route,
                    json=body
                    | {
                        "payload": {"text": "changed"},
                    },
                )
            ).status_code == 409
        finally:
            worker.resume.set()
        assert (await pending).status_code == 503
        assert (await client.post(route, json=body)).status_code == 409
    with db.connect() as conn:
        row = conn.execute("SELECT status,error_code FROM transaction_stages").fetchone()
        assert tuple(row) == ("failed", "worker_failed")


def test_escalation_and_terminal_cycle(
    db: Database,
    envelope: WorkerRequestEnvelope,
    result: WorkerResponseEnvelope,
) -> None:
    telemetry, journal = Telemetry(db), Journal(db)
    stages: tuple[Stage, ...] = ("S1", "S2", "S3")
    for index, stage in enumerate(stages, 1):
        request = envelope.model_copy(
            update={
                "stage": stage,
                "upstream_item": None
                if index == 1
                else ItemRef(entity_type_id=177, item_id=index - 1),
            }
        )
        current = WorkerResponseEnvelope.model_validate(
            result.model_dump()
            | {
                "stage": stage,
                "need_human": index == 1,
                "escalation_reasons": ["no_data"] if index == 1 else [],
            }
        )
        telemetry.reserve(request)
        telemetry.complete(request, current)
        key = PublicationKey(
            tenant_id=request.tenant_id,
            request_id=request.request_id,
            stage=stage,
            purpose="stage_result",
        )
        command = PublicationCommand(
            key=key,
            stage_attempt=1,
            portal_id="portal",
            approval_ref="decision-v1",
            entity_type_id=177,
            task_required=False,
            fields={},
            external_key_field="ufExternal",
        )
        if index == 1:
            with pytest.raises(Conflict, match="human decision"):
                journal.approve(command)
        journal.approve(command, human_approved=index == 1)
        token = journal.claim(key)
        assert token is not None
        assert journal.begin_step(key, token, "item")
        assert journal.record_step(key, token, "item", index)
        assert journal.finish(key, token)
    with db.connect() as conn:
        row = conn.execute(
            "SELECT outcome,processing_time_sec FROM transaction_telemetry",
        ).fetchone()
        assert row["outcome"] == "completed" and row["processing_time_sec"] >= 0
