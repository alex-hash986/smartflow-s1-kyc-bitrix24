import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr, StrictBool

from core.api import create_app
from core.b24_client import BitrixClient, PortalLimiter
from core.config import Settings
from core.contracts import Nonempty, StrictDTO, WorkerRequestEnvelope, WorkerResponseEnvelope
from core.database import Database
from core.idempotency import Delivery, Journal, PublicationCommand, PublicationKey, TaskCommand
from core.interfaces import BaseSmartWorker, WorkerResult
from core.llm_gateway import CycleBudget
from core.publisher import Publisher


class Input(StrictDTO):
    text: Nonempty


class Output(StrictDTO):
    answer: StrictBool


class TestWorker(BaseSmartWorker):
    __test__ = False
    request_payload = Input
    response_payload = Output

    def __init__(self, response: WorkerResponseEnvelope) -> None:
        self.response = response
        self.calls = 0

    async def run(self, request: WorkerRequestEnvelope, budget: CycleBudget) -> WorkerResult:
        self.calls += 1
        return WorkerResult(self.response, 0, 0)


@pytest.mark.parametrize("fail_task", [False, True])
@pytest.mark.asyncio
async def test_workflow_boundary(
    db: Database,
    delivery: Delivery,
    envelope: WorkerRequestEnvelope,
    result: WorkerResponseEnvelope,
    fail_task: bool,
) -> None:
    journal = Journal(db)
    calls: list[str] = []
    marker = ""

    async def send(request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        calls.append(method)
        if method == "crm.item.add":
            assert json.loads(request.content)["fields"]["ufExternal"] == marker
        if method == "tasks.task.add" and fail_task:
            raise httpx.ReadTimeout("simulated", request=request)
        return httpx.Response(
            200,
            json={
                "result": (
                    {
                        "task": {
                            "id": 42,
                            "ufCrmTask": ["Tb1_29"],
                            "description": f"[smartflow:{marker}]",
                        }
                    }
                    if method.startswith("tasks.")
                    else {"item": {"id": 29, "ufExternal": marker}}
                )
            },
        )

    async def skip_sleep(_: float) -> None:
        return None

    bitrix = BitrixClient(
        "https://crm.invalid/rest/1/mock/",
        PortalLimiter(sleep=skip_sleep),
        transport=httpx.MockTransport(send),
    )
    worker = TestWorker(result)
    app = create_app(
        Settings(sqlite_path=db.path, worker_token=SecretStr("local")),
        {"S1": worker},
        publisher=Publisher(journal, bitrix, "crm.invalid"),
    )
    headers = {"X-Smartflow-Token": "local"}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://kernel.invalid",
        ) as client,
    ):
        assert (await client.get("/health")).json() == {"status": "ok"}
        route = "/v1/workers/S1/run"
        body = envelope.model_dump(mode="json")
        assert (await client.post(route, json=body)).status_code == 401
        registered = await client.post(
            "/v1/deliveries", json=delivery.model_dump(mode="json"), headers=headers
        )
        assert registered.json()["request_id"] == str(envelope.request_id)
        invalid = await client.post(
            route, json=body | {"payload": {"text": "x", "extra": True}}, headers=headers
        )
        assert invalid.status_code == 422
        assert (await client.post(route, json=body, headers=headers)).json() == result.model_dump(
            mode="json"
        )
        assert (await client.post(route, json=body, headers=headers)).status_code == 200
        assert worker.calls == 1
        key = PublicationKey(
            tenant_id=envelope.tenant_id,
            request_id=envelope.request_id,
            stage="S1",
            purpose="stage_result",
        )
        dispatch = "/v1/publications/dispatch"
        key_json = key.model_dump(mode="json")
        assert (await client.post(dispatch, json=key_json, headers=headers)).status_code == 409
        marker = journal.approve(
            PublicationCommand(
                key=key,
                stage_attempt=1,
                portal_id="crm.invalid",
                approval_ref="policy:v1",
                entity_type_id=177,
                task_required=True,
                fields={"title": "x"},
                external_key_field="ufExternal",
                task=TaskCommand(title="t", responsible_id=7),
            )
        )
        outcome = await client.post(dispatch, json=key_json, headers=headers)
        assert outcome.status_code == (502 if fail_task else 200)
        replay = (await client.post(dispatch, json=key_json, headers=headers)).json()
        assert replay == {
            "status": "reconcile" if fail_task else "published",
            "item_id": 29,
            "task_id": None if fail_task else 42,
        }
        assert calls[:3] == ["crm.item.add", "crm.item.get", "tasks.task.add"]
        assert calls.count("tasks.task.add") == 1


def test_n8n_graph() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = json.loads(
        (root / "workflows/_base/base_smartflow_template.json").read_text(
            encoding="utf-8",
        )
    )
    nodes = {node["name"]: node for node in workflow["nodes"]}
    routes = workflow["connections"]
    assert workflow["active"] is False
    assert routes["Need Human"]["main"][0][0]["node"] == "Respond"
    assert routes["Need Human"]["main"][1][0]["node"] == "Wait 600ms"
    assert nodes["Wait 600ms"]["parameters"]["amount"] == 0.6
    assert nodes["Bitrix24 REST Dispatcher"]["parameters"]["url"].endswith(
        "/v1/publications/dispatch"
    )
