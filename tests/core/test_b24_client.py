import asyncio
import json

import httpx
import pytest

from core.b24_client import BitrixClient, BitrixError, PortalLimiter, task_binding


class Clock:
    time = 0.0

    def now(self) -> float:
        return self.time

    async def sleep(self, delay: float) -> None:
        self.time += delay


@pytest.mark.asyncio
async def test_methods_and_shared_rate_limiter() -> None:
    clock = Clock()
    calls: list[tuple[str, float]] = []

    def send(request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        calls.append((method, clock.time))
        body = json.loads(request.content)
        if method == "tasks.task.add":
            assert body["fields"]["UF_CRM_TASK"] == ["Tb1_29"]
        if method == "crm.timeline.comment.add":
            assert body["fields"]["ENTITY_TYPE"] == "DYNAMIC_177"
        if method == "crm.item.list":
            assert body["entityTypeId"] == 177
        responses = {
            "crm.type.list": {"types": [{"entityTypeId": 177}]},
            "crm.status.list": [{"STATUS_ID": "NEW"}],
            "crm.item.list": {"items": [{"id": 29}]},
            "crm.item.get": {"item": {"id": 29}},
            "crm.item.add": {"item": {"id": 29}},
            "crm.item.update": {"item": {"id": "29"}},
            "crm.timeline.comment.add": 33,
            "tasks.task.add": {"task": {"id": "42"}},
            "tasks.task.get": {"task": {"id": "42"}},
        }
        body = {"result": responses[method]}
        if method == "crm.item.list":
            body["next"] = 50
        return httpx.Response(200, json=body)

    limiter = PortalLimiter(clock=clock.now, sleep=clock.sleep)
    transport = httpx.MockTransport(send)
    first = BitrixClient("https://crm.invalid/rest/1/mock/", limiter, transport=transport)
    second = BitrixClient("https://crm.invalid/rest/2/mock/", limiter, transport=transport)
    try:
        types, statuses, items = await asyncio.gather(
            first.type_list(),
            second.status_list(),
            first.item_list(177),
        )
        assert types.items == [{"entityTypeId": 177}]
        assert statuses.items == [{"STATUS_ID": "NEW"}]
        assert items.items == [{"id": 29}] and items.next_start == 50
        assert await second.item_add(177, {"title": "x"}) == 29
        assert await second.item_update(177, 29, {"title": "y"}) == 29
        assert await first.item_get(177, 29) == {"id": 29}
        assert await first.timeline_comment_add(177, 29, "comment") == 33
        assert await first.task_add("title", 1, 177, 29) == 42
        assert await first.task_get(42) == {"id": "42"}
        assert [ts for _, ts in calls] == [i / 2 for i in range(9)]
        assert task_binding(177, 29) == "Tb1_29"
        with pytest.raises(BitrixError):
            task_binding(0, 29)
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.asyncio
async def test_retry_after_reads_and_never_repeat_ambiguous_write() -> None:
    clock = Clock()
    calls: list[float] = []

    def send(request: httpx.Request) -> httpx.Response:
        calls.append(clock.time)
        if len(calls) == 1:
            return httpx.Response(
                503, headers={"Retry-After": "2"}, json={"error": "QUERY_LIMIT_EXCEEDED"}
            )
        if len(calls) == 2:
            return httpx.Response(200, json={"result": {"types": []}})
        raise httpx.ReadTimeout("simulated", request=request)

    client = BitrixClient(
        "https://crm.invalid/rest/1/mock/",
        PortalLimiter(clock=clock.now, sleep=clock.sleep),
        transport=httpx.MockTransport(send),
    )
    try:
        assert (await client.type_list()).items == []
        with pytest.raises(BitrixError, match="transport_error") as problem:
            await client.item_add(177, {})
        assert problem.value.uncertain
        assert calls == [0, 2, 2.5]
    finally:
        await client.aclose()
