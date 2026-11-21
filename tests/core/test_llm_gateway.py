import asyncio
import json
from decimal import Decimal

import httpx
import pytest
from pydantic import AnyHttpUrl, StrictInt

from core.config import ModelRate, Settings
from core.contracts import StrictDTO
from core.llm_gateway import CycleBudget, LLMGateway


class Child(StrictDTO):
    count: StrictInt


class Output(StrictDTO):
    items: list[Child]


@pytest.fixture
def settings() -> Settings:
    rate = ModelRate(input_usd_per_million=Decimal(1), output_usd_per_million=Decimal(2))
    return Settings(
        base_url=AnyHttpUrl("https://llm.invalid/v1"),
        fast_llm_model="fast",
        smart_llm_model="smart",
        embedding_model="embedding",
        llm_model_rates=dict.fromkeys(("fast", "smart", "embedding"), rate),
        usd_to_rub_rate=Decimal(100),
    )


@pytest.mark.asyncio
async def test_structured_and_embedding(settings: Settings) -> None:
    models: list[str] = []

    def send(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        models.append(data["model"])
        if request.url.path == "/v1/embeddings":
            return httpx.Response(
                200, json={"data": [{"embedding": [0.4, 0.2]}], "usage": {"prompt_tokens": 4}}
            )
        assert request.url.path == "/v1/chat/completions"
        assert data["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"items":[{"count":1,"extra":2}],"x":0}\n```'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        )

    gateway = LLMGateway(settings, transport=httpx.MockTransport(send))
    try:
        budget = CycleBudget(Decimal(1))
        result = await gateway.chat_json("FAST_LLM", "prompt", Output, budget)
        assert result.value == Output(items=[Child(count=1)])
        assert (result.mode, result.cost_rub) == ("llm", Decimal(".0008"))
        await gateway.chat_json("SMART_LLM", "prompt", Output, budget)
        embedding = await gateway.embed("example", budget)
        assert embedding.value == [0.4, 0.2] and embedding.cost_rub == Decimal(".0004")
        assert models == ["fast", "smart", "embedding"]
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_preflight_and_parallel_reservations(settings: Settings) -> None:
    calls = 0
    sent = asyncio.Event()
    release = asyncio.Event()

    async def send(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        sent.set()
        await release.wait()
        return httpx.Response(200, json={"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    gateway = LLMGateway(settings, transport=httpx.MockTransport(send))
    try:
        budget = CycleBudget(Decimal(".01"))
        task = asyncio.create_task(
            gateway.chat_json("FAST_LLM", "x", Output, budget, max_output_tokens=1)
        )
        await asyncio.wait_for(sent.wait(), timeout=2)
        skipped = await gateway.chat_json("FAST_LLM", "x", Output, budget, max_output_tokens=1)
        assert skipped.fallback_reason == "budget_exceeded" and skipped.cost_rub == 0
        release.set()
        assert (await task).fallback_reason == "invalid_model_output"
        assert calls == 1
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("timeout", "provider_transport"),
        ("schema", "invalid_model_output"),
        ("usage", "usage_unavailable"),
        ("provider", "provider_error"),
        ("json", "invalid_provider_response"),
    ],
)
async def test_fallbacks(settings: Settings, case: str, reason: str) -> None:
    def send(request: httpx.Request) -> httpx.Response:
        if case == "timeout":
            raise httpx.ReadTimeout("test", request=request)
        if case == "json":
            return httpx.Response(200, text="!")
        if case == "provider":
            return httpx.Response(200, json={"error": {"message": "unknown"}})
        usage = {"usage": {"prompt_tokens": 3, "completion_tokens": 4}} if case == "schema" else {}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"items":[{"count":"wrong"}]}'}}],
                **usage,
            },
        )

    gateway = LLMGateway(settings, transport=httpx.MockTransport(send))
    try:
        budget = CycleBudget(Decimal(1))
        result = await gateway.chat_json("FAST_LLM", "prompt", Output, budget)
        assert (result.value, result.mode, result.fallback_reason) == (None, "fallback", reason)
        assert result.cost_rub == (Decimal(".0011") if case == "schema" else None)
        if result.cost_rub is None:
            assert (await gateway.embed("x", budget)).fallback_reason == "budget_exceeded"
    finally:
        await gateway.aclose()
