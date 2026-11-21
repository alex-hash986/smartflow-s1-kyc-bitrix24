import asyncio
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Generic, Literal, TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter

from core.config import ModelRate, Settings
from core.contracts import JSONObject, Mode

T = TypeVar("T", bound=BaseModel)
ResultT = TypeVar("ResultT")

Role = Literal["FAST_LLM", "SMART_LLM", "EMBEDDING"]


@dataclass(frozen=True)
class GatewayResult(Generic[ResultT]):
    value: ResultT | None
    mode: Mode
    fallback_reason: str | None
    cost_rub: Decimal | None
    tokens_in: int | None = None
    tokens_out: int | None = None


class CycleBudget:
    def __init__(self, limit: Decimal) -> None:
        self.limit = limit
        if not limit.is_finite() or limit < 0:
            raise ValueError("Invalid cycle limit")
        self._used = Decimal(0)
        self._uncertain = False
        self._lock = asyncio.Lock()

    async def reserve(self, amount: Decimal, ceiling: Decimal) -> bool:
        async with self._lock:
            if self._uncertain or self._used + amount > min(self.limit, ceiling):
                return False
            self._used += amount
            return True

    async def settle(self, reserved: Decimal, actual: Decimal | None) -> None:
        async with self._lock:
            if actual is None:
                self._uncertain = True
            else:
                self._used += actual - reserved


def _cost(rate: ModelRate, usd_to_rub: Decimal, tokens_in: int, tokens_out: int) -> Decimal:
    return (
        (rate.input_usd_per_million * tokens_in + rate.output_usd_per_million * tokens_out)
        * usd_to_rub
        / 1_000_000
    )


def _usage(raw: object) -> tuple[int, int] | None:
    if not isinstance(raw, dict):
        return None
    tokens_in, tokens_out = raw.get("prompt_tokens"), raw.get("completion_tokens", 0)
    if type(tokens_in) is int and type(tokens_out) is int and min(tokens_in, tokens_out) >= 0:
        return tokens_in, tokens_out
    return None


def _filter(value: object, schema: dict[str, object], root: dict[str, object]) -> object:
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        definitions = root.get("$defs")
        if isinstance(definitions, dict):
            target = definitions.get(ref.rsplit("/", 1)[-1])
            if isinstance(target, dict):
                return _filter(value, target, root)
    alternatives = schema.get("anyOf")
    if isinstance(alternatives, list):
        candidates = [
            item for item in alternatives if isinstance(item, dict) and item.get("type") != "null"
        ]
        if value is not None and len(candidates) == 1:
            return _filter(value, candidates[0], root)
    props = schema.get("properties")
    if isinstance(value, dict) and isinstance(props, dict):
        return {
            key: _filter(child, props[key], root) for key, child in value.items() if key in props
        }
    items = schema.get("items")
    if isinstance(value, list) and isinstance(items, dict):
        return [_filter(item, items, root) for item in value]
    return value


class LLMGateway:
    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        headers = {}
        if settings.api_key is not None:
            headers["Authorization"] = "Bearer " + settings.api_key.get_secret_value()
        self._client = httpx.AsyncClient(
            base_url=str(settings.base_url).rstrip("/") + "/",
            headers=headers,
            transport=transport,
            timeout=30,
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat_json(
        self,
        role: Literal["FAST_LLM", "SMART_LLM"],
        prompt: str,
        schema: type[T],
        budget: CycleBudget,
        *,
        max_output_tokens: int = 512,
    ) -> GatewayResult[T]:
        if not 1 <= max_output_tokens <= 32768:
            raise ValueError("Invalid output token limit")
        raw = await self._request(
            role,
            "chat/completions",
            {
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "max_tokens": max_output_tokens,
            },
            prompt,
            budget,
            max_output_tokens,
        )
        value: T | None = None
        reason = raw.fallback_reason
        if raw.value is not None:
            try:
                content = raw.value["choices"][0]["message"]["content"]
                match = re.fullmatch(
                    r"```(?:json)?\s*\n(.*?)\n\s*```", content.strip(), re.DOTALL | re.IGNORECASE
                )
                parsed = json.loads(match[1] if match else content)
                if not isinstance(parsed, dict):
                    raise TypeError("Expected a JSON object")
                shape = schema.model_json_schema()
                value = schema.model_validate(_filter(parsed, shape, shape))
            except (ValueError, AttributeError, KeyError, IndexError, TypeError):
                reason = "invalid_model_output"
        return GatewayResult(
            value,
            "llm" if reason is None else "fallback",
            reason,
            raw.cost_rub,
            raw.tokens_in,
            raw.tokens_out,
        )

    async def embed(self, text: str, budget: CycleBudget) -> GatewayResult[list[float]]:
        raw = await self._request("EMBEDDING", "embeddings", {"input": text}, text, budget, 0)
        value: list[float] | None = None
        reason = raw.fallback_reason
        if raw.value is not None:
            try:
                value = TypeAdapter(list[float]).validate_python(
                    raw.value["data"][0]["embedding"], strict=True
                )
                if not value or any(not float("-inf") < x < float("inf") for x in value):
                    raise ValueError("Invalid embedding")
            except (ValueError, KeyError, IndexError, TypeError):
                value, reason = None, "invalid_model_output"
        return GatewayResult(
            value,
            "llm" if reason is None else "fallback",
            reason,
            raw.cost_rub,
            raw.tokens_in,
            raw.tokens_out,
        )

    async def _request(
        self,
        role: Role,
        path: str,
        body: JSONObject,
        text: str,
        budget: CycleBudget,
        output_limit: int,
    ) -> GatewayResult[dict[str, Any]]:
        model = {
            "FAST_LLM": self.settings.fast_llm_model,
            "SMART_LLM": self.settings.smart_llm_model,
            "EMBEDDING": self.settings.embedding_model,
        }[role]
        rate = self.settings.llm_model_rates.get(model) if model else None
        if rate is None:
            return GatewayResult(None, "fallback", "model_or_rate_unconfigured", Decimal(0))
        if not text.strip() or len(text.encode()) > 65536:
            return GatewayResult(None, "fallback", "input_limit", Decimal(0))
        fx = self.settings.usd_to_rub_rate
        estimate = _cost(rate, fx, len(text.encode()) + 64, output_limit)
        if not await budget.reserve(estimate, self.settings.max_cycle_cost_rub):
            return GatewayResult(None, "fallback", "budget_exceeded", Decimal(0), 0, 0)
        actual = None
        try:
            response = await self._client.post(path, json={**body, "model": model})
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or "error" in data:
                return GatewayResult(None, "fallback", "provider_error", None)
            usage = _usage(data.get("usage"))
            if usage is None:
                return GatewayResult(None, "fallback", "usage_unavailable", None)
            actual = _cost(rate, fx, *usage)
            return GatewayResult(data, "llm", None, actual, *usage)
        except httpx.HTTPError:
            return GatewayResult(None, "fallback", "provider_transport", None)
        except ValueError:
            return GatewayResult(None, "fallback", "invalid_provider_response", None)
        finally:
            await asyncio.shield(budget.settle(estimate, actual))
