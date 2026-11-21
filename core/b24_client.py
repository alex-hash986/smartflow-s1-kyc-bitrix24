import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from random import uniform
from time import monotonic
from typing import Literal

import httpx
from pydantic import JsonValue, TypeAdapter, ValidationError

from core.contracts import JSONObject

Method = Literal[
    "profile",
    "scope",
    "method.get",
    "crm.type.add",
    "crm.category.list",
    "crm.item.fields",
    "crm.status.add",
    "crm.status.update",
    "userfieldconfig.list",
    "userfieldconfig.get",
    "userfieldconfig.add",
    "crm.type.list",
    "crm.status.list",
    "crm.item.list",
    "crm.item.get",
    "crm.item.add",
    "crm.item.update",
    "crm.timeline.comment.add",
    "tasks.task.add",
    "tasks.task.get",
]

_object = TypeAdapter(JSONObject)
_rows = TypeAdapter(list[JSONObject])


class BitrixError(RuntimeError):
    def __init__(self, code: str, *, uncertain: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.uncertain = uncertain


def task_binding(entity_type_id: int, item_id: int) -> str:
    return f"T{_id(entity_type_id):x}_{_id(item_id)}"


def _id(value: JsonValue) -> int:
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if type(value) is not int or not 0 < value < 2**63:
        raise BitrixError("invalid_id", uncertain=True)
    return value


class PortalLimiter:
    def __init__(
        self,
        rps: float = 2.0,
        *,
        clock: Callable[[], float] = monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not 0 < rps <= 2:
            raise ValueError("Expected 0 < rate <= 2 requests/s")
        self._interval = 1 / rps
        self._clock, self._sleep = clock, sleep
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            delay = self._next - self._clock()
            if delay > 0:
                await self._sleep(delay)
            self._next = self._clock() + self._interval

    async def defer(self, seconds: float) -> None:
        async with self._lock:
            self._next = max(self._next, self._clock() + seconds)


def _retry_delay(header: str | None, attempt: int) -> float:
    if header:
        try:
            delay = float(header)
        except ValueError:
            try:
                delay = (parsedate_to_datetime(header) - datetime.now(UTC)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                delay = 0
        if math.isfinite(delay) and delay > 0:
            return delay
    return 0.5 * 2.0**attempt + uniform(0, 0.1)


@dataclass(frozen=True)
class Page:
    items: list[JSONObject]
    next_start: int | None


class BitrixClient:
    def __init__(
        self,
        webhook_url: str,
        limiter: PortalLimiter,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        url = httpx.URL(webhook_url)
        if url.scheme != "https" or not url.host or url.query or url.fragment or url.userinfo:
            raise ValueError("Expected HTTPS webhook base URL")
        self.portal_id = url.host
        self.limiter = limiter
        self._http = httpx.AsyncClient(
            base_url=webhook_url.rstrip("/") + "/",
            timeout=20,
            transport=transport,
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _call(
        self,
        method: Method,
        body: JSONObject,
        *,
        write: bool = False,
    ) -> tuple[JsonValue, int | None]:
        for attempt in range(3):
            await self.limiter.wait()
            try:
                response = await self._http.post(method, json=body)
            except httpx.HTTPError:
                if write or attempt == 2:
                    raise BitrixError("transport_error", uncertain=write) from None
                await self.limiter.defer(_retry_delay(None, attempt))
                continue
            try:
                envelope = _object.validate_python(response.json())
            except (ValidationError, ValueError):
                envelope = {}
            code = envelope.get("error")
            limited = response.status_code == 429 or code in (
                "QUERY_LIMIT_EXCEEDED",
                "OPERATION_TIME_LIMIT",
            )
            if limited or (not write and response.status_code >= 500):
                delay = _retry_delay(response.headers.get("Retry-After"), attempt)
                await self.limiter.defer(delay)
                if not write and attempt < 2 and delay <= 30:
                    continue
            if not response.is_success or code is not None or "result" not in envelope:
                known = (
                    code
                    if code
                    in (
                        "QUERY_LIMIT_EXCEEDED",
                        "OPERATION_TIME_LIMIT",
                        "ACCESS_DENIED",
                        "ERROR_NOT_FOUND",
                    )
                    else "rest_error"
                )
                raise BitrixError(str(known), uncertain=write)
            next_start = envelope.get("next")
            return envelope["result"], None if next_start is None else _id(next_start)
        raise AssertionError("Unreachable retry state")

    async def setup_call(
        self,
        method: Method,
        body: JSONObject,
        *,
        write: bool = False,
    ) -> tuple[JsonValue, int | None]:
        return await self._call(method, body, write=write)

    async def type_list(self, filters: JSONObject | None = None, *, start: int = 0) -> Page:
        result, cursor = await self._call(
            "crm.type.list", {"filter": filters or {}, "start": start}
        )
        return Page(_rows.validate_python(_object.validate_python(result)["types"]), cursor)

    async def status_list(self, filters: JSONObject | None = None, *, start: int = 0) -> Page:
        result, cursor = await self._call(
            "crm.status.list", {"filter": filters or {}, "start": start}
        )
        return Page(_rows.validate_python(result), cursor)

    async def item_list(
        self,
        entity_type_id: int,
        filters: JSONObject | None = None,
        *,
        start: int = 0,
    ) -> Page:
        result, cursor = await self._call(
            "crm.item.list",
            {
                "entityTypeId": _id(entity_type_id),
                "filter": filters or {},
                "start": start,
            },
        )
        return Page(_rows.validate_python(_object.validate_python(result)["items"]), cursor)

    async def item_get(self, entity_type_id: int, item_id: int) -> JSONObject:
        result, _ = await self._call(
            "crm.item.get",
            {
                "entityTypeId": _id(entity_type_id),
                "id": _id(item_id),
            },
        )
        return _object.validate_python(_object.validate_python(result)["item"])

    async def item_add(self, entity_type_id: int, fields: JSONObject) -> int:
        return await self._item_write(
            "crm.item.add", {"entityTypeId": _id(entity_type_id), "fields": fields}
        )

    async def item_update(self, entity_type_id: int, item_id: int, fields: JSONObject) -> int:
        return await self._item_write(
            "crm.item.update",
            {
                "entityTypeId": _id(entity_type_id),
                "id": _id(item_id),
                "fields": fields,
            },
        )

    async def _item_write(self, method: Method, body: JSONObject) -> int:
        return await self._write_id(method, body, "item")

    async def _write_id(self, method: Method, body: JSONObject, field: str) -> int:
        result, _ = await self._call(method, body, write=True)
        try:
            return _id(_object.validate_python(_object.validate_python(result)[field])["id"])
        except (ValidationError, KeyError):
            raise BitrixError("invalid_write_result", uncertain=True) from None

    async def timeline_comment_add(self, entity_type_id: int, item_id: int, comment: str) -> int:
        result, _ = await self._call(
            "crm.timeline.comment.add",
            {
                "fields": {
                    "ENTITY_TYPE": f"DYNAMIC_{_id(entity_type_id)}",
                    "ENTITY_ID": _id(item_id),
                    "COMMENT": comment,
                }
            },
            write=True,
        )
        return _id(result)

    async def task_add(
        self,
        title: str,
        responsible_id: int,
        entity_type_id: int,
        item_id: int,
        *,
        description: str = "",
    ) -> int:
        return await self._write_id(
            "tasks.task.add",
            {
                "fields": {
                    "TITLE": title,
                    "RESPONSIBLE_ID": _id(responsible_id),
                    "DESCRIPTION": description,
                    "UF_CRM_TASK": [task_binding(entity_type_id, item_id)],
                }
            },
            "task",
        )

    async def task_get(self, task_id: int) -> JSONObject:
        result, _ = await self._call(
            "tasks.task.get",
            {
                "taskId": _id(task_id),
                "select": ["ID", "UF_CRM_TASK", "DESCRIPTION"],
            },
        )
        return _object.validate_python(_object.validate_python(result)["task"])
