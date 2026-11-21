"""Publish two synthetic S1 fixtures to the configured test portal."""

import argparse
import asyncio
import json
import secrets
import socket
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch
from uuid import UUID

import httpx
import uvicorn
from pydantic import SecretStr

from core.api import create_app
from core.b24_client import BitrixClient, BitrixError, PortalLimiter
from core.config import Settings
from core.contracts import JSONObject, WorkerRequestEnvelope, WorkerResponseEnvelope
from core.database import Database
from core.idempotency import Conflict, Journal, PublicationCommand, PublicationKey
from core.profile_loader import load_profile
from core.publisher import Publisher
from core.telemetry import Telemetry
from ops.runners.b24_setup_s1 import OBJECT, ROWS, identifier, pages
from ops.runners.benchmark_s1 import ROOT, Case, post, registry, source

PORTAL = "b24-c53p0d.bitrix24.ru"
SLUG = "portal_b24_b2b"
BASE = "http://127.0.0.1:8021"


async def comments(
    portal: BitrixClient,
    rest: httpx.AsyncClient,
    entity: int,
    item: int,
) -> list[JSONObject]:
    start = 0
    rows = []
    while True:
        await portal.limiter.wait()
        try:
            response = await rest.post(
                "crm.timeline.comment.list",
                json={
                    "filter": {"ENTITY_ID": item, "ENTITY_TYPE": f"DYNAMIC_{entity}"},
                    "select": ["ID", "COMMENT"],
                    "start": start,
                },
            )
            body = OBJECT.validate_python(response.json())
            if not response.is_success or "result" not in body:
                raise BitrixError("timeline_read_failed")
        except (httpx.HTTPError, ValueError):
            raise BitrixError("timeline_read_failed") from None
        rows.extend(ROWS.validate_python(body["result"]))
        if body.get("next") is None:
            return rows
        next_start = identifier(body["next"])
        if next_start <= start:
            raise BitrixError("timeline_invalid_cursor")
        start = next_start


async def scenario(
    client: httpx.AsyncClient,
    publisher: Publisher,
    rest: httpx.AsyncClient,
    cfg: Settings,
    batch: str,
    red: bool,
) -> JSONObject:
    start = perf_counter()
    profile = load_profile(cfg.profiles_path, SLUG)
    stage = profile.stages["S1"]
    assert profile.s1_kyc and profile.tenant_id and stage.stage_ids
    assert stage.entity_type_id and stage.responsible_id and stage.external_key_field
    assert not profile.s1_kyc.bitrix24.tasks_enabled
    entity = stage.entity_type_id
    mapping = profile.s1_kyc.bitrix24
    label = "B" if red else "A"
    case = Case(id=label, scenario="red" if red else "green")
    application = source(case, 94021 if red else 94020)
    delivery: JSONObject = {
        "tenant_id": profile.tenant_id,
        "source": "s1-live-smoke",
        "delivery_key": f"{batch}-{label}",
        "profile_slug": SLUG,
        "profile_revision": profile.revision,
        "channel": "synthetic",
        "is_after_hours": False,
        "run_origin": "pilot",
        "payload": application,
    }
    receipt = await post(client, "/v1/deliveries", delivery)
    stamp = str(receipt["received_at"])
    context = {**receipt, "tenant_id": profile.tenant_id}
    extract_context = {key: value for key, value in context.items() if key != "received_at"}
    parsed = OBJECT.validate_python(
        (await post(client, "/s1/parse", {**extract_context, "payload": application}))["payload"]
    )
    assert not parsed["need_human"]
    evaluation: JSONObject = {
        **context,
        "attempt": 1,
        "payload": {"parsed": parsed, "registry": registry(parsed, case.scenario, stamp)},
    }
    rated = WorkerResponseEnvelope.model_validate(await post(client, "/s1/evaluate", evaluation))
    assert rated.payload and rated.need_human is red
    result = rated.payload
    assert result["risk_level"] == ("CRITICAL" if red else "LOW")
    assert result["recommended_credit_limit_rub"] == ("0.00" if red else "250000.00")
    purpose = "human_review" if red else "stage_result"
    chosen_stage = stage.stage_ids.review if red else stage.stage_ids.approved
    key = PublicationKey.model_validate(
        {
            "tenant_id": profile.tenant_id,
            "request_id": context["request_id"],
            "stage": "S1",
            "purpose": purpose,
        }
    )
    values: JSONObject = {
        "INN": parsed["inn"],
        "OGRN": parsed["ogrn"],
        "RELIABILITY_SCORE": result["reliability_score"],
    }
    if not red:
        values.update(
            KYC_STATUS=mapping.kyc_status_values["APPROVED"],
            APPROVED_CREDIT_LIMIT_RUB=result["recommended_credit_limit_rub"],
        )
    fields: JSONObject = {
        "title": f"[S1.4 TEST] {batch} / {label}",
        "categoryId": stage.category_id,
        "assignedById": stage.responsible_id,
        "stageId": chosen_stage,
        **{
            code: values[name]
            for name, code in mapping.fields.items()
            if code is not None and name in values
        },
    }
    comment = str(result["dossier_markdown"])
    marker = publisher.journal.approve(
        PublicationCommand(
            key=key,
            stage_attempt=1,
            portal_id=PORTAL,
            approval_ref=f"ops:s1.4:{batch}:{label}",
            entity_type_id=entity,
            fields=fields,
            external_key_field=stage.external_key_field,
            inn_field=mapping.fields["INN"],
            task_required=False,
            timeline_comment=comment,
        ),
        allow_review_card=True,
    )
    key_body = key.model_dump(mode="json")
    published = await post(client, "/v1/publications/dispatch", key_body)
    assert published["status"] == "published"
    item_id, comment_id = identifier(published["item_id"]), identifier(published["comment_id"])
    if not red:
        assert await post(client, "/v1/deliveries", delivery) == receipt
        assert await post(client, "/s1/evaluate", evaluation) == rated.model_dump(mode="json")
        assert await post(client, "/v1/publications/dispatch", key_body) == published
    item = await publisher.client.item_get(entity, item_id)
    assert all(str(item[name]) == str(value) for name, value in fields.items())
    assert item[stage.external_key_field] == marker
    matched = await pages(
        publisher.client,
        "crm.item.list",
        {
            "entityTypeId": entity,
            "filter": {"=" + stage.external_key_field: marker},
            "select": ["id", stage.external_key_field],
        },
        "items",
    )
    assert [
        identifier(row["id"]) for row in matched if row.get(stage.external_key_field) == marker
    ] == [
        item_id,
    ]
    timeline = await comments(publisher.client, rest, entity, item_id)
    marked = [row for row in timeline if f"[smartflow:{marker}]" in str(row.get("COMMENT"))]
    assert len(marked) == 1 and identifier(marked[0]["ID"]) == comment_id
    if red:
        assert all(
            code in str(marked[0]["COMMENT"]) for code in ("BANKRUPTCY", "FNS_ADDRESS_UNRELIABLE")
        )
        assert all(
            item.get(mapping.fields[name] or "") in (None, "", 0, "0")
            for name in ("KYC_STATUS", "APPROVED_CREDIT_LIMIT_RUB")
        )
    gate = WorkerRequestEnvelope.model_validate(
        {
            **context,
            "schema_version": "1",
            "stage": "S2",
            "attempt": 1,
            "payload": {},
            "upstream_item": {"entity_type_id": entity, "item_id": item_id},
        }
    )
    # Roll back the readiness probe: S2 is not executed or reserved on the pilot database.
    probe_db = Database(cfg.sqlite_path.with_name(f"gate-{label}.sqlite3"))
    with publisher.journal.db.connect() as original, probe_db.connect() as clone:
        original.backup(clone)
    allowed = False
    try:
        allowed = Telemetry(probe_db).reserve(gate).status == "reserved"
    except Conflict:
        if not red:
            raise
    assert allowed is not red
    if allowed:
        assert str(item[mapping.fields["KYC_STATUS"] or ""]) == str(
            mapping.kyc_status_values["APPROVED"],
        )
    Telemetry(publisher.journal.db).record_cycle(
        profile.tenant_id,
        UUID(str(context["request_id"])),
        outcome="waiting_human" if red else "completed",
        processing_time_sec=perf_counter() - start,
        saved_minutes=None,
        ab_variant=None,
    )
    return {
        "scenario": label,
        "item_id": item_id,
        "comment_id": comment_id,
        "stage": chosen_stage,
        "fields": {code: item.get(code) for code in mapping.fields.values() if code is not None},
        "external_key": marker,
        "s2_ready": allowed,
        "request_id": context["request_id"],
        "risk": result["risk_level"],
        "need_human": rated.need_human,
        "recommended_limit": result["recommended_credit_limit_rub"],
    }


async def run(output: Path, batch: str) -> None:
    cfg = Settings(
        sqlite_path=output / "pilot.sqlite3",
        profiles_path=ROOT / "profiles",
        worker_token=SecretStr(secrets.token_urlsafe()),
        app_port=8021,
    )
    assert cfg.bitrix24_webhook_url and cfg.worker_token
    url = cfg.bitrix24_webhook_url.get_secret_value()
    portal = BitrixClient(url, PortalLimiter(cfg.bitrix24_rate_limit_rps))
    if portal.portal_id != PORTAL:
        await portal.aclose()
        raise ValueError("Unexpected test portal")
    try:
        who, _ = await portal.setup_call("profile", {})
        assert identifier(OBJECT.validate_python(who)["ID"]) == 1
    except BaseException:
        await portal.aclose()
        raise
    publisher = Publisher(Journal(Database(cfg.sqlite_path)), portal, PORTAL)
    app = create_app(cfg, publisher=publisher)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8021, access_log=False))
    task = asyncio.create_task(server.serve())
    results: list[JSONObject] = []
    try:
        for _ in range(150):
            if server.started:
                break
            if task.done():
                await task
                raise RuntimeError("Local stand failed to start")
            await asyncio.sleep(0.1)
        if not server.started:
            raise RuntimeError("Local stand did not start")
        async with (
            httpx.AsyncClient(
                base_url=BASE,
                headers={"X-Smartflow-Token": cfg.worker_token.get_secret_value()},
                timeout=90,
            ) as client,
            httpx.AsyncClient(base_url=url.rstrip("/") + "/", timeout=15) as rest,
        ):
            assert (await client.get("/health")).status_code == 200
            for red in (False, True):
                result = await scenario(client, publisher, rest, cfg, batch, red)
                results.append(result)
                payload = {"portal": PORTAL, "batch": batch, "results": results}
                (output / "smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(json.dumps(result), flush=True)
    finally:
        server.should_exit = True
        await task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--pin-ip", help="One resolved portal IP. TLS still verifies the portal host."
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    batch = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    original = socket.getaddrinfo

    def resolve(host: Any, *opts: Any, **kwargs: Any) -> Any:
        if args.pin_ip and host in (PORTAL, PORTAL.encode()):
            return original(args.pin_ip, *opts, **kwargs)
        return original(host, *opts, **kwargs)

    with patch.object(socket, "getaddrinfo", side_effect=resolve):
        asyncio.run(run(args.output, batch))


if __name__ == "__main__":
    main()
