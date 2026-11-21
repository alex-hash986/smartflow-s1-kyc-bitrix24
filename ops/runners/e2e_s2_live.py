"""Exercise S1 -> S2 on the test portal with operator-directed synthetic decisions."""

import argparse
import asyncio
import json
import secrets
import socket
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

import httpx
import uvicorn
from pydantic import JsonValue, SecretStr

from core.api import create_app
from core.b24_client import BitrixClient, PortalLimiter
from core.config import Settings
from core.contracts import JSONObject
from core.database import Database
from core.idempotency import Journal, PublicationKey
from core.profile_loader import load_profile
from core.publisher import Publisher
from ops.runners.b24_setup_s1 import OBJECT, ROWS, identifier, pages
from ops.runners.benchmark_s1 import ROOT, post
from ops.runners.smoke_live_s1 import BASE, PORTAL, SLUG, comments, scenario

PIN = "109.120.182.255"


def quote(card: int, inn: str, price: str, batch: str) -> JSONObject:
    document = {
        "currency": "RUB",
        "quoted_total": price,
        "deferment_days": 15,
        "items": [
            {
                "item_code": "TUBE",
                "name": "Tube",
                "unit": "pc",
                "quantity": "1",
                "unit_price": price,
                "unit_price_includes_vat": True,
                "vat_treatment": "NO_VAT",
                "vat_amount": "0",
                "line_total": price,
                "line_total_includes_vat": True,
                "delivery_days": 4,
            }
        ],
    }
    return {
        "quote_id": str(uuid5(NAMESPACE_URL, f"{batch}:{card}")),
        "supplier_inn": inn,
        "supplier_name": f"Synthetic S1 #{card}",
        "s1_item_id": card,
        "source_type": "TEXT",
        "text": json.dumps(document),
        "pdf_base64": None,
    }


async def tender(
    http: httpx.AsyncClient,
    publisher: Publisher,
    rest: httpx.AsyncClient,
    cfg: Settings,
    label: str,
    batch: str,
    anchor: int,
    quotes: list[JSONObject],
) -> JSONObject:
    profile = load_profile(cfg.profiles_path, SLUG)
    assert profile.s2_tender
    binding = profile.s2_tender.bitrix24
    entity = identifier(binding.entity_type_id)
    external_field, status_field = binding.fields.EXTERNAL_KEY, binding.fields.TENDER_STATUS
    assert external_field and status_field
    body: JSONObject = {
        "profile_slug": SLUG,
        "delivery_key": batch + label,
        "upstream_s1_item_id": anchor,
        "quotes": list[JsonValue](quotes),
        "items": [
            {
                "item_code": "TUBE",
                "name": "Tube",
                "unit": "pc",
                "quantity": "1",
                "target_unit_price_rub": "100000.00",
            }
        ],
        "required_deferment_days": 15,
        "target_budget_rub": "100000.00",
    }
    receipt = await post(http, "/s2/intake", body)
    context = {
        key: receipt[key] for key in ("tenant_id", "request_id", "profile_slug", "profile_revision")
    }
    gates = await post(
        http,
        "/s2/supplier-gates",
        {
            **context,
            "upstream_item": receipt["upstream_item"],
            "suppliers": [
                {"supplier_inn": q["supplier_inn"], "s1_item_id": q["s1_item_id"]} for q in quotes
            ],
        },
    )
    parsed = await post(
        http, "/s2/parse", {**context, "payload": {"quotes": list[JsonValue](quotes)}}
    )
    packet: JSONObject = {
        **receipt,
        "schema_version": "1",
        "stage": "S2",
        "payload": {
            **gates,
            "quotes": OBJECT.validate_python(parsed["payload"])["quotes"],
            "items": body["items"],
            "required_deferment_days": 15,
            "target_budget_rub": "100000.00",
        },
    }
    result = await post(http, "/s2/evaluate", packet)
    payload = OBJECT.validate_python(result["payload"])
    blocked = next(
        row for row in ROWS.validate_python(payload["participants"]) if row["s1_item_id"] == 3
    )
    assert blocked["rejection_reasons"] == ["S2_S1_UNCONFIRMED"]
    if label == "A":
        assert payload["winner_s1_item_id"] == 1 and payload["savings_rub"] == "10000.00"
    else:
        assert payload["winner_inn"] is None and result["escalation_reasons"] == [
            "S2_QUORUM_NOT_MET"
        ]
    call = {**context, "attempt": receipt["attempt"]}
    review_key = {**context, "stage": "S2", "purpose": "human_review"}
    review_key.pop("profile_slug")
    review_key.pop("profile_revision")
    final_key = {**review_key, "purpose": "stage_result"}
    try:
        prior_review = publisher.state(PublicationKey.model_validate(review_key))
    except ValueError:
        prior_review = None
    if prior_review is None:
        assert await post(http, "/s2/publication/prepare", call) == review_key
    review = await post(http, "/v1/publications/dispatch", review_key)
    assert review["status"] == "published"
    item_id = identifier(review["item_id"])
    snapshot = await publisher.client.item_get(entity, item_id)
    final_stage = binding.stage_ids.awarded if label == "A" else binding.stage_ids.cancelled
    final_status = (
        binding.status_values.AWARDED if label == "A" else binding.status_values.CANCELLED
    )
    if snapshot["stageId"] == binding.stage_ids.review:
        assert str(snapshot[status_field]) == str(binding.status_values.REVIEW_REQUIRED)
        review_comments = await comments(publisher.client, rest, entity, item_id)
        assert any(
            identifier(row["ID"]) == review["comment_id"]
            and "TUBE" in str(row["COMMENT"])
            and (label == "A" or "S2" in str(row["COMMENT"]))
            for row in review_comments
        )
        # The acceptance operator explicitly requested these synthetic decisions.
        assert (
            await publisher.client.item_update(
                entity, item_id, {"stageId": final_stage, status_field: final_status}
            )
            == item_id
        )
    else:
        assert snapshot["stageId"] == final_stage and str(snapshot[status_field]) == str(
            final_status
        )
    decision = await http.post("/s2/publication/decision", json=call)
    if decision.status_code != 204:
        raise RuntimeError(f"Decision {label}: {decision.status_code} {decision.text}")
    assert await post(http, "/s2/publication/prepare", call) == final_key
    final = await post(http, "/v1/publications/dispatch", final_key)
    assert final["status"] == "published" and final["item_id"] == item_id
    assert await post(http, "/s2/intake", body) == receipt
    assert (
        await post(http, "/s2/parse", {**context, "payload": {"quotes": list[JsonValue](quotes)}})
        == parsed
    )
    assert await post(http, "/v1/workers/S2/run", packet) == result
    assert await post(http, "/s2/publication/prepare", call) == final_key
    assert await post(http, "/v1/publications/dispatch", final_key) == final

    item = await publisher.client.item_get(entity, item_id)
    assert item["stageId"] == final_stage and str(item[status_field]) == str(final_status)
    assert str(item["categoryId"]) == str(binding.category_id)
    marker = item[external_field]
    assert isinstance(marker, str) and len(marker) == 64
    fields = {code: item.get(code) for code in binding.fields.model_dump().values()}
    for logical, expected in {
        "WINNER_INN": payload["winner_inn"],
        "WINNER_S1_ITEM_ID": payload["winner_s1_item_id"],
        "TOTAL_AMOUNT_RUB": payload["winner_total_rub"],
        "SAVINGS_RUB": payload["savings_rub"],
    }.items():
        actual = fields[getattr(binding.fields, logical)]
        assert actual in (None, "") if expected is None else str(actual) == str(expected)
    matched = await pages(
        publisher.client,
        "crm.item.list",
        {
            "entityTypeId": entity,
            "filter": {"=" + external_field: marker},
            "select": ["id", external_field],
        },
        "items",
    )
    assert [identifier(row["id"]) for row in matched if row.get(external_field) == marker] == [
        item_id
    ]
    timeline = await comments(publisher.client, rest, entity, item_id)
    for key, publication in ((review_key, review), (final_key, final)):
        state = publisher.state(PublicationKey.model_validate(key))
        entries = [row for row in timeline if f"[smartflow:{state.marker}]" in str(row["COMMENT"])]
        assert len(entries) == 1 and identifier(entries[0]["ID"]) == publication["comment_id"]
    return {
        "scenario": label,
        "anchor": anchor,
        "item_id": item_id,
        "request_id": receipt["request_id"],
        "review_stage": binding.stage_ids.review,
        "review_status": binding.status_values.REVIEW_REQUIRED,
        "review_comment": review["comment_id"],
        "final_stage": final_stage,
        "final_status": final_status,
        "final_comment": final["comment_id"],
        "fields": fields,
        "duplicate_cards": 0,
        "duplicate_marked_comments": 0,
        "matrix": payload["comparison_matrix_markdown"],
    }


async def run(output: Path, s1_db: Path) -> None:
    db_path = output / "pilot.sqlite3"
    if not db_path.exists():
        with (
            closing(sqlite3.connect(s1_db.resolve().as_uri() + "?mode=ro", uri=True)) as origin,
            closing(sqlite3.connect(db_path)) as target,
        ):
            origin.backup(target)
    cfg = Settings(
        sqlite_path=db_path,
        profiles_path=ROOT / "profiles",
        worker_token=SecretStr(secrets.token_urlsafe()),
    )
    assert cfg.bitrix24_webhook_url and cfg.worker_token
    url = cfg.bitrix24_webhook_url.get_secret_value()
    if url.split("/")[2] != PORTAL:
        raise ValueError("Configured webhook targets a different portal")
    portal = BitrixClient(url, PortalLimiter(cfg.bitrix24_rate_limit_rps))
    publisher = Publisher(Journal(Database(db_path)), portal, PORTAL)
    app = create_app(cfg, publisher=publisher)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8021, access_log=False))
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(150):
            if server.started:
                break
            if task.done():
                await task
                raise RuntimeError("FastAPI stopped before startup")
            await asyncio.sleep(0.1)
        assert server.started
        async with (
            httpx.AsyncClient(
                base_url=BASE,
                timeout=90,
                headers={
                    "X-Smartflow-Token": cfg.worker_token.get_secret_value(),
                },
            ) as http,
            httpx.AsyncClient(base_url=url.rstrip("/") + "/", timeout=20) as rest,
        ):
            assert (await http.get("/health")).status_code == 200
            who, _ = await portal.setup_call("profile", {})
            assert identifier(OBJECT.validate_python(who)["ID"]) == 1
            approved = await portal.item_get(1040, 1)
            blocked = await portal.item_get(1040, 3)
            assert str(approved["ufCrm7KycStatus"]) == "45"
            assert str(approved["ufCrm7ApprovedCreditLimitRub"]) == "250000.00"
            assert blocked["stageId"] == "DT1040_15:S1_REVIEW"
            results = []
            for label in ("A", "B"):
                batch = f"S2.4-{output.name}-{label}"
                fresh = await scenario(
                    http,
                    publisher,
                    rest,
                    cfg,
                    batch,
                    False,
                    94241 if label == "A" else 94242,
                    standalone=False,
                )
                anchor = identifier(fresh["item_id"])
                supplier = await portal.item_get(1040, anchor)
                quotes = [
                    quote(1, str(approved["ufCrm7Inn"]), "90000.00", batch),
                    quote(3, str(blocked["ufCrm7Inn"]), "80000.00", batch),
                ]
                if label == "A":
                    quotes.append(quote(anchor, str(supplier["ufCrm7Inn"]), "95000.00", batch))
                result = await tender(http, publisher, rest, cfg, label, batch, anchor, quotes)
                results.append(result)
                (output / "e2e.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(json.dumps({k: v for k, v in result.items() if k != "matrix"}), flush=True)
    finally:
        server.should_exit = True
        await task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-db", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = socket.getaddrinfo

    def resolve(host: str | bytes | None, *opts: Any, **kwargs: Any) -> Any:
        return original(PIN if host in (PORTAL, PORTAL.encode()) else host, *opts, **kwargs)

    with patch.object(socket, "getaddrinfo", side_effect=resolve):
        asyncio.run(run(args.output, args.s1_db))


if __name__ == "__main__":
    main()
