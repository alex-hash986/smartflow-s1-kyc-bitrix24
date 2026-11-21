"""Exercise S1 -> S2 -> S3 on the test portal with operator-directed synthetic decisions."""

import argparse
import asyncio
import json
import secrets
import socket
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
from ops.runners.b24_setup_s1 import OBJECT, identifier, pages
from ops.runners.benchmark_s1 import ROOT, post
from ops.runners.e2e_s2_live import quote
from ops.runners.smoke_live_s1 import BASE, PORTAL, SLUG, comments, scenario

PIN = "109.120.182.255"


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
    assert not result["need_human"]
    assert payload["winner_s1_item_id"] == quotes[0]["s1_item_id"]
    assert payload["winner_total_rub"] == "90000.00"
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
    final_stage = binding.stage_ids.awarded
    final_status = binding.status_values.AWARDED
    if snapshot["stageId"] == binding.stage_ids.review:
        assert str(snapshot[status_field]) == str(binding.status_values.REVIEW_REQUIRED)
        review_comments = await comments(publisher.client, rest, entity, item_id)
        assert any(
            identifier(row["ID"]) == review["comment_id"] and "TUBE" in str(row["COMMENT"])
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
        "received_at": receipt["received_at"],
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


async def acceptance(
    http: httpx.AsyncClient,
    publisher: Publisher,
    rest: httpx.AsyncClient,
    label: str,
    batch: str,
    awarded: JSONObject,
) -> JSONObject:
    context: JSONObject = {
        "tenant_id": SLUG,
        "profile_slug": SLUG,
        "profile_revision": "5",
        "request_id": awarded["request_id"],
    }
    s2_id = identifier(awarded["item_id"])
    with publisher.journal.db.connect() as conn:
        assert (
            conn.execute(
                "SELECT items_json FROM s2_award_items WHERE tenant_id=? AND request_id=?",
                (SLUG, context["request_id"]),
            ).fetchone()
            is not None
        )
    qty, amount, claim = (
        ("1.000", "90000.00", "0.00") if label == "A" else ("0.750", "67500.00", "22500.00")
    )
    number = f"{batch}-UPD"
    fields = OBJECT.validate_python(awarded["fields"])
    raw: JSONObject = {
        "doc_number": number,
        "doc_date": "2026-11-21",
        "currency": "RUB",
        "supplier_inn": fields["ufCrm9WinnerInn"],
        "lines": [
            {
                "item_code": "TUBE",
                "unit": "pc",
                "quantity": qty,
                "unit_price_rub": "90000.00",
                "line_total_rub": amount,
                "vat_treatment": "NO_VAT",
            }
        ],
    }
    intake: JSONObject = {
        **context,
        "s2_item_id": s2_id,
        "delivery_key": batch,
        "documents": [
            {
                "document_id": str(uuid5(NAMESPACE_URL, number)),
                "doc_type": "UPD",
                "doc_number": number,
                "doc_date": raw["doc_date"],
                "actual_delivery_days": 4,
                "source_type": "TEXT",
                "text": json.dumps(raw),
                "pdf_base64": None,
            }
        ],
    }
    receipt = await post(http, "/s3/intake", intake)
    call = {**context, "package_id": receipt["package_id"]}
    gate = await post(http, "/s3/tender-gate", call)
    award = OBJECT.validate_python(gate["award"])
    assert not gate["need_human"] and award["s2_item_id"] == s2_id
    parse = {**context, "payload": {"package_id": receipt["package_id"]}}
    parsed = await post(http, "/s3/parse", parse)
    packet: JSONObject = {
        **context,
        "schema_version": "1",
        "stage": "S3",
        "attempt": 1,
        "received_at": awarded["received_at"],
        "upstream_item": {"entity_type_id": 1044, "item_id": s2_id},
        "payload": {
            "package_id": receipt["package_id"],
            "award": award,
            "documents": OBJECT.validate_python(parsed["payload"])["documents"],
        },
    }
    result = await post(http, "/s3/evaluate", packet)
    payload = OBJECT.validate_python(result["payload"])
    outcome = "ACCEPTED" if label == "A" else "CLAIM_REQUIRED"
    assert payload["outcome"] == outcome
    assert payload["accepted_amount_rub"] == amount and payload["claim_amount_rub"] == claim
    entity = 1048
    status_field = "ufCrm11AcceptanceStatus"
    external_field = "ufCrm11ExternalKey"
    pub_call = {**call, "attempt": 1}
    key: JSONObject = {"tenant_id": SLUG, "request_id": context["request_id"], "stage": "S3"}
    review_key = {**key, "purpose": "human_review", "review_attempt": 1}
    final_key = {**key, "purpose": "stage_result"}
    try:
        publisher.state(PublicationKey.model_validate(review_key))
    except ValueError:
        assert await post(http, "/s3/publication/prepare", pub_call) == review_key
    review = await post(http, "/v1/publications/dispatch", review_key)
    item_id = identifier(review["item_id"])
    assert review["status"] == "published"
    first = await publisher.client.item_get(entity, item_id)
    stage, status = ("DT1048_19:SUCCESS", 57) if label == "A" else ("DT1048_19:FAIL", 59)
    if first["stageId"] == "DT1048_19:S3_REVIEW":
        assert str(first[status_field]) == "55"
        timeline = await comments(publisher.client, rest, entity, item_id)
        assert any(
            identifier(row["ID"]) == review["comment_id"] and claim in str(row["COMMENT"])
            for row in timeline
        )
        assert (
            await publisher.client.item_update(
                entity, item_id, {"stageId": stage, status_field: status}
            )
            == item_id
        )
    else:
        assert first["stageId"] == stage and str(first[status_field]) == str(status)
    resolved = await post(
        http,
        "/s3/publication/resolve",
        {
            **{k: v for k, v in context.items() if k != "request_id"},
            "item_id": item_id,
        },
    )
    assert resolved == pub_call
    decision = await http.post("/s3/publication/decision", json=pub_call)
    if decision.status_code != 204:
        raise RuntimeError(f"S3 decision {label}: {decision.status_code} {decision.text}")
    assert await post(http, "/s3/publication/prepare", pub_call) == final_key
    final = await post(http, "/v1/publications/dispatch", final_key)
    assert final["status"] == "published" and final["item_id"] == item_id
    assert await post(http, "/s3/intake", intake) == receipt
    assert await post(http, "/s3/parse", parse) == parsed
    assert await post(http, "/v1/workers/S3/run", packet) == result
    assert await post(http, "/v1/publications/dispatch", review_key) == review
    assert await post(http, "/s3/publication/prepare", pub_call) == final_key
    assert await post(http, "/v1/publications/dispatch", final_key) == final
    saved = await publisher.client.item_get(entity, item_id)
    assert saved["stageId"] == stage and str(saved[status_field]) == str(status)
    expected: JSONObject = {
        "categoryId": 19,
        "ufCrm_11_S2_ITEM_ID": s2_id,
        "ufCrm11SupplierInn": award["winner_inn"],
        "ufCrm11AcceptedAmountRub": amount,
        "ufCrm11ClaimAmountRub": claim,
        external_field: first[external_field],
    }
    assert all(str(saved[k]) == str(v) for k, v in expected.items())
    marker = saved[external_field]
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
    for key_ref, publication in ((review_key, review), (final_key, final)):
        state = publisher.state(PublicationKey.model_validate(key_ref))
        entries = [row for row in timeline if f"[smartflow:{state.marker}]" in str(row["COMMENT"])]
        assert len(entries) == 1 and identifier(entries[0]["ID"]) == publication["comment_id"]
    return {
        "scenario": label,
        "request_id": context["request_id"],
        "s2_item_id": s2_id,
        "item_id": item_id,
        "review_status": 55,
        "final_status": status,
        "final_stage": stage,
        "review_comment": review["comment_id"],
        "final_comment": final["comment_id"],
        "outcome": outcome,
        "accepted_amount_rub": amount,
        "claim_amount_rub": claim,
        "duplicate_cards": 0,
        "duplicate_marked_comments": 0,
        "report": payload["reconciliation_report_markdown"],
    }


async def run(output: Path) -> None:
    db_path = output / "pilot.sqlite3"
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
            profile = load_profile(cfg.profiles_path, SLUG, "5")
            assert profile.revision == "5" and profile.s3_claims and profile.s3_claims.enabled
            results = []
            for index, label in enumerate(("A", "B")):
                batch = f"S3.4-{output.name}-{label}"
                suppliers = []
                for offset, name in enumerate(("winner", "competitor")):
                    approved = await scenario(
                        http,
                        publisher,
                        rest,
                        cfg,
                        f"{batch}-{name}",
                        False,
                        94340 + index * 2 + offset,
                        standalone=False,
                    )
                    suppliers.append(approved)
                anchor = identifier(suppliers[0]["item_id"])
                quotes = []
                for supplier, price in zip(suppliers, ("90000.00", "95000.00"), strict=True):
                    card = identifier(supplier["item_id"])
                    item = await portal.item_get(1040, card)
                    quotes.append(quote(card, str(item["ufCrm7Inn"]), price, batch))
                awarded = await tender(http, publisher, rest, cfg, label, batch, anchor, quotes)
                accepted = await acceptance(http, publisher, rest, label, batch, awarded)
                result = {"s1": suppliers, "s2": awarded, "s3": accepted}
                results.append(result)
                (output / "e2e.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(json.dumps({k: v for k, v in accepted.items() if k != "report"}), flush=True)
    finally:
        server.should_exit = True
        await task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = socket.getaddrinfo

    def resolve(host: str | bytes | None, *opts: Any, **kwargs: Any) -> Any:
        return original(PIN if host in (PORTAL, PORTAL.encode()) else host, *opts, **kwargs)

    with patch.object(socket, "getaddrinfo", side_effect=resolve):
        asyncio.run(run(args.output))


if __name__ == "__main__":
    main()
