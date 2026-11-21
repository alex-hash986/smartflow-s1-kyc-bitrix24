"""Run the same 35 labelled synthetic tenders against two template profiles."""

import argparse
import asyncio
import json
from decimal import Decimal
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

import httpx
from pydantic import AnyHttpUrl, JsonValue, SecretStr

from core.api import create_app
from core.config import Settings
from core.contracts import JSONObject, WorkerRequestEnvelope, WorkerResponseEnvelope
from core.database import Database
from core.profile_loader import load_profile
from core.telemetry import Telemetry
from modules.s2_tender.contracts import S2EvaluateTenderRequest, S2ParseQuotesResponse
from modules.s2_tender.scoring import evaluate
from ops.runners.b24_setup_s1 import OBJECT
from ops.runners.benchmark_s1 import ROOT, post

PROFILES = ("default_b2b", "construction_supply")
INNS = ("7707083893", "7702070139", "7702235133", "7736050003", "7704211201")
PATTERNS = ("competitive", "s1_gate", "credit", "slow_dump", "budget", "quorum", "parse")
GATE_CODES = {"S2_S1_REJECTED", "S2_S1_UNCONFIRMED"}


def fixture(kind: str, case: int) -> tuple[list[JSONObject], list[JSONObject]]:
    approved = [("90", 5), ("94", 6), ("98", 7)]
    bids = {
        "competitive": approved,
        "s1_gate": approved + [("76", 3), ("77", 3)],
        "credit": approved + [("80", 3)],
        "slow_dump": [("70", 90), ("94", 1), ("99", 3)],
        "budget": [("90", 5), ("95", 6), ("104", 7), ("106", 8)],
        "quorum": approved[:1],
        "parse": approved[:2] + [("89", 5), ("84", 5)],
    }[kind]
    sources: list[JSONObject] = []
    snapshots: list[JSONObject] = []
    for index, (unit_price, days) in enumerate(bids):
        card = index + 1
        inn = INNS[index]
        price = str(Decimal(unit_price) + Decimal(case % 5) / 100)
        wrong = kind == "parse" and card == 3
        scanned = kind == "parse" and card == 4
        total = str(Decimal(price) + 1) if wrong else price
        document = {
            "currency": "RUB",
            "quoted_total": total,
            "deferment_days": 15,
            "items": [
                {
                    "item_code": "A",
                    "name": "Tube",
                    "unit": "pc",
                    "quantity": "1.000",
                    "unit_price": price,
                    "line_total": total,
                    "unit_price_includes_vat": True,
                    "line_total_includes_vat": True,
                    "vat_treatment": "NO_VAT",
                    "vat_amount": "0",
                    "delivery_days": days,
                }
            ],
        }
        sources.append(
            {
                "quote_id": str(uuid5(NAMESPACE_URL, f"smartflow:s2:benchmark:{case}:{card}")),
                "supplier_inn": inn,
                "supplier_name": f"Synthetic supplier {card}",
                "s1_item_id": card,
                "source_type": "PDF" if scanned else "TEXT",
                "text": None if scanned else json.dumps(document),
                "pdf_base64": "JVBERi0xLjcK" if scanned else None,
            }
        )
        if kind == "s1_gate" and card == 5:
            continue
        rejected = kind == "s1_gate" and card == 4
        snapshots.append(
            {
                "s1_item_id": card,
                "inn": inn,
                "ogrn": "1269900000237",
                "kyc_status": "REJECTED" if rejected else "APPROVED",
                "reliability_score": "100",
                "stage_result_published": True,
                "approved_credit_limit_rub": "0"
                if rejected
                else "79"
                if kind == "credit" and card == 4
                else "250",
            }
        )
    return sources, snapshots


def confusion(predicted: set[int], expected: set[int]) -> list[int]:
    return [len(predicted & expected), len(predicted - expected), len(expected - predicted)]


async def profile_run(
    client: httpx.AsyncClient, telemetry: Telemetry, profile: str
) -> tuple[JSONObject, list[JSONObject]]:
    config = load_profile(ROOT / "profiles", profile)
    assert config.s2_tender
    results: list[JSONObject] = []
    independent_labels = {"s1_gate": {4, 5}, "parse": {3}}
    times, s1, arithmetic = [], [0, 0, 0], [0, 0, 0]
    for index in range(35):
        kind = PATTERNS[index // 5]
        sources, snapshots = fixture(kind, index)
        start = perf_counter()
        delivery: JSONObject = {
            "tenant_id": profile,
            "profile_slug": profile,
            "profile_revision": config.revision,
            "source": "s2-benchmark",
            "delivery_key": str(index),
            "channel": "synthetic",
            "is_after_hours": False,
            "run_origin": "benchmark",
            "payload": {"fixture": kind},
        }
        receipt = await post(client, "/v1/deliveries", delivery)
        context = {key: value for key, value in receipt.items() if key != "received_at"}
        context["tenant_id"] = profile
        parse_body: JSONObject = {**context, "payload": {"quotes": list[JsonValue](sources)}}
        response = await post(client, "/s2/parse", parse_body)
        parsed = S2ParseQuotesResponse.model_validate(response["payload"])
        request = S2EvaluateTenderRequest.model_validate(
            {
                "quotes": parsed.quotes,
                "supplier_gate_snapshots": snapshots,
                "items": [
                    {
                        "item_code": "A",
                        "name": "Tube",
                        "unit": "pc",
                        "quantity": "1",
                        "target_unit_price_rub": "100",
                    }
                ],
                "target_budget_rub": "100",
                "required_deferment_days": 15,
            }
        )
        envelope = WorkerRequestEnvelope.model_validate(
            {
                **context,
                "received_at": receipt["received_at"],
                "schema_version": "1",
                "stage": "S2",
                "attempt": 1,
                "upstream_item": {"entity_type_id": 1040, "item_id": 1},
                "payload": request.model_dump(mode="json"),
            }
        )
        assert telemetry.reserve(envelope, verify_s2_in_worker=True).status == "reserved"
        result, reasons = evaluate(
            request,
            config.s2_tender,
            upstream_confirmed=True,
            verified_quote_ids={quote.quote_id for quote in parsed.quotes},
        )
        outcome = WorkerResponseEnvelope.model_validate(
            {
                **context,
                "schema_version": "1",
                "stage": "S2",
                "attempt": 1,
                "payload": result.model_dump(mode="json"),
                "need_human": bool(reasons),
                "escalation_reasons": reasons,
                "cost_rub": "0.00000000",
                "mode": "deterministic",
                "fallback_reason": None,
            }
        )
        telemetry.complete(envelope, outcome, tokens_in=0, tokens_out=0)
        elapsed = (perf_counter() - start) * 1000
        times.append(elapsed)
        expected = (
            None
            if kind == "quorum"
            or (profile == "construction_supply" and kind in ("budget", "parse"))
            else 2
            if kind == "slow_dump"
            else 1
        )
        assert result.winner_s1_item_id == expected, (profile, kind, result.winner_s1_item_id)
        if expected is not None:
            expected_price = Decimal(94 if kind == "slow_dump" else 90) + Decimal(index % 5) / 100
            assert result.savings_rub == 100 - expected_price
        flagged = {
            p.s1_item_id
            for p in result.participants
            if GATE_CODES.intersection(p.rejection_reasons)
        }
        mismatches = {
            q.s1_item_id for q in parsed.quotes if "S2_ARITHMETIC_MISMATCH" in q.escalation_reasons
        }
        s1 = [
            a + b
            for a, b in zip(
                s1,
                confusion(
                    flagged, independent_labels.get(kind, set()) if kind == "s1_gate" else set()
                ),
                strict=True,
            )
        ]
        arithmetic = [
            a + b
            for a, b in zip(
                arithmetic,
                confusion(
                    mismatches, independent_labels.get(kind, set()) if kind == "parse" else set()
                ),
                strict=True,
            )
        ]
        if kind == "credit":
            assert result.participants[-1].rejection_reasons == ["S2_CREDIT_LIMIT_EXCEEDED"]
        if kind == "budget":
            exclusions = {
                p.s1_item_id
                for p in result.participants
                if "S2_BUDGET_EXCEEDED" in p.rejection_reasons
            }
            assert exclusions == ({4} if profile == "default_b2b" else {3, 4})
        if kind == "parse":
            assert "S2_SOURCE_UNREADABLE" in parsed.quotes[3].escalation_reasons
        if index % 5 == 0:
            assert await post(client, "/v1/deliveries", delivery) == receipt
            assert await post(client, "/s2/parse", parse_body) == response
            assert telemetry.reserve(envelope, verify_s2_in_worker=True).response == outcome
        with telemetry.db.connect() as conn:
            row = conn.execute(
                """SELECT cost_rub_e8 FROM transaction_telemetry WHERE tenant_id=? AND request_id=?""",
                (profile, str(envelope.request_id)),
            ).fetchone()
        assert row["cost_rub_e8"] == 0
        results.append(
            {
                "case": index,
                "kind": kind,
                "request_id": receipt["request_id"],
                "ms": round(elapsed, 3),
                "winner": result.winner_s1_item_id,
                "need_human": bool(reasons),
                "cost_rub_e8": 0,
                "savings_rub": str(result.savings_rub) if result.savings_rub is not None else None,
            }
        )
    assert s1 == [10, 0, 0] and arithmetic == [5, 0, 0]
    review = sum(row["need_human"] is True for row in results)
    report = {
        "profile": profile,
        "cycles": 35,
        "automatic": 35 - review,
        "review": review,
        "p50_ms": round(median(times), 2),
        "p95_ms": round(quantiles(times, n=100, method="inclusive")[94], 2),
        "s1_tp_fp_fn": s1,
        "arithmetic_tp_fp_fn": arithmetic,
        "cycle_cost_rub": "0.00000000",
        "replays": 7,
    }
    return OBJECT.validate_python(report), results


async def run(output: Path) -> None:
    db = Database(output / "benchmark.sqlite3")
    app = create_app(
        Settings(
            sqlite_path=db.path,
            profiles_path=ROOT / "profiles",
            worker_token=SecretStr("synthetic-benchmark"),
            base_url=AnyHttpUrl("http://127.0.0.1:1"),
        )
    )
    summaries, rows = [], {}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://offline.invalid",
            headers={"X-Smartflow-Token": "synthetic-benchmark"},
        ) as client,
    ):
        for profile in PROFILES:
            summary, cases = await profile_run(client, Telemetry(db), profile)
            summaries.append(summary)
            rows[profile] = cases
    (output / "benchmark.json").write_text(
        json.dumps({"summary": summaries, "cases": rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    asyncio.run(run(args.output))


if __name__ == "__main__":
    main()
