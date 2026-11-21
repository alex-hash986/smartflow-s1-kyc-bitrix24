"""Measure paired S3 fixtures with a real SQLite extraction cache and synthetic awards."""

import argparse
import asyncio
import base64
import json
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

import httpx
from pydantic import JsonValue

from core.config import Settings
from core.contracts import JSONObject, WorkerRequestEnvelope, WorkerResponseEnvelope
from core.database import Database, utc_now
from core.idempotency import Delivery, Journal, PublicationCommand, PublicationKey
from core.llm_gateway import LLMGateway
from core.profile_loader import Profile, load_profile
from core.telemetry import Telemetry
from modules.s3_claims.contracts import (
    S3Context,
    S3DeliveryDocumentSource,
    S3EvaluateClaimsRequest,
    S3IntakeRequest,
    S3ParseCall,
    S3ParseDocsRequest,
    S3TenderAwardSnapshot,
    S3TenderGateRequest,
    S3TenderGateResponse,
)
from modules.s3_claims.module import S3Worker, cycle
from ops.runners.benchmark_s1 import ROOT
from ops.runners.benchmark_s2 import confusion

SLUGS = ("default_b2b", "construction_supply")
INN = "7707083893"


@dataclass(frozen=True)
class Case:
    name: str
    qty: str = "100"
    price: str = "1000"
    days: int = 4
    defect: str | None = None
    inn: str = INN
    pdf: bool = False
    claims: tuple[str | None, str | None] = ("0.00", "0.00")
    reasons: tuple[str, ...] = ()


CASES = (
    Case("clean"),
    Case("quantity_boundary", qty="99", claims=("1000.00", "0.00")),
    Case("price_boundary", price="1005", claims=("500.00", "0.00")),
    Case("shortage", qty="75", claims=("25000.00",) * 2),
    Case("defect_act", defect="5", claims=("5000.00",) * 2),
    Case("overcharge", price="1200", claims=("20000.00",) * 2),
    Case("delay", days=7, claims=("300.00",) * 2),
    Case("delay_cap", days=204, claims=("10000.00",) * 2),
    Case("overdelivery", qty="101", reasons=("S3_OVERDELIVERY",)),
    Case("price_drop", price="975", reasons=("S3_PRICE_DECREASE_REVIEW",)),
    Case("threshold", price="2200", claims=("120000.00",) * 2, reasons=("S3_CLAIM_THRESHOLD",)),
    Case(
        "supplier_mismatch",
        inn="7702070139",
        claims=(None, None),
        reasons=("S3_DOCUMENT_SUPPLIER_MISMATCH",),
    ),
    Case(
        "unreadable_pdf",
        pdf=True,
        claims=(None, None),
        reasons=("S3_DOCUMENT_SUPPLIER_MISMATCH", "S3_SOURCE_UNREADABLE"),
    ),
)


def sources(case: Case, index: int) -> list[S3DeliveryDocumentSource]:
    primary = f"BENCH-{index}-UPD"
    items = []
    for defect in (False, True) if case.defect else (False,):
        number = f"BENCH-{index}-ACT" if defect else primary
        raw = {
            "doc_number": number,
            "doc_date": "2026-11-21",
            "currency": "RUB",
            "supplier_inn": case.inn,
            "reference_number": primary if defect else None,
            "reference_date": "2026-11-21" if defect else None,
            "lines": [
                {
                    "item_code": "PIPE",
                    "unit": "pc",
                    "quantity": case.qty,
                    "defect_quantity": case.defect if defect else None,
                    "unit_price_rub": case.price,
                    "line_total_rub": format(Decimal(case.qty) * Decimal(case.price), ".2f"),
                    "vat_treatment": "NO_VAT",
                }
            ],
        }
        items.append(
            S3DeliveryDocumentSource.model_validate(
                {
                    "document_id": str(uuid5(NAMESPACE_URL, number)),
                    "doc_type": "DEFECT_ACT" if defect else "UPD",
                    "doc_number": number,
                    "doc_date": "2026-11-21",
                    "actual_delivery_days": case.days,
                    "source_type": "PDF" if case.pdf else "TEXT",
                    "text": None if case.pdf else json.dumps(raw),
                    "pdf_base64": base64.b64encode(b"unreadable").decode() if case.pdf else None,
                }
            )
        )
    return items


class BenchmarkWorker(S3Worker):
    def profile(self, call: S3Context) -> Profile:
        config = load_profile(self.profiles_path, call.profile_slug, call.profile_revision)
        assert config.slug == call.tenant_id and config.s3_claims and config.s2_tender
        # Keep both profile policies intact. Point the synthetic upstream publication at a fixture.
        config.portal_id = "benchmark.invalid"
        config.s2_tender.bitrix24.entity_type_id = 1044
        return config


def seed(db: Database, request: WorkerRequestEnvelope, item_id: int) -> S3TenderAwardSnapshot:
    identity = request.model_dump(exclude={"received_at", "upstream_item", "payload"})
    upstream = request.model_copy(update={"stage": "S2"})
    telemetry, journal = Telemetry(db), Journal(db)
    telemetry.reserve(upstream, verify_s2_in_worker=True)
    telemetry.complete(
        upstream,
        WorkerResponseEnvelope.model_validate(
            {
                **identity,
                "stage": "S2",
                "payload": {},
                "need_human": False,
                "escalation_reasons": [],
                "mode": "deterministic",
                "fallback_reason": None,
                "cost_rub": "0.00000000",
            }
        ),
    )
    key = PublicationKey(
        tenant_id=request.tenant_id,
        request_id=request.request_id,
        stage="S2",
        purpose="stage_result",
    )
    journal.approve(
        PublicationCommand(
            key=key,
            stage_attempt=1,
            portal_id="benchmark.invalid",
            approval_ref="benchmark:fixture",
            entity_type_id=1044,
            task_required=False,
            fields={},
            external_key_field="external",
        )
    )
    token = journal.claim(key)
    assert token and journal.begin_step(key, token, "item")
    assert journal.record_step(key, token, "item", item_id) and journal.finish(key, token)
    return S3TenderAwardSnapshot.model_validate(
        {
            "s2_item_id": item_id,
            "s1_item_id": 1,
            "tender_status": "AWARDED",
            "winner_inn": INN,
            "winner_name": "Synthetic supplier",
            "awarded_total_rub": "100000.00",
            "promised_delivery_days": 4,
            "promised_deferment_days": 0,
            "approved_credit_limit_rub": "250000.00",
            "awarded_items": [
                {
                    "item_code": "PIPE",
                    "name": "Pipe",
                    "unit": "pc",
                    "quantity": "100.000",
                    "unit_price_rub": "1000.00",
                    "line_total_rub": "100000.00",
                }
            ],
        }
    )


async def measure(
    worker: BenchmarkWorker,
    slug: str,
    profile_index: int,
) -> tuple[JSONObject, list[JSONObject]]:
    db = worker.db
    revision = load_profile(ROOT / "profiles", slug).revision
    expected_claims, actual_claims, expected_blocks, actual_blocks = set(), set(), set(), set()
    times: list[float] = []
    rows: list[JSONObject] = []
    manual, repeats = 0, 0
    for index in range(35):
        case = CASES[index % len(CASES)]
        receipt = Journal(db).register(
            Delivery(
                tenant_id=slug,
                source="s34-benchmark",
                delivery_key=str(index),
                profile_slug=slug,
                profile_revision=revision,
                channel="synthetic",
                is_after_hours=False,
                run_origin="benchmark",
                payload={"case": case.name},
            )
        )
        context = {**receipt.model_dump(exclude={"received_at"}), "tenant_id": slug}
        item_id = index + 1 + 100 * profile_index
        envelope = WorkerRequestEnvelope.model_validate(
            {
                **context,
                "received_at": receipt.received_at,
                "schema_version": "1",
                "stage": "S3",
                "attempt": 1,
                "upstream_item": {"entity_type_id": 1044, "item_id": item_id},
                "payload": {},
            }
        )
        award = seed(db, envelope, item_id)
        intake = S3IntakeRequest(
            **context, delivery_key=str(index), s2_item_id=item_id, documents=sources(case, index)
        )
        start = perf_counter()
        package = worker.intake(intake)
        gate = S3TenderGateResponse(
            **context,
            package_id=package.package_id,
            award=award,
            need_human=False,
            escalation_reasons=[],
        )
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO s3_tender_gates VALUES (?,?,?,?,?)",
                (
                    *cycle(gate),
                    str(gate.package_id),
                    gate.model_dump_json(),
                    utc_now(),
                ),
            )
        if case.days > 4:
            worker.record_penalty_terms(
                S3TenderGateRequest(**context, package_id=package.package_id),
                evidence_ref="benchmark:verified-contract",
                daily_penalty_pct=Decimal("0.10"),
                max_penalty_pct=Decimal("10.00"),
            )
        call = S3ParseCall(**context, payload=S3ParseDocsRequest(package_id=package.package_id))
        parsed = await worker.parse(call, Decimal(0))
        assert parsed != "running"
        envelope.payload = S3EvaluateClaimsRequest(
            package_id=package.package_id,
            award=award,
            documents=parsed.payload.documents,
        ).model_dump(mode="json")
        telemetry = Telemetry(db)
        assert (
            telemetry.reserve(envelope, pin_attempt=partial(worker.pin_attempt, envelope)).status
            == "reserved"
        )
        response = worker.compute(envelope).envelope
        telemetry.complete(envelope, response, tokens_in=0, tokens_out=0)
        elapsed = (perf_counter() - start) * 1000
        times.append(elapsed)
        result = response.payload
        assert result is not None and response.cost_rub == parsed.cost_rub == Decimal(0)
        claim = case.claims[profile_index]
        expected = (
            "ESCALATED" if case.reasons else "CLAIM_REQUIRED" if claim != "0.00" else "ACCEPTED"
        )
        assert result["outcome"] == expected, (slug, case.name, response)
        assert result["claim_amount_rub"] == claim
        assert response.escalation_reasons == list(case.reasons), (slug, case.name, response)
        manual += int(response.need_human)
        if claim not in (None, "0.00"):
            expected_claims.add(index)
        if result["claim_amount_rub"] not in (None, "0.00"):
            actual_claims.add(index)
        if claim is None:
            expected_blocks.add(index)
        if result["delay_days"] is None:
            actual_blocks.add(index)
        if index % 5 == 0:
            assert worker.intake(intake) == package
            assert await worker.parse(call, Decimal(0)) == parsed
            assert telemetry.reserve(envelope).response == response
            repeats += 1
        rows.append(
            {
                "fixture": case.name,
                "request_id": str(receipt.request_id),
                "package_id": str(package.package_id),
                "outcome": expected,
                "elapsed_ms": round(elapsed, 3),
                "claim_rub": claim,
                "reasons": list[JsonValue](response.escalation_reasons),
            }
        )
    with db.connect() as conn:
        costs = conn.execute(
            "SELECT COUNT(*),MIN(cost_rub_e8),MAX(cost_rub_e8) FROM transaction_telemetry WHERE tenant_id=?",
            (slug,),
        ).fetchone()
        assert tuple(costs) == (35, 0, 0)
        extractions = conn.execute(
            "SELECT COUNT(*) FROM s3_extractions WHERE tenant_id=?", (slug,)
        ).fetchone()
        assert extractions[0] == sum(2 if CASES[i % len(CASES)].defect else 1 for i in range(35))
    return {
        "profile": slug,
        "cycles": 35,
        "auto": 35 - manual,
        "manual": manual,
        "p50_ms": round(median(times), 2),
        "p95_ms": round(quantiles(times, n=100, method="inclusive")[94], 2),
        "claims_tp_fp_fn": list[JsonValue](confusion(actual_claims, expected_claims)),
        "blocking_tp_fp_fn": list[JsonValue](confusion(actual_blocks, expected_blocks)),
        "cost_rub": "0.00000000",
        "replays": repeats,
    }, rows


async def run(output: Path) -> None:
    db = Database(output / "benchmark.sqlite3")
    if db.path.exists():
        raise FileExistsError("Pick a fresh output directory for the benchmark")
    db.initialize()
    cfg = Settings(sqlite_path=db.path, profiles_path=ROOT / "profiles")

    def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Deterministic benchmark attempted an LLM call")

    gateway = LLMGateway(cfg, transport=httpx.MockTransport(no_network))
    worker = BenchmarkWorker(cfg.profiles_path, gateway, db)
    results = []
    try:
        for index, slug in enumerate(SLUGS):
            metrics, rows = await measure(worker, slug, index)
            print(json.dumps(metrics), flush=True)
            results.append({"summary": metrics, "runs": rows})
    finally:
        await gateway.aclose()
    (output / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("D:/cache/ai-smartflow-core/s34-bench"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args.output))


if __name__ == "__main__":
    main()
