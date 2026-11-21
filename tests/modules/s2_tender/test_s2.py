import asyncio
import base64
import json
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import yaml
from pydantic import AnyHttpUrl, SecretStr, ValidationError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from core.api import create_app
from core.b24_client import BitrixClient, PortalLimiter
from core.config import ModelRate, Settings
from core.contracts import JSONObject, RunOrigin, WorkerRequestEnvelope, WorkerResponseEnvelope
from core.database import Database
from core.idempotency import (
    Conflict,
    Delivery,
    Journal,
    PublicationCommand,
    PublicationDecision,
    PublicationKey,
    Receipt,
)
from core.llm_gateway import CycleBudget, LLMGateway
from core.profile_loader import S2Profile, load_profile
from core.publisher import Publisher
from core.telemetry import Telemetry
from modules.s2_tender.contracts import (
    S2EvaluateTenderRequest,
    S2EvaluateTenderResponse,
    S2ParsedQuote,
    S2ParseQuotesResponse,
    S2QuoteSource,
    S2SupplierGateSnapshot,
    S2TenderItem,
)
from modules.s2_tender.matcher import match_items
from modules.s2_tender.parser import ExtractedQuote, QuoteParser, normalize
from modules.s2_tender.scoring import evaluate

PROFILES = Path(__file__).resolve().parents[3] / "profiles"
IDENTITIES = [
    ("7707083893", "1027700132195"),
    ("500100732259", "304500116000157"),
    ("7702070139", "1027700132195"),
]


def rules(slug: str = "default_b2b") -> S2Profile:
    profile = load_profile(PROFILES, slug).s2_tender
    assert profile is not None
    return profile


def raw(price: str = "50.00", delivery: int = 5, deferment: int = 15) -> ExtractedQuote:
    total = Decimal(price) * 2
    return ExtractedQuote.model_validate(
        {
            "currency": "RUB",
            "quoted_total": str(total),
            "deferment_days": deferment,
            "items": [
                {
                    "item_code": "A",
                    "name": "Tube",
                    "unit": "pc",
                    "quantity": "2.000",
                    "unit_price": price,
                    "unit_price_includes_vat": True,
                    "vat_treatment": "NO_VAT",
                    "vat_rate_pct": None,
                    "vat_amount": "0.00",
                    "line_total": str(total),
                    "line_total_includes_vat": True,
                    "delivery_days": delivery,
                }
            ],
        }
    )


def source(index: int = 0, text: str | None = None) -> S2QuoteSource:
    return S2QuoteSource(
        quote_id=uuid4(),
        supplier_inn=IDENTITIES[index][0],
        supplier_name="Supply | <script>",
        s1_item_id=index + 1,
        source_type="TEXT",
        text=raw().model_dump_json() if text is None else text,
        pdf_base64=None,
    )


def pdf(text: str = "") -> str:
    writer = PdfWriter()
    page = writer.add_blank_page(612, 792)
    if text:
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {
                        NameObject("/F1"): DictionaryObject(
                            {
                                NameObject("/Type"): NameObject("/Font"),
                                NameObject("/Subtype"): NameObject("/Type1"),
                                NameObject("/BaseFont"): NameObject("/Helvetica"),
                            }
                        ),
                    }
                ),
            }
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 10 Tf 20 700 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = stream
    with BytesIO() as buf:
        writer.write(buf)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def gate(index: int, **changes: object) -> S2SupplierGateSnapshot:
    return S2SupplierGateSnapshot.model_validate(
        {
            "s1_item_id": index + 1,
            "inn": IDENTITIES[index][0],
            "ogrn": IDENTITIES[index][1],
            "kyc_status": "APPROVED",
            "reliability_score": "100.00",
            "approved_credit_limit_rub": "1000.00",
            "stage_result_published": True,
        }
        | changes
    )


def item() -> S2TenderItem:
    return S2TenderItem.model_validate(
        {
            "item_code": "A",
            "name": "Tube",
            "unit": "pc",
            "quantity": "2",
            "target_unit_price_rub": "50",
        }
    )


def request_data(
    quotes: list[S2ParsedQuote],
    snapshots: list[S2SupplierGateSnapshot],
) -> S2EvaluateTenderRequest:
    return S2EvaluateTenderRequest.model_validate(
        {
            "items": [item()],
            "supplier_gate_snapshots": snapshots,
            "quotes": quotes,
            "target_budget_rub": "100.00",
            "required_deferment_days": 15,
        }
    )


def score(
    quotes: list[S2ParsedQuote],
    snapshots: list[S2SupplierGateSnapshot],
    slug: str = "default_b2b",
) -> tuple[S2EvaluateTenderResponse, list[str]]:
    return evaluate(
        request_data(quotes, snapshots),
        rules(slug),
        upstream_confirmed=True,
        verified_quote_ids={quote.quote_id for quote in quotes},
    )


def settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=tmp_path / "test.db",
        profiles_path=PROFILES,
        worker_token=SecretStr("test"),
        base_url=AnyHttpUrl("https://llm.invalid/v1"),
        fast_llm_model="fast",
        llm_model_rates={
            "fast": ModelRate(input_usd_per_million=Decimal(1), output_usd_per_million=Decimal(1))
        },
        usd_to_rub_rate=Decimal(100),
        max_cycle_cost_rub=Decimal(1),
    )


@pytest.mark.parametrize("quantity", [2, 2.001, "2.0001", "2e0", "NaN"])
def test_public_dto_rejects_noncanonical_quantities(quantity: object) -> None:
    with pytest.raises(ValidationError):
        S2TenderItem.model_validate(item().model_dump(mode="json") | {"quantity": quantity})
    assert item().model_dump(mode="json")["quantity"] == "2.000"


@pytest.mark.asyncio
async def test_text_and_pdf_without_llm(tmp_path: Path) -> None:
    def unexpected(_: httpx.Request) -> httpx.Response:
        pytest.fail("Structured quote must not call FAST LLM")

    gateway = LLMGateway(settings(tmp_path), transport=httpx.MockTransport(unexpected))
    parser = QuoteParser(gateway)
    document = raw().model_dump_json()
    try:
        for quote_source in (
            source(text=document),
            source().model_copy(
                update={"source_type": "PDF", "text": None, "pdf_base64": pdf(document)}
            ),
        ):
            result = await parser.parse(quote_source, rules().extraction, CycleBudget(Decimal(0)))
            assert result.mode == "deterministic"
            assert result.quote.arithmetic_ok and not result.quote.need_human
            assert result.quote.items[0].vat_amount_rub == Decimal("0.00")
        scan = source().model_copy(update={"source_type": "PDF", "text": None, "pdf_base64": pdf()})
        blank = await parser.parse(scan, rules().extraction, CycleBudget(Decimal(0)))
        assert blank.quote.escalation_reasons == ["S2_SOURCE_UNREADABLE"]
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_fast_extraction_vat_and_budget(tmp_path: Path) -> None:
    calls: list[str] = []
    extracted = raw()
    line = extracted.items[0]
    line.unit_price = "50,00"
    line.unit_price_includes_vat = False
    line.vat_treatment, line.vat_rate_pct, line.vat_amount = "TAXABLE", "20", "20.00"
    line.line_total_includes_vat = False
    extracted.quoted_total = "120.00"

    def send(req: httpx.Request) -> httpx.Response:
        calls.append(json.loads(req.content)["model"])
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": extracted.model_dump_json()}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    gateway = LLMGateway(settings(tmp_path), transport=httpx.MockTransport(send))
    parser = QuoteParser(gateway)
    try:
        for quote_source in (
            source(text="Tube 2 x 50.00 plus VAT 20%, total 120.00 RUB"),
            source().model_copy(
                update={
                    "source_type": "PDF",
                    "text": None,
                    "pdf_base64": pdf("Tube 2 x 50 RUB plus VAT 20%"),
                }
            ),
        ):
            result = await parser.parse(quote_source, rules().extraction, CycleBudget(Decimal(1)))
            assert result.mode == "llm" and result.prompt_version == "1/baseline"
            assert result.cost_rub == Decimal(".015")
            assert result.quote.items[0].unit_price_rub == Decimal("60.00")
            assert result.quote.items[0].total_rub == Decimal("120.00")
            assert not result.quote.need_human
        blocked = await parser.parse(
            source(text="unstructured"), rules().extraction, CycleBudget(Decimal(0))
        )
        assert blocked.quote.escalation_reasons == ["S2_PARSE_BUDGET_EXCEEDED"]
        assert blocked.mode == "fallback" and calls == ["fast", "fast"]
    finally:
        await gateway.aclose()


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("line_total", "99.00", "S2_ARITHMETIC_MISMATCH"),
        ("vat_amount", "1.00", "S2_ARITHMETIC_MISMATCH"),
        ("quantity", "2.0001", "S2_UNSUPPORTED_PRECISION"),
        ("unit_price", "50.001", "S2_UNSUPPORTED_PRECISION"),
    ],
)
def test_arithmetic_and_precision(field: str, value: str, reason: str) -> None:
    extracted = raw()
    setattr(extracted.items[0], field, value)
    quote = normalize(source(), extracted)
    assert quote.need_human and reason in quote.escalation_reasons
    assert not quote.arithmetic_ok


def test_quote_total_mismatch() -> None:
    extracted = raw()
    extracted.quoted_total = "100.01"
    quote = normalize(source(), extracted)
    assert quote.items[0].arithmetic_ok
    assert not quote.arithmetic_ok
    assert quote.escalation_reasons == ["S2_ARITHMETIC_MISMATCH"]


def test_matching_and_incomplete_specification() -> None:
    line = normalize(source(), raw()).items[0]
    inferred = line.model_copy(update={"item_code": None, "name": "ＴＵＢＥ", "unit": "ＰＣ"})
    assert match_items([item()], [inferred]) == {"A": inferred}
    extra = item().model_copy(update={"item_code": "B"})
    assert match_items([item(), extra], [inferred]) is None
    assert match_items([item()], [line, line]) is None
    assert match_items([item()], [line.model_copy(update={"quantity": Decimal(3)})]) is None
    quote = normalize(source(), raw())
    incomplete = request_data([quote], [gate(0)]).model_copy(update={"items": [item(), extra]})
    result, reasons = evaluate(
        incomplete, rules(), upstream_confirmed=True, verified_quote_ids={quote.quote_id}
    )
    assert reasons == ["S2_QUORUM_NOT_MET"]
    assert result.participants[0].rejection_reasons == ["S2_ITEM_MATCH_FAILED"]


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"kyc_status": "REJECTED", "approved_credit_limit_rub": "0"}, "S2_S1_REJECTED"),
        ({"stage_result_published": False}, "S2_S1_UNCONFIRMED"),
        ({"approved_credit_limit_rub": "99.99"}, "S2_CREDIT_LIMIT_EXCEEDED"),
    ],
)
def test_gate_exclusion(changes: dict[str, object], reason: str) -> None:
    quotes = [normalize(source(i), raw()) for i in range(2)]
    result, reasons = score(quotes, [gate(0, **changes), gate(1)])
    assert reasons == ["S2_QUORUM_NOT_MET"] and result.winner_inn is None
    assert result.participants[-1].rejection_reasons == [reason]


def test_profile_budget_quorum_and_scoring() -> None:
    quotes = [
        normalize(source(i), raw(price, due))
        for i, (price, due) in enumerate([("48.00", 5), ("50.00", 4), ("52.00", 8)])
    ]
    gates = [gate(i) for i in range(3)]
    normal, reasons = score(quotes, gates)
    assert reasons == [] and normal.winner_inn == gates[1].inn
    assert normal.participants[0].score == Decimal("98.0000")
    assert normal.savings_rub == Decimal("0.00")
    assert "Supply &#124; &#60;script&#62;" in normal.comparison_matrix_markdown
    assert score(quotes, gates)[0].comparison_matrix_markdown == normal.comparison_matrix_markdown

    construction, reasons = score(quotes, gates, "construction_supply")
    assert reasons == ["S2_QUORUM_NOT_MET"]
    assert construction.participants[-1].rejection_reasons == ["S2_BUDGET_EXCEEDED"]
    assert construction.participants[0].score == Decimal("98.4000")

    quotes[1] = normalize(source(1), raw("50.00", 6))
    result, reasons = score(quotes[:2], gates)
    assert reasons == [] and result.savings_rub == Decimal("4.00")
    assert result.winner_total_rub == Decimal("96.00")


def register(
    journal: Journal,
    tenant: str,
    profile: str,
    revision: str,
    index: int,
    run_origin: RunOrigin = "mock",
) -> Receipt:
    return journal.register(
        Delivery.model_validate(
            {
                "tenant_id": tenant,
                "source": "test",
                "delivery_key": str(index),
                "payload": {},
                "profile_slug": profile,
                "profile_revision": revision,
                "channel": "test",
                "is_after_hours": False,
                "run_origin": run_origin,
            }
        )
    )


def publish_s1(journal: Journal, receipt: Receipt, tenant: str, index: int) -> None:
    request = WorkerRequestEnvelope.model_validate(
        {
            **receipt.model_dump(),
            "tenant_id": tenant,
            "stage": "S1",
            "attempt": 1,
            "schema_version": "1",
            "upstream_item": None,
            "payload": {},
        }
    )
    telemetry = Telemetry(journal.db)
    telemetry.reserve(request)
    telemetry.complete(
        request,
        WorkerResponseEnvelope.model_validate(
            {
                **request.model_dump(exclude={"received_at", "upstream_item"}),
                "need_human": False,
                "escalation_reasons": [],
                "cost_rub": "0.00000000",
                "mode": "deterministic",
                "fallback_reason": None,
            }
        ),
    )
    supplier = gate(index)
    key = PublicationKey(
        tenant_id=tenant, request_id=receipt.request_id, stage="S1", purpose="stage_result"
    )
    journal.approve(
        PublicationCommand(
            key=key,
            stage_attempt=1,
            portal_id="crm.invalid",
            approval_ref="fixture",
            entity_type_id=1040,
            task_required=False,
            external_key_field="ufCrm7ExternalKey",
            fields={
                "stageId": "DT1040_15:SUCCESS",
                "ufCrm7Inn": supplier.inn,
                "ufCrm7Ogrn": supplier.ogrn,
                "ufCrm7KycStatus": 45,
                "ufCrm7ReliabilityScore": str(supplier.reliability_score),
                "ufCrm7ApprovedCreditLimitRub": str(supplier.approved_credit_limit_rub),
            },
        )
    )
    token = journal.claim(key)
    assert token is not None and journal.begin_step(key, token, "item")
    assert journal.record_step(key, token, "item", supplier.s1_item_id)
    assert journal.finish(key, token)


def bound_profiles(tmp_path: Path) -> Path:
    slug = "portal_b24_b2b"
    document = yaml.safe_load((PROFILES / slug / "profile.yaml").read_text(encoding="utf-8"))
    document["portal_id"] = "crm.invalid"
    document["revision"] = "4"
    mapping = document["s2_tender"]["bitrix24"]
    mapping.update(entity_type_id=2200, category_id=4, responsible_id=1)
    mapping["stage_ids"] = {name: f"DT2200_4:{name}" for name in mapping["stage_ids"]}
    mapping["fields"] = {name: f"ufCrm22{name}" for name in mapping["fields"]}
    mapping["status_values"] = {name: i for i, name in enumerate(mapping["status_values"], 71)}
    path = tmp_path / "profiles" / slug
    path.mkdir(parents=True)
    (path / "profile.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return path.parent


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["award", "cancel", "pilot"])
async def test_s2_routes_and_publication(tmp_path: Path, scenario: str) -> None:
    tenant = "portal_b24_b2b"
    configuration = settings(tmp_path).model_copy(
        update={"profiles_path": bound_profiles(tmp_path)}
    )
    journal = Journal(Database(configuration.sqlite_path))
    items: list[JSONObject] = []
    comments: list[JSONObject] = []

    def crm(req: httpx.Request) -> httpx.Response:
        method = req.url.path.rsplit("/", 1)[-1].removesuffix(".json")
        body = json.loads(req.content)
        if method == "crm.item.list":
            return httpx.Response(200, json={"result": {"items": items}})
        if method == "crm.item.add":
            items.append({"id": 11, **body["fields"]})
        elif method == "crm.item.update":
            assert int(body["id"]) == 11 and len(items) == 1
            items[0].update(body["fields"])
        elif method == "crm.timeline.comment.add":
            comments.append(body["fields"])
            return httpx.Response(200, json={"result": len(comments)})
        else:
            assert method == "crm.item.get" and int(body["id"]) == 11
        return httpx.Response(200, json={"result": {"item": items[0]}})

    publisher = Publisher(
        journal,
        BitrixClient(
            "https://crm.invalid/rest/1/token", PortalLimiter(2), transport=httpx.MockTransport(crm)
        ),
        "crm.invalid",
    )
    app = create_app(configuration, publisher=publisher)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8021",
            headers={"X-Smartflow-Token": "test"},
        ) as client,
    ):
        delivery = register(
            journal, tenant, tenant, "4", 0, run_origin="pilot" if scenario == "pilot" else "mock"
        )
        upstream = register(journal, tenant, tenant, "4", 1)
        publish_s1(journal, delivery, tenant, 0)
        publish_s1(journal, upstream, tenant, 1)
        context = {"tenant_id": tenant, **delivery.model_dump(mode="json", exclude={"received_at"})}
        quotes = [source(0, raw("49.00").model_dump_json()), source(1)]
        parse_body = {**context, "payload": {"quotes": [q.model_dump(mode="json") for q in quotes]}}
        first = await client.post("/s2/parse", json=parse_body)
        assert first.status_code == 200, first.text
        parsed = S2ParseQuotesResponse.model_validate(first.json()["payload"])
        assert not parsed.need_human
        assert (await client.post("/s2/parse", json=parse_body)).json() == first.json()
        unconfirmed = register(journal, tenant, tenant, "4", 2)
        wrong_cycle = await client.post(
            "/s2/evaluate",
            json={
                **context,
                **unconfirmed.model_dump(mode="json"),
                "schema_version": "1",
                "stage": "S2",
                "attempt": 1,
                "upstream_item": {"entity_type_id": 1040, "item_id": 1},
                "payload": request_data([], []).model_dump(mode="json"),
            },
        )
        assert wrong_cycle.status_code == 200, wrong_cycle.text
        assert wrong_cycle.json()["fallback_reason"] == "S2_UPSTREAM_UNCONFIRMED"
        assert wrong_cycle.json()["payload"]["winner_inn"] is None
        changed = source(text="different").model_copy(update={"quote_id": quotes[0].quote_id})
        assert (
            await client.post(
                "/s2/parse",
                json=context
                | {
                    "payload": {
                        "quotes": [changed.model_dump(mode="json")],
                    }
                },
            )
        ).status_code == 409
        assert (
            await client.post(
                "/s2/parse",
                json=parse_body,
                headers={
                    "X-Smartflow-Token": "invalid",
                },
            )
        ).status_code == 401

        altered = parsed.quotes[1].model_copy(update={"supplier_name": "different"})
        snapshots = [gate(0, reliability_score="99.00"), gate(1)]
        body = {
            **context,
            "schema_version": "1",
            "stage": "S2",
            "attempt": 1,
            "received_at": delivery.model_dump(mode="json")["received_at"],
            "upstream_item": {"entity_type_id": 1040, "item_id": 1},
            "payload": request_data([parsed.quotes[0], altered], snapshots).model_dump(mode="json"),
        }
        response = await client.post("/s2/evaluate", json=body)
        assert response.status_code == 200, response.text
        rejected = response.json()["payload"]["participants"]
        assert {p["rejection_reasons"][0] for p in rejected} == {
            "S2_QUOTE_INVALID",
            "S2_S1_UNCONFIRMED",
        }
        assert response.json()["need_human"]
        replay = await client.post("/v1/workers/S2/run", json=body)
        assert replay.status_code == 200 and replay.json() == response.json()

        prepare = await client.post("/s2/publication/prepare", json=context | {"attempt": 1})
        assert prepare.status_code == 200, prepare.text
        review = prepare.json()
        assert review["purpose"] == "human_review"
        dispatched = await client.post("/v1/publications/dispatch", json=review)
        assert dispatched.status_code == 200 and dispatched.json()["item_id"] == 11, dispatched.text
        assert items[0]["ufCrm22TENDER_STATUS"] == 72 and items[0]["ufCrm22WINNER_INN"] is None

        if scenario == "cancel":
            journal.record_decision(
                PublicationDecision(
                    tenant_id=tenant,
                    request_id=delivery.request_id,
                    stage="S2",
                    attempt=1,
                    decision="cancel",
                    evidence_ref="bitrix:review:cancelled:11",
                )
            )
            finish = await client.post("/s2/publication/prepare", json=context | {"attempt": 1})
            assert finish.status_code == 200, finish.text
            assert finish.json()["purpose"] == "stage_result"
            dispatched = await client.post("/v1/publications/dispatch", json=finish.json())
            assert dispatched.status_code == 200 and dispatched.json()["item_id"] == 11
            assert items[0]["ufCrm22TENDER_STATUS"] == 73 and items[0]["ufCrm22WINNER_INN"] is None
            with pytest.raises(Conflict, match="closed"):
                Telemetry(journal.db).reserve(
                    WorkerRequestEnvelope.model_validate(
                        body
                        | {
                            "stage": "S3",
                            "upstream_item": {"entity_type_id": 2200, "item_id": 11},
                        }
                    )
                )
            return

        body["attempt"] = 2
        body["payload"] = request_data(parsed.quotes, [gate(0), gate(1)]).model_dump(mode="json")
        awarded = await client.post("/v1/workers/S2/run", json=body)
        assert awarded.status_code == 200, awarded.text
        assert awarded.json()["payload"]["savings_rub"] == "2.00"
        assert (await client.post("/s2/evaluate", json=body)).json() == awarded.json()
        if scenario == "pilot":
            blocked = await client.post("/s2/publication/prepare", json=context | {"attempt": 2})
            assert blocked.status_code == 409
            journal.record_decision(
                PublicationDecision(
                    tenant_id=tenant,
                    request_id=delivery.request_id,
                    stage="S2",
                    attempt=2,
                    decision="approve",
                    evidence_ref="bitrix:award:11",
                )
            )
        finalize = await client.post("/s2/publication/prepare", json=context | {"attempt": 2})
        assert finalize.status_code == 200 and finalize.json()["purpose"] == "stage_result"
        dispatched = await client.post("/v1/publications/dispatch", json=finalize.json())
        assert dispatched.status_code == 200 and dispatched.json()["item_id"] == 11, dispatched.text
        assert len(items) == 1 and items[0]["ufCrm22WINNER_INN"] == gate(0).inn
        assert len(comments) == 2
        assert comments[0]["COMMENT"] != comments[1]["COMMENT"]
        assert (
            await client.post("/s2/publication/prepare", json=context | {"attempt": 2})
        ).json() == (finalize.json())


@pytest.mark.asyncio
async def test_batch_reservation_and_cost_not_rebilled(tmp_path: Path) -> None:
    configuration = settings(tmp_path)
    sent, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def send(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        sent.set()
        await release.wait()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": raw().model_dump_json()}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    gateway = LLMGateway(configuration, transport=httpx.MockTransport(send))
    app = create_app(configuration, gateway=gateway)
    journal = Journal(Database(configuration.sqlite_path))
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8021",
                headers={"X-Smartflow-Token": "test"},
            ) as client,
        ):
            receipt = register(journal, "demo", "default_b2b", "2", 0)
            body = {
                "tenant_id": "demo",
                **receipt.model_dump(mode="json", exclude={"received_at"}),
                "payload": {"quotes": [source(text="unstructured quote").model_dump(mode="json")]},
            }
            pending = asyncio.create_task(client.post("/s2/parse", json=body))
            try:
                await asyncio.wait_for(sent.wait(), timeout=5)
                replay = await client.post("/s2/parse", json=body)
                assert replay.status_code == 202 and replay.headers["Retry-After"] == "1"
                competing = await client.post(
                    "/s2/parse",
                    json=body
                    | {
                        "payload": {
                            "quotes": [
                                source(1, text="unstructured quote").model_dump(mode="json")
                            ],
                        }
                    },
                )
                assert competing.status_code == 202
            finally:
                release.set()
                first = await pending
            assert first.status_code == 200 and first.json()["cost_rub"] == "0.01500000"
            assert (await client.post("/s2/parse", json=body)).json() == first.json()
            assert calls == 1
            assert Telemetry(journal.db).remaining_budget(
                "demo", receipt.request_id, Decimal(1)
            ) == (Decimal(".985"))
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_live_tender_intake_reads_s1_and_verifies_decision(tmp_path: Path) -> None:
    tenant = "portal_b24_b2b"
    configuration = settings(tmp_path).model_copy(
        update={"profiles_path": bound_profiles(tmp_path)}
    )
    journal = Journal(Database(configuration.sqlite_path))
    cards: list[JSONObject] = []

    def crm(req: httpx.Request) -> httpx.Response:
        method = req.url.path.rsplit("/", 1)[-1].removesuffix(".json")
        data = json.loads(req.content)
        if method == "crm.item.get" and data["entityTypeId"] == 1040:
            item_id = int(data["id"])
            supplier = gate(item_id - 1)
            return httpx.Response(
                200,
                json={
                    "result": {
                        "item": {
                            "id": item_id,
                            "categoryId": 15,
                            "stageId": "DT1040_15:SUCCESS",
                            "ufCrm7Inn": supplier.inn,
                            "ufCrm7Ogrn": supplier.ogrn,
                            "ufCrm7KycStatus": "45",
                            "ufCrm7ReliabilityScore": str(supplier.reliability_score),
                            "ufCrm7ApprovedCreditLimitRub": str(supplier.approved_credit_limit_rub),
                        }
                    }
                },
            )
        if method == "crm.item.list":
            return httpx.Response(200, json={"result": {"items": cards}})
        if method == "crm.item.add":
            cards.append({"id": 12, "updatedBy": 1, **data["fields"]})
        elif method == "crm.item.update":
            assert len(cards) == 1
            cards[0].update(data["fields"])
        elif method == "crm.timeline.comment.add":
            return httpx.Response(200, json={"result": 18})
        else:
            assert method == "crm.item.get"
        return httpx.Response(200, json={"result": {"item": cards[0]}})

    async def no_wait(_: float) -> None:
        return None

    publisher = Publisher(
        journal,
        BitrixClient(
            "https://crm.invalid/rest/1/token",
            PortalLimiter(2, sleep=no_wait),
            transport=httpx.MockTransport(crm),
        ),
        "crm.invalid",
    )
    app = create_app(configuration, publisher=publisher)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8021",
            headers={"X-Smartflow-Token": "test"},
        ) as client,
    ):
        anchor = register(journal, tenant, tenant, "4", 1)
        supplier = register(journal, tenant, tenant, "4", 2)
        publish_s1(journal, anchor, tenant, 0)
        publish_s1(journal, supplier, tenant, 1)
        quotes = [source(0, raw("49.00").model_dump_json()), source(1)]
        tender = {
            "profile_slug": tenant,
            "delivery_key": "tender-1",
            "upstream_s1_item_id": 1,
            "items": [item().model_dump(mode="json")],
            "target_budget_rub": "100.00",
            "required_deferment_days": 15,
            "quotes": [quote.model_dump(mode="json") for quote in quotes],
        }
        intake = await client.post("/s2/intake", json=tender)
        assert intake.status_code == 200, intake.text
        receipt = intake.json()
        assert receipt["request_id"] == str(anchor.request_id) and receipt["attempt"] == 1
        assert (await client.post("/s2/intake", json=tender)).json() == receipt
        assert (
            await client.post(
                "/s2/intake",
                json=tender
                | {
                    "target_budget_rub": "99.00",
                },
            )
        ).status_code == 409
        context = {
            key: receipt[key]
            for key in ("tenant_id", "request_id", "profile_slug", "profile_revision")
        }
        response = await client.post(
            "/s2/supplier-gates",
            json=context
            | {
                "upstream_item": receipt["upstream_item"],
                "suppliers": [
                    {"supplier_inn": quote.supplier_inn, "s1_item_id": quote.s1_item_id}
                    for quote in quotes
                ],
            },
        )
        assert response.status_code == 200, response.text
        snapshots = response.json()["supplier_gate_snapshots"]
        assert len(snapshots) == 2 and all(row["stage_result_published"] for row in snapshots)
        parsed = await client.post(
            "/s2/parse", json=context | {"payload": {"quotes": tender["quotes"]}}
        )
        assert parsed.status_code == 200, parsed.text
        evaluated = await client.post(
            "/s2/evaluate",
            json=receipt
            | {
                "schema_version": "1",
                "stage": "S2",
                "payload": {
                    "items": tender["items"],
                    "target_budget_rub": tender["target_budget_rub"],
                    "required_deferment_days": tender["required_deferment_days"],
                    "supplier_gate_snapshots": snapshots,
                    "quotes": parsed.json()["payload"]["quotes"],
                },
            },
        )
        assert evaluated.status_code == 200 and not evaluated.json()["need_human"], evaluated.text

        publication_call = context | {"attempt": 1}
        review = await client.post("/s2/publication/prepare", json=publication_call)
        assert review.status_code == 200 and review.json()["purpose"] == "human_review", review.text
        dispatch = await client.post("/v1/publications/dispatch", json=review.json())
        assert dispatch.status_code == 200 and dispatch.json()["status"] == "published", (
            dispatch.text
        )
        assert cards[0]["ufCrm22WINNER_INN"] is None
        with journal.db.connect() as conn:
            cycle = conn.execute(
                "SELECT outcome,need_human FROM transaction_telemetry WHERE tenant_id=? AND request_id=?",
                (tenant, str(anchor.request_id)),
            ).fetchone()
        assert tuple(cycle) == ("waiting_human", 1)
        rejected = await client.post("/s2/publication/decision", json=publication_call)
        assert rejected.status_code == 409

        cards[0].update(stageId="DT2200_4:awarded", ufCrm22TENDER_STATUS=71, updatedBy=2)
        refused = await client.post("/s2/publication/decision", json=publication_call)
        assert refused.status_code == 409
        cards[0]["updatedBy"] = 1
        approved = await client.post("/s2/publication/decision", json=publication_call)
        assert approved.status_code == 204, approved.text
        final = await client.post("/s2/publication/prepare", json=publication_call)
        assert final.status_code == 200 and final.json()["purpose"] == "stage_result"
        dispatch = await client.post("/v1/publications/dispatch", json=final.json())
        assert dispatch.status_code == 200 and dispatch.json()["status"] == "published", (
            dispatch.text
        )
        assert len(cards) == 1 and cards[0]["ufCrm22WINNER_INN"] == gate(0).inn
