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
from pydantic import AnyHttpUrl, JsonValue, SecretStr, ValidationError
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from core.api import create_app
from core.b24_client import BitrixClient, PortalLimiter
from core.config import ModelRate, Settings
from core.contracts import JSONObject, WorkerRequestEnvelope, WorkerResponseEnvelope
from core.database import Database, canonical_json
from core.idempotency import Delivery, Journal, PublicationCommand, PublicationKey, Receipt
from core.llm_gateway import LLMGateway
from core.profile_loader import S3Profile, load_profile
from core.publisher import Publisher
from core.telemetry import Telemetry
from modules.s3_claims.contracts import (
    S3AwardedItem,
    S3DeliveryDocumentSource,
    S3EvaluateClaimsRequest,
    S3TenderAwardSnapshot,
    S3TenderGateRequest,
)
from modules.s3_claims.module import S3Worker
from modules.s3_claims.parser import ExtractedDocument, SourceLimit, normalize, source_text
from modules.s3_claims.scoring import evaluate

ROOT = Path(__file__).resolve().parents[3]
SLUG = "portal_b24_b2b"
INN = "7707083893"
OTHER_INN = "7702070139"


def award() -> S3TenderAwardSnapshot:
    return S3TenderAwardSnapshot.model_validate(
        {
            "s2_item_id": 22,
            "s1_item_id": 11,
            "tender_status": "AWARDED",
            "winner_inn": INN,
            "winner_name": "Тест <img src=x>",
            "awarded_total_rub": "1000.00",
            "promised_delivery_days": 5,
            "promised_deferment_days": 0,
            "approved_credit_limit_rub": "5000.00",
            "awarded_items": [
                {
                    "item_code": "A",
                    "name": "Болт",
                    "unit": "шт",
                    "quantity": "10.000",
                    "unit_price_rub": "100.00",
                    "line_total_rub": "1000.00",
                },
            ],
        }
    )


def source(
    qty: str = "10",
    price: str = "100",
    *,
    days: int = 5,
    defect: str | None = None,
    inn: str = INN,
    code: str | None = "A",
) -> S3DeliveryDocumentSource:
    number = "A-1" if defect is None else "B-1"
    document = {
        "doc_number": number,
        "doc_date": "2026-11-21",
        "currency": "RUB",
        "supplier_inn": inn,
        "reference_number": None if defect is None else "A-1",
        "reference_date": None if defect is None else "2026-11-21",
        "lines": [
            {
                "item_code": code,
                "name": " БОЛТ ",
                "unit": "ШТ",
                "quantity": qty,
                "defect_quantity": defect,
                "unit_price_rub": price,
                "line_total_rub": str((Decimal(qty) * Decimal(price)).quantize(Decimal(".01"))),
                "vat_treatment": "GROSS",
            },
        ],
    }
    return S3DeliveryDocumentSource.model_validate(
        {
            "document_id": str(uuid4()),
            "doc_type": "UPD" if defect is None else "DEFECT_ACT",
            "doc_number": number,
            "doc_date": "2026-11-21",
            "actual_delivery_days": days,
            "source_type": "TEXT",
            "text": json.dumps(document),
            "pdf_base64": None,
        }
    )


def rules(**updates: str) -> S3Profile:
    profile = load_profile(ROOT / "profiles", "default_b2b").s3_claims
    assert profile is not None
    return S3Profile.model_validate({**profile.model_dump(), **updates})


@pytest.mark.parametrize(
    ("qty", "price", "defect", "days", "confirmed", "updates", "outcome", "claim", "reasons"),
    [
        ("10", "100", None, 5, False, {}, "ACCEPTED", "0.00", []),
        (
            "9",
            "101",
            None,
            5,
            False,
            {"price_tolerance_pct": "1", "quantity_tolerance_pct": "10"},
            "ACCEPTED",
            "0.00",
            [],
        ),
        ("8", "100", None, 5, False, {}, "CLAIM_REQUIRED", "200.00", []),
        ("10", "100", "2", 5, False, {}, "CLAIM_REQUIRED", "200.00", []),
        ("10", "110", None, 5, False, {}, "CLAIM_REQUIRED", "100.00", []),
        ("10", "100", None, 7, True, {}, "CLAIM_REQUIRED", "2.00", []),
        ("10", "100", None, 200, True, {}, "CLAIM_REQUIRED", "100.00", []),
        ("10", "100", None, 7, False, {}, "ESCALATED", None, ["S3_PENALTY_TERMS_UNCONFIRMED"]),
        ("11", "100", None, 5, False, {}, "ESCALATED", "0.00", ["S3_OVERDELIVERY"]),
        ("10", "99", None, 5, False, {}, "ESCALATED", "0.00", ["S3_PRICE_DECREASE_REVIEW"]),
        (
            "9",
            "100",
            None,
            5,
            False,
            {"manual_claim_threshold_rub": "100"},
            "ESCALATED",
            "100.00",
            ["S3_CLAIM_THRESHOLD"],
        ),
    ],
)
def test_decimal_calculation(
    qty: str,
    price: str,
    defect: str | None,
    days: int,
    confirmed: bool,
    updates: dict[str, str],
    outcome: str,
    claim: str | None,
    reasons: list[str],
) -> None:
    documents = [source(qty, price, days=days, code=None)]
    if defect is not None:
        documents.append(source(qty, price, defect=defect, days=days))
    parsed = [
        normalize(doc, ExtractedDocument.model_validate_json(doc.text or "{}"), documents)
        for doc in documents
    ]
    result, flagged = evaluate(
        S3EvaluateClaimsRequest(package_id=uuid4(), award=award(), documents=parsed),
        rules(**updates),
        penalty_confirmed=confirmed,
    )
    assert result.outcome == outcome
    assert result.claim_amount_rub == (Decimal(claim) if claim is not None else None)
    assert flagged == reasons
    assert "<img" not in result.reconciliation_report_markdown


@pytest.mark.parametrize(
    "value", [1, 1.0, "1.0000", "1e0", "+1", "NaN", "-0", "1234567890123456789"]
)
def test_strict_quantity(value: object) -> None:
    with pytest.raises(ValidationError):
        S3AwardedItem.model_validate(
            {
                **award().awarded_items[0].model_dump(mode="json"),
                "quantity": value,
            }
        )


def test_document_identity_supplier_and_arithmetic() -> None:
    doc = source(inn=OTHER_INN)
    raw = ExtractedDocument.model_validate_json(doc.text or "{}")
    parsed = normalize(doc, raw, [doc])
    _, flagged = evaluate(
        S3EvaluateClaimsRequest(package_id=uuid4(), award=award(), documents=[parsed]),
        rules(),
    )
    assert flagged == ["S3_DOCUMENT_SUPPLIER_MISMATCH"]
    raw.doc_number = "unexpected"
    raw.lines[0].line_total_rub = "11"
    invalid = normalize(doc, raw, [doc])
    assert not invalid.arithmetic_ok
    assert invalid.escalation_reasons == ["S3_ARITHMETIC_MISMATCH", "S3_DOCUMENT_INCOMPLETE"]


def test_pdf_text_layer_and_page_limit() -> None:
    raw = source()
    writer = PdfWriter()
    page = writer.add_blank_page(612, 792)
    writer.add_blank_page(612, 792)
    name = NameObject
    page[name("/Resources")] = DictionaryObject(
        {
            name("/Font"): DictionaryObject(
                {
                    name("/F1"): DictionaryObject(
                        {
                            name("/Type"): name("/Font"),
                            name("/Subtype"): name("/Type1"),
                            name("/BaseFont"): name("/Helvetica"),
                        }
                    ),
                }
            ),
        }
    )
    stream = DecodedStreamObject()
    document = json.loads(raw.text or "{}")
    document["lines"][0].update(name="bolt", unit="each")
    text = json.dumps(document, ensure_ascii=True)
    stream.set_data(f"BT /F1 10 Tf 20 700 Td ({text}) Tj ET".encode("ascii"))
    page[name("/Contents")] = stream
    with BytesIO() as buffer:
        writer.write(buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode()
    pdf = S3DeliveryDocumentSource.model_validate(
        {
            **raw.model_dump(mode="json"),
            "source_type": "PDF",
            "text": None,
            "pdf_base64": encoded,
        }
    )
    with pytest.raises(SourceLimit):
        source_text(pdf, rules().extraction, 1)
    extracted = ExtractedDocument.model_validate_json(source_text(pdf, rules().extraction, 2))
    assert normalize(pdf, extracted, [pdf]).arithmetic_ok


def test_profile_requires_distinct_complete_active_bindings() -> None:
    profile = rules()
    with pytest.raises(ValidationError):
        S3Profile.model_validate({**profile.model_dump(), "enabled": True})
    duplicate = profile.model_dump()
    duplicate["bitrix24"]["fields"].update(SUPPLIER_INN="field1", EXTERNAL_KEY="field1")
    with pytest.raises(ValidationError):
        S3Profile.model_validate(duplicate)


def active_profiles(tmp_path: Path) -> Path:
    document = yaml.safe_load(
        (ROOT / "profiles" / SLUG / "profile.yaml").read_text(encoding="utf-8")
    )
    document.update(portal_id="crm.invalid", revision="5")
    draft = document["s3_claims"]
    draft.update(enabled=True)
    mapping = draft["bitrix24"]
    mapping.update(type_id=25, entity_type_id=1234, category_id=6, responsible_id=1)
    mapping["stage_ids"] = {name: f"DT1234_6:{name}" for name in mapping["stage_ids"]}
    mapping["fields"] = {name: f"ufCrm25{name}" for name in mapping["fields"]}
    mapping["status_values"] = {name: idx for idx, name in enumerate(mapping["status_values"], 101)}
    folder = tmp_path / "profiles" / SLUG
    folder.mkdir(parents=True)
    (folder / "profile.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return folder.parent


def publish_fixture(
    journal: Journal,
    receipt: Receipt,
    stage: str,
    item_id: int,
    entity_id: int,
    external_field: str,
    fields: JSONObject,
    payload: JSONObject,
) -> None:
    request = WorkerRequestEnvelope.model_validate(
        {
            **receipt.model_dump(),
            "tenant_id": SLUG,
            "schema_version": "1",
            "stage": stage,
            "attempt": 1,
            "upstream_item": (None if stage == "S1" else {"item_id": 11, "entity_type_id": 1040}),
            "payload": {},
        }
    )
    telemetry = Telemetry(journal.db)
    telemetry.reserve(request)
    telemetry.complete(
        request,
        WorkerResponseEnvelope.model_validate(
            {
                **request.model_dump(exclude={"received_at", "upstream_item", "payload"}),
                "payload": payload,
                "need_human": False,
                "escalation_reasons": [],
                "mode": "deterministic",
                "cost_rub": "0.00000000",
                "fallback_reason": None,
            }
        ),
    )
    key = PublicationKey(
        tenant_id=SLUG, request_id=receipt.request_id, stage=request.stage, purpose="stage_result"
    )
    marker = journal.approve(
        PublicationCommand(
            key=key,
            stage_attempt=1,
            portal_id="crm.invalid",
            approval_ref="fixture",
            entity_type_id=entity_id,
            task_required=False,
            fields=fields,
            external_key_field=external_field,
        )
    )
    fields[external_field] = marker
    token = journal.claim(key)
    assert token is not None
    assert journal.begin_step(key, token, "item")
    assert journal.record_step(key, token, "item", item_id)
    assert journal.finish(key, token)


def seed(journal: Journal) -> tuple[Receipt, JSONObject]:
    receipt = journal.register(
        Delivery(
            tenant_id=SLUG,
            source="test",
            delivery_key="S1",
            profile_slug=SLUG,
            profile_revision="5",
            payload={},
            channel="test",
            is_after_hours=False,
            run_origin="mock",
        )
    )
    publish_fixture(
        journal,
        receipt,
        "S1",
        11,
        1040,
        "ufCrm7ExternalKey",
        {
            "stageId": "DT1040_15:SUCCESS",
            "ufCrm7Inn": INN,
            "ufCrm7Ogrn": "1027700132195",
            "ufCrm7KycStatus": 45,
            "ufCrm7ReliabilityScore": "90.00",
            "ufCrm7ApprovedCreditLimitRub": "5000.00",
        },
        {},
    )
    quote_id = str(uuid4())
    identity: JSONObject = {
        "quote_id": quote_id,
        "supplier_inn": INN,
        "supplier_name": "Поставщик",
        "s1_item_id": 11,
    }
    fields: JSONObject = {
        "id": 22,
        "stageId": "DT1044_17:SUCCESS",
        "ufCrm9TenderStatus": 49,
        "ufCrm9WinnerInn": INN,
        "ufCrm_9_WINNER_S1_ITEM_ID": 11,
        "ufCrm9TotalAmountRub": "1000.00",
    }
    publish_fixture(
        journal,
        receipt,
        "S2",
        22,
        1044,
        "ufCrm9ExternalKey",
        fields,
        {
            "winner_inn": INN,
            "winner_s1_item_id": 11,
            "winner_total_rub": "1000.00",
            "savings_rub": "0.00",
            "comparison_matrix_markdown": "Матрица",
            "participants": [
                {
                    **identity,
                    "rank": 1,
                    "score": "100.0000",
                    "total_rub": "1000.00",
                    "delivery_days": 5,
                    "deferment_days": 0,
                    "rejection_reasons": [],
                    "penalties": [],
                }
            ],
        },
    )
    items: list[JsonValue] = [
        {
            "line_no": 1,
            "item_code": "A",
            "name": "Bolt",
            "unit": "each",
            "quantity": "10.000",
            "unit_price_rub": "100.00",
            "vat_treatment": "NO_VAT",
            "vat_rate_pct": None,
            "vat_amount_rub": "0.00",
            "total_rub": "1000.00",
            "delivery_days": 5,
            "arithmetic_ok": True,
        }
    ]
    quote: JSONObject = {
        **identity,
        "quoted_total_rub": "1000.00",
        "deferment_days": 0,
        "arithmetic_ok": True,
        "need_human": False,
        "escalation_reasons": [],
        "items": items,
    }
    with journal.db.transaction() as conn:
        conn.execute(
            """INSERT INTO s2_extractions
               (tenant_id,request_id,quote_id,input_sha256,started_at,response_json,cost_rub_e8)
               VALUES (?,?,?,?,?,?,0)""",
            (
                SLUG,
                str(receipt.request_id),
                quote_id,
                "a" * 64,
                receipt.received_at.isoformat(),
                canonical_json(
                    {
                        "quote": quote,
                        "mode": "deterministic",
                        "fallback_reason": None,
                        "cost_rub": "0.00000000",
                        "prompt_version": None,
                    }
                ),
            ),
        )
        conn.execute(
            "INSERT INTO s2_award_items (tenant_id,request_id,attempt,items_json) VALUES (?,?,1,?)",
            (
                SLUG,
                str(receipt.request_id),
                json.dumps([item.model_dump(mode="json") for item in award().awarded_items]),
            ),
        )
    return receipt, fields


@pytest.mark.asyncio
async def test_s3_api_review_replacement_and_final(tmp_path: Path) -> None:
    profiles = active_profiles(tmp_path)
    s3 = load_profile(profiles, SLUG).s3_claims
    assert s3 is not None
    status_field = s3.bitrix24.fields.ACCEPTANCE_STATUS
    key_field = s3.bitrix24.fields.EXTERNAL_KEY
    assert status_field is not None and key_field is not None
    settings = Settings(
        sqlite_path=tmp_path / "test.db",
        profiles_path=profiles,
        worker_token=SecretStr("test"),
        base_url=AnyHttpUrl("https://llm.invalid/v1"),
        fast_llm_model="fast",
        max_cycle_cost_rub=Decimal("1.00"),
        usd_to_rub_rate=Decimal(100),
        llm_model_rates={
            "fast": ModelRate(input_usd_per_million=Decimal(1), output_usd_per_million=Decimal(1))
        },
    )
    db = Database(settings.sqlite_path)
    journal = Journal(db)
    item: JSONObject = {}
    comments: list[str] = []
    tender: JSONObject = {}
    portal_unavailable = False

    def crm(req: httpx.Request) -> httpx.Response:
        method = req.url.path.rsplit("/", 1)[-1].removesuffix(".json")
        body = json.loads(req.content)
        if method == "crm.item.get" and body["entityTypeId"] == 1044:
            if portal_unavailable:
                return httpx.Response(403, json={"error": "access_denied"})
            return httpx.Response(200, json={"result": {"item": tender}})
        if method == "crm.item.list":
            return httpx.Response(200, json={"result": {"items": [item] if item else []}})
        if method == "crm.item.add":
            assert not item
            item.update({"id": 33, **body["fields"]})
        elif method == "crm.item.update":
            assert body["id"] == 33
            item.update(body["fields"])
        elif method == "crm.timeline.comment.add":
            comments.append(body["fields"]["COMMENT"])
            return httpx.Response(200, json={"result": len(comments)})
        else:
            assert method == "crm.item.get" and body["id"] == 33
        return httpx.Response(200, json={"result": {"item": item}})

    async def instant(delay: float) -> None:
        pass

    publisher = Publisher(
        journal,
        BitrixClient(
            "https://crm.invalid/rest/1/token",
            PortalLimiter(2, sleep=instant),
            transport=httpx.MockTransport(crm),
        ),
        "crm.invalid",
    )
    started, release = asyncio.Event(), asyncio.Event()
    model_calls = 0

    async def ai(req: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        model_calls += 1
        request = json.loads(req.content)
        assert request["messages"][0]["role"] == "system"
        started.set()
        await release.wait()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": source().text}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 50},
            },
        )

    gateway = LLMGateway(settings, transport=httpx.MockTransport(ai))
    app = create_app(settings, publisher=publisher, gateway=gateway)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        receipt, upstream_card = seed(journal)
        tender.update(upstream_card)
        context = {"tenant_id": SLUG, **receipt.model_dump(mode="json", exclude={"received_at"})}
        primary = source()
        first = {
            **context,
            "delivery_key": "d1",
            "s2_item_id": 22,
            "documents": [{**primary.model_dump(mode="json"), "text": "УПД, текст для модели"}],
        }
        assert (await client.post("/s3/intake", json=first)).status_code == 401
        client.headers["X-Smartflow-Token"] = "test"

        async def post(path: str, body: dict[str, object], status: int = 200) -> httpx.Response:
            response = await client.post(path, json=body)
            assert response.status_code == status, response.text
            return response

        first_id = (await post("/s3/intake", first)).json()["package_id"]
        assert (await post("/s3/intake", first)).json()["package_id"] == first_id
        await post("/s3/intake", {**first, "documents": [source("9").model_dump(mode="json")]}, 409)
        call = {**context, "package_id": first_id}
        portal_unavailable = True
        await post("/s3/tender-gate", call, 503)
        portal_unavailable = False
        await post("/s3/tender-gate", {**call, "unexpected": True}, 422)
        good_gate = (await post("/s3/tender-gate", call)).json()
        assert good_gate["award"]["winner_inn"] == INN
        tender["ufCrm9TenderStatus"] = 51
        assert (await post("/s3/tender-gate", call)).json()["escalation_reasons"] == [
            "S3_TENDER_NOT_AWARDED",
        ]
        tender["ufCrm9TenderStatus"] = 49
        tender["ufCrm_9_WINNER_S1_ITEM_ID"] = 12
        assert (await post("/s3/tender-gate", call)).json()["escalation_reasons"] == [
            "S3_WINNER_MISMATCH",
        ]
        tender["ufCrm_9_WINNER_S1_ITEM_ID"] = 11
        await post("/s3/tender-gate", call)
        parse = {**context, "payload": {"package_id": first_id}}
        task = asyncio.create_task(post("/s3/parse", parse))
        await asyncio.wait_for(started.wait(), 5)
        assert (await post("/s3/parse", parse, 202)).headers["Retry-After"] == "1"
        release.set()
        original = (await task).json()
        assert original["mode"] == "llm" and model_calls == 1
        assert (await post("/s3/parse", parse)).json() == original
        assert model_calls == 1

        async def evaluate_package(package_id: str, attempt: int, route: str) -> dict[str, object]:
            gate = (await post("/s3/tender-gate", {**context, "package_id": package_id})).json()
            documents = (
                await post(
                    "/s3/parse",
                    {
                        **context,
                        "payload": {"package_id": package_id},
                    },
                )
            ).json()["payload"]["documents"]
            envelope = {
                **context,
                "received_at": receipt.received_at.isoformat(),
                "stage": "S3",
                "attempt": attempt,
                "schema_version": "1",
                "upstream_item": {"entity_type_id": 1044, "item_id": 22},
                "payload": {
                    "package_id": package_id,
                    "award": gate["award"],
                    "documents": documents,
                },
            }
            result = (await post(route, envelope)).json()
            assert result == (await post("/s3/evaluate", envelope)).json()
            return envelope

        first_envelope = await evaluate_package(first_id, 1, "/v1/workers/S3/run")
        await post(
            "/s3/evaluate",
            {
                **first_envelope,
                "payload": {
                    "package_id": first_id,
                    "award": None,
                    "documents": [],
                },
            },
            409,
        )
        review_1 = (await post("/s3/publication/prepare", {**call, "attempt": 1})).json()
        assert review_1["review_attempt"] == 1
        assert (await post("/v1/publications/dispatch", review_1)).json()["status"] == "published"
        stable_key = item[key_field]

        scan = primary.model_copy(
            update={
                "source_type": "PDF",
                "pdf_base64": base64.b64encode(b"invalid PDF").decode(),
                "text": None,
            }
        )
        scan_id = (
            await post(
                "/s3/intake",
                {
                    **first,
                    "delivery_key": "scan",
                    "documents": [scan.model_dump(mode="json")],
                },
            )
        ).json()["package_id"]
        assert (
            await post(
                "/s3/parse",
                {
                    **context,
                    "payload": {"package_id": scan_id},
                },
            )
        ).json()["payload"]["escalation_reasons"] == ["S3_SOURCE_UNREADABLE"]

        second = {
            **first,
            "delivery_key": "d2",
            "documents": [source("9", days=7).model_dump(mode="json")],
        }
        second_id = (await post("/s3/intake", second)).json()["package_id"]
        await post("/s3/parse", parse, 409)
        await post("/s3/tender-gate", {**context, "package_id": second_id})
        worker = S3Worker(profiles, gateway, db)
        worker.record_penalty_terms(
            S3TenderGateRequest.model_validate({**context, "package_id": second_id}),
            evidence_ref="contract:verified:fixture",
            daily_penalty_pct=Decimal("0.10"),
            max_penalty_pct=Decimal("10.00"),
        )
        envelope = await evaluate_package(second_id, 2, "/s3/evaluate")
        assert (await post("/s3/evaluate", envelope)).json()["payload"][
            "claim_amount_rub"
        ] == "102.00"
        call2 = {**context, "package_id": second_id, "attempt": 2}
        await post("/s3/publication/decision", {**call, "attempt": 1}, 409)
        review_2 = (await post("/s3/publication/prepare", call2)).json()
        assert review_2["review_attempt"] == 2
        assert (await post("/v1/publications/dispatch", review_2)).json()["item_id"] == 33
        assert item[key_field] == stable_key and second_id in str(item["title"])
        event_context = {key: value for key, value in context.items() if key != "request_id"}
        resolved = await post("/s3/publication/resolve", {**event_context, "item_id": 33})
        assert resolved.json() == call2
        await post("/s3/publication/resolve", {**event_context, "item_id": 999}, 409)
        await post("/s3/publication/decision", call2, 409)
        item.update(
            {
                status_field: s3.bitrix24.status_values.CLAIM_REQUIRED,
                "stageId": s3.bitrix24.stage_ids.claim_required,
                "updatedBy": 1,
            }
        )
        await post("/s3/publication/decision", call2, 204)
        await post("/s3/publication/decision", call2, 204)
        final = (await post("/s3/publication/prepare", call2)).json()
        assert final["purpose"] == "stage_result" and "review_attempt" not in final
        await post("/s3/intake", {**second, "delivery_key": "d3"}, 409)
        assert (await post("/v1/publications/dispatch", final)).json()["item_id"] == 33
        item[status_field] = s3.bitrix24.status_values.CLOSED
        assert (await post("/v1/publications/dispatch", final)).json()["status"] == "published"
        assert len(comments) == 3 and len({c.rsplit("[smartflow:", 1)[1] for c in comments}) == 3
        assert (
            item[key_field] == stable_key and item[status_field] == s3.bitrix24.status_values.CLOSED
        )
        with db.connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM s3_packages").fetchone()[0] == 3
            billed = conn.execute("SELECT cost_rub_e8 FROM transaction_telemetry").fetchone()[0]
            actual_spend = conn.execute("SELECT SUM(cost_rub_e8) FROM s3_extractions").fetchone()[0]
            assert billed == actual_spend > 0
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    await gateway.aclose()
