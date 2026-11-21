import base64
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from uuid import UUID

import httpx
import pytest
import yaml
from pydantic import SecretStr, ValidationError
from pypdf import PdfWriter
from pypdf.generic import ContentStream, DecodedStreamObject, DictionaryObject, NameObject

from core.api import create_app
from core.b24_client import BitrixClient, PortalLimiter
from core.config import ModelRate, Settings
from core.database import Database
from core.idempotency import Journal, PublicationKey
from core.llm_gateway import CycleBudget, LLMGateway
from core.profile_loader import Profile, S1Profile, load_profile
from core.publisher import Publisher
from core.telemetry import Telemetry
from modules.s1_kyc.contracts import S1EvaluateRequest, S1ParseRequest, S1ParseResponse
from modules.s1_kyc.parser import RequisitesParser
from modules.s1_kyc.requisites import valid_inn, valid_kpp, valid_ogrn, valid_ogrnip
from modules.s1_kyc.scoring import evaluate

PROFILES = Path(__file__).resolve().parents[3] / "profiles"
ENTITY = {
    "inn": "7707083893",
    "ogrn": "1027700132195",
    "kpp": "773601001",
    "legal_name": "Company",
    "requested_limit_rub": "750000.00",
}
IP = {
    **ENTITY,
    "inn": "500100732259",
    "ogrn": "304500116000157",
    "kpp": None,
    "legal_name": "Individual",
}
TEXT = "INN: 7707083893; OGRN: 1027700132195; KPP: 773601001\nLegal name: Company\nRequested limit: 750 000.00 RUB"
IP_TEXT = (
    "INN: 500100732259; OGRNIP: 304500116000157\nLegal name: Individual\nRequested limit: 750000"
)
NOW = datetime(2026, 11, 20, 12, tzinfo=UTC)


def profile(slug: str = "default_b2b") -> S1Profile:
    loaded = load_profile(PROFILES, slug)
    assert loaded.s1_kyc is not None
    return loaded.s1_kyc


def data(ip: bool = False, **changes: object) -> S1EvaluateRequest:
    parsed = S1ParseResponse.model_validate(
        {
            **(IP if ip else ENTITY),
            "need_human": False,
            "escalation_reasons": [],
        }
    )
    return S1EvaluateRequest.model_validate(
        {
            "parsed": parsed,
            "registry": {
                "inn": parsed.inn,
                "ogrn": parsed.ogrn,
                "legal_entity_status": "ACTIVE",
                "registration_age_months": 36,
                "charter_capital_rub": None if ip else "10000.00",
                "tax_debt_rub": "0.00",
                "defendant_claims_rub": "0.00",
                "fns_address_unreliable": False,
                "bankruptcy": False,
                "observed_at": NOW,
                **changes,
            },
        }
    )


def settings(tmp_path: Path) -> Settings:
    return Settings(
        sqlite_path=tmp_path / "test.db",
        profiles_path=PROFILES,
        worker_token=SecretStr("test"),
        fast_llm_model="fast",
        llm_model_rates={
            "fast": ModelRate(
                input_usd_per_million=Decimal(1),
                output_usd_per_million=Decimal(1),
            )
        },
    )


def pdf(text: str = "") -> str:
    writer = PdfWriter()
    page = writer.add_blank_page(600, 800)
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
        lines = "".join(f"({line}) Tj T*" for line in text.splitlines())
        stream.set_data(("BT /F1 12 Tf 40 700 Td 14 TL " + lines + " ET").encode())
        page.replace_contents(ContentStream(stream, writer))
    output = BytesIO()
    writer.write(output)
    return base64.b64encode(output.getvalue()).decode()


def model_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": json.dumps(ENTITY)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
    )


def test_checksums() -> None:
    assert valid_inn("7707083893") and not valid_inn("7707083894")
    assert valid_inn("500100732259")
    assert not valid_inn("500100732249") and not valid_inn("500100732250")
    assert valid_ogrn("1027700132195") and not valid_ogrn("1027700132196")
    assert valid_ogrnip("304500116000157") and not valid_ogrnip("304500116000158")
    assert valid_kpp("773601001") and not valid_kpp("77360100A")
    assert not valid_inn("７707083893")
    with pytest.raises(ValidationError):
        data(tax_debt_rub=0)
    with pytest.raises(ValidationError):
        data(ip=True, charter_capital_rub="0")
    with pytest.raises(ValidationError):
        data(ogrn=IP["ogrn"])


@pytest.mark.parametrize("source", ["text", "pdf", "ip"])
@pytest.mark.asyncio
async def test_deterministic_parse(tmp_path: Path, source: str) -> None:
    def unexpected(_: httpx.Request) -> httpx.Response:
        pytest.fail("A complete source must not call the model")

    gateway = LLMGateway(settings(tmp_path), transport=httpx.MockTransport(unexpected))
    try:
        request = (
            S1ParseRequest(source_type="PDF", text=None, pdf_base64=pdf(TEXT))
            if source == "pdf"
            else S1ParseRequest(
                source_type="TEXT", text=IP_TEXT if source == "ip" else TEXT, pdf_base64=None
            )
        )
        result = await RequisitesParser(gateway).parse(
            request,
            profile().extraction,
            CycleBudget(Decimal(0)),
        )
    finally:
        await gateway.aclose()
    assert not result.payload.need_human
    assert result.mode == "deterministic"
    assert result.payload.model_dump(
        mode="json",
        exclude={
            "need_human",
            "escalation_reasons",
        },
    ) == (IP if source == "ip" else ENTITY)


@pytest.mark.asyncio
async def test_llm_budget_and_empty_scan(tmp_path: Path) -> None:
    calls = []

    def transport(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert json.loads(request.content)["model"] == "fast"
        return model_response()

    gateway = LLMGateway(settings(tmp_path), transport=httpx.MockTransport(transport))
    request = S1ParseRequest(
        source_type="TEXT", text=TEXT.replace("Legal name:", "Applicant"), pdf_base64=None
    )
    try:
        parser = RequisitesParser(gateway)
        result = await parser.parse(request, profile().extraction, CycleBudget(Decimal(1)))
        assert result.mode == "llm" and not result.payload.need_human
        assert result.payload.legal_name == "Company"
        assert result.prompt_version == "1/baseline"
        budget = await parser.parse(request, profile().extraction, CycleBudget(Decimal(0)))
        assert budget.payload.need_human and budget.fallback_reason == "budget_exceeded"
        assert "PARSE_BUDGET_EXCEEDED" in budget.payload.escalation_reasons
        scan = await parser.parse(
            S1ParseRequest(source_type="PDF", text=None, pdf_base64=pdf()),
            profile().extraction,
            CycleBudget(Decimal(1)),
        )
        assert scan.payload.need_human
        assert "PARSE_SOURCE_UNREADABLE" in scan.payload.escalation_reasons
        ambiguous = await parser.parse(
            S1ParseRequest(source_type="TEXT", text=TEXT + "\nINN: 500100732259", pdf_base64=None),
            profile().extraction,
            CycleBudget(Decimal(1)),
        )
        assert ambiguous.payload.inn is None
        assert "PARSE_AMBIGUOUS_FIELD" in ambiguous.payload.escalation_reasons
        assert len(calls) == 1
    finally:
        await gateway.aclose()


@pytest.mark.parametrize(
    ("slug", "cap"),
    [
        ("default_b2b", Decimal("750000.00")),
        ("construction_supply", Decimal("500000.00")),
    ],
)
def test_green(slug: str, cap: Decimal) -> None:
    result = evaluate(data(ip=True), profile(slug), now=NOW)
    assert result.reliability_score == Decimal("100.00")
    assert result.risk_level == "LOW" and not result.need_human
    assert result.recommended_credit_limit_rub == cap
    assert result.payment_terms.deferment_days == 30


def test_penalties() -> None:
    request = data(registration_age_months=10, defendant_claims_rub="600000.00")
    base = evaluate(request, profile(), now=NOW)
    strict = evaluate(request, profile("construction_supply"), now=NOW)
    assert (base.reliability_score, base.risk_level) == (Decimal("65.00"), "MEDIUM")
    assert base.payment_terms.deferment_days == 14
    assert base.recommended_credit_limit_rub == Decimal("300000.00")
    assert (strict.reliability_score, strict.risk_level) == (Decimal("30.00"), "CRITICAL")
    assert strict.need_human and not strict.recommended_credit_limit_rub


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("bankruptcy", "BANKRUPTCY"),
        ("fns_address_unreliable", "FNS_ADDRESS_UNRELIABLE"),
    ],
)
def test_stop_factors(field: str, code: str) -> None:
    result = evaluate(data(**{field: True}), profile(), now=NOW)
    assert result.risk_level == "CRITICAL" and result.need_human
    assert result.stop_factors == [code] and code in result.dossier_markdown
    assert result.recommended_credit_limit_rub == 0


def test_freshness_and_completeness() -> None:
    stale = evaluate(
        data(observed_at=NOW - timedelta(hours=25), bankruptcy=True),
        profile(),
        now=NOW,
    )
    assert stale.fallback_reason == "REGISTRY_STALE" and stale.stop_factors == []
    incomplete = evaluate(data(tax_debt_rub=None), profile(), now=NOW)
    assert incomplete.reliability_score is None and incomplete.need_human
    assert incomplete.fallback_reason == "REGISTRY_INCOMPLETE"
    with pytest.raises(ValueError, match="future"):
        evaluate(data(), profile(), now=NOW - timedelta(seconds=1))


@pytest.mark.asyncio
async def test_http(tmp_path: Path) -> None:
    app = create_app(settings(tmp_path))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app),
            base_url="http://test",
            headers={"X-Smartflow-Token": "test"},
        ) as client,
    ):
        response = await client.post(
            "/v1/deliveries",
            json={
                "tenant_id": "test",
                "source": "test",
                "delivery_key": "s1",
                "profile_slug": "default_b2b",
                "profile_revision": "2",
                "channel": "test",
                "is_after_hours": False,
                "run_origin": "mock",
                "payload": {"text": TEXT},
            },
        )
        assert response.status_code == 200
        receipt = response.json()
        context = {**receipt, "tenant_id": "test"}
        parse_body = {key: value for key, value in context.items() if key != "received_at"}
        parse_body["payload"] = {"source_type": "TEXT", "text": TEXT, "pdf_base64": None}
        assert (
            await client.post(
                "/s1/parse",
                json=parse_body,
                headers={
                    "X-Smartflow-Token": "wrong",
                },
            )
        ).status_code == 401
        parsed = await client.post("/s1/parse", json=parse_body)
        assert parsed.status_code == 200 and not parsed.json()["payload"]["need_human"]
        assert parsed.json()["request_id"] == receipt["request_id"]
        assert (await client.post("/s1/parse", json=parse_body)).json() == parsed.json()
        changed = {**parse_body, "payload": {**parse_body["payload"], "text": IP_TEXT}}
        assert (await client.post("/s1/parse", json=changed)).status_code == 409
        request = data(observed_at=datetime.now(UTC)).model_dump(mode="json")
        request["parsed"] = parsed.json()["payload"]
        body = {**context, "attempt": 1, "payload": request}
        result = await client.post("/s1/evaluate", json=body)
        assert result.status_code == 200, result.text
        assert result.json()["payload"]["risk_level"] == "LOW"
        assert result.json()["cost_rub"] == "0.00000000"
        assert (await client.post("/s1/evaluate", json=body)).json() == result.json()


@pytest.mark.asyncio
async def test_paid_parse_replays_without_spending_twice(tmp_path: Path) -> None:
    calls = 0

    def transport(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return model_response()

    config = settings(tmp_path)
    gateway = LLMGateway(config, transport=httpx.MockTransport(transport))
    app = create_app(config, gateway=gateway)
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app),
                base_url="http://test",
                headers={"X-Smartflow-Token": "test"},
            ) as client,
        ):
            receipt = await client.post(
                "/v1/deliveries",
                json={
                    "tenant_id": "test",
                    "source": "test",
                    "delivery_key": "s1-paid",
                    "profile_slug": "default_b2b",
                    "profile_revision": "2",
                    "channel": "test",
                    "is_after_hours": False,
                    "run_origin": "mock",
                    "payload": {},
                },
            )
            assert receipt.status_code == 200
            body = receipt.json()
            body.pop("received_at")
            body.update(
                tenant_id="test",
                payload={
                    "source_type": "TEXT",
                    "text": TEXT.replace("Legal name:", "Applicant"),
                    "pdf_base64": None,
                },
            )
            first = await client.post("/s1/parse", json=body)
            assert first.status_code == 200 and first.json()["cost_rub"] == "0.01500000"
            second = await client.post("/s1/parse", json=body)
            assert second.json() == first.json() and calls == 1
            request_id = UUID(body["request_id"])
            remaining = Telemetry(Database(config.sqlite_path)).remaining_budget(
                "test",
                request_id,
                config.max_cycle_cost_rub,
            )
            assert remaining == config.max_cycle_cost_rub - Decimal("0.015")
    finally:
        await gateway.aclose()


@pytest.mark.parametrize("issue", ["bindings", "score", "stop", "mapping"])
def test_rejects_invalid_profile(issue: str) -> None:
    raw = yaml.safe_load((PROFILES / "default_b2b" / "profile.yaml").read_text(encoding="utf-8"))
    s1 = raw["s1_kyc"]
    if issue == "bindings":
        raw["template"] = False
    elif issue == "score":
        s1["scoring"]["risk_bands"]["LOW"]["min_score"] = "101.00"
    elif issue == "stop":
        s1["scoring"]["stop_factors"]["BANKRUPTCY"]["value"] = "true"
    else:
        del s1["bitrix24"]["fields"]["OGRN"]
    with pytest.raises(ValidationError):
        Profile.model_validate(raw)


def test_live_profile_bindings() -> None:
    bound = load_profile(PROFILES, "portal_b24_b2b")
    archived = load_profile(PROFILES, bound.slug, "3")
    prior = load_profile(PROFILES, bound.slug, "4")
    assert not bound.template and bound.revision == "5"
    assert prior.s1_kyc == bound.s1_kyc and prior.stages == bound.stages
    assert prior.s3_claims is not None and not prior.s3_claims.enabled
    assert bound.s3_claims is not None and bound.s3_claims.enabled
    assert archived.revision == "3" and archived.s1_kyc == bound.s1_kyc
    stage = bound.stages["S1"]
    assert stage.stage_ids is not None and stage.stage_ids.initial == stage.stage_id
    assert stage.entity_type_id == 1040 and stage.category_id == 15
    assert bound.s1_kyc is not None
    assert bound.s1_kyc.bitrix24.kyc_status_values == {"APPROVED": 45, "REJECTED": 47}
    assert all((value or "").startswith("ufCrm") for value in bound.s1_kyc.bitrix24.fields.values())
    assert stage.external_key_field == "ufCrm7ExternalKey"
    assert not bound.s1_kyc.bitrix24.tasks_enabled


@pytest.mark.parametrize(
    "route", ["approved", "registry_review", "parse_review", "with_task", "uncertain"]
)
@pytest.mark.asyncio
async def test_s1_publication_flow(tmp_path: Path, route: str) -> None:
    bound = load_profile(PROFILES, "portal_b24_b2b")
    assert bound.s1_kyc is not None and bound.portal_id is not None
    stage = bound.stages["S1"]
    assert stage.stage_ids is not None
    bound.s1_kyc.bitrix24.tasks_enabled = route == "with_task"
    folder = tmp_path / bound.slug
    folder.mkdir()
    (folder / "profile.yaml").write_text(
        yaml.safe_dump(bound.model_dump(mode="json")),
        encoding="utf-8",
    )
    calls: list[str] = []
    item: dict[str, object] = {}
    task: dict[str, object] = {}

    async def no_delay(_: float) -> None:
        return None

    def bitrix(request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        calls.append(method)
        payload = json.loads(request.content)
        match method:
            case "crm.item.list":
                result: object = {"items": []}
            case "crm.item.add":
                item.update(payload["fields"], id=29)
                result = {"item": item}
            case "crm.item.get":
                result = {"item": item}
            case "crm.timeline.comment.add":
                assert payload["fields"]["ENTITY_ID"] == 29
                assert "[smartflow:" in payload["fields"]["COMMENT"]
                if route == "uncertain":
                    return httpx.Response(503)
                result = 30
            case "tasks.task.add":
                assert payload["fields"]["UF_CRM_TASK"] == ["T410_29"]
                task.update(
                    ufCrmTask=payload["fields"]["UF_CRM_TASK"],
                    description=payload["fields"]["DESCRIPTION"],
                    id=31,
                )
                result = {"task": task}
            case "tasks.task.get":
                result = {"task": task}
            case _:
                pytest.fail(f"Unexpected Bitrix method: {method}")
        return httpx.Response(200, json={"result": result})

    config = settings(tmp_path).model_copy(update={"profiles_path": tmp_path})
    portal = BitrixClient(
        f"https://{bound.portal_id}/rest/test/",
        PortalLimiter(sleep=no_delay),
        transport=httpx.MockTransport(bitrix),
    )
    publisher = Publisher(Journal(Database(config.sqlite_path)), portal, bound.portal_id)
    app = create_app(config, publisher=publisher)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app),
            base_url="http://test",
            headers={"X-Smartflow-Token": "test"},
        ) as client,
    ):
        registered = await client.post(
            "/v1/deliveries",
            json={
                "tenant_id": bound.tenant_id,
                "source": "test",
                "delivery_key": "s1",
                "profile_slug": bound.slug,
                "profile_revision": bound.revision,
                "channel": "test",
                "is_after_hours": False,
                "run_origin": "mock",
                "payload": {},
            },
        )
        assert registered.status_code == 200
        receipt = registered.json()
        context = {**receipt, "tenant_id": bound.tenant_id}
        received_at = context.pop("received_at")
        extraction_review = route in {"parse_review", "with_task"}
        extracted = await client.post(
            "/s1/parse",
            json={
                **context,
                "payload": {
                    "source_type": "TEXT",
                    "text": "" if extraction_review else TEXT,
                    "pdf_base64": None,
                },
            },
        )
        assert extracted.status_code == 200
        if not extraction_review:
            result = data(
                observed_at=datetime.now(UTC),
                bankruptcy=(route == "registry_review"),
            ).model_dump(mode="json")
            result["parsed"] = extracted.json()["payload"]
            evaluated = await client.post(
                "/s1/evaluate",
                json={
                    **context,
                    "attempt": 1,
                    "received_at": received_at,
                    "payload": result,
                },
            )
            assert evaluated.status_code == 200
        prepare_body = {**context, "attempt": 1}
        prepared = await client.post("/s1/publication/prepare", json=prepare_body)
        assert prepared.status_code == 200, prepared.text
        key = prepared.json()
        assert key["purpose"] == (
            "stage_result" if route in {"approved", "uncertain"} else "human_review"
        )
        published = await client.post("/v1/publications/dispatch", json=key)
        if route == "uncertain":
            assert published.status_code == 502
            replay = await client.post("/v1/publications/dispatch", json=key)
            assert replay.json()["status"] == "reconcile"
            assert calls.count("crm.timeline.comment.add") == 1
            journal = Journal(Database(config.sqlite_path))
            with journal.db.connect() as conn:
                token = conn.execute("SELECT owner_token FROM crm_publications").fetchone()[0]
            publication_key = PublicationKey.model_validate(key)
            assert not journal.resolve(publication_key, token, "test", previous_stopped=True)
            assert journal.confirm_reconciled_step(
                publication_key,
                token,
                "comment",
                30,
                previous_stopped=True,
            )
            assert journal.resolve(publication_key, token, "test", previous_stopped=True)
            assert (await client.post("/v1/publications/dispatch", json=key)).json()[
                "status"
            ] == "published"
            assert calls.count("crm.timeline.comment.add") == 1
            return
        assert published.status_code == 200, published.text
        assert published.json() == {
            "item_id": 29,
            "task_id": 31 if route == "with_task" else None,
            "comment_id": 30,
            "status": "published",
        }
        assert item["stageId"] == (
            stage.stage_ids.approved if route == "approved" else stage.stage_ids.review
        )
        assert (await client.post("/s1/publication/prepare", json=prepare_body)).json() == key
        assert (await client.post("/v1/publications/dispatch", json=key)).json() == published.json()
        assert calls.count("crm.item.add") == calls.count("crm.timeline.comment.add") == 1
        assert calls.count("tasks.task.add") == (route == "with_task")


def test_database_upgrades_the_legacy_publication_constraint(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    db = Database(path)
    root = PROFILES.parent
    schema = (
        (root / "core" / "schema.sql")
        .read_text(encoding="utf-8")
        .split("CREATE TABLE IF NOT EXISTS crm_publication_comments")[0]
    )
    old_schema = schema.replace(
        "    CHECK (status <> 'publishing'",
        "    CHECK (purpose <> 'human_review' OR task_required = 1),\n"
        "    CHECK (status <> 'publishing'",
    )
    with db.connect() as conn:
        conn.executescript(old_schema)
    db.initialize()
    db.initialize()
    with db.connect() as conn:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='crm_publications'"
        ).fetchone()[0]
        assert "purpose <> 'human_review'" not in sql
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
