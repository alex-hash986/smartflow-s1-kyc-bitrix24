from pathlib import Path

import pytest

from core.contracts import WorkerRequestEnvelope, WorkerResponseEnvelope
from core.database import Database
from core.idempotency import Delivery, Journal


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "kernel.sqlite")
    database.initialize()
    return database


@pytest.fixture
def delivery() -> Delivery:
    return Delivery(
        tenant_id="tenant",
        source="source",
        delivery_key="delivery",
        profile_slug="demo",
        profile_revision="v1",
        channel="crm",
        is_after_hours=False,
        run_origin="mock",
        payload={"text": "hi"},
    )


@pytest.fixture
def envelope(db: Database, delivery: Delivery) -> WorkerRequestEnvelope:
    receipt = Journal(db).register(delivery)
    return WorkerRequestEnvelope(
        schema_version="1",
        tenant_id=delivery.tenant_id,
        request_id=receipt.request_id,
        profile_slug=receipt.profile_slug,
        profile_revision=receipt.profile_revision,
        received_at=receipt.received_at,
        stage="S1",
        attempt=1,
        upstream_item=None,
        payload=delivery.payload,
    )


@pytest.fixture
def result(envelope: WorkerRequestEnvelope) -> WorkerResponseEnvelope:
    return WorkerResponseEnvelope.model_validate(
        {
            **envelope.model_dump(exclude={"received_at", "upstream_item", "payload"}),
            "payload": {"answer": True},
            "need_human": False,
            "escalation_reasons": [],
            "cost_rub": "0.00000000",
            "mode": "deterministic",
            "fallback_reason": None,
        }
    )
