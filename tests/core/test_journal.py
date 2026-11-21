import sqlite3
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from threading import Barrier
from uuid import uuid4

import pytest

from core.contracts import ItemRef, WorkerRequestEnvelope, WorkerResponseEnvelope
from core.database import Database
from core.idempotency import (
    Conflict,
    Delivery,
    Journal,
    PublicationCommand,
    PublicationKey,
    TaskCommand,
)
from core.telemetry import Telemetry


def test_database_and_delivery(db: Database, delivery: Delivery) -> None:
    db.initialize()
    journal = Journal(db)
    original = journal.register(delivery)
    assert journal.register(delivery.model_copy(update={"profile_revision": "v2"})) == original
    with pytest.raises(Conflict):
        journal.register(delivery.model_copy(update={"payload": {"text": "changed"}}))
    assert journal.register(delivery.model_copy(update={"tenant_id": "other"})) != original
    with db.connect() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("SELECT count(*) FROM transaction_telemetry").fetchone()[0] == 2
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM idempotency_records")


def test_attempt_replay_and_cycle_rollup(
    db: Database,
    envelope: WorkerRequestEnvelope,
    result: WorkerResponseEnvelope,
) -> None:
    telemetry = Telemetry(db)
    assert telemetry.reserve(envelope).status == "reserved"
    assert telemetry.reserve(envelope).status == "running"
    with pytest.raises(Conflict):
        telemetry.reserve(envelope.model_copy(update={"payload": {"changed": True}}))
    with pytest.raises(Conflict):
        telemetry.complete(envelope, result.model_copy(update={"request_id": uuid4()}))
    assert telemetry.fail(
        envelope,
        "interrupted",
        interrupted=True,
        tokens_in=3,
        tokens_out=0,
        cost_rub=Decimal(".00000001"),
    )
    with pytest.raises(Conflict):
        telemetry.complete(envelope, result)
    second = envelope.model_copy(update={"attempt": 2})
    response = WorkerResponseEnvelope.model_validate(
        result.model_dump()
        | {
            "attempt": 2,
            "cost_rub": Decimal(".000000015"),
            "mode": "fallback",
            "fallback_reason": "budget_exceeded",
            "need_human": True,
            "escalation_reasons": ["unconfirmed"],
            "payload": None,
        }
    )
    assert telemetry.reserve(second).status == "reserved"
    telemetry.complete(second, response, tokens_in=1, tokens_out=2, prompt_versions=["v1"])
    assert telemetry.reserve(second).response == response
    assert telemetry.remaining_budget(envelope.tenant_id, envelope.request_id, Decimal(1)) == (
        Decimal(".99999997")
    )
    assert telemetry.record_cycle(
        envelope.tenant_id,
        envelope.request_id,
        outcome="waiting_human",
        processing_time_sec=3,
        saved_minutes=-2,
        ab_variant="b",
    )
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM transaction_telemetry").fetchone()
        assert (row["tokens_in"], row["tokens_out"], row["cost_rub_e8"]) == (4, 2, 3)
        assert (row["mode"], row["fallback_reason"]) == ("fallback", "budget_exceeded")
        assert (row["need_human"], row["saved_minutes"], row["run_origin"]) == (1, -2, "mock")


def test_atomic_claim_and_recovery(
    db: Database,
    envelope: WorkerRequestEnvelope,
    result: WorkerResponseEnvelope,
) -> None:
    telemetry, journal = Telemetry(db), Journal(db)
    telemetry.reserve(envelope)
    telemetry.complete(envelope, result)
    key = PublicationKey(
        tenant_id=envelope.tenant_id,
        request_id=envelope.request_id,
        stage="S1",
        purpose="stage_result",
    )
    command = PublicationCommand(
        key=key,
        stage_attempt=1,
        portal_id="portal",
        approval_ref="policy-v1",
        entity_type_id=177,
        task_required=True,
        fields={},
        external_key_field="ufExternal",
        task=TaskCommand(title="review", responsible_id=7),
    )
    assert journal.approve(command) == journal.approve(command)
    with pytest.raises(Conflict):
        journal.approve(command.model_copy(update={"fields": {"changed": True}}))
    gate = Barrier(8)

    def claim(_: int) -> str | None:
        gate.wait()
        return Journal(db).claim(key)

    with ThreadPoolExecutor(max_workers=8) as pool:
        tokens = [token for token in pool.map(claim, range(8)) if token is not None]
    assert len(tokens) == 1
    token = tokens[0]
    assert not journal.begin_step(key, token, "task")
    assert not journal.begin_step(key, "wrong", "item")
    assert journal.begin_step(key, token, "item")
    assert not journal.begin_step(key, token, "item")
    assert journal.reconcile(key, token)
    assert journal.claim(key) is None
    assert not journal.resolve(key, token, "audit-1", previous_stopped=True)
    with pytest.raises(ValueError):
        journal.confirm_reconciled_step(key, token, "item", 29, previous_stopped=False)
    assert journal.confirm_reconciled_step(key, token, "item", 29, previous_stopped=True)
    assert journal.resolve(key, token, "audit-1", previous_stopped=True)
    recovered = journal.claim(key)
    assert recovered is not None and recovered != token
    assert not journal.finish(key, recovered)
    assert journal.begin_step(key, recovered, "task")
    assert journal.record_step(key, recovered, "task", 42)
    assert journal.finish(key, recovered)
    assert journal.claim(key) is None
    downstream = envelope.model_copy(
        update={
            "stage": "S2",
            "upstream_item": ItemRef(entity_type_id=177, item_id=30),
        }
    )
    with pytest.raises(Conflict):
        telemetry.reserve(downstream)
    assert (
        telemetry.reserve(
            downstream.model_copy(
                update={
                    "upstream_item": ItemRef(entity_type_id=177, item_id=29),
                }
            )
        ).status
        == "reserved"
    )
