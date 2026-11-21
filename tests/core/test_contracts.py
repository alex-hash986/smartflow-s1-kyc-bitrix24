from datetime import UTC
from decimal import Decimal

import pytest
from pydantic import ValidationError

from core.contracts import WorkerRequestEnvelope, WorkerResponseEnvelope, cost_atoms, round_money


@pytest.mark.parametrize(
    "change",
    [
        {"unknown": True},
        {"attempt": "1"},
        {"attempt": True},
        {"received_at": 123},
        {"received_at": "2026-11-20T12:00:00"},
        {"upstream_item": {"entity_type_id": 1, "item_id": 2}},
        {"stage": "S2"},
        {"tenant_id": " "},
        {"payload": []},
        {"stage": "S2", "upstream_item": {"entity_type_id": 1, "item_id": 2, "unknown": 3}},
    ],
)
def test_request_rejects_invalid_wire_format(
    envelope: WorkerRequestEnvelope,
    change: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        WorkerRequestEnvelope.model_validate(envelope.model_dump(mode="json") | change)


def test_request_strips_and_normalizes(envelope: WorkerRequestEnvelope) -> None:
    obj = WorkerRequestEnvelope.model_validate(
        envelope.model_dump(mode="json")
        | {
            "received_at": "2026-11-20T12:00:00+07:00",
            "tenant_id": " tenant ",
        }
    )
    assert obj.received_at.hour == 5 and obj.received_at.tzinfo == UTC
    assert obj.tenant_id == "tenant"


@pytest.mark.parametrize(
    "change",
    [
        {"cost_rub": 1.0},
        {"cost_rub": "1.00"},
        {"cost_rub": "92233720368.54775808"},
        {"need_human": True},
        {"need_human": "false"},
        {"payload": None},
        {"mode": "fallback"},
        {"fallback_reason": "invalid"},
        {"extra": 1},
    ],
)
def test_invalid_response(result: WorkerResponseEnvelope, change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WorkerResponseEnvelope.model_validate(result.model_dump(mode="json") | change)


def test_half_up_and_exact_storage(result: WorkerResponseEnvelope) -> None:
    assert round_money(Decimal("1.005")) == Decimal("1.01")
    assert round_money(Decimal("-1.005")) == Decimal("-1.01")
    assert cost_atoms(Decimal("0.000000015")) == 2
    assert cost_atoms(None) is None
    assert cost_atoms(Decimal("92233720368.54775807")) == 2**63 - 1
    response = WorkerResponseEnvelope.model_validate(
        result.model_dump() | {"cost_rub": Decimal("0.000000015")}
    )
    assert response.cost_rub == Decimal("0.00000002")
    assert response.model_dump(mode="json")["cost_rub"] == "0.00000002"
