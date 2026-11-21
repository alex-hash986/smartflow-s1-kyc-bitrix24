import json
from pathlib import Path
from typing import Any

from core.profile_loader import load_profile

ROOT = Path(__file__).resolve().parents[3]


def test_s2_workflow_has_authenticated_complete_paths() -> None:
    workflow = json.loads(
        (ROOT / "workflows/s2_tender_evaluation.json").read_text(encoding="utf-8")
    )
    assert workflow["active"] is False
    assert workflow["settings"]["saveDataSuccessExecution"] == "none"
    assert workflow["settings"]["saveDataErrorExecution"] == "none"
    nodes: dict[str, dict[str, Any]] = {node["name"]: node for node in workflow["nodes"]}
    assert len(nodes) == len(workflow["nodes"]) == len({node["id"] for node in nodes.values()})
    webhooks = {
        name: node["parameters"]["path"]
        for name, node in nodes.items()
        if node["type"] == "n8n-nodes-base.webhook"
    }
    assert webhooks == {
        "Tender webhook": "s2-tender-evaluation",
        "Decision webhook": "s2-tender-decision",
    }
    for name in webhooks:
        assert nodes[name]["parameters"]["authentication"] == "headerAuth"
        assert nodes[name]["parameters"]["responseMode"] == "responseNode"

    base = "http://host.docker.internal:8021"
    routes = {
        "Register S2 delivery": "/s2/intake",
        "Read current S1 cards": "/s2/supplier-gates",
        "Parse quotes": "/s2/parse",
        "Evaluate tender": "/s2/evaluate",
        "Verify decision in Bitrix24": "/s2/publication/decision",
        "Prepare publication": "/s2/publication/prepare",
        "Dispatch publication": "/v1/publications/dispatch",
    }
    for name, route in routes.items():
        node = nodes[name]
        assert node["type"] == "n8n-nodes-base.httpRequest"
        params = node["parameters"]
        assert params["method"] == "POST" and params["url"] == base + route
        assert params["genericAuthType"] == "httpHeaderAuth"
        # n8n expression interpolation terminates at the first closing delimiter.
        assert params["jsonBody"].startswith("={{") and params["jsonBody"].endswith("}}")
        assert "}}" not in params["jsonBody"][3:-2]
        assert params["options"]["response"]["response"] == {
            "fullResponse": True,
            "neverError": True,
        }
        assert "credentials" not in node

    edges = workflow["connections"]

    def paths(name: str, lineage: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
        assert name in nodes and name not in lineage
        lineage += (name,)
        if name == "Respond":
            assert name not in edges
            return [lineage]
        outputs = edges[name]["main"]
        if nodes[name]["type"] == "n8n-nodes-base.if":
            assert outputs[1] == [{"node": "Safe response", "type": "main", "index": 0}]
        return [
            path for output in outputs for link in output for path in paths(link["node"], lineage)
        ]

    visited: set[str] = set()
    for trigger, success_routes in {
        "Tender webhook": list(routes)[:4] + list(routes)[-2:],
        "Decision webhook": list(routes)[-3:],
    }.items():
        alternatives = paths(trigger)
        visited.update(name for path in alternatives for name in path)
        success = [path for path in alternatives if "Dispatch publication" in path]
        assert len(success) == 1
        assert [name for name in success[0] if name in routes] == success_routes
    assert visited == set(nodes)

    profile = load_profile(ROOT / "profiles", "portal_b24_b2b", "4")
    current = load_profile(ROOT / "profiles", "portal_b24_b2b")
    assert current.s2_tender == profile.s2_tender
    assert profile.revision == "4" and profile.s2_tender is not None
    assert (
        f"profile_revision:'{profile.revision}'"
        in (nodes["Normalize decision notice"]["parameters"]["jsCode"])
    )
    assert profile.s1_kyc is not None and profile.s1_kyc.bitrix24.tasks_enabled is False
    assert profile.s2_tender.bitrix24.stage_ids.initial is not None
