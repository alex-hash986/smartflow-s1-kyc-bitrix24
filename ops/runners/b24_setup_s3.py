"""Discover or provision S3 on the configured test portal."""

import argparse
import asyncio
import ipaddress
import json
import socket
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from unittest.mock import patch

import yaml

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.b24_client import BitrixClient, BitrixError, PortalLimiter
from core.config import Settings
from core.contracts import JSONObject
from core.profile_loader import Profile
from ops.runners.b24_setup_s1 import OBJECT, identifier, one, pages
from ops.runners.b24_setup_s2 import READ_METHODS, WRITE_METHODS, discover_fields, discover_stages

PORTAL = "b24-c53p0d.bitrix24.ru"
TITLE = "Приёмка и рекламации"
STATUS = ("REVIEW_REQUIRED", "ACCEPTED", "CLAIM_REQUIRED", "CLOSED")
FIELDS = {
    "ACCEPTANCE_STATUS": "enumeration",
    "S2_ITEM_ID": "integer",
    "SUPPLIER_INN": "string",
    "ACCEPTED_AMOUNT_RUB": "string",
    "CLAIM_AMOUNT_RUB": "string",
    "EXTERNAL_KEY": "string",
}
STAGES = {
    "initial": ("NEW", "", "Приёмка"),
    "review": ("S3_REVIEW", "", "Ручная проверка"),
    "accepted": ("SUCCESS", "S", "Принято"),
    "claim_required": ("FAIL", "F", "Претензия"),
    "closed": ("S3_CLOSED", "", "Закрыто"),
}
PIN = "109.120.182.255"


def write_profile(root: Path, portal: str, bindings: JSONObject) -> None:
    directory = root / "portal_b24_b2b"
    path = directory / "profile.yaml"
    original = path.read_text(encoding="utf-8")
    profile = yaml.safe_load(original)
    if profile["revision"] not in ("4", "5") or profile["portal_id"] != portal:
        raise ValueError("The portal profile does not match revisions 4 or 5")
    if profile["s1_kyc"]["bitrix24"]["tasks_enabled"] is not False:
        raise ValueError("The working S1 profile must keep tasks disabled")
    if profile["revision"] == "5":
        if profile["s3_claims"]["bitrix24"] != bindings:
            raise ValueError("Revision 5 bindings differ from the discovered portal")
        Profile.model_validate(profile)
        return
    profile["revision"] = "5"
    profile["s3_claims"].update(
        enabled=True,
        tasks_enabled=False,
        penalty_confirmation="verified_contract_evidence",
        bitrix24=bindings,
    )
    Profile.model_validate(profile)
    archived = directory / "profile.v4.yaml"
    if archived.exists() and archived.read_text(encoding="utf-8") != original:
        raise ValueError("Revision 4 archive differs from current configuration")
    if not archived.exists():
        with archived.open("x", encoding="utf-8") as stream:
            stream.write(original)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as stream:
        temp_path = Path(stream.name)
        yaml.safe_dump(profile, stream, allow_unicode=True, sort_keys=False)
    try:
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


async def run(settings: Settings, apply: bool) -> bool:
    if settings.bitrix24_webhook_url is None:
        raise ValueError("BITRIX24_WEBHOOK_URL is missing")
    client = BitrixClient(
        settings.bitrix24_webhook_url.get_secret_value(),
        PortalLimiter(settings.bitrix24_rate_limit_rps),
    )
    try:
        if client.portal_id != PORTAL:
            raise ValueError("Configured portal is outside S3 setup scope")
        who, _ = await client.setup_call("profile", {})
        if identifier(OBJECT.validate_python(who)["ID"]) != 1:
            raise ValueError("S3 setup requires the portal webhook for user 1")
        for method in (*READ_METHODS, *(WRITE_METHODS if apply else ())):
            result, _ = await client.setup_call("method.get", {"name": method})
            if OBJECT.validate_python(result).get("isAvailable") is not True:
                raise ValueError(f"Required portal method unavailable: {method}")

        filters: JSONObject = {"filter": {"title": TITLE}}
        row = one(await pages(client, "crm.type.list", filters, "types"), "title", TITLE)
        if row is None and apply:
            await client.setup_call(
                "crm.type.add",
                {
                    "fields": {
                        "title": TITLE,
                        "isStagesEnabled": "Y",
                        "isCategoriesEnabled": "Y",
                        "isAutomationEnabled": "N",
                    }
                },
                write=True,
            )
            row = one(await pages(client, "crm.type.list", filters, "types"), "title", TITLE)
        if row is None:
            print(json.dumps({"portal_id": PORTAL, "process_found": False}))
            return False
        type_id, entity = identifier(row["id"]), identifier(row["entityTypeId"])
        categories = await pages(
            client, "crm.category.list", {"entityTypeId": entity}, "categories"
        )
        defaults = [row for row in categories if row.get("isDefault") in ("Y", True)]
        if len(defaults) != 1:
            raise ValueError("Could not identify the default CRM category")
        category = int(str(defaults[0]["id"]))
        stages = await discover_stages(
            client,
            entity,
            category,
            apply,
            definitions=STAGES,
            custom=frozenset({"review", "closed"}),
        )
        fields, status_ids = await discover_fields(
            client, type_id, entity, apply, definitions=FIELDS, options_codes=STATUS
        )
        details: JSONObject = {
            "type_id": type_id,
            "entity_type_id": entity,
            "category_id": category,
            "responsible_id": 1,
            "stage_ids": dict[str, Any](stages),
            "fields": dict[str, Any](fields),
            "status_values": dict[str, Any](status_ids),
        }
        print(json.dumps(details, ensure_ascii=True))
        complete = (
            set(stages) == set(STAGES)
            and set(fields) == set(FIELDS)
            and set(status_ids) == set(STATUS)
        )
        if apply and complete:
            write_profile(settings.profiles_path, client.portal_id, details)
        return complete
    finally:
        await client.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Discover without making changes")
    mode.add_argument("--apply", action="store_true", help="Provision the S3 test process")
    parser.add_argument(
        "--pin-ip", type=ipaddress.ip_address, default=PIN, help="Pin DNS while retaining TLS SNI"
    )
    args = parser.parse_args()
    original = socket.getaddrinfo

    def resolve(host: str | bytes | None, *opts: Any, **kwargs: Any) -> Any:
        if args.pin_ip and host in (PORTAL, PORTAL.encode()):
            host = str(args.pin_ip)
        return original(host, *opts, **kwargs)

    try:
        with patch.object(socket, "getaddrinfo", side_effect=resolve):
            return 0 if asyncio.run(run(Settings(), args.apply)) else 2
    except BitrixError as error:
        print(f"S3 setup stopped: {error.code}; uncertain={error.uncertain}")
    except ValueError as error:
        print(
            f"S3 setup stopped: {error}"
            if type(error) is ValueError
            else "Invalid S3 configuration"
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
