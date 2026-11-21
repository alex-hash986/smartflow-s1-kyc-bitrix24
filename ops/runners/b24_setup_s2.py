"""Discover or provision the S2 tender SPA on the configured test portal."""

import argparse
import asyncio
import ipaddress
import json
import socket
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from unittest.mock import patch

import yaml

from core.b24_client import BitrixClient, BitrixError, Method, PortalLimiter
from core.config import Settings
from core.contracts import JSONObject
from core.profile_loader import Profile
from ops.runners.b24_setup_s1 import OBJECT, ROWS, identifier, one, pages

PORTAL = "b24-c53p0d.bitrix24.ru"
TITLE = "Тендер и свод КП"
STATUS = ("AWARDED", "REVIEW_REQUIRED", "CANCELLED")
FIELDS = {
    "TENDER_STATUS": "enumeration",
    "WINNER_INN": "string",
    "WINNER_S1_ITEM_ID": "integer",
    "TOTAL_AMOUNT_RUB": "string",
    "SAVINGS_RUB": "string",
    "EXTERNAL_KEY": "string",
}
STAGES = {
    "initial": ("NEW", "", "Приём КП / расчёт"),
    "review": ("S2_REVIEW", "", "Согласование тендера"),
    "awarded": ("SUCCESS", "S", "Тендер присуждён"),
    "cancelled": ("FAIL", "F", "Тендер отменён"),
}
READ_METHODS: tuple[Method, ...] = (
    "crm.type.list",
    "crm.category.list",
    "crm.status.list",
    "userfieldconfig.list",
    "userfieldconfig.get",
    "crm.item.fields",
)
WRITE_METHODS: tuple[Method, ...] = (
    "crm.type.add",
    "crm.status.add",
    "crm.status.update",
    "userfieldconfig.add",
    "crm.item.add",
    "crm.item.update",
    "crm.item.get",
    "crm.item.list",
    "crm.timeline.comment.add",
)


async def discover_stages(
    client: BitrixClient,
    entity_id: int,
    category_id: int,
    apply: bool,
    *,
    definitions: dict[str, tuple[str, str, str]] = STAGES,
    custom: frozenset[str] = frozenset({"review"}),
) -> dict[str, str]:
    directory = f"DYNAMIC_{entity_id}_STAGE_{category_id}"
    prefix = f"DT{entity_id}_{category_id}:"
    rows = await pages(client, "crm.status.list", {"filter": {"ENTITY_ID": directory}})
    stages = {}
    for key, (suffix, semantics, name) in definitions.items():
        code = prefix + suffix
        row = one(rows, "STATUS_ID", code)
        if row is None:
            if key not in custom:
                raise ValueError(f"Missing built-in stage {code}")
            if apply:
                await client.setup_call(
                    "crm.status.add",
                    {
                        "fields": {
                            "ENTITY_ID": directory,
                            "STATUS_ID": suffix,
                            "NAME": name,
                            "SEMANTICS": "",
                            "SORT": 30 if key == "review" else 35,
                        }
                    },
                    write=True,
                )
            continue
        if (row.get("SEMANTICS") or "") != semantics:
            raise ValueError(f"Unexpected semantics for {code}")
        if apply and row.get("NAME") != name:
            await client.setup_call(
                "crm.status.update",
                {"id": identifier(row["ID"]), "fields": {"NAME": name}},
                write=True,
            )
        stages[key] = code
    if apply:
        return await discover_stages(
            client, entity_id, category_id, False, definitions=definitions, custom=custom
        )
    return stages


async def discover_fields(
    client: BitrixClient,
    type_id: int,
    entity_id: int,
    apply: bool,
    *,
    definitions: dict[str, str] = FIELDS,
    options_codes: tuple[str, ...] = STATUS,
) -> tuple[dict[str, str], dict[str, int]]:
    owner = f"CRM_{type_id}"
    rows = await pages(
        client,
        "userfieldconfig.list",
        {"moduleId": "crm", "filter": {"entityId": owner}},
        "fields",
    )
    original_names = {}
    status_ids = {}
    for logical, field_type in definitions.items():
        original = f"UF_{owner}_{logical}"
        row = one(rows, "fieldName", original)
        if row is None and apply:
            definition: JSONObject = {
                "entityId": owner,
                "fieldName": original,
                "userTypeId": field_type,
                "multiple": "N",
                "mandatory": "N",
                "editFormLabel": {"ru": logical},
            }
            if field_type == "enumeration":
                definition["enum"] = [
                    {"value": code, "xmlId": code, "sort": 100 * index}
                    for index, code in enumerate(options_codes, 1)
                ]
            result, _ = await client.setup_call(
                "userfieldconfig.add",
                {"moduleId": "crm", "field": definition},
                write=True,
            )
            row = OBJECT.validate_python(OBJECT.validate_python(result)["field"])
        if row is None:
            continue
        if (
            row.get("userTypeId") != field_type
            or row.get("multiple") not in ("N", False, 0)
            or row.get("mandatory") not in ("N", False, 0)
        ):
            raise ValueError(f"Incompatible portal field {original}")
        original_names[logical] = original
        if field_type == "enumeration":
            result, _ = await client.setup_call(
                "userfieldconfig.get",
                {"moduleId": "crm", "id": identifier(row["id"])},
            )
            options = ROWS.validate_python(
                OBJECT.validate_python(OBJECT.validate_python(result)["field"])["enum"]
            )
            if len(options) != len(options_codes):
                raise ValueError(f"Unexpected options for {logical}")
            for status in options_codes:
                option = one(options, "value", status)
                if option is None:
                    raise ValueError(f"Missing {logical} option {status}")
                status_ids[status] = identifier(option["id"])

    raw, _ = await client.setup_call("crm.item.fields", {"entityTypeId": entity_id})
    metadata = OBJECT.validate_python(OBJECT.validate_python(raw)["fields"])
    fields = {}
    for logical, original in original_names.items():
        matches = [
            code
            for code in metadata
            if code.startswith("ufCrm")
            and code.casefold().replace("_", "") == original.casefold().replace("_", "")
        ]
        if len(matches) != 1:
            raise ValueError(f"Could not resolve universal REST code for {original}")
        fields[logical] = matches[0]
    return fields, status_ids


def write_profile(root: Path, portal: str, bindings: JSONObject) -> None:
    directory = root / "portal_b24_b2b"
    path = directory / "profile.yaml"
    original = path.read_text(encoding="utf-8")
    profile = yaml.safe_load(original)
    if profile["revision"] not in ("3", "4") or profile["portal_id"] != portal:
        raise ValueError("The portal profile does not match revisions 3 or 4")
    if profile["s1_kyc"]["bitrix24"]["tasks_enabled"] is not False:
        raise ValueError("The working S1 profile must keep tasks disabled")
    if profile["revision"] == "4":
        if profile["s2_tender"]["bitrix24"] != bindings:
            raise ValueError("Revision 4 bindings differ from the discovered portal")
        Profile.model_validate(profile)
        return
    profile["revision"] = "4"
    profile["s2_tender"]["bitrix24"] = bindings
    Profile.model_validate(profile)
    archived = directory / "profile.v3.yaml"
    if archived.exists() and archived.read_text(encoding="utf-8") != original:
        raise ValueError("Revision 3 archive differs from current configuration")
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
            raise ValueError("Configured portal is outside S2 setup scope")
        who, _ = await client.setup_call("profile", {})
        if identifier(OBJECT.validate_python(who)["ID"]) != 1:
            raise ValueError("S2 setup requires the portal webhook for user 1")
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
        stages = await discover_stages(client, entity, category, apply)
        fields, status_ids = await discover_fields(client, type_id, entity, apply)
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
            set(stages) == set(STAGES) and set(fields) == set(FIELDS) and len(status_ids) == 3
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
    mode.add_argument("--apply", action="store_true", help="Provision the S2 test process")
    parser.add_argument(
        "--pin-ip", type=ipaddress.ip_address, help="Pin DNS while retaining TLS SNI"
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
        print(f"S2 setup stopped: {error.code}; uncertain={error.uncertain}")
    except ValueError as error:
        print(
            f"S2 setup stopped: {error}"
            if type(error) is ValueError
            else "Invalid S2 configuration"
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
