"""Discover or provision the S1 test SPA. Run from the repository root."""

import argparse
import asyncio
import json
from pathlib import Path

import yaml
from pydantic import JsonValue, TypeAdapter

from core.b24_client import BitrixClient, BitrixError, Method, PortalLimiter
from core.config import Settings
from core.contracts import JSONObject
from core.profile_loader import Profile

TITLE = "Проверка контрагента"
FIELDS = (
    "INN",
    "OGRN",
    "KYC_STATUS",
    "RELIABILITY_SCORE",
    "APPROVED_CREDIT_LIMIT_RUB",
    "EXTERNAL_KEY",
)
STAGES = {
    "review": ("На проверке СБ", "", "S1_REVIEW", 30),
    "approved": ("Одобрен", "S", "S1_APPROVED", 90),
    "rejected": ("Отклонён", "F", "S1_REJECTED", 100),
}
REQUIRED: tuple[Method, ...] = (
    "crm.type.list",
    "crm.type.add",
    "crm.category.list",
    "crm.item.fields",
    "crm.item.list",
    "crm.item.get",
    "crm.item.add",
    "crm.item.update",
    "crm.status.list",
    "crm.status.add",
    "crm.status.update",
    "tasks.task.add",
    "tasks.task.get",
    "crm.timeline.comment.add",
    "userfieldconfig.list",
    "userfieldconfig.get",
    "userfieldconfig.add",
)
OBJECT = TypeAdapter(JSONObject)
ROWS = TypeAdapter(list[JSONObject])


def identifier(value: JsonValue) -> int:
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    if type(value) is not int or value < 1:
        raise ValueError("Invalid Bitrix identifier")
    return value


def one(rows: list[JSONObject], key: str, value: JsonValue) -> JSONObject | None:
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous portal configuration for {key}={value}")
    return matches[0] if matches else None


async def pages(
    client: BitrixClient, method: Method, body: JSONObject, key: str | None = None
) -> list[JSONObject]:
    rows = []
    start = 0
    while True:
        result, cursor = await client.setup_call(method, {**body, "start": start})
        rows.extend(
            ROWS.validate_python(result if key is None else OBJECT.validate_python(result)[key])
        )
        if cursor is None:
            return rows
        if cursor <= start:
            raise ValueError("Invalid Bitrix pagination")
        start = cursor


async def process(client: BitrixClient, apply: bool) -> JSONObject | None:
    rows = await pages(client, "crm.type.list", {"filter": {"title": TITLE}}, "types")
    found = one(rows, "title", TITLE)
    if found is None and apply:
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
        return await process(client, False)
    return found


async def stages(client: BitrixClient, entity: int, category: int, apply: bool) -> dict[str, str]:
    directory = f"DYNAMIC_{entity}_STAGE_{category}"
    rows = await pages(client, "crm.status.list", {"filter": {"ENTITY_ID": directory}})
    initial = [row for row in rows if str(row.get("STATUS_ID")).endswith(":NEW")]
    if len(initial) != 1:
        raise ValueError(f"Could not identify the initial stage in {directory}")
    output = {"initial": str(initial[0]["STATUS_ID"])}
    for key, (title, semantics, code, sort) in STAGES.items():
        row = one(rows, "NAME", title)
        if row is None and apply:
            if key == "review":
                await client.setup_call(
                    "crm.status.add",
                    {
                        "fields": {
                            "ENTITY_ID": directory,
                            "STATUS_ID": code,
                            "NAME": title,
                            "SEMANTICS": semantics,
                            "SORT": sort,
                        }
                    },
                    write=True,
                )
            else:
                # Rename the built-in terminal stage and preserve its ID and semantics.
                existing = [row for row in rows if row.get("SEMANTICS") == semantics]
                if len(existing) != 1:
                    raise ValueError(f"Could not identify a unique terminal stage: {key}")
                await client.setup_call(
                    "crm.status.update",
                    {
                        "id": identifier(existing[0]["ID"]),
                        "fields": {"NAME": title},
                    },
                    write=True,
                )
        elif row is not None:
            if (row.get("SEMANTICS") or "") != semantics:
                raise ValueError(f"Unexpected semantics for stage {title}")
            output[key] = str(row["STATUS_ID"])
    if apply and len(output) != 4:
        return await stages(client, entity, category, False)
    return output


async def fields(
    client: BitrixClient, type_id: int, entity: int, apply: bool
) -> tuple[dict[str, str], dict[str, int]]:
    owner = f"CRM_{type_id}"
    base: JSONObject = {"moduleId": "crm", "filter": {"entityId": owner}}
    existing = await pages(client, "userfieldconfig.list", base, "fields")
    enum: dict[str, int] = {}
    names = {}
    for logical in FIELDS:
        original = f"UF_{owner}_{logical}"
        field_type = "enumeration" if logical == "KYC_STATUS" else "string"
        row = one(existing, "fieldName", original)
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
                    {"value": "APPROVED", "xmlId": "APPROVED", "sort": 100},
                    {"value": "REJECTED", "xmlId": "REJECTED", "sort": 200},
                ]
            raw, _ = await client.setup_call(
                "userfieldconfig.add",
                {
                    "moduleId": "crm",
                    "field": definition,
                },
                write=True,
            )
            row = OBJECT.validate_python(OBJECT.validate_python(raw)["field"])
        if row is None:
            continue
        if row.get("userTypeId") != field_type or row.get("multiple") not in ("N", False, 0):
            raise ValueError(f"Incompatible portal field: {original}")
        names[logical] = original
        if field_type == "enumeration":
            raw, _ = await client.setup_call(
                "userfieldconfig.get",
                {
                    "moduleId": "crm",
                    "id": identifier(row["id"]),
                },
            )
            options = ROWS.validate_python(
                OBJECT.validate_python(OBJECT.validate_python(raw)["field"])["enum"]
            )
            for code in ("APPROVED", "REJECTED"):
                option = one(options, "value", code)
                if option is None:
                    raise ValueError(f"Missing KYC_STATUS value: {code}")
                enum[code] = identifier(option["id"])
    raw, _ = await client.setup_call("crm.item.fields", {"entityTypeId": entity})
    metadata = OBJECT.validate_python(OBJECT.validate_python(raw)["fields"])
    output = {}
    for logical, original in names.items():
        # Match the actual camelCase code returned by the portal, including simplified names.
        matches = [
            name
            for name in metadata
            if name.casefold().replace("_", "") == original.casefold().replace("_", "")
        ]
        if len(matches) != 1 or not matches[0].startswith("ufCrm"):
            raise ValueError(f"Could not resolve universal REST code for {original}")
        output[logical] = matches[0]
    return output, enum


def write_profile(root: Path, details: dict[str, JsonValue]) -> None:
    template = yaml.safe_load((root / "default_b2b" / "profile.yaml").read_text(encoding="utf-8"))
    bindings = OBJECT.validate_python(details["fields"])
    stage_ids = OBJECT.validate_python(details["stages"])
    template.update(
        slug="portal_b24_b2b",
        revision="3",
        template=False,
        tenant_id="portal_b24_b2b",
        portal_id=details["portal_id"],
    )
    template["stages"]["S1"] = {
        "entity_type_id": details["entityTypeId"],
        "category_id": details["categoryId"],
        "stage_id": stage_ids["initial"],
        "external_key_field": bindings.pop("EXTERNAL_KEY"),
        "responsible_id": details["responsible_id"],
        "stage_ids": stage_ids,
    }
    template["s1_kyc"]["bitrix24"] = {
        "fields": bindings,
        "kyc_status_values": details["kyc_status_values"],
        "tasks_enabled": details["tasks_enabled"],
    }
    Profile.model_validate(template)
    path = root / "portal_b24_b2b" / "profile.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(template, allow_unicode=True, sort_keys=False)
    if path.exists() and yaml.safe_load(path.read_text(encoding="utf-8")) != template:
        raise ValueError("Existing working profile differs; review and assign a new revision")
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)


async def run(settings: Settings, *, apply: bool) -> bool:
    if settings.bitrix24_webhook_url is None:
        raise ValueError("BITRIX24_WEBHOOK_URL is missing")
    client = BitrixClient(
        settings.bitrix24_webhook_url.get_secret_value(),
        PortalLimiter(settings.bitrix24_rate_limit_rps),
    )
    try:
        who, _ = await client.setup_call("profile", {})
        responsible = identifier(OBJECT.validate_python(who)["ID"])
        scope, _ = await client.setup_call("scope", {})
        print(
            json.dumps(
                {"portal_id": client.portal_id, "responsible_id": responsible, "scope": scope}
            )
        )
        methods = {}
        for name in REQUIRED:
            result, _ = await client.setup_call("method.get", {"name": name})
            methods[name] = OBJECT.validate_python(result).get("isAvailable") is True
        print(json.dumps({"methods": methods}, sort_keys=True))
        found = await process(client, apply and methods["crm.type.add"])
        if found is None:
            print("SPA not found.")
            return False
        type_id, entity = identifier(found["id"]), identifier(found["entityTypeId"])
        categories = await pages(
            client, "crm.category.list", {"entityTypeId": entity}, "categories"
        )
        defaults = [row for row in categories if row.get("isDefault") in ("Y", True)]
        if len(defaults) != 1:
            raise ValueError("Could not identify the default CRM category")
        category = int(str(defaults[0]["id"]))
        details: dict[str, JsonValue] = {
            "portal_id": client.portal_id,
            "responsible_id": responsible,
            "entityTypeId": entity,
            "type_id": type_id,
            "categoryId": category,
        }
        print(json.dumps(details))
        stage_ids = await stages(
            client,
            entity,
            category,
            apply and methods["crm.status.add"] and methods["crm.status.update"],
        )
        details["stages"] = dict[str, JsonValue](stage_ids)
        print(json.dumps({"stages": stage_ids}))
        if not (methods["userfieldconfig.list"] and methods["userfieldconfig.get"]):
            print("Missing userfieldconfig scope. Field and enum discovery blocked.")
            return False
        bindings, enum = await fields(
            client, type_id, entity, apply and methods["userfieldconfig.add"]
        )
        details.update(
            fields=dict[str, JsonValue](bindings),
            kyc_status_values=dict[str, JsonValue](enum),
            tasks_enabled=methods["tasks.task.add"] and methods["tasks.task.get"],
        )
        print(json.dumps(details, ensure_ascii=True))
        complete = (
            set(bindings) == set(FIELDS)
            and len(enum) == 2
            and len(OBJECT.validate_python(details["stages"])) == 4
        )
        if complete and apply:
            write_profile(settings.profiles_path, details)
        return complete
    finally:
        await client.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Provision the configured test portal")
    args = parser.parse_args()
    try:
        return 0 if asyncio.run(run(Settings(), apply=args.apply)) else 2
    except BitrixError as error:
        print(f"Bitrix24 setup stopped: {error.code}; uncertain={error.uncertain}")
    except ValueError as error:
        if type(error) is ValueError:
            print(f"Bitrix24 setup stopped: {error}")
        else:
            print("Bitrix24 setup stopped: invalid configuration or response")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
