import asyncio
from dataclasses import dataclass
from typing import Literal

from core.b24_client import BitrixClient, task_binding
from core.contracts import JSONObject
from core.idempotency import Conflict, Journal, PublicationCommand, PublicationKey

PublicationStatus = Literal["approved", "publishing", "published", "reconcile"]


@dataclass(frozen=True)
class PublicationState:
    command: PublicationCommand
    marker: str
    status: PublicationStatus
    item_id: int | None
    task_id: int | None
    comment_id: int | None = None


class Publisher:
    def __init__(self, journal: Journal, client: BitrixClient, portal_id: str) -> None:
        self.journal, self.client, self.portal_id = journal, client, portal_id

    def state(self, key: PublicationKey) -> PublicationState:
        with self.journal.db.connect() as conn:
            row = conn.execute(
                """SELECT p.command_json,p.external_key,p.status,p.item_id,p.task_id,c.comment_id
                   FROM crm_publications AS p LEFT JOIN crm_publication_comments AS c
                   USING (tenant_id,request_id,stage,purpose)
                   WHERE p.tenant_id=? AND p.request_id=? AND p.stage=? AND p.purpose=?""",
                self.journal._key(key),
            ).fetchone()
        if row is None:
            raise Conflict("Publication has not been approved")
        command = PublicationCommand.model_validate_json(row["command_json"])
        if command.portal_id != self.portal_id:
            raise Conflict("Publication belongs to another portal")
        return PublicationState(
            command,
            row["external_key"],
            row["status"],
            row["item_id"],
            row["task_id"],
            row["comment_id"],
        )

    async def dispatch(self, key: PublicationKey) -> PublicationState:
        state = await asyncio.to_thread(self.state, key)
        token = await asyncio.to_thread(self.journal.claim, key)
        if token is None:
            return await asyncio.to_thread(self.state, key)
        try:
            # Read again after the claim so a competing reconciliation cannot leave stale IDs.
            state = await asyncio.to_thread(self.state, key)
            await self._publish(key, token, state)
        except (Exception, asyncio.CancelledError):
            await asyncio.shield(asyncio.to_thread(self.journal.reconcile, key, token))
            raise
        return await asyncio.to_thread(self.state, key)

    async def _publish(self, key: PublicationKey, token: str, state: PublicationState) -> None:
        command, item_id = state.command, state.item_id
        card_marker = command.card_external_key or state.marker
        if item_id is None:
            existing = await self._find_item(state) if command.timeline_comment else None
            if command.card_external_key is not None and key.purpose == "stage_result":
                with self.journal.db.connect() as conn:
                    review = conn.execute(
                        """SELECT item_id FROM crm_publications WHERE tenant_id=? AND request_id=?
                           AND stage=? AND purpose=? AND status='published'""",
                        (
                            key.tenant_id,
                            str(key.request_id),
                            key.stage,
                            f"human_review:{command.stage_attempt}"
                            if key.stage == "S3"
                            else "human_review",
                        ),
                    ).fetchone()
                if review is not None and (
                    existing is None or str(existing.get("id")) != str(review["item_id"])
                ):
                    raise Conflict("The confirmed review card could not be found")
            if not await asyncio.to_thread(self.journal.begin_step, key, token, "item"):
                raise Conflict("Publication item step is not available")
            fields = {**command.fields, command.external_key_field: card_marker}
            if existing is None:
                item_id = await self.client.item_add(command.entity_type_id, fields)
            else:
                existing_id = int(str(existing["id"]))
                if self._matches(existing, fields):
                    item_id = existing_id
                else:
                    item_id = await self.client.item_update(
                        command.entity_type_id, existing_id, fields
                    )
                    if item_id != existing_id:
                        raise Conflict("CRM update returned an unexpected item")
            if not await asyncio.to_thread(self.journal.record_step, key, token, "item", item_id):
                raise Conflict("Publication item ownership changed")
        item = await self.client.item_get(command.entity_type_id, item_id)
        if command.item_id is None and item.get(command.external_key_field) != card_marker:
            raise Conflict("CRM item marker could not be verified")
        if command.timeline_comment and not self._matches(item, command.fields):
            raise Conflict("CRM publication fields could not be verified")
        if command.timeline_comment and state.comment_id is None:
            if not await asyncio.to_thread(self.journal.begin_step, key, token, "comment"):
                raise Conflict("Publication comment step is not available")
            comment_id = await self.client.timeline_comment_add(
                command.entity_type_id,
                item_id,
                f"{command.timeline_comment}\n\n[smartflow:{state.marker}]",
            )
            if not await asyncio.to_thread(
                self.journal.record_step, key, token, "comment", comment_id
            ):
                raise Conflict("Publication comment ownership changed")
        task_id = state.task_id
        if command.task_required and task_id is None:
            if not await asyncio.to_thread(self.journal.begin_step, key, token, "task"):
                raise Conflict("Publication task step is not available")
            task = command.task
            if task is None:
                raise Conflict("Approved task fields are missing")
            task_id = await self.client.task_add(
                task.title,
                task.responsible_id,
                command.entity_type_id,
                item_id,
                description=f"{task.description}\n[smartflow:{state.marker}]",
            )
            if not await asyncio.to_thread(self.journal.record_step, key, token, "task", task_id):
                raise Conflict("Publication task ownership changed")
        if task_id is not None:
            task_item = await self.client.task_get(task_id)
            bindings = task_item.get("ufCrmTask", task_item.get("UF_CRM_TASK"))
            if (
                not isinstance(bindings, list)
                or task_binding(command.entity_type_id, item_id) not in bindings
            ):
                raise Conflict("CRM task binding could not be verified")
            description = task_item.get("description", task_item.get("DESCRIPTION"))
            if not isinstance(description, str) or f"[smartflow:{state.marker}]" not in description:
                raise Conflict("CRM task marker could not be verified")
        if not await asyncio.to_thread(self.journal.finish, key, token):
            raise Conflict("Publication is incomplete")

    async def _find_item(self, state: PublicationState) -> JSONObject | None:
        command = state.command
        card_marker = command.card_external_key or state.marker
        matches: list[JSONObject] = []
        start = 0
        while True:
            page = await self.client.item_list(
                command.entity_type_id,
                {"=" + command.external_key_field: card_marker},
                start=start,
            )
            matches.extend(
                item for item in page.items if item.get(command.external_key_field) == card_marker
            )
            if page.next_start is None:
                break
            if page.next_start <= start:
                raise Conflict("CRM pagination did not advance")
            start = page.next_start
        if len(matches) > 1:
            raise Conflict("CRM marker is ambiguous")
        if not matches:
            return None
        match = await self.client.item_get(command.entity_type_id, int(str(matches[0]["id"])))
        if match.get(command.external_key_field) != card_marker:
            raise Conflict("CRM marker changed during lookup")
        if (
            command.inn_field is not None
            and command.inn_field in command.fields
            and (match.get(command.inn_field) != command.fields[command.inn_field])
        ):
            raise Conflict("CRM marker belongs to another counterparty")
        return match

    @staticmethod
    def _matches(item: JSONObject, fields: JSONObject) -> bool:
        return all(
            item.get(key) == expected
            or (expected is None and item.get(key) == "")
            or (type(expected) is int and item.get(key) == str(expected))
            for key, expected in fields.items()
        )
