import asyncio
import base64
import hashlib
import json
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import ValidationError

from core.contracts import Mode, WorkerRequestEnvelope, WorkerResponseEnvelope, cost_atoms
from core.database import Database, digest, utc_now
from core.idempotency import Conflict, PublicationCommand
from core.interfaces import BaseSmartWorker, WorkerResult
from core.llm_gateway import CycleBudget, LLMGateway
from core.profile_loader import Profile, load_profile
from core.publisher import Publisher
from core.telemetry import Telemetry
from modules.s2_tender.contracts import S2EvaluateTenderResponse
from modules.s2_tender.extractions import ExtractionRecord
from modules.s2_tender.module import published_gate
from modules.s3_claims.contracts import (
    S3AwardedItem,
    S3Context,
    S3DeliveryDocumentSource,
    S3EvaluateClaimsRequest,
    S3EvaluateClaimsResponse,
    S3IntakeReceipt,
    S3IntakeRequest,
    S3ParseCall,
    S3ParseDocsResponse,
    S3ParseReply,
    S3TenderAwardSnapshot,
    S3TenderGateRequest,
    S3TenderGateResponse,
    as_money,
)
from modules.s3_claims.parser import AcceptanceParser, ParseRecord, normalize
from modules.s3_claims.scoring import evaluate


def cycle(call: S3Context) -> tuple[str, str]:
    return call.tenant_id, str(call.request_id)


def context(call: S3Context | WorkerRequestEnvelope) -> S3Context:
    return S3Context.model_validate(call.model_dump(include=set(S3Context.model_fields)))


def source_hash(source: S3DeliveryDocumentSource) -> str:
    data = source.model_dump(mode="json", exclude={"pdf_base64"})
    if source.pdf_base64 is not None:
        data["pdf_sha256"] = hashlib.sha256(base64.b64decode(source.pdf_base64)).hexdigest()
    return digest(data)


def package(conn: sqlite3.Connection, call: S3Context, package_id: UUID) -> sqlite3.Row:
    row = conn.execute(
        """SELECT * FROM s3_packages WHERE tenant_id=? AND request_id=?
           AND package_id=? AND active=1""",
        (*cycle(call), str(package_id)),
    ).fetchone()
    if row is None:
        raise Conflict("S3 package is unknown or superseded")
    assert isinstance(row, sqlite3.Row)
    return row


def published_tender(conn: sqlite3.Connection, call: S3Context) -> sqlite3.Row:
    row = conn.execute(
        """SELECT p.*, s.response_json FROM crm_publications AS p
           JOIN transaction_stages AS s ON s.tenant_id=p.tenant_id AND s.request_id=p.request_id
               AND s.stage=p.stage AND s.attempt=p.stage_attempt
           WHERE p.tenant_id=? AND p.request_id=? AND p.stage='S2' AND p.purpose='stage_result'
           AND p.status='published' AND s.status='completed'""",
        cycle(call),
    ).fetchone()
    if row is None:
        raise Conflict("S3 requires published S2 for the cycle")
    assert isinstance(row, sqlite3.Row)
    return row


def read_gate(conn: sqlite3.Connection, call: S3Context, package_id: UUID) -> S3TenderGateResponse:
    row = conn.execute(
        """SELECT response_json FROM s3_tender_gates WHERE tenant_id=? AND request_id=?
           AND package_id=?""",
        (*cycle(call), str(package_id)),
    ).fetchone()
    if row is None:
        raise Conflict("S3 tender gate has not been checked")
    return S3TenderGateResponse.model_validate_json(row[0])


class S3Worker(BaseSmartWorker):
    request_payload = S3EvaluateClaimsRequest
    response_payload = S3EvaluateClaimsResponse

    def __init__(self, profiles_path: Path, gateway: LLMGateway, db: Database) -> None:
        self.profiles_path = profiles_path
        self.db = db
        self.parser = AcceptanceParser(gateway)
        self.telemetry = Telemetry(db)

    def profile(self, call: S3Context) -> Profile:
        try:
            profile = load_profile(self.profiles_path, call.profile_slug, call.profile_revision)
        except (OSError, ValueError) as exc:
            raise Conflict("Unknown S3 profile") from exc
        if (
            profile.tenant_id != call.tenant_id
            or profile.template
            or profile.s3_claims is None
            or not profile.s3_claims.enabled
            or profile.s2_tender is None
        ):
            raise Conflict("S3 profile is inactive or belongs to another tenant")
        return profile

    def intake(self, call: S3IntakeRequest) -> S3IntakeReceipt:
        profile = self.profile(call)
        sha = digest([call.s2_item_id, *[source_hash(doc) for doc in call.documents]])
        with self.db.transaction() as conn:
            delivery = conn.execute(
                """SELECT profile_slug,profile_revision FROM idempotency_records
                   WHERE tenant_id=? AND request_id=?""",
                cycle(call),
            ).fetchone()
            if delivery is None or tuple(delivery) != (call.profile_slug, call.profile_revision):
                raise Conflict("S3 intake does not match the cycle")
            tender = published_tender(conn, call)
            assert profile.s2_tender is not None
            if (
                tender["item_id"] != call.s2_item_id
                or tender["portal_id"] != profile.portal_id
                or tender["entity_type_id"] != profile.s2_tender.bitrix24.entity_type_id
            ):
                raise Conflict("S3 package does not match the published tender")
            prior = conn.execute(
                """SELECT package_id,input_sha256 FROM s3_packages
                   WHERE tenant_id=? AND request_id=? AND delivery_key=?""",
                (*cycle(call), call.delivery_key),
            ).fetchone()
            if prior is not None:
                if prior["input_sha256"] != sha:
                    raise Conflict("S3 delivery content changed")
                return S3IntakeReceipt(
                    **context(call).model_dump(),
                    package_id=prior["package_id"],
                    s2_item_id=call.s2_item_id,
                )
            if conn.execute(
                """SELECT 1 FROM crm_publications WHERE tenant_id=? AND request_id=? AND stage='S3'
                   AND (purpose='stage_result' OR status IN ('publishing','reconcile'))""",
                cycle(call),
            ).fetchone():
                raise Conflict("S3 publication prevents replacing this package")
            package_id = uuid4()
            conn.execute(
                "UPDATE s3_packages SET active=0 WHERE tenant_id=? AND request_id=? AND active=1",
                cycle(call),
            )
            conn.execute(
                "INSERT INTO s3_packages VALUES (?,?,?,?,?,?,?,1,?)",
                (
                    *cycle(call),
                    str(package_id),
                    call.delivery_key,
                    sha,
                    call.model_dump_json(),
                    call.s2_item_id,
                    utc_now(),
                ),
            )
        return S3IntakeReceipt(
            **context(call).model_dump(),
            package_id=package_id,
            s2_item_id=call.s2_item_id,
        )

    async def gate(
        self, call: S3TenderGateRequest, publisher: Publisher, *, save: bool = True
    ) -> S3TenderGateResponse:
        profile = self.profile(call)
        assert profile.s2_tender is not None
        mapping = profile.s2_tender.bitrix24
        with self.db.connect() as conn:
            current = package(conn, call, call.package_id)
            row = published_tender(conn, call)
        if context(S3IntakeRequest.model_validate_json(current["input_json"])) != context(call):
            raise Conflict("S3 gate context differs from intake")
        command = PublicationCommand.model_validate_json(row["command_json"])
        if profile.portal_id != publisher.portal_id or row["portal_id"] != publisher.portal_id:
            raise Conflict("S3 gate belongs to another portal")
        item = await publisher.client.item_get(command.entity_type_id, current["s2_item_id"])
        fields = mapping.fields
        award = None
        reason = "S3_AWARD_UNCONFIRMED"
        expected = mapping.status_values.AWARDED
        if expected is not None and (
            str(item.get(fields.TENDER_STATUS or "")) != str(expected)
            or str(command.fields.get(fields.TENDER_STATUS or "")) != str(expected)
        ):
            reason = "S3_TENDER_NOT_AWARDED"
        elif expected is not None and all(
            (
                str(item.get("id")) == str(current["s2_item_id"]),
                command.entity_type_id == mapping.entity_type_id,
                item.get("stageId") == mapping.stage_ids.awarded,
                item.get(fields.EXTERNAL_KEY or "")
                == (command.card_external_key or row["external_key"]),
            )
        ):
            response = WorkerResponseEnvelope.model_validate_json(row["response_json"])
            result = S2EvaluateTenderResponse.model_validate(response.payload)
            inn, s1_id = result.winner_inn, result.winner_s1_item_id
            if (
                inn is None
                or s1_id is None
                or item.get(fields.WINNER_INN or "") != inn
                or str(item.get(fields.WINNER_S1_ITEM_ID or "")) != str(s1_id)
                or command.fields.get(fields.WINNER_INN or "") != inn
                or str(command.fields.get(fields.WINNER_S1_ITEM_ID or "")) != str(s1_id)
            ):
                reason = "S3_WINNER_MISMATCH"
            elif not response.need_human:
                try:
                    if (
                        as_money(item.get(fields.TOTAL_AMOUNT_RUB or ""))
                        == as_money(command.fields.get(fields.TOTAL_AMOUNT_RUB or ""))
                        == result.winner_total_rub
                    ):
                        award = self.award(
                            profile, call, result, current["s2_item_id"], row["stage_attempt"]
                        )
                except ValidationError:
                    award = None
                except ValueError:
                    award = None
        reply = S3TenderGateResponse(
            **call.model_dump(),
            award=award,
            need_human=award is None,
            escalation_reasons=[reason] if award is None else [],
        )
        if save:
            with self.db.transaction() as conn:
                package(conn, call, call.package_id)
                conn.execute(
                    """INSERT INTO s3_tender_gates VALUES (?,?,?,?,?)
                       ON CONFLICT (tenant_id,request_id,package_id) DO UPDATE SET
                       response_json=excluded.response_json,checked_at=excluded.checked_at""",
                    (*cycle(call), str(call.package_id), reply.model_dump_json(), utc_now()),
                )
        return reply

    def award(
        self,
        profile: Profile,
        call: S3Context,
        result: S2EvaluateTenderResponse,
        item_id: int,
        attempt: int,
    ) -> S3TenderAwardSnapshot | None:
        winners = [
            p
            for p in result.participants
            if p.rank == 1
            and p.supplier_inn == result.winner_inn
            and p.s1_item_id == result.winner_s1_item_id
        ]
        stage = profile.stages.get("S1")
        if len(winners) != 1 or stage is None:
            return None
        winner = winners[0]
        if (
            winner.total_rub is None
            or winner.delivery_days is None
            or winner.deferment_days is None
        ):
            return None
        with self.db.connect() as conn:
            s1 = conn.execute(
                """SELECT * FROM crm_publications WHERE tenant_id=? AND portal_id=? AND stage='S1'
                   AND status='published' AND purpose='stage_result' AND entity_type_id=? AND item_id=?""",
                (call.tenant_id, profile.portal_id, stage.entity_type_id, winner.s1_item_id),
            ).fetchall()
            row = conn.execute(
                """SELECT response_json FROM s2_extractions WHERE tenant_id=? AND request_id=?
                   AND quote_id=? AND response_json IS NOT NULL""",
                (*cycle(call), str(winner.quote_id)),
            ).fetchone()
            approved = conn.execute(
                """SELECT items_json FROM s2_award_items
                   WHERE tenant_id=? AND request_id=? AND attempt=?""",
                (*cycle(call), attempt),
            ).fetchone()
        if len(s1) != 1 or row is None or approved is None:
            return None
        supplier = published_gate(profile, s1[0])
        quote = ExtractionRecord.model_validate_json(row[0]).quote
        if (
            supplier is None
            or supplier.kyc_status != "APPROVED"
            or supplier.inn != winner.supplier_inn
            or quote.supplier_inn != winner.supplier_inn
            or quote.s1_item_id != winner.s1_item_id
            or quote.quoted_total_rub != result.winner_total_rub
            or winner.total_rub != result.winner_total_rub
            or quote.deferment_days != winner.deferment_days
            or max((line.delivery_days or 0 for line in quote.items), default=0)
            != winner.delivery_days
            or quote.need_human
            or not quote.arithmetic_ok
        ):
            return None
        return S3TenderAwardSnapshot(
            s2_item_id=item_id,
            s1_item_id=winner.s1_item_id,
            tender_status="AWARDED",
            winner_inn=winner.supplier_inn,
            winner_name=winner.supplier_name,
            awarded_total_rub=winner.total_rub,
            promised_delivery_days=winner.delivery_days,
            promised_deferment_days=winner.deferment_days,
            approved_credit_limit_rub=supplier.approved_credit_limit_rub,
            awarded_items=[
                S3AwardedItem.model_validate(line) for line in json.loads(approved["items_json"])
            ],
        )

    def reserve_parse(
        self, call: S3ParseCall
    ) -> tuple[list[S3DeliveryDocumentSource], list[ParseRecord] | None] | Literal["running"]:
        package_id = str(call.payload.package_id)
        key = (*cycle(call), package_id)
        with self.db.transaction() as conn:
            row = package(conn, call, call.payload.package_id)
            intake = S3IntakeRequest.model_validate_json(row["input_json"])
            if context(intake) != context(call):
                raise Conflict("S3 parse context differs from intake")
            rows = conn.execute(
                "SELECT * FROM s3_extractions WHERE tenant_id=? AND request_id=? AND package_id=?",
                key,
            ).fetchall()
            if rows:
                by_id = {row["document_id"]: row for row in rows}
                ordered = [by_id[str(doc.document_id)] for doc in intake.documents]
                if any(
                    row["source_sha256"] != source_hash(doc)
                    for row, doc in zip(ordered, intake.documents, strict=True)
                ):
                    raise Conflict("S3 extraction source changed")
                if any(row["response_json"] is None for row in ordered):
                    return "running"
                return intake.documents, [
                    ParseRecord.model_validate_json(row["response_json"]) for row in ordered
                ]
            if conn.execute(
                """SELECT 1 FROM s3_extractions WHERE tenant_id=? AND request_id=?
                   AND response_json IS NULL""",
                cycle(call),
            ).fetchone():
                return "running"
            conn.executemany(
                """INSERT INTO s3_extractions
                   (tenant_id,request_id,package_id,document_id,source_sha256,profile_revision,
                    parser_version,prompt_version,prompt_variant,started_at)
                   VALUES (?,?,?,?,?,?, '1',?,?,?)""",
                [
                    (
                        *key,
                        str(doc.document_id),
                        source_hash(doc),
                        call.profile_revision,
                        self.parser.prompt.version,
                        self.parser.prompt.variant,
                        utc_now(),
                    )
                    for doc in intake.documents
                ],
            )
            return intake.documents, None

    def finish_parse(
        self,
        call: S3ParseCall,
        record: ParseRecord,
        *,
        evidence_ref: str | None = None,
    ) -> None:
        with self.db.transaction() as conn:
            saved = conn.execute(
                """UPDATE s3_extractions SET response_json=?,cost_rub_e8=?,mode=?,fallback_reason=?,
                   tokens_in=?,tokens_out=?,finished_at=?,recovery_ref=?
                   WHERE tenant_id=? AND request_id=? AND package_id=? AND document_id=?
                   AND response_json IS NULL""",
                (
                    record.model_dump_json(),
                    cost_atoms(record.cost_rub),
                    record.mode,
                    record.fallback_reason,
                    record.tokens_in,
                    record.tokens_out,
                    utc_now(),
                    evidence_ref,
                    *cycle(call),
                    str(call.payload.package_id),
                    str(record.document.document_id),
                ),
            )
            if saved.rowcount != 1:
                raise Conflict("S3 extraction reservation is already resolved or missing")
            Telemetry.refresh_extraction(conn, cycle(call), None)

    def reconcile_interrupted_parse(
        self,
        call: S3ParseCall,
        *,
        document_id: UUID,
        verified_cost_rub: Decimal | None,
        evidence_ref: str,
    ) -> None:
        """Close a reserved document after a trusted operator verifies the executor stopped."""
        self.profile(call)
        if not evidence_ref.strip():
            raise ValueError("S3 extraction reconciliation requires an evidence reference")
        with self.db.connect() as conn:
            row = conn.execute(
                """SELECT input_json FROM s3_packages WHERE tenant_id=? AND request_id=?
                   AND package_id=?""",
                (*cycle(call), str(call.payload.package_id)),
            ).fetchone()
        if row is None:
            raise Conflict("Unknown S3 package")
        intake = S3IntakeRequest.model_validate_json(row["input_json"])
        if context(intake) != context(call):
            raise Conflict("S3 extraction context differs from intake")
        source = next((doc for doc in intake.documents if doc.document_id == document_id), None)
        if source is None:
            raise Conflict("Unknown S3 extraction document")
        self.finish_parse(
            call,
            ParseRecord(
                document=normalize(source, None, intake.documents, "S3_PARSE_LLM_UNAVAILABLE"),
                mode="fallback",
                fallback_reason="interrupted",
                cost_rub=verified_cost_rub,
                tokens_in=None,
                tokens_out=None,
            ),
            evidence_ref=evidence_ref.strip(),
        )

    async def parse(self, call: S3ParseCall, limit: Decimal) -> S3ParseReply | Literal["running"]:
        profile = self.profile(call).s3_claims
        assert profile is not None
        reservation = await asyncio.to_thread(self.reserve_parse, call)
        if reservation == "running":
            return reservation
        sources, records = reservation
        if records is None:
            remaining = await asyncio.to_thread(
                self.telemetry.remaining_budget,
                call.tenant_id,
                call.request_id,
                limit,
                ignore_pending_s3=True,
            )
            budget = CycleBudget(remaining)
            records = []
            for source in sources:
                record = await self.parser.parse(
                    source,
                    sources,
                    profile.extraction,
                    profile.extraction.max_pdf_pages,
                    budget,
                )
                await asyncio.to_thread(self.finish_parse, call, record)
                records.append(record)
        docs = [record.document for record in records]
        flags = sorted({reason for doc in docs for reason in doc.escalation_reasons})
        fallback = next((r.fallback_reason for r in records if r.mode == "fallback"), None)
        mode: Mode = (
            "fallback"
            if fallback
            else "llm"
            if any(r.mode == "llm" for r in records)
            else ("deterministic")
        )
        cost = (
            None
            if any(r.cost_rub is None for r in records)
            else sum((r.cost_rub or Decimal(0) for r in records), Decimal(0))
        )
        prompt = next((r for r in records if r.prompt_version is not None), None)
        return S3ParseReply(
            **context(call).model_dump(),
            payload=S3ParseDocsResponse(
                package_id=call.payload.package_id,
                documents=docs,
                need_human=bool(flags),
                escalation_reasons=flags,
            ),
            mode=mode,
            cost_rub=cost,
            fallback_reason=fallback,
            prompt_version=prompt.prompt_version if prompt else None,
            prompt_variant=prompt.prompt_variant if prompt else None,
        )

    def record_penalty_terms(
        self,
        call: S3TenderGateRequest,
        *,
        evidence_ref: str,
        daily_penalty_pct: Decimal,
        max_penalty_pct: Decimal,
    ) -> None:
        """Record terms verified by a trusted contract adapter, outside the worker API."""
        self.profile(call)
        daily, cap = as_money(daily_penalty_pct), as_money(max_penalty_pct)
        if not evidence_ref.strip() or not daily <= cap <= 100:
            raise ValueError("Invalid verified S3 penalty terms")
        with self.db.transaction() as conn:
            package(conn, call, call.package_id)
            gate = read_gate(conn, call, call.package_id)
            if gate.award is None or context(gate) != context(call):
                raise Conflict("Penalty evidence requires the confirmed S3 deal")
            tender = published_tender(conn, call)
            key = (*cycle(call), tender["stage_attempt"])
            terms = (
                evidence_ref.strip(),
                digest(gate.award.model_dump(mode="json")),
                "AWARDED_TOTAL_RUB",
                str(daily),
                str(cap),
            )
            prior = conn.execute(
                """SELECT evidence_ref,award_sha256,base,daily_penalty_pct,max_penalty_pct
                   FROM s3_penalty_terms WHERE tenant_id=? AND request_id=? AND s2_attempt=?""",
                key,
            ).fetchone()
            if prior is not None:
                if tuple(prior) != terms:
                    raise Conflict("Verified terms for this S2 award have already been pinned")
                return
            conn.execute(
                """INSERT INTO s3_penalty_terms
                   (tenant_id,request_id,s2_attempt,evidence_ref,award_sha256,base,
                    daily_penalty_pct,max_penalty_pct) VALUES (?,?,?,?,?,?,?,?)""",
                (*key, *terms),
            )

    def pin_attempt(self, request: WorkerRequestEnvelope, conn: sqlite3.Connection) -> None:
        call = context(request)
        profile = self.profile(call).s3_claims
        assert profile is not None
        payload = S3EvaluateClaimsRequest.model_validate(request.payload)
        active = package(conn, call, payload.package_id)
        tender = published_tender(conn, call)
        if request.upstream_item is None or request.upstream_item.item_id != active["s2_item_id"]:
            raise Conflict("S3 evaluation does not match the package tender")
        if conn.execute(
            """SELECT 1 FROM crm_publications WHERE tenant_id=? AND request_id=?
               AND stage='S3' AND purpose='stage_result'""",
            cycle(call),
        ).fetchone():
            raise Conflict("S3 final publication already prepared")
        gate = read_gate(conn, call, payload.package_id)
        if gate.award != payload.award:
            raise Conflict("S3 award does not match the verified snapshot")
        intake = S3IntakeRequest.model_validate_json(active["input_json"])
        rows = conn.execute(
            """SELECT document_id,response_json FROM s3_extractions
               WHERE tenant_id=? AND request_id=? AND package_id=?""",
            (*cycle(call), str(payload.package_id)),
        ).fetchall()
        records = {row["document_id"]: row["response_json"] for row in rows}
        saved = [records.get(str(doc.document_id)) for doc in intake.documents]
        if (
            any(record is None for record in saved)
            or [
                ParseRecord.model_validate_json(record).document
                for record in saved
                if record is not None
            ]
            != payload.documents
        ):
            raise Conflict("S3 documents do not match the completed extraction")
        terms = conn.execute(
            """SELECT evidence_ref,award_sha256,base,daily_penalty_pct,max_penalty_pct
               FROM s3_penalty_terms WHERE tenant_id=? AND request_id=? AND s2_attempt=?""",
            (*cycle(call), tender["stage_attempt"]),
        ).fetchone()
        penalty_sha = None
        if (
            terms is not None
            and payload.award is not None
            and (
                terms["award_sha256"] == digest(payload.award.model_dump(mode="json"))
                and terms["base"] == "AWARDED_TOTAL_RUB"
                and terms["evidence_ref"].strip()
                and as_money(terms["daily_penalty_pct"]) == Decimal(profile.daily_penalty_pct)
                and as_money(terms["max_penalty_pct"]) == Decimal(profile.max_penalty_pct)
            )
        ):
            penalty_sha = digest(list(terms))
        conn.execute(
            """INSERT INTO s3_attempts
               (tenant_id,request_id,package_id,attempt,gate_json,penalty_sha256)
               VALUES (?,?,?,?,?,?)""",
            (
                *cycle(call),
                str(payload.package_id),
                request.attempt,
                gate.model_dump_json(),
                penalty_sha,
            ),
        )

    def compute(self, request: WorkerRequestEnvelope) -> WorkerResult:
        bound = self.profile(context(request))
        profile = bound.s3_claims
        assert profile is not None
        with self.db.connect() as conn:
            row = conn.execute(
                """SELECT gate_json,penalty_sha256 FROM s3_attempts
                   WHERE tenant_id=? AND request_id=? AND attempt=?""",
                (*cycle(context(request)), request.attempt),
            ).fetchone()
        if row is None:
            raise Conflict("S3 evaluation must be reserved")
        gate = S3TenderGateResponse.model_validate_json(row["gate_json"])
        payload = S3EvaluateClaimsRequest.model_validate(request.payload)
        result, reasons = evaluate(
            payload,
            profile,
            gate_reasons=gate.escalation_reasons,
            penalty_confirmed=row["penalty_sha256"] is not None,
        )
        if payload.award is not None:
            s1 = bound.stages["S1"].entity_type_id
            s2 = request.upstream_item.entity_type_id if request.upstream_item else None
            result.reconciliation_report_markdown += (
                f"\n\n[S1](/crm/type/{s1}/details/{payload.award.s1_item_id}/)"
                f" | [S2](/crm/type/{s2}/details/{payload.award.s2_item_id}/)"
            )
        fallback = result.delay_days is None
        return WorkerResult(
            WorkerResponseEnvelope(
                **request.model_dump(exclude={"payload", "received_at", "upstream_item"}),
                payload=result.model_dump(mode="json"),
                need_human=bool(reasons),
                escalation_reasons=reasons,
                cost_rub=Decimal(0),
                mode="fallback" if fallback else "deterministic",
                fallback_reason=reasons[0] if fallback else None,
            ),
            tokens_in=0,
            tokens_out=0,
        )

    async def run(self, request: WorkerRequestEnvelope, budget: CycleBudget) -> WorkerResult:
        return await asyncio.to_thread(self.compute, request)
