import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal
from uuid import UUID

from core.contracts import COST_UNIT, Mode, StrictDTO, cost_atoms
from core.database import Database, digest, utc_now
from core.idempotency import Conflict
from core.telemetry import Telemetry
from modules.s2_tender.contracts import (
    S2Context,
    S2ParseCall,
    S2ParsedQuote,
    S2ParseQuotesResponse,
    S2QuoteSource,
)
from modules.s2_tender.parser import ParseResult


class ExtractionRecord(StrictDTO):
    quote: S2ParsedQuote
    mode: Mode
    fallback_reason: str | None
    cost_rub: str | None
    prompt_version: str | None


class S2ParseReply(S2Context):
    payload: S2ParseQuotesResponse
    mode: Mode
    fallback_reason: str | None
    cost_rub: str | None
    prompt_version: str | None


@dataclass(frozen=True)
class Batch:
    records: dict[UUID, ExtractionRecord]
    pending: list[S2QuoteSource]


class ExtractionJournal:
    def __init__(self, db: Database) -> None:
        self.db = db

    def reserve(self, call: S2ParseCall) -> Batch | Literal["running"]:
        key = (call.tenant_id, str(call.request_id))
        records: dict[UUID, ExtractionRecord] = {}
        pending: list[S2QuoteSource] = []
        inserts = []
        with self.db.transaction() as conn:
            delivery = conn.execute(
                """SELECT profile_slug,profile_revision FROM idempotency_records
                   WHERE tenant_id=? AND request_id=?""",
                key,
            ).fetchone()
            if delivery is None or tuple(delivery) != (call.profile_slug, call.profile_revision):
                raise Conflict("Unregistered request or different delivery context")
            running = False
            for source in call.payload.quotes:
                quote_id = str(source.quote_id)
                fingerprint = digest(source.model_dump(mode="json"))
                prior = conn.execute(
                    """SELECT input_sha256,response_json FROM s2_extractions
                       WHERE tenant_id=? AND request_id=? AND quote_id=?""",
                    (*key, quote_id),
                ).fetchone()
                if prior is None:
                    pending.append(source)
                    inserts.append((*key, quote_id, fingerprint, utc_now()))
                elif prior["input_sha256"] != fingerprint:
                    raise Conflict("Extraction source has changed")
                elif prior["response_json"] is None:
                    running = True
                else:
                    records[source.quote_id] = ExtractionRecord.model_validate_json(
                        prior["response_json"]
                    )
            if running:
                return "running"
            if pending:
                # A cycle owns one extraction batch at a time, including across processes.
                busy = conn.execute(
                    """SELECT 1 FROM s2_extractions WHERE tenant_id=? AND request_id=?
                       AND response_json IS NULL
                       UNION ALL
                       SELECT 1 FROM s1_extractions WHERE tenant_id=? AND request_id=?
                       AND response_json IS NULL
                       UNION ALL
                       SELECT 1 FROM transaction_stages WHERE tenant_id=? AND request_id=?
                       AND status='running' LIMIT 1""",
                    (*key, *key, *key),
                ).fetchone()
                if busy is not None:
                    return "running"
                final = conn.execute(
                    """SELECT 1 FROM crm_publications WHERE tenant_id=? AND request_id=?
                       AND stage='S2' AND purpose='stage_result'""",
                    key,
                ).fetchone()
                if final is not None:
                    raise Conflict("S2 final publication has already been prepared")
                conn.executemany(
                    """INSERT INTO s2_extractions
                       (tenant_id,request_id,quote_id,input_sha256,started_at) VALUES (?,?,?,?,?)""",
                    inserts,
                )
        return Batch(records, pending)

    def complete(
        self, call: S2Context, results: dict[UUID, ParseResult]
    ) -> dict[UUID, ExtractionRecord]:
        key = (call.tenant_id, str(call.request_id))
        records: dict[UUID, ExtractionRecord] = {}
        with self.db.transaction() as conn:
            for quote_id, result in results.items():
                atoms = cost_atoms(result.cost_rub)
                record = ExtractionRecord(
                    quote=result.quote,
                    mode=result.mode,
                    fallback_reason=result.fallback_reason,
                    cost_rub=None if atoms is None else format(Decimal(atoms) * COST_UNIT, ".8f"),
                    prompt_version=result.prompt_version,
                )
                updated = conn.execute(
                    """UPDATE s2_extractions SET response_json=?,finished_at=?,mode=?,
                       fallback_reason=?,cost_rub_e8=?,tokens_in=?,tokens_out=?
                       WHERE tenant_id=? AND request_id=? AND quote_id=? AND response_json IS NULL""",
                    (
                        record.model_dump_json(),
                        utc_now(),
                        result.mode,
                        result.fallback_reason,
                        atoms,
                        result.tokens_in,
                        result.tokens_out,
                        *key,
                        str(quote_id),
                    ),
                )
                if updated.rowcount != 1:
                    raise Conflict("Extraction is not reserved or already completed")
                records[quote_id] = record
            Telemetry.refresh_extraction(conn, key, None)
        return records

    @staticmethod
    def verified(conn: sqlite3.Connection, key: tuple[str, str], quote: S2ParsedQuote) -> bool:
        row = conn.execute(
            """SELECT response_json FROM s2_extractions
               WHERE tenant_id=? AND request_id=? AND quote_id=?""",
            (*key, str(quote.quote_id)),
        ).fetchone()
        return (
            row is not None
            and row["response_json"] is not None
            and ExtractionRecord.model_validate_json(row["response_json"]).quote == quote
        )


def make_reply(call: S2ParseCall, records: list[ExtractionRecord]) -> S2ParseReply:
    quotes = [record.quote for record in records]
    reasons = sorted({reason for quote in quotes for reason in quote.escalation_reasons})
    fallback = next(
        (record.fallback_reason for record in records if record.mode == "fallback"), None
    )
    costs = [record.cost_rub for record in records]
    cost = (
        None
        if None in costs
        else sum((Decimal(value) for value in costs if value is not None), Decimal(0))
    )
    return S2ParseReply(
        **call.model_dump(exclude={"payload"}),
        payload=S2ParseQuotesResponse(
            quotes=quotes, need_human=bool(reasons), escalation_reasons=reasons
        ),
        mode="fallback"
        if fallback
        else "llm"
        if any(r.mode == "llm" for r in records)
        else "deterministic",
        fallback_reason=fallback,
        cost_rub=None if cost is None else format(cost, ".8f"),
        prompt_version=next((r.prompt_version for r in records if r.prompt_version), None),
    )
