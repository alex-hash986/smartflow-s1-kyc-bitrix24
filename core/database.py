import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import JsonValue


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: JsonValue) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            if conn.execute("PRAGMA journal_mode = WAL").fetchone()[0] != "wal":
                raise RuntimeError("SQLite WAL unavailable")
            script = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute("PRAGMA legacy_alter_table=ON")
            try:
                # Reserve before inspecting the schema so concurrent startups cannot repeat migration.
                conn.execute("BEGIN IMMEDIATE")
                table = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name='crm_publications' AND type='table'"
                ).fetchone()
                definition = "CREATE TABLE IF NOT EXISTS crm_publications ("
                if table is not None and "purpose <> 'human_review' OR task_required" in table[0]:
                    ddl = definition + script.split(definition, 1)[1].split(") STRICT;", 1)[0]
                    conn.execute("ALTER TABLE crm_publications RENAME TO crm_publications_legacy")
                    conn.execute(ddl + ") STRICT")
                    conn.execute(
                        "INSERT INTO crm_publications SELECT * FROM crm_publications_legacy"
                    )
                    conn.execute("DROP TABLE crm_publications_legacy")
                for statement in script.split(";"):
                    if statement.strip():
                        conn.execute(statement)
                if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    raise RuntimeError("SQLite foreign-key check failed")
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.execute("PRAGMA legacy_alter_table=OFF")
                conn.execute("PRAGMA foreign_keys=ON")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connect() as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
