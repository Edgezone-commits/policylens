"""SQLite log of every question the agent has answered.

WHY log at all, for a tool with one user:

  * The evaluation needs a record it did not have to re-run. A 50-question
    batch costs real time and a slice of the daily quota; being able to ask
    "what did the system actually say last Tuesday" without paying for it
    again is the difference between measuring and guessing.
  * The confidence formula in responder.py is a claim that has to be checked
    against outcomes. That check is only possible if the inputs (verified
    ratio, retrieval similarity, retries) are stored next to the answer.
  * An abstention is the system's most interesting output, and the one a
    reviewer will want to inspect. Counting how often it happens, and on what,
    needs persistence.

WHY SQLite and not a JSONL file: the questions worth asking of this log are
aggregates -- abstention rate by document, mean confidence by status, which
questions needed a retry -- and those are one SQL query each against a table
and a file-scan-plus-parse each against a log file. It is also already a
dependency, being in the standard library.

Rows are written best-effort: a logging failure must never cost the user an
answer they already have. See record().
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src import config
from src.agent.graph import AgentAnswer

logger = logging.getLogger(__name__)

# One table, wide, with the JSON blobs kept as text. WHY not normalised tables
# for citations and trace entries: nothing queries inside them -- they are read
# back whole, to redisplay an answer -- while every question that matters is
# about the scalar columns. Three tables and two joins would buy nothing and
# cost every reader of this file five minutes.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    question          TEXT    NOT NULL,
    source_document   TEXT,
    status            TEXT    NOT NULL,
    intent            TEXT,
    answer            TEXT,
    confidence        REAL,
    top_similarity    REAL,
    retries           INTEGER NOT NULL DEFAULT 0,
    verified_count    INTEGER NOT NULL DEFAULT 0,
    unverified_count  INTEGER NOT NULL DEFAULT 0,
    llm_calls         INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    latency_seconds   REAL,
    citations_json    TEXT    NOT NULL DEFAULT '[]',
    trace_json        TEXT    NOT NULL DEFAULT '[]',
    error             TEXT
);

-- The log is almost always read newest-first, and "how often do we abstain"
-- groups by status. Both are cheap at this scale anyway; the indexes cost
-- nothing and keep that true if the evaluation writes thousands of rows.
CREATE INDEX IF NOT EXISTS idx_queries_asked_at ON queries (asked_at DESC);
CREATE INDEX IF NOT EXISTS idx_queries_status ON queries (status);
"""


class QueryLog:
    """Append-only record of answered questions, backed by one SQLite file."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if str(db_path) == ":memory:":
            # Rejected with an explanation rather than silently misbehaving:
            # this class opens a connection per operation (see _connect), and
            # each connection to ":memory:" gets its OWN empty database, so the
            # schema would vanish between the write and the read. A test wants
            # tmp_path / "queries.db", which exercises the real code path.
            raise ValueError(
                "QueryLog cannot use ':memory:' -- it opens one connection per "
                "operation, so an in-memory database would be empty every "
                "time. Pass a file path (in a test, tmp_path)."
            )
        self.db_path = Path(db_path or config.SQLITE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection for one operation, then close it.

        A connection per operation rather than one long-lived connection:
        FastAPI runs sync endpoints in a thread pool, and a SQLite connection
        may not be shared across threads. Opening one is microseconds against
        an answer that took seconds of LLM latency, so the simple option is
        also the right one here.
        """
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        # Rows as mappings, so callers get dicts and never index by position --
        # which is what silently breaks when a column is added.
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connect() as connection:
            # WAL lets the Streamlit UI read the log while the API is writing
            # to it. With the default journal, the reader would hit "database
            # is locked" exactly when a question was being answered.
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)
        logger.debug("Query log ready at %s", self.db_path)

    # -- writing ------------------------------------------------------------

    def record(self, answer: AgentAnswer, latency_seconds: float | None = None) -> int | None:
        """Store one answered question. Returns its row id, or None on failure.

        Deliberately swallows its own errors. The caller has already produced a
        verified answer at the cost of several LLM calls; losing it to a full
        disk or a locked database would be absurd. A failure is logged as a
        warning and the answer is still returned to the user.
        """
        usage = answer.usage
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO queries (
                        question, source_document, status, intent, answer,
                        confidence, top_similarity, retries, verified_count,
                        unverified_count, llm_calls, total_tokens,
                        latency_seconds, citations_json, trace_json, error
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        answer.question,
                        # Which document the question was SCOPED to, if any --
                        # not where the answer came from, which is in the
                        # citations. Kept separate because "the user restricted
                        # the search" and "the evidence happened to come from
                        # one document" are different facts.
                        answer.source_document,
                        answer.status,
                        answer.intent,
                        answer.answer,
                        answer.confidence,
                        answer.top_similarity,
                        answer.retries,
                        answer.verified_count,
                        answer.unverified_count,
                        usage.calls,
                        usage.prompt_tokens + usage.completion_tokens,
                        latency_seconds,
                        json.dumps([dict(citation) for citation in answer.citations]),
                        json.dumps([dict(entry) for entry in answer.trace]),
                        answer.error or None,
                    ),
                )
                return int(cursor.lastrowid or 0)
        except (sqlite3.Error, OSError, TypeError) as error:
            logger.warning("Could not log the query: %s", error)
            return None

    # -- reading ------------------------------------------------------------

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """The most recent questions, newest first, with JSON columns decoded."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM queries ORDER BY id DESC LIMIT ?", (max(1, limit),)
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get(self, query_id: int) -> dict[str, Any] | None:
        """One logged question by id."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM queries WHERE id = ?", (query_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def stats(self) -> dict[str, Any]:
        """Aggregates over the whole log.

        The abstention rate is the headline number: a system that never
        abstains is not being honest, and one that always abstains is not
        useful, so this is the figure to watch while the thresholds are tuned.
        """
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
            by_status = {
                row["status"]: row["count"]
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM queries GROUP BY status"
                ).fetchall()
            }
            # Averaged over answered questions only. Including abstentions,
            # whose confidence is fixed at 0.0 by definition, would drag the
            # mean towards the abstention rate and measure nothing.
            answered = connection.execute(
                """
                SELECT AVG(confidence) AS mean_confidence,
                       AVG(latency_seconds) AS mean_latency,
                       AVG(llm_calls) AS mean_llm_calls,
                       SUM(total_tokens) AS total_tokens,
                       SUM(retries) AS total_retries
                FROM queries
                WHERE status IN ('ANSWERED', 'PARTIALLY_VERIFIED')
                """
            ).fetchone()

        abstentions = by_status.get("INSUFFICIENT_EVIDENCE", 0) + by_status.get(
            "OUT_OF_SCOPE", 0
        )
        return {
            "total_queries": total,
            "by_status": by_status,
            # Errors are excluded from the denominator: a run that broke made no
            # decision about evidence, so counting it either way would be wrong.
            "abstention_rate": round(
                abstentions / max(1, total - by_status.get("ERROR", 0)), 3
            ),
            "mean_confidence": _round_or_none(answered["mean_confidence"], 3),
            "mean_latency_seconds": _round_or_none(answered["mean_latency"], 2),
            "mean_llm_calls": _round_or_none(answered["mean_llm_calls"], 2),
            "total_tokens": int(answered["total_tokens"] or 0),
            "total_retries": int(answered["total_retries"] or 0),
        }

    # -- maintenance --------------------------------------------------------

    def clear(self) -> int:
        """Delete every row. Used by tests and before a clean evaluation run."""
        with self._connect() as connection:
            before = connection.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
            connection.execute("DELETE FROM queries")
        logger.info("Cleared %d row(s) from the query log", before)
        return int(before)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Decode a row, turning the JSON text columns back into structures.

        A malformed blob yields an empty list rather than an exception: the
        scalar columns of an old row are still worth reading, and a log viewer
        should not break on one bad write.
        """
        record = dict(row)
        for column, key in (("citations_json", "citations"), ("trace_json", "trace")):
            raw = record.pop(column, None)
            try:
                record[key] = json.loads(raw) if raw else []
            except (TypeError, json.JSONDecodeError):
                logger.warning("Row %s has an unreadable %s", record.get("id"), column)
                record[key] = []
        return record


def _round_or_none(value: Any, digits: int) -> float | None:
    """Round a SQL average, which is NULL when there are no matching rows."""
    return None if value is None else round(float(value), digits)
