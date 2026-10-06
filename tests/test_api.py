"""Tests for the HTTP layer and the query log.

A fake agent is injected, so nothing here loads the embedding model, needs an
API key or touches the network. What is actually being tested is the shell
around the agent: the status codes, the upload path, and the promise that an
abstention is a 200.

That last one is the test that earns its place. If an abstention ever becomes a
4xx or 5xx, the project's most important behaviour starts reading as a failure
in every client, log and dashboard that sees it.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.agent.graph import AgentAnswer
from src.agent.state import Citation
from src.api.app import create_app
from src.db.queries import QueryLog


def citation(verified: bool = True, **overrides: Any) -> Citation:
    base = Citation(
        chunk_id="Nepal_Labour_Act_2074.pdf::p7::c12",
        quote=(
            "A labour may be kept on probation for a maximum period of six "
            "months from the date of appointment."
        ),
        source_document="Nepal_Labour_Act_2074.pdf",
        page_number=7,
        section="Section 13",
        verified=verified,
        method="exact" if verified else "llm_judge",
        match_score=100.0 if verified else 72.0,
        note="" if verified else "the passage does not say 'one year'",
    )
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


class FakeStore:
    def __init__(self, documents: list[str] | None = None, chunks: int = 279) -> None:
        self._documents = (
            documents
            if documents is not None
            else ["Nepal_Labour_Act_2074.pdf", "Apache_License_2.0.txt"]
        )
        self._chunks = chunks
        self.deleted: list[str] = []
        self.added: list[str] = []

    def list_documents(self) -> list[dict[str, Any]]:
        return [
            {"source_document": name, "chunk_count": 100} for name in self._documents
        ]

    def document_names(self) -> list[str]:
        return list(self._documents)

    def count(self) -> int:
        return self._chunks

    def delete_document(self, source_document: str) -> int:
        self._documents.remove(source_document)
        self.deleted.append(source_document)
        return 42

    def add_chunks(self, chunks: list[Any]) -> int:
        self.added.extend(chunk.metadata["chunk_id"] for chunk in chunks)
        self._chunks += len(chunks)
        return len(chunks)


class FakeAgent:
    """Returns a prepared answer and records what it was asked."""

    def __init__(self, answer: AgentAnswer | None = None, store: FakeStore | None = None):
        self.store = store or FakeStore()
        self.answer = answer or AgentAnswer(
            question="",
            status="ANSWERED",
            answer="A worker may be kept on probation for at most six months.",
            citations=[citation()],
            confidence=0.88,
            intent="DOCUMENT_QA",
            top_similarity=0.71,
            verified_count=1,
        )
        self.asked: list[tuple[str, str | None]] = []

    def ask(self, question: str, source_document: str | None = None) -> AgentAnswer:
        self.asked.append((question, source_document))
        self.answer.question = question
        self.answer.source_document = source_document
        return self.answer


@pytest.fixture
def log(tmp_path) -> QueryLog:
    return QueryLog(tmp_path / "queries.db")


def build_client(agent: FakeAgent, log: QueryLog) -> TestClient:
    return TestClient(create_app(agent=agent, query_log=log))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------


def test_an_answer_comes_back_with_its_citations_and_verdicts(log):
    agent = FakeAgent()
    with build_client(agent, log) as client:
        response = client.post("/ask", json={"question": "Probation period?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ANSWERED"
    assert body["confidence"] == 0.88
    assert body["citations"][0]["verified"] is True
    assert body["citations"][0]["section"] == "Section 13"
    assert body["query_id"] == 1
    assert body["latency_seconds"] >= 0.0


def test_an_abstention_is_a_200(log):
    """The whole project in one assertion.

    An abstention is a correct, reasoned response. A 4xx would tell every
    client the request was wrong; a 5xx would make honesty look like an
    outage.
    """
    agent = FakeAgent(
        AgentAnswer(
            question="",
            status="INSUFFICIENT_EVIDENCE",
            answer="I could not find anything in the indexed documents.",
        )
    )
    with build_client(agent, log) as client:
        response = client.post("/ask", json={"question": "Who won the World Cup?"})

    assert response.status_code == 200
    assert response.json()["status"] == "INSUFFICIENT_EVIDENCE"
    assert response.json()["citations"] == []


def test_an_out_of_scope_question_is_also_a_200(log):
    agent = FakeAgent(
        AgentAnswer(question="", status="OUT_OF_SCOPE", answer="Not these documents.")
    )
    with build_client(agent, log) as client:
        response = client.post("/ask", json={"question": "Recipe for cake?"})

    assert response.status_code == 200
    assert response.json()["status"] == "OUT_OF_SCOPE"


def test_a_partially_verified_answer_keeps_the_failing_citation_visible(log):
    agent = FakeAgent(
        AgentAnswer(
            question="",
            status="PARTIALLY_VERIFIED",
            answer="Probation is six months; gratuity is 10%.",
            citations=[citation(), citation(verified=False)],
            confidence=0.52,
            verified_count=1,
            unverified_count=1,
        )
    )
    with build_client(agent, log) as client:
        body = client.post("/ask", json={"question": "Probation and gratuity?"}).json()

    failing = [item for item in body["citations"] if not item["verified"]]
    assert len(failing) == 1
    # The note is what makes the failure checkable rather than just flagged.
    assert "one year" in failing[0]["note"]


def test_a_pipeline_failure_is_a_503_and_is_still_logged(log):
    """Retrying later might work, and a failed run is what you want to inspect."""
    agent = FakeAgent(
        AgentAnswer(
            question="",
            status="ERROR",
            answer="",
            error="No documents are indexed.",
        )
    )
    with build_client(agent, log) as client:
        response = client.post("/ask", json={"question": "Probation period?"})

    assert response.status_code == 503
    assert "No documents are indexed" in response.json()["detail"]
    assert log.recent()[0]["status"] == "ERROR"


def test_scoping_to_a_document_is_passed_through(log):
    agent = FakeAgent()
    with build_client(agent, log) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Probation period?",
                "document": "Nepal_Labour_Act_2074.pdf",
            },
        )

    assert response.status_code == 200
    assert agent.asked == [("Probation period?", "Nepal_Labour_Act_2074.pdf")]
    assert response.json()["source_document"] == "Nepal_Labour_Act_2074.pdf"


def test_scoping_to_an_unindexed_document_is_a_404_that_says_what_exists(log):
    agent = FakeAgent()
    with build_client(agent, log) as client:
        response = client.post(
            "/ask", json={"question": "Probation period?", "document": "typo.pdf"}
        )

    assert response.status_code == 404
    assert "Nepal_Labour_Act_2074.pdf" in response.json()["detail"]
    assert agent.asked == []  # no LLM calls spent on a typo


@pytest.mark.parametrize("question", ["", "  ", "ab"])
def test_a_question_too_short_to_mean_anything_is_rejected(log, question):
    agent = FakeAgent()
    with build_client(agent, log) as client:
        response = client.post("/ask", json={"question": question})

    assert response.status_code == 422
    assert agent.asked == []


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def test_documents_are_listed_with_their_chunk_counts(log):
    with build_client(FakeAgent(), log) as client:
        body = client.get("/documents").json()

    assert [item["source_document"] for item in body] == [
        "Nepal_Labour_Act_2074.pdf",
        "Apache_License_2.0.txt",
    ]
    assert body[0]["chunk_count"] == 100


def test_uploading_a_text_document_indexes_it(log, tmp_path, monkeypatch):
    from src import config as config_module

    # Redirect the keep-a-copy directory so the test does not write into the
    # real data/raw/.
    monkeypatch.setattr(config_module, "RAW_DOCS_DIR", tmp_path / "raw")

    agent = FakeAgent(store=FakeStore(chunks=0))
    content = (
        "13. Probation Period: A labour may be kept on probation for a maximum "
        "period of six months from the date of appointment. The employer may "
        "terminate the contract if the work is not satisfactory.\n"
    ) * 5

    with build_client(agent, log) as client:
        response = client.post(
            "/documents",
            files={"file": ("policy.txt", io.BytesIO(content.encode()), "text/plain")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] is True
    assert body["chunks"] > 0
    assert body["chunks_indexed"] == body["chunks"]
    # Kept only because it indexed, so a re-index does not need the file again.
    assert (tmp_path / "raw" / "policy.txt").exists()


def test_an_unsupported_file_type_is_a_415(log):
    with build_client(FakeAgent(), log) as client:
        response = client.post(
            "/documents",
            files={"file": ("notes.docx", io.BytesIO(b"PK\x03\x04"), "application/zip")},
        )

    assert response.status_code == 415
    assert ".pdf" in response.json()["detail"]


def test_an_empty_upload_is_rejected(log):
    with build_client(FakeAgent(), log) as client:
        response = client.post(
            "/documents", files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
        )

    assert response.status_code == 422


def test_a_document_with_no_extractable_text_is_reported_not_kept(
    log, tmp_path, monkeypatch
):
    """A scanned PDF indexes nothing, and must not sit in data/raw looking indexed."""
    from src import config as config_module

    monkeypatch.setattr(config_module, "RAW_DOCS_DIR", tmp_path / "raw")

    with build_client(FakeAgent(store=FakeStore(chunks=0)), log) as client:
        response = client.post(
            "/documents",
            files={"file": ("blank.txt", io.BytesIO(b"   \n  \n"), "text/plain")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] is False
    assert body["skipped_reason"]
    assert not (tmp_path / "raw" / "blank.txt").exists()


def test_deleting_a_document_removes_its_chunks(log):
    agent = FakeAgent()
    with build_client(agent, log) as client:
        response = client.delete("/documents/Apache_License_2.0.txt")

    assert response.status_code == 200
    assert response.json()["chunks_removed"] == 42
    assert agent.store.deleted == ["Apache_License_2.0.txt"]


def test_deleting_a_document_that_is_not_there_is_a_404(log):
    with build_client(FakeAgent(), log) as client:
        assert client.delete("/documents/nope.pdf").status_code == 404


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def test_health_reports_ok_when_the_index_and_the_key_are_both_ready(log, monkeypatch):
    from src.api import app as app_module

    monkeypatch.setattr(
        app_module.litellm,
        "validate_environment",
        lambda model: {"keys_in_environment": True},
    )

    with build_client(FakeAgent(), log) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["chunks_indexed"] == 279
    assert body["llm_key_present"] is True
    assert len(body["documents"]) == 2


def test_health_is_degraded_when_the_index_is_empty(log, monkeypatch):
    """The failure that is otherwise invisible until a question comes back empty."""
    from src.api import app as app_module

    monkeypatch.setattr(
        app_module.litellm,
        "validate_environment",
        lambda model: {"keys_in_environment": True},
    )

    agent = FakeAgent(store=FakeStore(documents=[], chunks=0))
    with build_client(agent, log) as client:
        body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["chunks_indexed"] == 0


def test_health_is_degraded_without_an_api_key(log, monkeypatch):
    from src.api import app as app_module

    monkeypatch.setattr(
        app_module.litellm,
        "validate_environment",
        lambda model: {"keys_in_environment": False},
    )

    with build_client(FakeAgent(), log) as client:
        body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["llm_key_present"] is False


def test_recent_queries_come_back_newest_first(log):
    agent = FakeAgent()
    with build_client(agent, log) as client:
        client.post("/ask", json={"question": "First question?"})
        client.post("/ask", json={"question": "Second question?"})
        body = client.get("/queries", params={"limit": 5}).json()

    assert [row["question"] for row in body] == [
        "Second question?",
        "First question?",
    ]
    assert body[0]["citations"][0]["verified"] is True


def test_stats_report_the_abstention_rate(log):
    answered = FakeAgent()
    abstained = FakeAgent(
        AgentAnswer(question="", status="INSUFFICIENT_EVIDENCE", answer="No evidence.")
    )

    with build_client(answered, log) as client:
        client.post("/ask", json={"question": "Probation period?"})
    with build_client(abstained, log) as client:
        client.post("/ask", json={"question": "Who won the World Cup?"})
        body = client.get("/stats").json()

    assert body["total_queries"] == 2
    assert body["by_status"] == {"ANSWERED": 1, "INSUFFICIENT_EVIDENCE": 1}
    assert body["abstention_rate"] == 0.5
    # Averaged over answered questions only: including abstentions, whose
    # confidence is 0.0 by definition, would just measure the abstention rate.
    assert body["mean_confidence"] == 0.88


# ---------------------------------------------------------------------------
# The query log itself
# ---------------------------------------------------------------------------


def test_the_log_round_trips_citations_and_the_trace(log):
    answer = AgentAnswer(
        question="Probation period?",
        status="ANSWERED",
        answer="Six months.",
        citations=[citation()],
        confidence=0.88,
        trace=[{"node": "router", "summary": "Classified as DOCUMENT_QA", "detail": {}}],
    )
    row_id = log.record(answer, latency_seconds=3.5)

    stored = log.get(row_id)
    assert stored is not None
    assert stored["question"] == "Probation period?"
    assert stored["latency_seconds"] == 3.5
    assert stored["citations"][0]["chunk_id"] == citation()["chunk_id"]
    assert stored["trace"][0]["node"] == "router"


def test_a_logging_failure_does_not_cost_the_user_their_answer(log, monkeypatch):
    """The answer already cost several LLM calls; a full disk must not void it."""
    import sqlite3

    def explode(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(log, "_connect", explode)

    agent = FakeAgent()
    with build_client(agent, log) as client:
        response = client.post("/ask", json={"question": "Probation period?"})

    assert response.status_code == 200
    assert response.json()["status"] == "ANSWERED"
    assert response.json()["query_id"] is None  # not logged, and says so


def test_an_unreadable_json_column_does_not_break_the_log_reader(log):
    """An old or half-written row should still give up its scalar columns."""
    log.record(AgentAnswer(question="q", status="ANSWERED", answer="a"))
    with log._connect() as connection:
        connection.execute("UPDATE queries SET citations_json = '{not json'")

    row = log.recent()[0]
    assert row["question"] == "q"
    assert row["citations"] == []


def test_in_memory_databases_are_refused_with_an_explanation():
    """Silently losing every write would be the worse alternative."""
    with pytest.raises(ValueError, match="connection per"):
        QueryLog(":memory:")


def test_stats_on_an_empty_log_are_zeroes_not_crashes(log):
    stats = log.stats()

    assert stats["total_queries"] == 0
    assert stats["mean_confidence"] is None  # no rows to average, not 0.0
    assert stats["abstention_rate"] == 0.0
