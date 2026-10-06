"""Tests for the retrieval layer, especially the relevance gate.

This layer had no tests until now, and the gate in `VectorStore.retrieve` is
the single most consequential piece of logic in the system: it is the only
place that decides whether evidence exists at all, and therefore the
foundation of the agent's ability to abstain. Without it a vector search
always "succeeds" -- ask about chocolate cake and it hands back the five
least-unrelated paragraphs of labour law, which the model will then answer
from, with citations, confidently.

A FAKE EMBEDDING MODEL is injected throughout. That is not a shortcut, it is
what makes these tests meaningful:

  * Determinism. Similarity is arithmetic here, so a threshold test asserts
    the gate's logic rather than a model's opinion of two sentences. With real
    embeddings, "is 0.31 below the threshold" depends on MiniLM's weights and
    the test would drift when the model is swapped.
  * Speed. Loading all-MiniLM-L6-v2 takes ~45 seconds; these run in under a
    second, so they can be run on every change.

What is deliberately NOT tested here: whether MiniLM scores a real legal
question above 0.32. That is a property of the model and corpus, not of this
code, and it is measured where it belongs -- scripts/calibrate_threshold.py
fits the threshold on real score distributions, and scripts/evaluate.py
--recall-only reports Recall@5 against the live index.
"""

from __future__ import annotations

import math

import pytest
from langchain_core.documents import Document

from src.retrieval.store import VectorStore

ACT = "Nepal_Labour_Act_2074.pdf"
RULES = "Nepal_Labour_Rules_2075_Summary_PKF.pdf"


class FakeEmbeddings:
    """Maps text to a point on the unit circle, so cosine similarity is exact.

    Each registered text gets an angle; similarity between two texts is
    cos(difference). An unregistered text lands at pi, diametrically opposite
    angle 0, which is how an off-topic query is simulated without relying on a
    real model to find it off-topic.

    pi and not pi/2: a right angle is only 0 similarity to angle 0, and is
    0.985 similar to a chunk registered at 1.4 -- so "unregistered" would have
    come back as the closest match to the off-topic query. At pi the
    similarity to every registered angle here (0.0, 1.2, 1.4) is negative,
    which is what "unrelated to everything" has to mean for the gate test to
    be testing the gate.
    """

    def __init__(self, angles: dict[str, float]) -> None:
        self.angles = angles

    def _vector(self, text: str) -> list[float]:
        angle = self.angles.get(text.strip(), math.pi)
        return [math.cos(angle), math.sin(angle)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


# Texts long enough to be plausible chunks, keyed by the angle they embed to.
PROBATION = (
    "13. Probation Period: A labour may be kept on probation for a maximum "
    "period of six months from the date of appointment."
)
GRATUITY = (
    "53. To receive gratuity: The employer shall deduct an amount equivalent "
    "to eight point three three percent of the basic remuneration."
)
BASIC_PAY = (
    "Rule 77. Basic remuneration of a regular labor shall be equivalent to "
    "60% of last 3 months average gross salary."
)

ANGLES = {
    "probation query": 0.0,
    PROBATION: 0.0,            # identical -> similarity 1.0
    GRATUITY: 1.2,             # ~0.36 to the probation query
    BASIC_PAY: 1.4,            # ~0.17 to the probation query
    "gratuity query": 1.2,
}


@pytest.fixture
def store(tmp_path) -> VectorStore:
    """A real Chroma store on a temp path, with a fake embedder.

    Real Chroma because the cosine-distance configuration and the metadata
    round-trip are part of what is being tested; a mocked collection would
    assert how this code calls Chroma rather than what Chroma gives back.
    """
    store = VectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_docs",
        embedding_model=FakeEmbeddings(ANGLES),  # type: ignore[arg-type]
    )
    store.add_chunks(
        [
            Document(
                page_content=PROBATION,
                metadata={
                    "source_document": ACT,
                    "page_number": 7,
                    "chunk_index": 0,
                    "chunk_id": "act::p7::c0",
                    "section": "Section 13",
                },
            ),
            Document(
                page_content=GRATUITY,
                metadata={
                    "source_document": ACT,
                    "page_number": 20,
                    "chunk_index": 1,
                    "chunk_id": "act::p20::c1",
                    "section": "Section 53",
                },
            ),
            Document(
                page_content=BASIC_PAY,
                metadata={
                    "source_document": RULES,
                    "page_number": 12,
                    "chunk_index": 0,
                    "chunk_id": "rules::p12::c0",
                    "section": "Rule 77",
                },
            ),
        ]
    )
    return store


# ---------------------------------------------------------------------------
# The gate: does evidence exist at all
# ---------------------------------------------------------------------------


def test_nothing_is_returned_when_the_best_chunk_is_below_the_threshold(store):
    """An unregistered query sits at 0 similarity to everything."""
    results = store.retrieve("what is the recipe for chocolate cake", threshold=0.32)

    assert results == []


def test_a_relevant_query_returns_its_chunk(store):
    results = store.retrieve("probation query", threshold=0.32)

    assert results
    assert results[0]["chunk_id"] == "act::p7::c0"
    assert results[0]["similarity"] == pytest.approx(1.0, abs=0.01)


def test_weak_chunks_are_filtered_out_even_once_the_gate_passes(store):
    """A [1.00, 0.36, 0.17] set is better truncated than passed on whole.

    The weak chunks add no evidence but do give the model extra material to
    misattribute a citation to.
    """
    results = store.retrieve("probation query", top_k=5, threshold=0.30)

    returned = {r["chunk_id"] for r in results}
    assert "act::p7::c0" in returned      # 1.00
    assert "act::p20::c1" in returned     # ~0.36, above
    assert "rules::p12::c0" not in returned  # ~0.17, below


def test_a_threshold_of_zero_disables_the_gate(store):
    """Which is what the recall measurement needs: ranking, not the gate."""
    results = store.retrieve("probation query", top_k=5, threshold=0.0)

    assert len(results) == 3


def test_results_come_back_ordered_by_descending_similarity(store):
    results = store.retrieve("probation query", top_k=5, threshold=0.0)

    scores = [r["similarity"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_how_many_chunks_come_back(store):
    results = store.retrieve("probation query", top_k=1, threshold=0.0)

    assert len(results) == 1


def test_an_empty_query_returns_nothing_without_raising(store):
    """The API rejects these, but the store must not crash if one arrives."""
    assert store.retrieve("   ", threshold=0.0) == []


# ---------------------------------------------------------------------------
# Scoping and metadata
# ---------------------------------------------------------------------------


def test_scoping_to_one_document_excludes_the_others(store):
    results = store.retrieve(
        "probation query", top_k=5, threshold=0.0, source_document=RULES
    )

    assert results
    assert {r["source_document"] for r in results} == {RULES}


def test_every_result_carries_the_metadata_a_citation_needs(store):
    results = store.retrieve("probation query", threshold=0.0)

    result = results[0]
    assert result["source_document"] == ACT
    assert result["page_number"] == 7
    assert result["section"] == "Section 13"
    assert result["chunk_id"] == "act::p7::c0"
    assert results[0]["text"].startswith("13. Probation Period")


def test_a_chunk_can_be_looked_back_up_by_id(store):
    """The verifier re-reads the cited passage BY ID rather than trusting state.

    If this lookup breaks, every citation becomes unverifiable at once.
    """
    chunk = store.get_chunk("act::p20::c1")

    assert chunk is not None
    assert "eight point three three percent" in chunk["text"]
    assert chunk["section"] == "Section 53"


def test_looking_up_an_unknown_chunk_id_returns_none(store):
    """A fabricated chunk_id must not resolve to some other passage."""
    assert store.get_chunk("act::p99::c99") is None


# ---------------------------------------------------------------------------
# Inventory and deletion
# ---------------------------------------------------------------------------


def test_documents_are_listed_with_their_chunk_counts(store):
    listed = {d["source_document"]: d["chunk_count"] for d in store.list_documents()}

    assert listed == {ACT: 2, RULES: 1}
    assert store.count() == 3
    assert sorted(store.document_names()) == sorted([ACT, RULES])


def test_deleting_a_document_removes_only_its_chunks(store):
    removed = store.delete_document(RULES)

    assert removed == 1
    assert store.count() == 2
    assert store.document_names() == [ACT]
    assert store.get_chunk("rules::p12::c0") is None
    assert store.get_chunk("act::p7::c0") is not None


def test_re_adding_the_same_chunk_id_overwrites_rather_than_duplicates(store):
    """Deterministic ids exist for this: a re-ingest must not double the index."""
    before = store.count()

    store.add_chunks(
        [
            Document(
                page_content=PROBATION,
                metadata={
                    "source_document": ACT,
                    "page_number": 7,
                    "chunk_index": 0,
                    "chunk_id": "act::p7::c0",
                    "section": "Section 13",
                },
            )
        ]
    )

    assert store.count() == before


def test_adding_no_chunks_is_a_no_op(store):
    assert store.add_chunks([]) == 0
    assert store.count() == 3
