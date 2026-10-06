"""The state passed between graph nodes, and the shapes it carries.

WHY a single explicit state object rather than nodes calling each other: every
node reads and writes this one dict, so the whole reasoning path of an answer
is inspectable at any point. That is what makes the agent auditable -- the API
returns the trace, and the evaluation harness reads status and confidence from
the same fields the UI shows.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

# What the router decided the question is.
Intent = Literal["DOCUMENT_QA", "COMPARISON", "OUT_OF_SCOPE"]

# How the run ended. These are the only four outcomes, and each is a distinct
# thing to tell the user:
#   ANSWERED              -- an answer, with at least one verified citation.
#   PARTIALLY_VERIFIED    -- an answer, but some citations failed verification;
#                            shown with the failures marked, not hidden.
#   INSUFFICIENT_EVIDENCE -- the corpus does not support an answer. Not a
#                            failure: this is the behaviour the retrieval
#                            threshold exists to produce.
#   OUT_OF_SCOPE          -- the question is not about the indexed documents.
#   ERROR                 -- the pipeline itself broke (no LLM key, say).
Status = Literal[
    "ANSWERED",
    "PARTIALLY_VERIFIED",
    "INSUFFICIENT_EVIDENCE",
    "OUT_OF_SCOPE",
    "ERROR",
]


class Citation(TypedDict, total=False):
    """One quoted span, plus the verdict on whether it is really there.

    `verified` is the whole point of the project: a citation the system could
    not confirm is still shown, but labelled, so a reader never has to assume
    a quote is genuine.
    """

    chunk_id: str
    quote: str
    source_document: str
    page_number: int | None
    section: str | None
    verified: bool
    # "exact" | "fuzzy" | "llm_judge" | "quote_too_short" | "unknown_chunk" |
    # "not_in_chunk" -- how the verdict was reached, so a reader can weigh it.
    method: str
    match_score: float
    note: str


class RetrievedChunk(TypedDict, total=False):
    """A chunk as the vector store returns it (see VectorStore.retrieve)."""

    chunk_id: str
    text: str
    source_document: str
    page_number: int | None
    section: str | None
    sections: str
    similarity: float


class TraceEntry(TypedDict, total=False):
    """One step in the reasoning path, as shown to the user.

    Kept as structured entries rather than log lines because the UI groups them
    by node and the evaluation harness counts them (e.g. how often a retry
    happened), neither of which works against free text.
    """

    node: str
    summary: str
    detail: dict[str, Any]


class AgentState(TypedDict, total=False):
    """Everything one question accumulates as it moves through the graph."""

    # -- input ----------------------------------------------------------
    question: str
    # Restrict retrieval to one document, when the caller (not the router)
    # asks for it -- e.g. the UI's document picker.
    source_document: str | None

    # -- router ---------------------------------------------------------
    intent: Intent
    # Documents the router thinks the question is about; [] means "all".
    target_documents: list[str]
    # The question rewritten in the documents' own vocabulary. Retrieval uses
    # this; the analyzer always answers the user's original wording.
    search_query: str
    router_reason: str

    # -- retriever ------------------------------------------------------
    chunks: list[RetrievedChunk]
    top_similarity: float
    retrieval_top_k: int

    # -- analyzer -------------------------------------------------------
    answer: str
    citations: list[Citation]
    # The analyzer's own verdict that the evidence does not answer the
    # question, which is independent of the retrieval threshold: chunks can
    # clear it and still be about the wrong thing.
    analyzer_abstained: bool

    # -- verifier -------------------------------------------------------
    verified_count: int
    unverified_count: int
    verified_ratio: float
    # Set when the graph was built without the verifier. Distinguishes
    # "nothing could be confirmed" from "nothing was checked", which the
    # responder must not confuse: the first is grounds to withdraw the
    # answer, the second is the evaluation's control arm.
    verification_skipped: bool

    # -- loop control ---------------------------------------------------
    retry_count: int
    # Set when a node cannot continue at all (LLM unreachable, empty index).
    error: str

    # -- bookkeeping ----------------------------------------------------
    # One entry per LLM call (see llm.call_record), concatenated across nodes.
    # Lives in the state rather than in a client attribute so that two
    # concurrent questions -- two API requests sharing one LLMClient -- cannot
    # mix up each other's counts.
    llm_calls: Annotated[list[dict[str, Any]], operator.add]

    # -- output ---------------------------------------------------------
    status: Status
    confidence: float
    # `operator.add` so each node contributes only its own new entries and the
    # graph concatenates them. With the default (overwrite) reducer every node
    # would have to carry the whole list forward, and a node that forgot would
    # silently erase the history before it.
    trace: Annotated[list[TraceEntry], operator.add]


def new_state(question: str, source_document: str | None = None) -> AgentState:
    """Build the initial state for a question.

    Every field the graph reads is given a value here, so no node has to guard
    against a missing key (and `state["retry_count"] + 1` cannot raise).
    """
    return AgentState(
        question=question.strip(),
        source_document=source_document,
        intent="DOCUMENT_QA",
        target_documents=[],
        search_query=question.strip(),
        router_reason="",
        chunks=[],
        top_similarity=0.0,
        retrieval_top_k=0,
        answer="",
        citations=[],
        analyzer_abstained=False,
        verified_count=0,
        unverified_count=0,
        verified_ratio=0.0,
        retry_count=0,
        error="",
        llm_calls=[],
        status="ERROR",  # replaced by the responder; visible if it never ran
        confidence=0.0,
        trace=[],
    )


def trace_entry(node: str, summary: str, **detail: Any) -> TraceEntry:
    """Shorthand for one reasoning-path entry."""
    return TraceEntry(node=node, summary=summary, detail=detail)
