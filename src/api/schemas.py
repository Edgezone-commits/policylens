"""Request and response shapes for the API.

WHY declare these at all, when the agent already returns a dict: the response
model is the project's public contract, and writing it down is what stops the
API from quietly changing shape every time a node gains a field. It is also
what produces the /docs page, which is the fastest way for anyone to see what
the system returns without reading the agent.

The one rule these models encode: a citation always carries its verdict.
`verified` and `method` are required fields, not optional extras, so there is
no way to serialise a citation that does not say whether it was checked.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """A question to put to the agent."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to answer from the indexed documents.",
        examples=["What is the maximum probation period for a worker?"],
    )
    document: str | None = Field(
        None,
        description=(
            "Restrict the search to one indexed document, by the filename "
            "shown in /documents. Overrides the router's own choice."
        ),
        examples=["Nepal_Labour_Act_2074.pdf"],
    )


class CitationModel(BaseModel):
    """One quoted span, with the verdict on whether the passage really says it."""

    chunk_id: str = Field(..., description="Id of the passage the quote came from.")
    quote: str = Field(..., description="The quoted text, as the model produced it.")
    source_document: str
    page_number: int | None = None
    section: str | None = Field(
        None, description='Best-effort locator, e.g. "Section 13" or "Rule 23".'
    )
    verified: bool = Field(
        ...,
        description=(
            "Whether the quote was confirmed against the passage it cites. "
            "False means the system could not confirm it -- the citation is "
            "still shown, labelled, rather than hidden."
        ),
    )
    method: str = Field(
        ...,
        description=(
            "How the verdict was reached: exact, fuzzy, llm_judge, "
            "judge_unavailable, or unknown_chunk."
        ),
    )
    match_score: float = Field(
        ..., description="Character-level match of quote to passage, 0-100."
    )
    note: str = Field(
        "", description="Why a citation failed, phrased so a reader can check it."
    )


class TraceEntryModel(BaseModel):
    """One step in the agent's reasoning path."""

    node: str
    summary: str
    detail: dict[str, Any] = Field(default_factory=dict)


class UsageModel(BaseModel):
    """What the answer cost, in calls and tokens."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    fallback_calls: int = Field(
        0, description="Calls that had to use the fallback model."
    )
    retries: int = Field(0, description="LLM-level retries, e.g. after a 429.")
    per_call: list[dict[str, Any]] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    """The agent's full answer, including how it got there."""

    question: str
    status: Literal[
        "ANSWERED",
        "PARTIALLY_VERIFIED",
        "INSUFFICIENT_EVIDENCE",
        "OUT_OF_SCOPE",
        "ERROR",
    ] = Field(
        ...,
        description=(
            "ANSWERED: every citation verified. PARTIALLY_VERIFIED: answered, "
            "but some citations failed -- see their notes. "
            "INSUFFICIENT_EVIDENCE: the documents do not support an answer "
            "(a correct outcome, not a failure). OUT_OF_SCOPE: not a question "
            "about these documents. ERROR: the pipeline itself failed."
        ),
    )
    answer: str
    source_document: str | None = None
    citations: list[CitationModel] = Field(default_factory=list)
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Computed from the verified fraction, the retrieval similarity and "
            "the retry count -- never self-reported by the model. 0.0 for any "
            "status that carries no answer."
        ),
    )
    intent: str = ""
    retries: int = Field(0, description="Agent-level retries of the whole retrieval.")
    top_similarity: float = 0.0
    verified_count: int = 0
    unverified_count: int = 0
    trace: list[TraceEntryModel] = Field(default_factory=list)
    usage: UsageModel = Field(default_factory=UsageModel)
    error: str = ""
    latency_seconds: float = 0.0
    query_id: int | None = Field(
        None, description="Row id in the query log, if the answer was logged."
    )


class DocumentModel(BaseModel):
    """An indexed document."""

    source_document: str
    chunk_count: int


class HealthResponse(BaseModel):
    """Whether the system can actually answer a question right now.

    More than a liveness probe on purpose: the three things that stop this
    system working are an empty index, a missing API key and a retired model
    name, and all three are invisible until a question fails. They are
    reported here instead.
    """

    status: Literal["ok", "degraded"] = Field(
        ...,
        description=(
            "degraded means a question would fail or be answered poorly -- "
            "see the fields below for which part is not ready."
        ),
    )
    chunks_indexed: int
    documents: list[DocumentModel] = Field(default_factory=list)
    model: str = Field(..., description="The configured primary LLM.")
    fallback_models: list[str] = Field(
        default_factory=list,
        description=(
            "Models tried in order when the one before it fails. The free-tier "
            "daily quota is charged per model, so the chain is also what makes "
            "more than a handful of questions a day possible on one key."
        ),
    )
    llm_key_present: bool = Field(
        ..., description="Whether the primary model's API key is set."
    )
    embedding_model: str
    similarity_threshold: float
    queries_logged: int = 0


class IngestResponse(BaseModel):
    """What happened to an uploaded document."""

    source_document: str
    pages: int
    chunks: int
    succeeded: bool
    skipped_reason: str | None = None
    chunks_indexed: int = Field(
        0, description="Total chunks in the index after this upload."
    )


class DeleteResponse(BaseModel):
    """What was removed from the index."""

    source_document: str
    chunks_removed: int


class QueryLogEntry(BaseModel):
    """One logged question.

    Lenient on purpose (`extra="allow"`): the log table is append-only and will
    gain columns, and an older client should keep working rather than reject a
    row for carrying a field it has never heard of.
    """

    model_config = {"extra": "allow"}

    id: int
    asked_at: str
    question: str
    status: str
    answer: str | None = None
    confidence: float | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)


class StatsResponse(BaseModel):
    """Aggregates over the query log."""

    total_queries: int
    by_status: dict[str, int] = Field(default_factory=dict)
    abstention_rate: float = Field(
        0.0,
        description=(
            "Share of non-error questions that ended in an abstention. The "
            "headline honesty metric: near zero means the system is answering "
            "things it cannot support, near one means it is refusing things it "
            "could."
        ),
    )
    mean_confidence: float | None = None
    mean_latency_seconds: float | None = None
    mean_llm_calls: float | None = None
    total_tokens: int = 0
    total_retries: int = 0
