"""The agent graph: five nodes, three exits, one retry loop.

    router ──OUT_OF_SCOPE──────────────────────────────────┐
      │                                                    │
      └─▶ retriever ──no evidence (nothing new to try)──────┤
            │   ▲                                          │
            │   └──────────────── retry ◀──┐               │
            ▼                              │               │
          analyzer ──abstained─────────────┤               │
            │                              │               │
            ▼                              │               │
          verifier ──nothing verified──────┘               │
            │                                              │
            └─▶ responder ◀─────────────────────────────────┘
                   │
                  END

WHY LangGraph rather than a sequential chain: the retry is a genuine cycle with
conditional entry -- verifier can send control BACKWARDS to retriever, and
whether it does depends on state (how many citations were verified, how many
retries have already been spent). A linear chain cannot express that; it would
have to be an explicit while-loop around the whole pipeline, which is where the
state bookkeeping stops being inspectable.

WHY retry at all, given retrieval is deterministic: the retry does not repeat
the same search. It widens top_k (5 -> 10) and adds the user's original wording
alongside the router's rewrite, so a passage that sat at rank 7, or that only
matches the user's phrasing, gets a second chance. The two conditions that
trigger it -- the analyzer finding the passages off-target, and every citation
failing verification -- are both evidence that retrieval, not generation, was
the weak link.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src import config
from src.agent.llm import LLMClient, LLMUsage, get_llm_client
from src.agent.nodes.analyzer import build_analyzer_node
from src.agent.nodes.responder import build_responder_node
from src.agent.nodes.retriever import build_retriever_node
from src.agent.nodes.router import build_router_node
from src.agent.nodes.verifier import build_verifier_node
from src.agent.state import AgentState, Citation, TraceEntry, new_state, trace_entry
from src.retrieval.store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------
# Each decision function returns the name of the next node. They are kept
# together, and kept free of side effects, so the whole control flow of the
# agent can be read in one screen -- which is also what makes it testable
# without running the graph.


def _retries_left(state: AgentState) -> bool:
    return state.get("retry_count", 0) < config.MAX_RETRIES


def after_router(state: AgentState) -> Literal["retriever", "responder"]:
    if state.get("error") or state.get("intent") == "OUT_OF_SCOPE":
        return "responder"
    return "retriever"


def after_retriever(state: AgentState) -> Literal["analyzer", "retry", "responder"]:
    if state.get("chunks"):
        return "analyzer"

    # Nothing cleared the threshold. A retry is only worth a round trip if it
    # would actually search for something different: the retry adds the user's
    # original wording, so it helps when the router rewrote the question, and
    # is pure waste when the rewrite was the question. Widening top_k alone
    # cannot help here -- the gate compares the BEST match to the threshold,
    # and more candidates do not improve the best one.
    original = (state.get("question") or "").strip().lower()
    rewritten = (state.get("search_query") or "").strip().lower()
    if state.get("retry_count", 0) == 0 and rewritten and rewritten != original:
        return "retry"
    return "responder"


def after_analyzer(state: AgentState) -> Literal["verifier", "retry", "responder"]:
    if state.get("error"):
        return "responder"
    if state.get("analyzer_abstained"):
        # The passages were topically close enough to retrieve but did not
        # answer the question. A wider net may reach the one that does.
        return "retry" if _retries_left(state) else "responder"
    return "verifier"


def after_verifier(state: AgentState) -> Literal["retry", "responder"]:
    # Retry only when NOTHING could be verified, which means the answer rests
    # on quotes that are not in the cited passages. A partially verified answer
    # goes through: it is reported as PARTIALLY_VERIFIED with the failures
    # marked, and retrying it would risk losing the citations that were sound.
    if state.get("verified_count", 0) == 0 and _retries_left(state):
        return "retry"
    return "responder"


def retry_node(state: AgentState) -> dict:
    """Spend one retry and record why, then hand back to the retriever.

    A node rather than a flag set by the edge functions, because LangGraph's
    conditional edges only choose a route -- they cannot write state -- and the
    retry counter has to be incremented exactly once per loop or the loop will
    not terminate.
    """
    attempt = state.get("retry_count", 0) + 1
    if state.get("analyzer_abstained"):
        reason = "the retrieved passages did not answer the question"
    elif state.get("citations"):
        reason = "no citation could be confirmed against its passage"
    else:
        reason = "the rewritten query found nothing; retrying with the question as asked"

    logger.info("Retry %d/%d: %s", attempt, config.MAX_RETRIES, reason)
    return {
        "retry_count": attempt,
        # Cleared so a retry cannot inherit the previous attempt's conclusions:
        # if the second pass abstains, the first pass's unverified citations
        # must not still be sitting in the state when the responder reads it.
        "citations": [],
        "answer": "",
        "analyzer_abstained": False,
        "trace": [
            trace_entry(
                "retry",
                f"Retrying ({attempt}/{config.MAX_RETRIES}) with a wider search: {reason}",
                top_k=config.RETRIEVAL_TOP_K_ON_RETRY,
            )
        ],
    }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class AgentAnswer:
    """What one question produced, in the shape the API and CLI both want."""

    question: str
    status: str
    answer: str
    # The document the question was SCOPED to, if the caller restricted it.
    # Distinct from where the evidence came from, which is in the citations.
    source_document: str | None = None
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    intent: str = ""
    retries: int = 0
    top_similarity: float = 0.0
    verified_count: int = 0
    unverified_count: int = 0
    trace: list[TraceEntry] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)
    error: str = ""

    @property
    def answered(self) -> bool:
        """Did this produce an answer, verified or partly verified?"""
        return self.status in ("ANSWERED", "PARTIALLY_VERIFIED")

    @property
    def abstained(self) -> bool:
        """Did the agent decline for lack of evidence or scope?

        Separate from `answered` rather than its negation, because ERROR is
        neither: the evaluation has to be able to exclude broken runs instead
        of scoring them as abstentions.
        """
        return self.status in ("INSUFFICIENT_EVIDENCE", "OUT_OF_SCOPE")

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "status": self.status,
            "answer": self.answer,
            "source_document": self.source_document,
            "citations": [dict(citation) for citation in self.citations],
            "confidence": self.confidence,
            "intent": self.intent,
            "retries": self.retries,
            "top_similarity": self.top_similarity,
            "verified_count": self.verified_count,
            "unverified_count": self.unverified_count,
            "trace": [dict(entry) for entry in self.trace],
            "usage": self.usage.as_dict(),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph(llm: LLMClient, store: VectorStore):
    """Wire the nodes into a compiled LangGraph.

    The LLM client and vector store are passed in, not constructed here, so
    tests can run the real control flow against a scripted LLM and an
    in-memory store -- the retry and abstention paths are the most important
    behaviour in the system and must be testable without network calls.
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", build_router_node(llm, store))
    graph.add_node("retriever", build_retriever_node(store))
    graph.add_node("analyzer", build_analyzer_node(llm))
    graph.add_node("verifier", build_verifier_node(llm, store))
    graph.add_node("retry", retry_node)
    graph.add_node("responder", build_responder_node())

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", after_router)
    graph.add_conditional_edges("retriever", after_retriever)
    graph.add_conditional_edges("analyzer", after_analyzer)
    graph.add_conditional_edges("verifier", after_verifier)
    graph.add_edge("retry", "retriever")
    graph.add_edge("responder", END)

    return graph.compile()


class PolicyLensAgent:
    """The agent as a single callable object.

    Holds the compiled graph, the vector store and the LLM client, all of which
    are expensive to build (the embedding model alone is ~100MB) and must be
    reused across questions. The FastAPI app creates one at startup; the CLI
    creates one per invocation.
    """

    def __init__(
        self,
        store: VectorStore | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.store = store or VectorStore()
        self.llm = llm or get_llm_client()
        self.graph = build_graph(self.llm, self.store)

    def ask(self, question: str, source_document: str | None = None) -> AgentAnswer:
        """Answer one question.

        `source_document` restricts retrieval to a single indexed document,
        for the UI's document picker. It overrides whatever the router infers.

        Never raises for an ordinary failure: an unreachable model or an empty
        index comes back as status ERROR with the reason in `error`, so one bad
        question in a batch evaluation does not abort the run.
        """
        question = (question or "").strip()
        if not question:
            return AgentAnswer(
                question="",
                status="ERROR",
                answer="",
                source_document=source_document,
                error="The question was empty.",
            )

        logger.info("Question: %s", question)
        initial = new_state(question, source_document)

        try:
            final: AgentState = self.graph.invoke(initial)
        except Exception as error:  # noqa: BLE001 -- see docstring
            # The graph should handle its own failures; anything reaching here
            # is a bug or an environment problem. Report it as a failed run
            # rather than letting it escape into the API layer as a 500.
            logger.exception("Agent graph raised")
            return AgentAnswer(
                question=question,
                status="ERROR",
                answer="",
                source_document=source_document,
                error=f"{type(error).__name__}: {error}",
            )

        citations = list(final.get("citations") or [])
        verified_count = final.get("verified_count", 0)

        return AgentAnswer(
            question=question,
            status=final.get("status", "ERROR"),
            answer=final.get("answer", ""),
            source_document=source_document,
            citations=citations,
            confidence=final.get("confidence", 0.0),
            intent=final.get("intent", ""),
            retries=final.get("retry_count", 0),
            top_similarity=final.get("top_similarity", 0.0),
            verified_count=verified_count,
            unverified_count=max(0, len(citations) - verified_count),
            trace=list(final.get("trace") or []),
            usage=LLMUsage.from_records(list(final.get("llm_calls") or [])),
            error=final.get("error", ""),
        )
