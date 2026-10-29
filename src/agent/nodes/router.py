"""Router node: decide what kind of question this is, and how to search for it.

The router does two jobs and deliberately not a third:

  * It separates questions about the indexed documents from questions about
    something else, so a request to write a poem never reaches retrieval.
  * It rewrites the question into the documents' vocabulary, because the
    embedding model matches wording, and a user's "can my boss keep me on
    trial?" is a poor vector neighbour of the Act's "probation period".

It does NOT decide whether the answer exists. That judgement belongs to the
retrieval threshold, which is calibrated against measured score distributions
(see scripts/calibrate_threshold.py); the router has read no evidence and would
be guessing. So the prompt pushes it towards DOCUMENT_QA whenever a question
plausibly touches the subject matter, and an abstention, when it comes, comes
from evidence rather than from an opinion.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from src.agent.llm import LLMClient, LLMError, call_record
from src.agent.prompts import ROUTER_SYSTEM, build_router_prompt
from src.agent.state import AgentState, Intent, trace_entry
from src.retrieval.store import VectorStore

logger = logging.getLogger(__name__)

_VALID_INTENTS: tuple[Intent, ...] = ("DOCUMENT_QA", "COMPARISON", "OUT_OF_SCOPE")


def _simplify(text: str) -> str:
    """Lowercase and strip everything but letters and digits.

    Lets "Nepal Labour Act 2074" match "Nepal_Labour_Act_2074.pdf" without
    caring about separators, case or the extension.
    """
    return "".join(character for character in text.lower() if character.isalnum())


def _match_document(name: str, known: list[str]) -> str | None:
    """Resolve a document name the model returned to one that really exists.

    The model is asked to copy filenames exactly and mostly does, but it also
    drops the extension, changes underscores to spaces, or returns a title it
    read off the first page. An unmatched name must be discarded rather than
    passed through: a metadata filter on a filename that is not in the index
    matches zero chunks, which would turn a good question into a silent
    "no evidence".
    """
    candidate = (name or "").strip()
    if not candidate:
        return None

    for document in known:
        if candidate == document:
            return document

    target = _simplify(candidate)
    if not target:
        return None

    for document in known:
        simplified = _simplify(document)
        # Substring either way: "Labour Act" should match
        # "Nepal_Labour_Act_2074.pdf", and the model sometimes returns a name
        # longer than the filename ("Nepal Labour Act 2074 (PDF)").
        if target == simplified or target in simplified or simplified in target:
            return document

    logger.warning("Router named an unknown document %r; ignoring it", candidate)
    return None


def build_router_node(
    llm: LLMClient, store: VectorStore
) -> Callable[[AgentState], dict]:
    """Create the router node bound to an LLM client and a vector store.

    Dependencies are injected rather than imported inside the node so the
    graph can be built against a stub LLM in tests, which is what makes the
    retry and abstention paths testable without spending quota.
    """

    def router_node(state: AgentState) -> dict:
        question = state["question"]

        if not question:
            return {
                "intent": "OUT_OF_SCOPE",
                "router_reason": "the question was empty",
                "trace": [trace_entry("router", "Empty question; nothing to route")],
            }

        documents = store.document_names()
        if not documents:
            # No index means no possible evidence. Say so as an error rather
            # than as an abstention: "I found no evidence" would be misleading
            # when the truth is that nothing has been ingested yet.
            return {
                "error": "No documents are indexed. Run: python scripts/ingest_docs.py data/raw/",
                "trace": [trace_entry("router", "Index is empty")],
            }

        try:
            payload, response = llm.complete_json(
                build_router_prompt(question, documents),
                system=ROUTER_SYSTEM,
                simple_task=True,  # classification; no thinking budget needed
                required_keys=("intent",),
            )
        except LLMError as error:
            # Degrade rather than fail. A router outage should not take the
            # whole system down, and its two outputs both have safe defaults:
            # treat the question as document QA, and search with the user's own
            # words. The answer may be slightly worse; it is still verified.
            logger.warning("Router unavailable (%s); defaulting to DOCUMENT_QA", error)
            return {
                "intent": "DOCUMENT_QA",
                "target_documents": [],
                "search_query": question,
                "router_reason": f"router unavailable ({type(error).__name__})",
                "trace": [
                    trace_entry(
                        "router",
                        "Router unavailable; searching with the question as asked",
                        error=str(error),
                    )
                ],
            }

        intent = str(payload.get("intent", "")).strip().upper()
        if intent not in _VALID_INTENTS:
            # An unrecognised label is not a reason to refuse: fall back to the
            # permissive branch, where evidence still has to be found.
            logger.warning("Router returned unknown intent %r; using DOCUMENT_QA", intent)
            intent = "DOCUMENT_QA"

        named = payload.get("target_documents") or []
        if not isinstance(named, list):
            named = [named]
        targets: list[str] = []
        for item in named:
            resolved = _match_document(str(item), documents)
            if resolved and resolved not in targets:
                targets.append(resolved)

        # A caller-supplied document (the UI's picker) is an instruction, not a
        # suggestion, so it overrides whatever the router inferred.
        if state.get("source_document"):
            targets = [state["source_document"]]

        search_query = str(payload.get("search_query") or "").strip() or question

        # A COMPARISON that resolved to fewer than two documents is a
        # comparison in name only; retrieving per-document would then just be a
        # single-document search with extra steps.
        if intent == "COMPARISON" and len(targets) < 2:
            logger.info(
                "Router said COMPARISON but named %d document(s); treating as "
                "DOCUMENT_QA",
                len(targets),
            )
            intent = "DOCUMENT_QA"

        reason = str(payload.get("reason") or "").strip()
        logger.info(
            "Routed as %s%s | search query: %r",
            intent,
            f" over {targets}" if targets else "",
            search_query,
        )

        return {
            "intent": intent,
            "target_documents": targets,
            "search_query": search_query,
            "router_reason": reason,
            "llm_calls": [call_record(response, "router")],
            "trace": [
                trace_entry(
                    "router",
                    f"Classified as {intent}"
                    + (f" over {', '.join(targets)}" if targets else ""),
                    intent=intent,
                    target_documents=targets,
                    search_query=search_query,
                    reason=reason,
                )
            ],
        }

    return router_node
