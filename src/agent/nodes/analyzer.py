"""Analyzer node: write the answer, with a quote behind every claim.

The node's real work is not the LLM call -- it is refusing to trust the reply.
A model asked for structured citations will, often enough to matter:

  * cite evidence [6] when five excerpts were supplied,
  * return a quote of four words that matches any paragraph of legal prose,
  * answer a question the excerpts do not address, because it knows the answer
    from pre-training and the question looked answerable.

Each of those produces a citation that LOOKS checkable and is not, which is
worse than an uncited answer: it borrows the credibility of a citation without
earning it. So every citation is resolved back to a real chunk here, before the
verifier ever reads it, and anything that cannot be resolved is dropped with a
reason recorded in the trace.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from src import config
from src.agent.llm import LLMClient, LLMError, call_record
from src.agent.prompts import ANALYZER_SYSTEM, build_analyzer_prompt
from src.agent.state import AgentState, Citation, RetrievedChunk, trace_entry

logger = logging.getLogger(__name__)


def _resolve_citations(
    raw_citations: Any, chunks: list[RetrievedChunk]
) -> tuple[list[Citation], list[str]]:
    """Turn the model's citation list into Citations tied to real chunks.

    Returns (citations, rejections) where each rejection is a human-readable
    reason. Rejections are kept rather than silently dropped because they are
    the clearest signal that the analyzer prompt needs work, and because the
    reasoning path should not claim a citation existed when it did not.
    """
    citations: list[Citation] = []
    rejections: list[str] = []

    if not isinstance(raw_citations, list):
        return citations, [f"citations was {type(raw_citations).__name__}, not a list"]

    for position, item in enumerate(raw_citations, start=1):
        if not isinstance(item, dict):
            rejections.append(f"citation {position} was not an object")
            continue

        quote = str(item.get("quote") or "").strip()
        if not quote:
            rejections.append(f"citation {position} had no quote")
            continue

        # The model cites by evidence number (see prompts.py for why it is not
        # given chunk ids at all). Accept the common aliases; anything
        # non-numeric or out of range is unresolvable.
        reference = item.get("evidence", item.get("index", item.get("number")))
        try:
            evidence_number = int(str(reference).strip())
        except (TypeError, ValueError):
            rejections.append(
                f"citation {position} referenced {reference!r}, not an evidence number"
            )
            continue

        if not 1 <= evidence_number <= len(chunks):
            # A fabricated reference: the model cited an excerpt it was never
            # shown. There is no chunk to check the quote against, so the
            # citation cannot be verified and must not be presented.
            rejections.append(
                f"citation {position} referenced evidence [{evidence_number}] "
                f"but only {len(chunks)} excerpt(s) were supplied"
            )
            continue

        chunk = chunks[evidence_number - 1]
        if len(quote) < config.MIN_CITATION_QUOTE_CHARS:
            # Recorded as a rejection, not passed on as unverified: a quote
            # this short cannot be checked either way (see
            # MIN_CITATION_QUOTE_CHARS in config), so showing it would be
            # showing an unfalsifiable citation.
            rejections.append(
                f"citation {position} quoted only {len(quote)} characters "
                f"(minimum {config.MIN_CITATION_QUOTE_CHARS})"
            )
            continue

        citations.append(
            Citation(
                chunk_id=chunk["chunk_id"],
                quote=quote,
                source_document=chunk["source_document"],
                page_number=chunk.get("page_number"),
                section=chunk.get("section"),
                # Verification happens in the next node; until then no claim is
                # made either way.
                verified=False,
                method="unchecked",
                match_score=0.0,
                note="",
            )
        )

    return citations, rejections


def build_analyzer_node(llm: LLMClient) -> Callable[[AgentState], dict]:
    """Create the analyzer node bound to an LLM client."""

    def analyzer_node(state: AgentState) -> dict:
        chunks = state.get("chunks") or []
        if not chunks:
            # The graph should not route here without evidence; if it does,
            # abstain rather than invite an uncited answer.
            return {
                "analyzer_abstained": True,
                "trace": [
                    trace_entry("analyzer", "No evidence supplied; nothing to analyse")
                ],
            }

        try:
            payload, response = llm.complete_json(
                build_analyzer_prompt(state["question"], chunks),
                system=ANALYZER_SYSTEM,
                # Thinking stays ON here: this is the one step that has to read
                # five excerpts, reconcile them and compose a cited answer.
                simple_task=False,
                required_keys=("answer",),
                # Not the default cap: this reply carries a verbatim quote
                # per citation, not just prose. See the config comment.
                max_tokens=config.LLM_MAX_TOKENS_FOR_CITED_ANSWER,
            )
        except LLMError as error:
            # Unlike the router, there is no safe default for the answer
            # itself. Record the error and let the responder report it; a
            # fabricated or empty answer presented as an abstention would
            # misrepresent what happened.
            logger.error("Analyzer failed: %s", error)
            return {
                "error": f"Could not generate an answer: {error}",
                "trace": [trace_entry("analyzer", "LLM call failed", error=str(error))],
            }

        answer = str(payload.get("answer") or "").strip()
        abstained = bool(payload.get("insufficient_evidence", False))

        if abstained or not answer:
            # The analyzer read the evidence and judged it insufficient. This
            # is a second, independent abstention gate: chunks can clear the
            # similarity threshold and still be about the wrong provision --
            # similarity is topical, and "annual leave" scores well against a
            # question on sick leave.
            reason = str(payload.get("reason") or "").strip()
            logger.info("Analyzer abstained on the retrieved evidence")
            return {
                "answer": "",
                "citations": [],
                "analyzer_abstained": True,
                "llm_calls": [call_record(response, "analyzer")],
                "trace": [
                    trace_entry(
                        "analyzer",
                        "Evidence retrieved but it does not answer the question",
                        reason=reason,
                        passages_read=len(chunks),
                    )
                ],
            }

        citations, rejections = _resolve_citations(payload.get("citations"), chunks)

        if rejections:
            logger.warning(
                "Dropped %d unusable citation(s): %s",
                len(rejections),
                "; ".join(rejections),
            )

        if not citations:
            # An answer with no usable citation cannot be verified at all.
            # Treat it as an abstention rather than publishing an unsupported
            # answer -- the whole contract of this system is that a claim comes
            # with a checkable quote.
            logger.info("Answer had no usable citations; treating as unsupported")
            return {
                "answer": "",
                "citations": [],
                "analyzer_abstained": True,
                "llm_calls": [call_record(response, "analyzer")],
                "trace": [
                    trace_entry(
                        "analyzer",
                        "Answer discarded: it carried no citation that could be "
                        "checked against a retrieved passage",
                        rejections=rejections,
                    )
                ],
            }

        logger.info(
            "Drafted an answer (%d words) with %d citation(s)",
            len(answer.split()),
            len(citations),
        )

        return {
            "answer": answer,
            "citations": citations,
            "analyzer_abstained": False,
            "llm_calls": [call_record(response, "analyzer")],
            "trace": [
                trace_entry(
                    "analyzer",
                    f"Drafted an answer with {len(citations)} citation(s)",
                    citations=[
                        {
                            "chunk_id": citation["chunk_id"],
                            "source_document": citation["source_document"],
                            "page_number": citation.get("page_number"),
                            "section": citation.get("section"),
                        }
                        for citation in citations
                    ],
                    rejected=rejections,
                )
            ],
        }

    return analyzer_node
