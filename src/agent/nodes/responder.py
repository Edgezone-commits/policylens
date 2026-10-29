"""Responder node: decide what outcome this was, and how confident to be.

Every path through the graph ends here, including the ones that produced no
answer. WHY one exit: the status, the confidence number and the user-facing
wording of an abstention are decisions that must be made the same way every
time, and four nodes each writing their own ending is how a system ends up
abstaining in two subtly different voices.

No LLM call is made here. Both outputs are computed from observable facts
already in the state.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from src import config
from src.agent.state import AgentState, Status, trace_entry

logger = logging.getLogger(__name__)

# Wording for the outcomes that carry no answer. Written out rather than
# generated: an abstention is the system's most important message, and it
# should read the same every time and explain what to do next.
_NO_EVIDENCE_MESSAGE = (
    "I could not find anything in the indexed documents that answers this. "
    "Rather than answer from general knowledge, I am stopping here -- any "
    "answer I gave would be unsupported by the documents you provided."
)

_OUT_OF_SCOPE_MESSAGE = (
    "This question is not about the indexed documents, so I cannot answer it "
    "from them. I only answer from the documents that have been ingested."
)


def compute_confidence(
    verified_ratio: float, top_similarity: float, retry_count: int
) -> float:
    """Score the answer from observable signals, not from self-report.

    confidence = 0.6 * (fraction of citations verified)
               + 0.4 * (similarity of the best retrieved passage)
               - 0.10 * (retries)

    WHY these signals: both are facts the system measured, and both can be
    checked against correctness in the Phase 6 evaluation -- which is the test
    of whether this formula deserves to exist. Asking the model to rate its own
    confidence is the obvious alternative and a bad one: self-reported
    confidence clusters near 0.9 whether the answer is right or wrong, so it
    adds a number that looks informative and is not.

    WHY verification is weighted highest: an answer whose quotes check out is
    grounded in the documents even if retrieval was mediocre, whereas a strong
    similarity score with failing citations means the right passage was found
    and then misquoted -- the more dangerous of the two failures.

    The retry penalty reflects that a first attempt which failed verification
    is evidence the question sits at the edge of what the corpus supports, even
    when the retry succeeds.

    Weights live in config so Phase 6 can re-fit them against measured
    accuracy instead of leaving them as a guess.
    """
    score = (
        config.CONFIDENCE_WEIGHT_VERIFIED_RATIO * verified_ratio
        + config.CONFIDENCE_WEIGHT_RETRIEVAL_SCORE * max(0.0, min(1.0, top_similarity))
        - config.CONFIDENCE_PENALTY_PER_RETRY * retry_count
    )
    return round(max(0.0, min(1.0, score)), 3)


def build_responder_node(verify: bool = True) -> Callable[[AgentState], dict]:
    """Create the responder node. Reads state; holds no dependencies.

    `verify=False` matches a graph built without the verifier. It changes one
    thing only: an answer whose citations were never CHECKED is not treated as
    an answer whose citations all FAILED. Those look identical in the state
    (verified_count == 0) and mean opposite things -- the first is the
    evaluation's control arm, the second is grounds to withdraw the answer.
    """

    def responder_node(state: AgentState) -> dict:
        # -- pipeline failure -------------------------------------------
        if state.get("error"):
            # Distinct from an abstention on purpose. "I found no evidence" and
            # "I could not run" are different claims, and conflating them would
            # let a broken key read as a documented absence of evidence.
            return {
                "status": "ERROR",
                "answer": state.get("answer") or "",
                "confidence": 0.0,
                "trace": [
                    trace_entry(
                        "responder", "Run failed", error=state["error"]
                    )
                ],
            }

        # -- out of scope -----------------------------------------------
        if state.get("intent") == "OUT_OF_SCOPE":
            return {
                "status": "OUT_OF_SCOPE",
                "answer": _OUT_OF_SCOPE_MESSAGE,
                "citations": [],
                "confidence": 0.0,
                "trace": [
                    trace_entry(
                        "responder",
                        "Declined: the question is not about the indexed documents",
                        reason=state.get("router_reason", ""),
                    )
                ],
            }

        citations = state.get("citations") or []
        verified_count = state.get("verified_count", 0)
        # Trust the graph that built this node over the state, which a test or
        # a partially run graph could leave unset.
        verification_skipped = not verify or state.get("verification_skipped", False)

        # -- nothing to stand on ----------------------------------------
        # Covers all three ways an answer can fail to be supported: retrieval
        # found nothing above the threshold, the analyzer read the evidence and
        # declined, or every citation failed verification. The user is told the
        # same thing in each case, because for their purposes it is the same
        # thing: this system will not assert what it cannot show.
        no_evidence = not state.get("chunks")
        analyzer_declined = state.get("analyzer_abstained", False)
        nothing_verified = (
            bool(citations) and verified_count == 0 and not verification_skipped
        )

        if no_evidence or analyzer_declined or nothing_verified:
            if no_evidence:
                detail = (
                    "no passage scored above the "
                    f"{config.RETRIEVAL_SIMILARITY_THRESHOLD:.2f} relevance threshold"
                )
            elif analyzer_declined:
                detail = "the retrieved passages do not address the question"
            else:
                detail = (
                    f"none of the {len(citations)} proposed citation(s) could be "
                    "confirmed against the passages they cited"
                )

            logger.info("Abstaining: %s", detail)
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "answer": _NO_EVIDENCE_MESSAGE,
                # Unverified citations are NOT carried into an abstention: the
                # answer they were attached to has been withdrawn, so showing
                # them would be offering evidence for a claim no longer made.
                "citations": [],
                "confidence": 0.0,
                "trace": [
                    trace_entry(
                        "responder",
                        f"Insufficient evidence -- {detail}",
                        attempts=state.get("retry_count", 0) + 1,
                        top_similarity=state.get("top_similarity", 0.0),
                        rejected_citations=len(citations),
                    )
                ],
            }

        # -- an answer, fully or partly verified -------------------------
        unverified_count = len(citations) - verified_count
        if verification_skipped:
            # PARTIALLY_VERIFIED would be a lie here: it means "we checked and
            # some quotes failed", and nothing was checked. ANSWERED is the
            # honest label for an unchecked draft, and the citations still
            # carry method="unchecked" so no individual quote claims
            # otherwise.
            status: Status = "ANSWERED"
        else:
            status = "ANSWERED" if unverified_count == 0 else "PARTIALLY_VERIFIED"
        confidence = compute_confidence(
            verified_ratio=state.get("verified_ratio", 0.0),
            top_similarity=state.get("top_similarity", 0.0),
            retry_count=state.get("retry_count", 0),
        )

        logger.info(
            "%s with confidence %.2f (%d/%d citations verified, %d retry)",
            status,
            confidence,
            verified_count,
            len(citations),
            state.get("retry_count", 0),
        )

        return {
            "status": status,
            "confidence": confidence,
            "trace": [
                trace_entry(
                    "responder",
                    f"{status} with confidence {confidence:.2f} "
                    f"({verified_count}/{len(citations)} citations verified)",
                    verified=verified_count,
                    unverified=unverified_count,
                    top_similarity=state.get("top_similarity", 0.0),
                    retries=state.get("retry_count", 0),
                )
            ],
        }

    return responder_node
