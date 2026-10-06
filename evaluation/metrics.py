"""Grading and aggregation for the Phase 6 evaluation.

Kept apart from the harness that calls the agent so the grader can be unit
tested on strings alone -- no model, no key, no quota. A grader nobody can
test is a number nobody should trust.

The vocabulary used throughout:

  CORRECT           the system did the right thing, whichever thing that was:
                    answered an answerable question with the decisive fact, or
                    abstained on an unanswerable one.
  WRONG_ANSWER      answered an answerable question, but the decisive fact is
                    not in the answer.
  FALSE_ABSTENTION  refused a question the corpus answers. Unhelpful.
  FALSE_ANSWER      answered a question the corpus does not answer. This is
                    the one that matters: a confident, cited-looking claim the
                    documents do not support.
  ERROR             the pipeline did not run (no key, quota gone). Excluded
                    from the rates rather than counted as either kind of
                    mistake, because a run that never happened made no claim.

WHY the two error rates are reported separately instead of as one accuracy
number: they are not equally expensive. A false abstention costs the user a
question they could have had answered, and they can see it happened. A false
answer looks exactly like a correct one -- same shape, same citations, same
confidence band -- and for a legal document that is the failure that does real
damage. Averaging them into "accuracy" would let an improvement in the cheap
error disguise a regression in the expensive one.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

from evaluation.dataset import Question

# Statuses that mean "the system asserted an answer".
ANSWERED_STATUSES = frozenset({"ANSWERED", "PARTIALLY_VERIFIED"})
# Statuses that mean "the system declined to answer", which is a decision, not
# a failure. OUT_OF_SCOPE and INSUFFICIENT_EVIDENCE reach it by different
# routes (the router versus the evidence gates) and both count as abstaining.
ABSTAINED_STATUSES = frozenset({"INSUFFICIENT_EVIDENCE", "OUT_OF_SCOPE"})

_WHITESPACE = re.compile(r"\s+")
# Characters the model varies freely and which must not decide a grade:
# curly quotes, en/em dashes, and non-breaking spaces all appear in answers
# drawn from a PDF.
_TRANSLATIONS = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
    }
)


def normalise(text: str) -> str:
    """Lower-case and flatten whitespace and quoting, nothing more.

    Deliberately conservative. Stripping punctuation wholesale would turn
    "8.33" into "833" and make the gratuity rate ungradeable, and stripping
    hyphens would silently equate "forty-eight" with "fortyeight" rather than
    letting the dataset list the spellings it will accept.
    """
    return _WHITESPACE.sub(" ", text.translate(_TRANSLATIONS).lower()).strip()


def facts_present(answer: str, expected_facts: Iterable[Iterable[str]]) -> list[bool]:
    """Which of the expected fact groups the answer satisfies.

    One group is satisfied by ANY of its spellings; the answer is correct only
    if every group is satisfied. Returns the per-group verdicts rather than a
    bare bool so a failing question can say which fact was missing, which is
    what makes a low score actionable instead of merely bad news.
    """
    haystack = normalise(answer)
    return [
        any(normalise(spelling) in haystack for spelling in group)
        for group in expected_facts
    ]


@dataclass
class Graded:
    """One question, what the system did, and the verdict."""

    question: str
    topic: str
    answerable: bool
    status: str
    outcome: str
    confidence: float
    # Per-group fact verdicts, empty for unanswerable questions.
    facts: list[bool] = field(default_factory=list)
    missing_facts: list[list[str]] = field(default_factory=list)
    # True only when EVERY document the answer needs was cited. For a
    # cross-document question that is the whole point: citing one of the
    # two is a half answer, and a metric that accepted it would report
    # cross-document grounding the system never achieved.
    cited_expected_documents: bool | None = None
    cross_document: bool = False
    verified_citations: int = 0
    total_citations: int = 0
    top_similarity: float = 0.0
    retries: int = 0
    llm_calls: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    answer: str = ""
    error: str = ""

    @property
    def correct(self) -> bool:
        return self.outcome == "CORRECT"


def grade(item: Question, answer: dict[str, Any], latency_seconds: float = 0.0) -> Graded:
    """Score one answer against its label.

    `answer` is AgentAnswer.as_dict(), so this grades exactly what the API
    returns -- not some internal view of it that a client would never see.
    """
    status = answer.get("status", "ERROR")
    citations = answer.get("citations") or []
    usage = answer.get("usage") or {}
    text = answer.get("answer") or ""

    verified = sum(1 for citation in citations if citation.get("verified"))
    cited_documents = {
        citation.get("source_document") for citation in citations if citation
    }

    common = {
        "question": item.question,
        "topic": item.topic,
        "answerable": item.answerable,
        "cross_document": item.cross_document,
        "status": status,
        "confidence": float(answer.get("confidence") or 0.0),
        "verified_citations": verified,
        "total_citations": len(citations),
        "top_similarity": float(answer.get("top_similarity") or 0.0),
        "retries": int(answer.get("retries") or 0),
        "llm_calls": int(usage.get("calls") or 0),
        "total_tokens": int(usage.get("prompt_tokens") or 0)
        + int(usage.get("completion_tokens") or 0),
        "latency_seconds": round(latency_seconds, 2),
        "answer": text,
        "error": answer.get("error") or "",
    }

    if status == "ERROR":
        # Not scored. A pipeline that could not run made no claim about the
        # documents, so counting it as either error type would invent a result.
        return Graded(outcome="ERROR", **common)

    if not item.answerable:
        outcome = "CORRECT" if status in ABSTAINED_STATUSES else "FALSE_ANSWER"
        return Graded(outcome=outcome, **common)

    if status in ABSTAINED_STATUSES:
        return Graded(outcome="FALSE_ABSTENTION", **common)

    verdicts = facts_present(text, item.expected_facts)
    missing = [
        list(group)
        for group, ok in zip(item.expected_facts, verdicts)
        if not ok
    ]
    cited_expected = (
        all(document in cited_documents for document in item.expected_documents)
        if item.expected_documents
        else None
    )
    return Graded(
        outcome="CORRECT" if all(verdicts) else "WRONG_ANSWER",
        facts=verdicts,
        missing_facts=missing,
        cited_expected_documents=cited_expected,
        **common,
    )


def _mean(values: list[float]) -> float | None:
    """Mean, or None when there is nothing to average.

    None rather than 0.0 on purpose: "no questions of this kind ran" and "they
    all scored zero" are different facts, and a report that prints 0.0 for the
    first one is lying.
    """
    return round(statistics.fmean(values), 3) if values else None


def aggregate(results: list[Graded]) -> dict[str, Any]:
    """Roll the graded questions up into the report's numbers."""
    scored = [r for r in results if r.outcome != "ERROR"]
    errors = [r for r in results if r.outcome == "ERROR"]
    answerable = [r for r in scored if r.answerable]
    unanswerable = [r for r in scored if not r.answerable]

    asserted = [r for r in answerable if r.status in ANSWERED_STATUSES]
    correct = [r for r in scored if r.correct]
    wrong_answers = [r for r in answerable if r.outcome == "WRONG_ANSWER"]
    false_abstentions = [r for r in answerable if r.outcome == "FALSE_ABSTENTION"]
    false_answers = [r for r in unanswerable if r.outcome == "FALSE_ANSWER"]

    citations_total = sum(r.total_citations for r in scored)
    citations_verified = sum(r.verified_citations for r in scored)
    recall_checked = [
        r for r in asserted if r.cited_expected_documents is not None
    ]
    cross = [r for r in answerable if r.cross_document]
    cross_asserted = [r for r in cross if r.status in ANSWERED_STATUSES]

    return {
        "questions": len(results),
        "scored": len(scored),
        "errors": len(errors),
        # -- the two error rates, kept apart ----------------------------
        "false_abstention_rate": _rate(len(false_abstentions), len(answerable)),
        "false_answer_rate": _rate(len(false_answers), len(unanswerable)),
        # -- did it get the answer right when it gave one ---------------
        "answer_accuracy": _rate(
            len([r for r in asserted if r.correct]), len(asserted)
        ),
        "overall_correct_rate": _rate(len(correct), len(scored)),
        "counts": {
            "CORRECT": len(correct),
            "WRONG_ANSWER": len(wrong_answers),
            "FALSE_ABSTENTION": len(false_abstentions),
            "FALSE_ANSWER": len(false_answers),
            "ERROR": len(errors),
        },
        "by_status": _counter(r.status for r in results),
        # -- grounding ---------------------------------------------------
        "citation_verification_rate": _rate(citations_verified, citations_total),
        "citations_total": citations_total,
        "citation_recall": _rate(
            len([r for r in recall_checked if r.cited_expected_documents]),
            len(recall_checked),
        ),
        # -- cross-document, broken out ----------------------------------
        #
        # Reported on its own because it is the harder half and a combined
        # figure hides it: 28 single-document questions would swamp 10
        # cross-document ones, and a system that silently answers every
        # cross-document question from one document would still look fine.
        "cross_document_questions": len(cross),
        "cross_document_accuracy": _rate(
            len([r for r in cross_asserted if r.correct]), len(cross_asserted)
        ),
        "cross_document_both_cited": _rate(
            len([r for r in cross_asserted if r.cited_expected_documents]),
            len(cross_asserted),
        ),
        # -- is the confidence formula worth keeping ---------------------
        #
        # The claim in responder.compute_confidence is that confidence tracks
        # correctness. That is only true if these two means separate, so the
        # gap is reported as the headline: a gap near zero means the number is
        # decoration and should either be re-fitted or dropped.
        "mean_confidence_correct": _mean([r.confidence for r in asserted if r.correct]),
        "mean_confidence_incorrect": _mean(
            [r.confidence for r in asserted if not r.correct]
        ),
        "confidence_separation": _separation(asserted),
        # -- what it cost ------------------------------------------------
        "mean_latency_seconds": _mean([r.latency_seconds for r in scored]),
        "mean_llm_calls": _mean([float(r.llm_calls) for r in scored]),
        "total_tokens": sum(r.total_tokens for r in results),
    }


def _separation(asserted: list[Graded]) -> float | None:
    """mean confidence when correct minus mean confidence when wrong."""
    right = [r.confidence for r in asserted if r.correct]
    wrong = [r.confidence for r in asserted if not r.correct]
    if not right or not wrong:
        # Needs both sides to mean anything. With no wrong answers there is no
        # separation to measure, which is a good problem and still not a number.
        return None
    return round(statistics.fmean(right) - statistics.fmean(wrong), 3)


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def _counter(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Retrieval, measured on its own
# ---------------------------------------------------------------------------
#
# WHY this is separate from everything above, and worth having twice over:
#
#   * It needs NO LLM. It asks the vector store directly, so the whole 53
#     questions can be measured for free, as often as wanted, while the daily
#     quota caps end-to-end runs at a handful of questions. On a free tier
#     that is the difference between a metric and an anecdote.
#   * It isolates blame. If an answer is wrong, "was the evidence even
#     retrieved" is the first question, and citation_recall cannot answer it:
#     a citation is only produced AFTER the analyzer has read the chunks, so a
#     low figure there could mean bad retrieval or a bad draft. Recall@k
#     separates the two.


def recall_at_k(retrieved_documents: Iterable[str], expected: Iterable[str]) -> bool:
    """Did the top-k passages include every document the answer needs?

    `all` rather than `any`, for the same reason cited_expected_documents uses
    it: for a cross-document question, retrieving one of the two documents is
    not enough to answer it, so counting it as a hit would overstate what the
    retriever delivered.
    """
    found = set(retrieved_documents)
    wanted = set(expected)
    return bool(wanted) and wanted <= found


def aggregate_recall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up per-question retrieval results.

    Each row: {"cross_document": bool, "hit": bool, "best_similarity": float,
    "above_threshold": int}.
    """
    if not rows:
        return {
            "questions": 0,
            "recall_at_k": None,
            "recall_at_k_single": None,
            "recall_at_k_cross": None,
            "mean_best_similarity": None,
        }

    single = [r for r in rows if not r["cross_document"]]
    cross = [r for r in rows if r["cross_document"]]
    return {
        "questions": len(rows),
        "recall_at_k": _rate(sum(r["hit"] for r in rows), len(rows)),
        "recall_at_k_single": _rate(sum(r["hit"] for r in single), len(single)),
        "recall_at_k_cross": _rate(sum(r["hit"] for r in cross), len(cross)),
        "mean_best_similarity": _mean([r["best_similarity"] for r in rows]),
    }
