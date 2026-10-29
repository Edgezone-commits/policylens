"""Run the Phase 6 evaluation and write the report.

    python scripts/evaluate.py                  # the whole set
    python scripts/evaluate.py --limit 6        # the first 6 questions
    python scripts/evaluate.py --only adversarial
    python scripts/evaluate.py --no-save        # print, write nothing

This is the measurement every other phase defers to. config.py, responder.py
and chunker.py all carry comments saying some number is "checked in Phase 6";
this is that check, and until it runs those numbers are guesses with good
reasons attached.

What it measures, and why each one earns its place
--------------------------------------------------
FALSE ANSWER RATE      -- answers given to questions the corpus cannot answer.
                          The headline. Everything else in this project
                          (retrieval threshold, three abstention gates, quote
                          verification) exists to keep this near zero, and the
                          adversarial questions are written to break it.
FALSE ABSTENTION RATE  -- refusals of questions the corpus does answer. The
                          cost of the above. Reported beside it because
                          driving one to zero by sacrificing the other is not
                          an improvement, and a single "accuracy" number would
                          hide exactly that trade.
ANSWER ACCURACY        -- of the answers it did give, how many contain the
                          decisive fact.
CITATION VERIFICATION  -- share of quotes confirmed against the passage cited.
RETRIEVAL RECALL       -- did the answer cite the document that holds the
                          answer.
CONFIDENCE SEPARATION  -- mean confidence when right minus when wrong. The
                          test of whether compute_confidence() deserves to
                          exist: near zero and it is decoration.

Quota, which shapes how this is run
-----------------------------------
The free tier allows about twenty requests per MODEL per day and one answer
costs two to five calls, so the full set is not always affordable in one
sitting. Hence --limit and --only, and hence the report recording how much of
the set actually ran: a partial run is a legitimate result as long as it says
it is partial. A question whose pipeline could not run at all is graded ERROR
and excluded from the rates rather than counted as a mistake, because a run
that never happened made no claim about the documents.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import dataset
from evaluation.metrics import (
    Graded,
    aggregate,
    aggregate_recall,
    grade,
    recall_at_k,
)
from src import config
from src.agent.graph import PolicyLensAgent
from src.retrieval.store import VectorStore

logger = logging.getLogger("evaluate")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"

GROUPS = {
    "single": dataset.SINGLE_DOCUMENT,
    "cross": dataset.CROSS_DOCUMENT,
    "answerable": dataset.ANSWERABLE,
    "out-of-domain": dataset.OUT_OF_DOMAIN,
    "adversarial": dataset.ADVERSARIAL,
    "unanswerable": dataset.UNANSWERABLE,
    "all": dataset.ALL_QUESTIONS,
}


def measure_recall(
    questions: list[dataset.Question], store: VectorStore, k: int
) -> tuple[list[dict], dict]:
    """Measure retrieval alone, with no LLM involved.

    Worth running every time, and separately from the rest: it asks the vector
    store directly, so the whole set can be measured for free as often as
    wanted, while the daily quota caps end-to-end runs at a handful of
    questions. It also isolates blame -- if an answer is wrong, "was the
    evidence even retrieved" is the first question, and the citation-based
    recall cannot answer it, because citations are produced only after the
    analyzer has read the chunks.

    Only answerable questions are measured. For an unanswerable one there is
    no chunk that SHOULD come back, so recall is undefined rather than zero.

    A LOWER BOUND, not the retriever's live behaviour. This searches on the
    question AS ASKED; the running system searches on the router's rewritten
    query, which is often better targeted. Measured on this corpus the gap is
    real: "Must a female labour do anything after her maternity leave ends?"
    misses here and was answered correctly end to end, because the rewrite
    reached the Rules chunk that the raw wording did not. Treated as a floor
    the figure is still worth having -- a miss means the raw question cannot
    reach the evidence -- but it must not be quoted as the retriever's ceiling.
    """
    rows: list[dict] = []
    for item in questions:
        if not item.expected_documents:
            continue
        # threshold=0.0: this measures the retriever's ranking, not the
        # relevance gate. The gate is calibrated separately in
        # scripts/calibrate_threshold.py, and mixing the two would make a
        # recall miss indistinguishable from a deliberate abstention.
        results = store.retrieve(item.question, top_k=k, threshold=0.0)
        documents = [
            (result.get("metadata") or result).get("source_document")
            for result in results
        ]
        best = max((result.get("similarity", 0.0) for result in results), default=0.0)
        rows.append(
            {
                "question": item.question,
                "topic": item.topic,
                "cross_document": item.cross_document,
                "expected": list(item.expected_documents),
                "retrieved": documents,
                "hit": recall_at_k(documents, item.expected_documents),
                "best_similarity": round(best, 4),
            }
        )
    return rows, aggregate_recall(rows)


def run(questions: list[dataset.Question], agent: PolicyLensAgent) -> list[Graded]:
    """Ask every question and grade the answer."""
    results: list[Graded] = []
    for index, item in enumerate(questions, start=1):
        kind = "answerable" if item.answerable else "unanswerable"
        print(f"[{index}/{len(questions)}] ({kind}) {item.question}")

        started = time.monotonic()
        answer = agent.ask(item.question)
        latency = time.monotonic() - started

        result = grade(item, answer.as_dict(), latency_seconds=latency)
        results.append(result)

        mark = {
            "CORRECT": "ok",
            "WRONG_ANSWER": "WRONG",
            "FALSE_ABSTENTION": "FALSE ABSTENTION",
            "FALSE_ANSWER": "FALSE ANSWER",
            "ERROR": "error",
        }[result.outcome]
        detail = f"{result.status} conf={result.confidence:.2f}"
        if result.total_citations:
            detail += (
                f" cites={result.verified_citations}/{result.total_citations} verified"
            )
        if result.missing_facts:
            detail += f" missing={result.missing_facts}"
        if result.error:
            detail += f" :: {result.error[:90]}"
        print(f"      -> {mark}  {detail}  ({latency:.1f}s)\n")

        if result.outcome == "ERROR" and "every model in the chain" in result.error:
            # Nothing below this point can succeed either, and continuing would
            # fill the report with ERROR rows that say only "the quota ran out".
            print("Stopping: the whole model chain is out of quota.\n")
            break

    return results


def print_report(results: list[Graded], totals: dict) -> None:
    """Print the report a reader is expected to defend."""
    line = "=" * 74
    print(f"\n{line}\nPHASE 6 EVALUATION\n{line}")
    print(f"Dataset        : {dataset.summary()}")
    print(f"Ran            : {totals['questions']} question(s), "
          f"{totals['scored']} scored, {totals['errors']} not run")
    print(f"Model          : {config.LLM_MODEL}")
    print(f"Chain          : {', '.join(config.LLM_FALLBACK_MODELS)}")
    print(f"Threshold      : {config.RETRIEVAL_SIMILARITY_THRESHOLD}")

    print(f"\n{line}\nHONESTY -- the two error types, kept apart\n{line}")
    print(f"  False ANSWER rate     {_pct(totals['false_answer_rate']):>8}   "
          f"answered a question the corpus cannot answer")
    print(f"  False ABSTENTION rate {_pct(totals['false_abstention_rate']):>8}   "
          f"refused a question the corpus answers")

    print(f"\n{line}\nCORRECTNESS AND GROUNDING\n{line}")
    print(f"  Answer accuracy          {_pct(totals['answer_accuracy']):>8}   "
          f"of the answers given, how many carry the decisive fact")
    print(f"  Citation verification    "
          f"{_pct(totals['citation_verification_rate']):>8}   "
          f"of {totals['citations_total']} quotes, confirmed against the passage")
    print(f"  Citation recall          {_pct(totals['citation_recall']):>8}   "
          f"cited every document the answer needs")
    print(f"  Overall correct          {_pct(totals['overall_correct_rate']):>8}   "
          f"right decision, whichever decision was right")

    if totals["cross_document_questions"]:
        print(f"\n{line}\nCROSS-DOCUMENT, BROKEN OUT\n{line}")
        print(
            "Reported on its own because it is the harder half and a combined "
            "figure hides it:\nanswering every cross-document question from "
            "one document alone would still look fine.\n"
        )
        print(f"  Questions             {totals['cross_document_questions']:>8}")
        print(f"  Answer accuracy       "
              f"{_pct(totals['cross_document_accuracy']):>8}")
        print(f"  Cited BOTH documents  "
              f"{_pct(totals['cross_document_both_cited']):>8}   "
              f"citing one of two is a half answer")

    print(f"\n{line}\nIS THE CONFIDENCE FORMULA WORTH KEEPING\n{line}")
    correct = totals["mean_confidence_correct"]
    wrong = totals["mean_confidence_incorrect"]
    separation = totals["confidence_separation"]
    print(f"  mean confidence, correct answers    {_num(correct):>7}")
    print(f"  mean confidence, wrong answers      {_num(wrong):>7}")
    print(f"  separation                          {_num(separation):>7}")
    if separation is None:
        print("  -> not measurable from this run: it needs both a correct and "
              "an incorrect answer to compare.")
    elif separation >= 0.10:
        print("  -> confidence tracks correctness; the formula is doing work.")
    else:
        print("  -> confidence barely separates right from wrong. Re-fit the "
              "weights in config or stop reporting the number.")

    print(f"\n{line}\nOUTCOMES\n{line}")
    for name, count in totals["counts"].items():
        if count:
            print(f"  {name:<18} {count}")

    failures = [r for r in results if r.outcome in {"FALSE_ANSWER", "WRONG_ANSWER",
                                                    "FALSE_ABSTENTION"}]
    if failures:
        print(f"\n{line}\nEVERY FAILURE, IN FULL\n{line}")
        print("Listed because an aggregate that hides its failures cannot be "
              "defended.\n")
        for r in failures:
            print(f"  [{r.outcome}] {r.question}")
            print(f"      topic   : {r.topic}")
            print(f"      status  : {r.status}  confidence {r.confidence:.2f}")
            if r.missing_facts:
                print(f"      missing : {r.missing_facts}")
            if r.answer:
                print(f"      said    : {r.answer[:200]}")
            print()

    print(f"{line}\nCOST\n{line}")
    print(f"  mean latency   {_num(totals['mean_latency_seconds'])} s")
    print(f"  mean LLM calls {_num(totals['mean_llm_calls'])}")
    print(f"  total tokens   {totals['total_tokens']}")


def print_recall(rows: list[dict], totals: dict, k: int) -> None:
    """Print the retrieval-only report."""
    line = "=" * 74
    print(f"\n{line}\nRETRIEVAL ONLY -- Recall@{k}, no LLM involved\n{line}")
    print(
        "Measured straight against the vector store, so it costs no quota and "
        "covers every\nanswerable question rather than the handful a daily "
        "limit allows end to end.\n"
        "A LOWER BOUND: this searches on the question as asked, while the "
        "running system\nsearches on the router's rewritten query, which is "
        "often better targeted.\n"
    )
    print(f"  Recall@{k}, all answerable   "
          f"{_pct(totals['recall_at_k']):>8}  ({totals['questions']} questions)")
    print(f"  Recall@{k}, single-document  "
          f"{_pct(totals['recall_at_k_single']):>8}")
    print(f"  Recall@{k}, cross-document   "
          f"{_pct(totals['recall_at_k_cross']):>8}  "
          f"needs BOTH documents in the top {k}")
    print(f"  mean best similarity      {_num(totals['mean_best_similarity']):>8}")

    misses = [r for r in rows if not r["hit"]]
    if misses:
        print(f"\n{line}\nEVERY MISS\n{line}")
        print(
            "A miss here is an upper bound on the answer: evidence that was "
            "never retrieved\ncannot be cited.\n"
        )
        for r in misses:
            print(f"  {r['question']}")
            print(f"      wanted   : {r['expected']}")
            print(f"      retrieved: {r['retrieved']}")
            print(f"      best similarity {r['best_similarity']}\n")


def save_recall(rows: list[dict], totals: dict, k: int) -> Path:
    """Write the retrieval-only run to evaluation/results/ as JSON.

    Named apart from the end-to-end file because the two measure different
    things, and a reader who found one where they expected the other would
    quote a retrieval number as an answer-accuracy number.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"recall_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(
        json.dumps(
            {
                "ran_at": datetime.now().isoformat(timespec="seconds"),
                "k": k,
                "dataset": dataset.summary(),
                "embedding_model": config.EMBEDDING_MODEL,
                "note": (
                    "Retrieval only: no LLM was called, and the relevance "
                    "threshold was not applied. This measures the retriever's "
                    "ranking, not the abstention gate."
                ),
                "totals": totals,
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def save(results: list[Graded], totals: dict, verification: str = "on") -> Path:
    """Write the full run to evaluation/results/ as JSON.

    The arm is recorded in the file, because an unverified run and a
    verified one are not comparable and a results file that does not say
    which it is would eventually be quoted as if it were the other.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if verification == "on" else "_noverify"
    path = RESULTS_DIR / f"evaluation_{datetime.now():%Y%m%d_%H%M%S}{suffix}.json"
    path.write_text(
        json.dumps(
            {
                "ran_at": datetime.now().isoformat(timespec="seconds"),
                "dataset": dataset.summary(),
                "verification": verification,
                "config": {
                    "model": config.LLM_MODEL,
                    "fallback_models": list(config.LLM_FALLBACK_MODELS),
                    "similarity_threshold": config.RETRIEVAL_SIMILARITY_THRESHOLD,
                    "embedding_model": config.EMBEDDING_MODEL,
                    "confidence_weights": {
                        "verified_ratio": config.CONFIDENCE_WEIGHT_VERIFIED_RATIO,
                        "retrieval_score": config.CONFIDENCE_WEIGHT_RETRIEVAL_SCORE,
                        "penalty_per_retry": config.CONFIDENCE_PENALTY_PER_RETRY,
                    },
                },
                "totals": totals,
                # Every question, not just the aggregate: the per-question rows
                # are what make a disputed number checkable.
                "results": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=sorted(GROUPS),
        default="all",
        help="Run one group of the set (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many questions, to stay inside the daily quota",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Print the report, write no file"
    )
    parser.add_argument(
        "--recall-only",
        action="store_true",
        help=(
            "Measure Recall@k against the vector store and stop. Needs no LLM, "
            "so it runs the whole set for free and ignores the daily quota."
        ),
    )
    parser.add_argument(
        "--k", type=int, default=config.RETRIEVAL_TOP_K, help="k for Recall@k"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "The control arm: build the graph WITHOUT the verifier, so the "
            "analyzer's draft is published unchecked. Use it to measure what "
            "verification is worth -- not a production mode."
        ),
    )
    args = parser.parse_args()

    config.configure_logging()

    questions = GROUPS[args.only]
    if args.limit is not None:
        questions = questions[: max(1, args.limit)]

    if args.recall_only:
        store = VectorStore()
        if store.count() == 0:
            print("The index is empty. Run scripts/ingest_docs.py first.")
            return 1
        rows, recall_totals = measure_recall(questions, store, args.k)
        if not rows:
            print(
                "No answerable questions in this group, so recall is "
                "undefined: there is no chunk that SHOULD come back."
            )
            return 1
        print_recall(rows, recall_totals, args.k)
        if not args.no_save:
            path = save_recall(rows, recall_totals, args.k)
            print(f"\nFull per-question retrieval results: "
                  f"{path.relative_to(Path.cwd())}")
        return 0

    agent = PolicyLensAgent(verify=not args.no_verify)
    if agent.store.count() == 0:
        print("The index is empty. Run scripts/ingest_docs.py first.")
        return 1

    if args.no_verify:
        print(
            "CONTROL ARM: the verifier is OFF, so the analyzer's draft is "
            "published unchecked.\nCitation verification will read 0% by "
            "construction -- nothing was checked. Compare the\nanswer "
            "accuracy and false answer rate against a verified run, not the "
            "citation\nfigures or the confidence.\n"
        )

    print(f"Evaluating against {agent.store.count()} chunks "
          f"from {len(agent.store.document_names())} document(s)\n")

    results = run(questions, agent)
    if not results:
        print("Nothing ran.")
        return 1

    totals = aggregate(results)
    print_report(results, totals)

    if not args.no_save:
        path = save(results, totals, "on" if not args.no_verify else "off")
        print(f"\nFull per-question results: {path.relative_to(Path.cwd())}")

    # Non-zero if the system asserted something the documents do not support.
    # That is the one outcome this project treats as a failed build rather than
    # a number to note, so CI can hold the line on it.
    false_answers = totals["counts"]["FALSE_ANSWER"]
    if false_answers:
        print(f"\nFAIL: {false_answers} false answer(s). "
              f"See the failure list above.")
        return 1
    return 0


if __name__ == "__main__":
    config.configure_logging()
    raise SystemExit(main())
