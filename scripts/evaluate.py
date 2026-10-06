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

from evaluation import dataset  # noqa: E402
from evaluation.metrics import Graded, aggregate, grade  # noqa: E402
from src import config  # noqa: E402
from src.agent.graph import PolicyLensAgent  # noqa: E402

logger = logging.getLogger("evaluate")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"

GROUPS = {
    "answerable": dataset.ANSWERABLE,
    "out-of-domain": dataset.OUT_OF_DOMAIN,
    "adversarial": dataset.ADVERSARIAL,
    "unanswerable": dataset.OUT_OF_DOMAIN + dataset.ADVERSARIAL,
    "all": dataset.ALL_QUESTIONS,
}


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
    print(f"  Retrieval recall         {_pct(totals['retrieval_recall']):>8}   "
          f"cited the document that holds the answer")
    print(f"  Overall correct          {_pct(totals['overall_correct_rate']):>8}   "
          f"right decision, whichever decision was right")

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


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def save(results: list[Graded], totals: dict) -> Path:
    """Write the full run to evaluation/results/ as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"evaluation_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(
        json.dumps(
            {
                "ran_at": datetime.now().isoformat(timespec="seconds"),
                "dataset": dataset.summary(),
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
    args = parser.parse_args()

    config.configure_logging()

    questions = GROUPS[args.only]
    if args.limit is not None:
        questions = questions[: max(1, args.limit)]

    agent = PolicyLensAgent()
    if agent.store.count() == 0:
        print("The index is empty. Run scripts/ingest_docs.py first.")
        return 1

    print(f"Evaluating against {agent.store.count()} chunks "
          f"from {len(agent.store.document_names())} document(s)\n")

    results = run(questions, agent)
    if not results:
        print("Nothing ran.")
        return 1

    totals = aggregate(results)
    print_report(results, totals)

    if not args.no_save:
        path = save(results, totals)
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
