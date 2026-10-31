"""Calibrate the retrieval relevance threshold on real score distributions.

    python scripts/calibrate_threshold.py

The threshold is the single most consequential number in the retrieval layer:
it decides when the agent says "I have evidence" versus "I do not". Picking it
by intuition would mean the system's honesty rests on a guess, so this script
measures it.

Method
------
Two labelled probe sets are run against the live index, and only the top-1
similarity of each is recorded (the gate in VectorStore.retrieve looks at the
best chunk, so that is the quantity to calibrate):

  * SHOULD_RETRIEVE   -- questions whose answers really are in the corpus.
                         A threshold above these scores causes a FALSE
                         ABSTENTION: the system refuses a question it could
                         have answered.
  * SHOULD_ABSTAIN    -- questions on topics the corpus never mentions.
                         A threshold below these scores causes a FALSE ANSWER:
                         the system answers from irrelevant text, which is how
                         confident hallucinations happen.

The chosen threshold maximises balanced accuracy -- the mean of the two
per-class accuracies -- rather than plain accuracy. WHY: plain accuracy can be
gamed by whichever probe set is larger, and these two error types are not
equally cheap. A false abstention is visibly unhelpful; a false answer is
confidently wrong, which for a legal document is the worse failure.

IMPORTANT -- these probes are deliberately NOT the Phase 6 evaluation
questions. Tuning a parameter on the same questions used to report final
metrics would make those metrics measure the tuning, not the system. This set
is small, hand-written, and used only to fix one scalar.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.retrieval.store import VectorStore

logger = logging.getLogger("calibrate_threshold")

# Answers to these are present in the indexed documents. Verified by reading
# the documents, not assumed -- e.g. probation is Section 13 of the Labour Act
# (six months), not Section 11.
SHOULD_RETRIEVE: list[str] = [
    "What is the maximum probation period for a worker?",
    "How much does the employer contribute to gratuity?",
    "How many hours may a labour be made to work in a day?",
    "What leave is given on the death of a family member?",
    "What rate is paid for overtime work?",
    "How is the minimum remuneration fixed?",
    "How many weekly holidays is a labour entitled to?",
    "What is the entitlement to maternity leave?",
    "How much is contributed to the provident fund?",
    "When can an employer terminate an employment contract?",
    "Does this licence grant a patent licence to the user?",
    "Is the software provided with any warranty?",
    "May I use the licensor's trade marks?",
    "What must an employment contract disclose?",
]

# Topics the corpus has nothing to say about. Kept clearly out of domain on
# purpose: questions that merely SOUND related (e.g. work visas, which brush
# against the Act's foreign-worker permit provisions) are a different and
# harder failure mode, and belong in the Phase 6 adversarial set rather than
# here, where they would blur the boundary this threshold is meant to draw.
SHOULD_ABSTAIN: list[str] = [
    "What is the recipe for chocolate cake?",
    "What is the capital city of Brazil?",
    "How do I change a flat car tyre?",
    "What is the syntax of a Python list comprehension?",
    "What are the early symptoms of diabetes?",
    "Who won the 2022 FIFA World Cup?",
    "How do I train a convolutional neural network?",
    "What is the current price of Bitcoin?",
    "Explain the leg-before-wicket rule in cricket.",
    "How long should sourdough bread prove?",
    "What is the boiling point of water at sea level?",
    "How do I reset a Wi-Fi router?",
]

# Thresholds to sweep. 0.00-0.60 in steps of 0.01 covers the whole plausible
# range for this embedding model, whose scores here span roughly 0.10-0.75.
_SWEEP = [round(0.01 * step, 2) for step in range(0, 61)]


def top_similarity(store: VectorStore, question: str) -> tuple[float, str]:
    """Return the best similarity for a question, and where it came from."""
    # threshold=0.0 so nothing is filtered -- we want the raw score, which is
    # exactly what the gate will later be compared against.
    results = store.retrieve(question, top_k=1, threshold=0.0)
    if not results:
        return 0.0, "-"
    best = results[0]
    where = f"{best['source_document']} p{best['page_number']} {best['section'] or '-'}"
    return best["similarity"], where


def evaluate_threshold(
    threshold: float, retrieve_scores: list[float], abstain_scores: list[float]
) -> dict[str, float]:
    """Score one candidate threshold against both probe sets."""
    # An answerable question is handled correctly when its best chunk clears
    # the bar, so the agent proceeds to answer.
    retrieved_ok = sum(1 for score in retrieve_scores if score >= threshold)
    # An unanswerable question is handled correctly when nothing clears the
    # bar, so the agent abstains.
    abstained_ok = sum(1 for score in abstain_scores if score < threshold)

    recall = retrieved_ok / len(retrieve_scores)
    abstention = abstained_ok / len(abstain_scores)
    return {
        "threshold": threshold,
        "answerable_retained": recall,
        "unanswerable_abstained": abstention,
        "balanced_accuracy": (recall + abstention) / 2,
        "false_abstentions": len(retrieve_scores) - retrieved_ok,
        "false_answers": len(abstain_scores) - abstained_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write the full result to evaluation/results/ as JSON",
    )
    args = parser.parse_args()

    config.configure_logging()
    # The per-query INFO lines would bury the report; the report IS the output.
    logging.getLogger("src.retrieval.store").setLevel(logging.WARNING)

    store = VectorStore()
    if store.count() == 0:
        logger.error("Index is empty. Run: python scripts/ingest_docs.py data/raw/")
        return 1

    print(
        f"\nCalibrating against {store.count()} chunks in "
        f"{len(store.list_documents())} document(s)\n"
    )

    print("=" * 78)
    print("PROBE SET A -- answers ARE in the corpus (should retrieve)")
    print("=" * 78)
    retrieve_scores: list[float] = []
    for question in SHOULD_RETRIEVE:
        score, where = top_similarity(store, question)
        retrieve_scores.append(score)
        # Keep the locator intact: truncating it to fit a column once made a
        # correct "Section 106" read as "Section 10" and sent me hunting a bug
        # that did not exist.
        print(f"  {score:.3f}  {question[:52]:<53} {where}")

    print("\n" + "=" * 78)
    print("PROBE SET B -- answers are NOT in the corpus (should abstain)")
    print("=" * 78)
    abstain_scores: list[float] = []
    for question in SHOULD_ABSTAIN:
        score, _ = top_similarity(store, question)
        abstain_scores.append(score)
        print(f"  {score:.3f}  {question[:52]}")

    print("\n" + "=" * 78)
    print("SCORE DISTRIBUTIONS")
    print("=" * 78)
    print(f"{'':<22}{'min':>8}{'median':>9}{'mean':>8}{'max':>8}")
    for label, scores in (
        ("answerable (A)", retrieve_scores),
        ("unanswerable (B)", abstain_scores),
    ):
        print(
            f"{label:<22}{min(scores):>8.3f}{statistics.median(scores):>9.3f}"
            f"{statistics.mean(scores):>8.3f}{max(scores):>8.3f}"
        )

    # The gap between the two sets is what makes a threshold possible at all.
    # If B's maximum exceeded A's minimum, NO single threshold could separate
    # them and the honest conclusion would be that this signal is insufficient.
    gap_low, gap_high = max(abstain_scores), min(retrieve_scores)
    print(
        f"\nSeparation: highest unanswerable = {gap_low:.3f}, "
        f"lowest answerable = {gap_high:.3f}"
    )
    if gap_high > gap_low:
        print(
            f"  -> cleanly separable; any threshold in "
            f"({gap_low:.3f}, {gap_high:.3f}] classifies every probe correctly"
        )
    else:
        print(
            "  -> OVERLAPPING: no threshold separates these sets perfectly. "
            "Some error is unavoidable; the sweep below picks the best "
            "available trade-off."
        )

    sweep = [evaluate_threshold(t, retrieve_scores, abstain_scores) for t in _SWEEP]
    best_score = max(row["balanced_accuracy"] for row in sweep)
    best_rows = [row for row in sweep if row["balanced_accuracy"] == best_score]
    # Among equally good thresholds, take the midpoint of the optimal band
    # rather than an endpoint: an endpoint sits right against a probe score, so
    # a slightly different document or question would flip its classification.
    recommended = round((best_rows[0]["threshold"] + best_rows[-1]["threshold"]) / 2, 3)

    print("\n" + "=" * 78)
    print("THRESHOLD SWEEP (every 0.05 shown)")
    print("=" * 78)
    print(
        f"{'thresh':>7}{'answerable kept':>18}{'unanswerable':>15}"
        f"{'balanced':>11}{'false abst.':>13}{'false ans.':>12}"
    )
    for row in sweep:
        if abs(row["threshold"] * 100 % 5) > 1e-6:
            continue
        marker = " <--" if row in best_rows else ""
        print(
            f"{row['threshold']:>7.2f}{row['answerable_retained']:>17.0%}"
            f"{row['unanswerable_abstained']:>15.0%}"
            f"{row['balanced_accuracy']:>11.1%}"
            f"{row['false_abstentions']:>13}{row['false_answers']:>12}{marker}"
        )

    print("\n" + "=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    print(
        f"  Optimal band:   {best_rows[0]['threshold']:.2f} to "
        f"{best_rows[-1]['threshold']:.2f}  "
        f"(balanced accuracy {best_score:.1%})"
    )
    print(f"  Recommended:    {recommended:.2f}  (midpoint of the band)")
    print(
        f"  Currently set:  {config.RETRIEVAL_SIMILARITY_THRESHOLD:.2f}  "
        f"(RETRIEVAL_SIMILARITY_THRESHOLD in src/config.py)"
    )
    # Judge the configured value by its MARGIN, not by its distance from the
    # recommendation. A threshold can sit inside the optimal band and still be
    # fragile: at 0.30 the nearest off-topic probe scored 0.297, so a 0.003
    # shift would have produced a confident wrong answer. Distance-to-optimal
    # hides that; margin-to-nearest-error does not.
    configured = config.RETRIEVAL_SIMILARITY_THRESHOLD
    margin_below = configured - gap_low  # headroom before a false answer
    margin_above = gap_high - configured  # headroom before a false abstention
    print(f"\n  Margin at the configured {configured:.2f}:")
    print(
        f"    {margin_below:+.3f} to the nearest false answer "
        f"(highest off-topic probe)"
    )
    print(
        f"    {margin_above:+.3f} to the nearest false abstention "
        f"(weakest answerable probe)"
    )

    if margin_below <= 0 or margin_above <= 0:
        print(
            f"\n  -> The configured value MISCLASSIFIES a probe. "
            f"Update config.py to {recommended:.2f}"
        )
    elif min(margin_below, margin_above) < 0.02:
        print(
            f"\n  -> Correct but thin (under 0.02 of margin). "
            f"Consider {recommended:.2f} for more headroom."
        )
    else:
        print(
            "\n  -> Configured value classifies every probe with margin "
            "on both sides."
        )
    print()

    if args.save:
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "indexed_documents": store.list_documents(),
            "total_chunks": store.count(),
            "probes": {
                "answerable": dict(zip(SHOULD_RETRIEVE, retrieve_scores, strict=True)),
                "unanswerable": dict(zip(SHOULD_ABSTAIN, abstain_scores, strict=True)),
            },
            "separation": {
                "highest_unanswerable": gap_low,
                "lowest_answerable": gap_high,
                "cleanly_separable": gap_high > gap_low,
            },
            "sweep": sweep,
            "recommended_threshold": recommended,
            "configured_threshold": config.RETRIEVAL_SIMILARITY_THRESHOLD,
        }
        out_dir = config.PROJECT_ROOT / "evaluation" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"threshold_calibration_{stamp}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved to {out_path.relative_to(config.PROJECT_ROOT)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
