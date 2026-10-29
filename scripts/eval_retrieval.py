"""Measure retrieval alone: hit rate@k, recall@k and MRR for k = 1, 3, 5, 10.

    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --ks 1 3 5 10 20
    python scripts/eval_retrieval.py --no-save

WHY this is a separate script from evaluate.py rather than another flag on it:
it calls no LLM. Nothing here touches the daily quota, so it runs on the whole
answerable set as often as wanted, while an end-to-end run is rationed to a
handful of questions a day. Anything measurable for free should be measurable
without the thing that costs money being in the same code path.

WHAT IT ADDS over the Recall@5 that `evaluate.py --recall-only` already
reports: rank. That figure asks "did the top five contain every document the
answer needs", and a retriever that puts the evidence fifth scores the same as
one that puts it first. Hit rate at k=1 and MRR separate those, and the shape
of the curve from k=1 to k=10 is what says whether a retrieval change helped
ranking or merely widened the net.

READ THIS BEFORE QUOTING THE NUMBERS. Two limits, both structural:

  * Ground truth here is DOCUMENT-level (dataset.Question.expected_documents),
    not a set of gold chunk ids. "Rank 1" means the top chunk came from a
    document the answer needs -- and the Act is 234 of the index's 279 chunks,
    so that is easy to satisfy and sometimes satisfied by accident. The
    figures are generous by construction. They are for comparing retrieval
    changes against each other, not for claiming the retriever is good.
  * It searches on the question AS ASKED. The running system searches on the
    router's rewritten query, which is usually better targeted, so these are a
    FLOOR on live behaviour rather than a measurement of it.

The relevance threshold is deliberately not applied (threshold=0.0). This
measures ranking; the gate is calibrated separately by
scripts/calibrate_threshold.py, and mixing them would make a ranking miss
indistinguishable from a deliberate abstention.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import dataset
from evaluation.metrics import DEFAULT_KS, aggregate_ranked
from src import config
from src.retrieval.store import VectorStore

logger = logging.getLogger("eval_retrieval")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"


def measure(
    questions: list[dataset.Question], store: VectorStore, depth: int
) -> list[dict]:
    """Retrieve once per question at the deepest k, and record the ranking.

    One search per question rather than one per k: the top 3 of a top-10
    search is the top 3, so searching four times would cost four times as much
    to produce the same rows -- and, more importantly, would let the four
    depths disagree if the store were ever non-deterministic.

    Only answerable questions are measured. An unanswerable one has no chunk
    that SHOULD come back, so its rank is undefined rather than zero.
    """
    rows: list[dict] = []
    for item in questions:
        if not item.expected_documents:
            continue
        results = store.retrieve(item.question, top_k=depth, threshold=0.0)
        rows.append(
            {
                "question": item.question,
                "topic": item.topic,
                "cross_document": item.cross_document,
                "expected": list(item.expected_documents),
                # Ordered best-first, one entry per retrieved chunk. Documents
                # repeat, which is the point: three Act chunks ahead of the
                # first Rules chunk is exactly the failure this measures.
                "retrieved": [
                    (result.get("metadata") or result).get("source_document")
                    for result in results
                ],
                "similarities": [
                    round(result.get("similarity", 0.0), 4) for result in results
                ],
            }
        )
    return rows


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_report(rows: list[dict], totals: dict) -> None:
    line = "=" * 78
    print(f"\n{line}\nRETRIEVAL METRICS -- no LLM, no quota\n{line}")
    print(
        f"{totals['questions']} answerable questions, searched to depth "
        f"{totals['searched_depth']}, embedding {config.EMBEDDING_MODEL}.\n"
        "Document-level ground truth, searched on the question as asked: a "
        "FLOOR on the\nlive retriever, which searches the router's rewrite.\n"
    )

    print("  Hit rate@k -- at least one needed document in the top k")
    print(f"    {'k':>4}  {'all':>8}  {'single':>8}  {'cross':>8}")
    for k in totals["ks"]:
        row = totals["at_k"][str(k)]
        print(
            f"    {k:>4}  {_pct((row['all'] or {}).get('hit_rate')):>8}  "
            f"{_pct((row['single_document'] or {}).get('hit_rate')):>8}  "
            f"{_pct((row['cross_document'] or {}).get('hit_rate')):>8}"
        )

    print("\n  Recall@k -- EVERY needed document in the top k")
    print(f"    {'k':>4}  {'all':>8}  {'single':>8}  {'cross':>8}")
    for k in totals["ks"]:
        row = totals["at_k"][str(k)]
        print(
            f"    {k:>4}  {_pct((row['all'] or {}).get('recall_strict')):>8}  "
            f"{_pct((row['single_document'] or {}).get('recall_strict')):>8}  "
            f"{_pct((row['cross_document'] or {}).get('recall_strict')):>8}"
        )

    print("\n  Recall@k, fraction of needed documents found (partial credit)")
    print(f"    {'k':>4}  {'all':>8}  {'single':>8}  {'cross':>8}")
    for k in totals["ks"]:
        row = totals["at_k"][str(k)]
        print(
            f"    {k:>4}  {_num((row['all'] or {}).get('recall_fraction')):>8}  "
            f"{_num((row['single_document'] or {}).get('recall_fraction')):>8}  "
            f"{_num((row['cross_document'] or {}).get('recall_fraction')):>8}"
        )

    mrr = totals["mrr"]
    print(
        f"\n  MRR  all {_num(mrr['all'])}   single {_num(mrr['single_document'])}"
        f"   cross {_num(mrr['cross_document'])}"
    )
    print(
        "\n  Read the gap between hit rate and recall on the cross-document "
        "row: it is\n  the difference between reaching one needed document "
        "and reaching both."
    )


def markdown(rows: list[dict], totals: dict) -> str:
    """A summary a README or a PR can quote, generated rather than retyped."""
    mrr = totals["mrr"]
    lines = [
        "# Retrieval metrics",
        "",
        f"Generated by `scripts/eval_retrieval.py` on "
        f"{datetime.now():%Y-%m-%d %H:%M}. No LLM was called.",
        "",
        f"- Questions: **{totals['questions']}** (answerable only)",
        f"- Search depth: {totals['searched_depth']}",
        f"- Embedding model: `{config.EMBEDDING_MODEL}`",
        f"- Index: {totals.get('chunks_indexed', 'n/a')} chunks",
        "",
        "Ground truth is document-level and the search uses the question as",
        "asked, without the router's rewrite. Both make these a floor on the",
        "live retriever rather than a measurement of it.",
        "",
        "## Hit rate@k — at least one needed document in the top k",
        "",
        "| k | all | single-document | cross-document |",
        "|---|---|---|---|",
    ]
    for k in totals["ks"]:
        row = totals["at_k"][str(k)]
        lines.append(
            f"| {k} | {_pct((row['all'] or {}).get('hit_rate'))} | "
            f"{_pct((row['single_document'] or {}).get('hit_rate'))} | "
            f"{_pct((row['cross_document'] or {}).get('hit_rate'))} |"
        )

    lines += [
        "",
        "## Recall@k — every needed document in the top k",
        "",
        "| k | all | single-document | cross-document |",
        "|---|---|---|---|",
    ]
    for k in totals["ks"]:
        row = totals["at_k"][str(k)]
        lines.append(
            f"| {k} | {_pct((row['all'] or {}).get('recall_strict'))} | "
            f"{_pct((row['single_document'] or {}).get('recall_strict'))} | "
            f"{_pct((row['cross_document'] or {}).get('recall_strict'))} |"
        )

    lines += [
        "",
        "## MRR",
        "",
        "| set | MRR |",
        "|---|---|",
        f"| all | {_num(mrr['all'])} |",
        f"| single-document | {_num(mrr['single_document'])} |",
        f"| cross-document | {_num(mrr['cross_document'])} |",
        "",
        "Reciprocal rank of the **first** chunk from a needed document, so a",
        "cross-document question scores well here for reaching either half of",
        "the evidence. Compare it against the strict recall row above.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help="The k values to report (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--only",
        choices=["answerable", "single", "cross"],
        default="answerable",
        help="Which answerable questions to measure (default: all of them)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Print the report, write no file"
    )
    parser.add_argument(
        "--index",
        default=None,
        help=(
            "Measure a different Chroma directory than config.CHROMA_DIR. "
            "WHY this exists: retrieval metrics are only comparable across "
            "runs if the corpus is the same, and a local index drifts as "
            "documents are uploaded through the API. Pointing this at an "
            "index built from a known document set makes a number "
            "reproducible instead of a property of whatever happens to be "
            "indexed today."
        ),
    )
    parser.add_argument(
        "--out-prefix",
        default="retrieval",
        help=(
            "Basename for the written files (default: retrieval -> "
            "retrieval.json, retrieval.md). Lets a second corpus be recorded "
            "without overwriting the first."
        ),
    )
    args = parser.parse_args()

    config.configure_logging()

    groups = {
        "answerable": dataset.ANSWERABLE,
        "single": dataset.SINGLE_DOCUMENT,
        "cross": dataset.CROSS_DOCUMENT,
    }
    questions = groups[args.only]

    store = VectorStore(persist_dir=args.index) if args.index else VectorStore()
    if store.count() == 0:
        print("The index is empty. Run scripts/ingest_docs.py first.")
        return 1

    ks = sorted({k for k in args.ks if k > 0})
    if not ks:
        print("No positive k values given.")
        return 1

    rows = measure(questions, store, depth=max(ks))
    if not rows:
        print("No answerable questions in this group; rank is undefined.")
        return 1

    totals = aggregate_ranked(rows, ks)
    totals["chunks_indexed"] = store.count()
    print_report(rows, totals)

    if args.no_save:
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # A FIXED filename, unlike evaluate.py's timestamped files. This one is
    # cheap to regenerate and is quoted by the README, so the README needs a
    # stable path to point at; the timestamp lives in the file's own contents.
    json_path = RESULTS_DIR / f"{args.out_prefix}.json"
    json_path.write_text(
        json.dumps(
            {
                "ran_at": datetime.now().isoformat(timespec="seconds"),
                "dataset": dataset.summary(),
                "group": args.only,
                "embedding_model": config.EMBEDDING_MODEL,
                "chunks_indexed": store.count(),
                "documents": store.list_documents(),
                "note": (
                    "Retrieval only: no LLM, and the relevance threshold was "
                    "not applied. Document-level ground truth, searched on the "
                    "question as asked rather than the router's rewrite -- a "
                    "floor on the live retriever."
                ),
                "totals": totals,
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path = RESULTS_DIR / f"{args.out_prefix}.md"
    md_path.write_text(markdown(rows, totals), encoding="utf-8")
    print(f"\nWrote {json_path.name} and {md_path.name} to evaluation/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
