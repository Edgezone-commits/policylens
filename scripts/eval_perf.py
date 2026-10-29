"""Latency and cost as distributions, derived from recorded evaluation runs.

    python scripts/eval_perf.py                 # every verified run on disk
    python scripts/eval_perf.py --from evaluation/results/evaluation_20261006_132958.json
    python scripts/eval_perf.py --include-unverified
    python scripts/eval_perf.py --no-save

WHY this reads recorded runs instead of asking questions itself: a p95 needs a
sample, and a sample costs 2-5 LLM calls per question against a quota of about
twenty requests per model per day. The runs are already on disk with
per-question `latency_seconds`, `llm_calls` and `total_tokens` in them, so the
distribution can be computed as often as wanted for free -- and recomputed by
anyone checking the README without spending a single call.

WHY percentiles rather than the mean the run already prints: the mean is the
statistic a retry destroys. One question that fails over to a second model
takes several times as long as a typical one, and across 38 questions that
single case moves the mean by seconds while the median does not notice. p50 is
what a user usually waits; p95 is what the slow tail costs them.

WHAT THIS IS NOT: a benchmark. These are wall-clock seconds against a free
tier with a deliberate 4-second throttle between calls
(config.LLM_MIN_SECONDS_BETWEEN_CALLS), measured on one machine, on one day,
over tens of questions. The throttle alone accounts for several seconds of a
two-call answer. Read it as "what this system costs to use as configured", not
as a property of the model or of the retrieval.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import performance_summary
from src import config

logger = logging.getLogger("eval_perf")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "results"


def load_runs(
    paths: list[Path], include_unverified: bool
) -> tuple[list[dict], list[str]]:
    """Read the per-question rows out of recorded runs.

    Returns (rows, sources). Rows are deduplicated by question text, keeping
    the FIRST occurrence: the same question appears in several runs during a
    day of partial runs, and counting it twice would weight whichever question
    was retried most heavily in the distribution.

    Unverified (control-arm) runs are excluded by default. They are a
    different system -- no verifier node, so fewer LLM calls by construction --
    and merging their latencies into a verified distribution would quietly
    report a faster system than the one being described.
    """
    rows: list[dict] = []
    sources: list[str] = []
    seen: set[str] = set()

    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Skipping %s: %s", path.name, error)
            continue

        verification = payload.get("verification", "unknown")
        if verification != "on" and not include_unverified:
            logger.info("Skipping %s (verification=%s)", path.name, verification)
            continue

        new = 0
        for row in payload.get("results") or []:
            question = row.get("question", "")
            if question in seen:
                continue
            seen.add(question)
            rows.append(row)
            new += 1
        if new:
            sources.append(
                f"{path.name} ({new} questions, verification={verification})"
            )

    return rows, sources


def _num(value: float | None, unit: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{unit}"


def _int(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def print_report(summary: dict, sources: list[str]) -> None:
    line = "=" * 78
    print(f"\n{line}\nLATENCY AND COST -- distributions, not means\n{line}")
    print(f"{summary['questions']} questions from {len(sources)} recorded run(s):")
    for source in sources:
        print(f"  - {source}")
    if summary["excluded_errors"]:
        print(
            f"\n  {summary['excluded_errors']} ERROR question(s) excluded: a run "
            "that failed on a\n  missing key returns in milliseconds and would "
            "pull the distribution towards zero."
        )

    latency = summary["latency_seconds"]
    calls = summary["llm_calls"]
    tokens = summary["tokens_per_question"]

    print(f"\n  {'':<22}{'p50':>10}{'p95':>10}{'min':>10}{'max':>10}{'mean':>10}")
    print(
        f"  {'latency (s)':<22}{_num(latency['p50']):>10}{_num(latency['p95']):>10}"
        f"{_num(latency['min']):>10}{_num(latency['max']):>10}{_num(latency['mean']):>10}"
    )
    print(
        f"  {'LLM calls':<22}{_num(calls['p50']):>10}{_num(calls['p95']):>10}"
        f"{_num(calls['min']):>10}{_num(calls['max']):>10}{_num(calls['mean']):>10}"
    )
    print(
        f"  {'tokens / question':<22}{_int(tokens['p50']):>10}{_int(tokens['p95']):>10}"
        f"{_int(tokens['min']):>10}{_int(tokens['max']):>10}{_int(tokens['mean']):>10}"
    )
    print(f"\n  tokens, all questions: {summary['tokens_total']:,}")
    print(
        f"\n  Includes a deliberate {config.LLM_MIN_SECONDS_BETWEEN_CALLS:.0f}s throttle "
        "between LLM calls\n  (config.LLM_MIN_SECONDS_BETWEEN_CALLS), which is most of "
        "a typical answer's\n  latency. It is there to stay inside free-tier "
        "per-minute limits."
    )


def markdown(summary: dict, sources: list[str]) -> str:
    latency = summary["latency_seconds"]
    calls = summary["llm_calls"]
    tokens = summary["tokens_per_question"]
    return "\n".join(
        [
            "# Latency and cost",
            "",
            f"Generated by `scripts/eval_perf.py` on {datetime.now():%Y-%m-%d %H:%M} "
            "from runs already recorded in this directory. No LLM was called.",
            "",
            f"- Questions: **{summary['questions']}**"
            + (
                f" ({summary['excluded_errors']} ERROR excluded)"
                if summary["excluded_errors"]
                else ""
            ),
            "- Sources:",
            *[f"  - `{source}`" for source in sources],
            "",
            "| | p50 | p95 | min | max | mean |",
            "|---|---|---|---|---|---|",
            f"| Latency (s) | {_num(latency['p50'])} | {_num(latency['p95'])} | "
            f"{_num(latency['min'])} | {_num(latency['max'])} | "
            f"{_num(latency['mean'])} |",
            f"| LLM calls | {_num(calls['p50'])} | {_num(calls['p95'])} | "
            f"{_num(calls['min'])} | {_num(calls['max'])} | {_num(calls['mean'])} |",
            f"| Tokens / question | {_int(tokens['p50'])} | {_int(tokens['p95'])} | "
            f"{_int(tokens['min'])} | {_int(tokens['max'])} | {_int(tokens['mean'])} |",
            "",
            f"Tokens across all questions: **{summary['tokens_total']:,}**.",
            "",
            "These are wall-clock seconds as configured, including a deliberate",
            f"{config.LLM_MIN_SECONDS_BETWEEN_CALLS:.0f}s throttle between LLM calls to "
            "stay inside free-tier per-minute",
            "limits — most of a typical answer's latency is that wait. Not a model",
            "benchmark.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from",
        dest="sources",
        nargs="+",
        default=None,
        help="Specific result files to read (default: every evaluation_*.json)",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help=(
            "Also read control-arm runs. Off by default: a graph without the "
            "verifier makes fewer calls, so merging the two would describe a "
            "faster system than the one being reported."
        ),
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Print the report, write no file"
    )
    args = parser.parse_args()

    config.configure_logging()

    if args.sources:
        paths = [Path(source) for source in args.sources]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            print(f"No such file(s): {', '.join(str(path) for path in missing)}")
            return 1
    else:
        paths = sorted(RESULTS_DIR.glob("evaluation_*.json"))

    if not paths:
        print(
            "No recorded runs in evaluation/results/. Run "
            "`python scripts/evaluate.py` first -- that one does cost quota."
        )
        return 1

    rows, sources = load_runs(paths, args.include_unverified)
    if not rows:
        print("No usable per-question rows in those files.")
        return 1

    summary = performance_summary(rows)
    print_report(summary, sources)

    if args.no_save:
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ran_at": datetime.now().isoformat(timespec="seconds"),
        "derived_from": sources,
        "throttle_seconds": config.LLM_MIN_SECONDS_BETWEEN_CALLS,
        "note": (
            "Derived from recorded runs, so no LLM was called to produce it. "
            "Wall-clock seconds as configured, including the inter-call "
            "throttle; not a model benchmark."
        ),
        "summary": summary,
    }
    (RESULTS_DIR / "perf.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "perf.md").write_text(markdown(summary, sources), encoding="utf-8")
    print("\nWrote perf.json and perf.md to evaluation/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
