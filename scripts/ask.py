"""CLI to ask the agent a question.

    python scripts/ask.py "What is the maximum probation period?"
    python scripts/ask.py "How much gratuity?" --document Nepal_Labour_Act_2074.pdf
    python scripts/ask.py "Compare leave in the Act and the Rules" --trace
    python scripts/ask.py "What is the probation period?" --json

This exists before the API and the UI on purpose: it is the shortest path to
seeing the whole pipeline run, and the only one that still works when FastAPI
or Streamlit is mid-change. --trace prints the reasoning path, which is what
makes an abstention or a failed citation explicable rather than mysterious.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.agent.graph import AgentAnswer, PolicyLensAgent  # noqa: E402

logger = logging.getLogger("ask")

_WIDTH = 78

# How each outcome is introduced. An abstention is not an error and must not
# read like one -- "no answer found" is the system working as designed, and the
# wording is what tells a reader that.
_STATUS_HEADINGS = {
    "ANSWERED": "ANSWER (all citations verified)",
    "PARTIALLY_VERIFIED": "ANSWER (some citations could NOT be verified -- see below)",
    "INSUFFICIENT_EVIDENCE": "NO ANSWER -- the documents do not support one",
    "OUT_OF_SCOPE": "NOT ANSWERED -- outside the indexed documents",
    "ERROR": "FAILED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask the PolicyLens agent a question about the indexed documents."
    )
    parser.add_argument("question", help="The question to ask")
    parser.add_argument(
        "--document",
        help="Restrict retrieval to one indexed document (filename as ingested)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print the agent's reasoning path, node by node",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the full result as JSON (for piping into an evaluation)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging"
    )
    return parser.parse_args()


def print_answer(answer: AgentAnswer, show_trace: bool) -> None:
    print()
    print("=" * _WIDTH)
    print(_STATUS_HEADINGS.get(answer.status, answer.status))
    print("=" * _WIDTH)

    if answer.error:
        print(f"\n{answer.error}\n")
        return

    print()
    print(textwrap.fill(answer.answer, width=_WIDTH))

    if answer.citations:
        print()
        print("-" * _WIDTH)
        print("CITATIONS")
        print("-" * _WIDTH)
        for index, citation in enumerate(answer.citations, start=1):
            # The verdict goes first: a reader scanning the list should see
            # which quotes are confirmed before reading any of them.
            mark = "VERIFIED  " if citation["verified"] else "UNVERIFIED"
            locator = citation["source_document"]
            if citation.get("page_number"):
                locator += f", page {citation['page_number']}"
            if citation.get("section"):
                locator += f", {citation['section']}"

            print(f"\n[{index}] {mark}  {locator}")
            print(
                f"     method: {citation['method']} "
                f"(match {citation['match_score']:.0f}/100)"
            )
            for line in textwrap.wrap(
                f'"{citation["quote"]}"', width=_WIDTH - 5
            ):
                print(f"     {line}")
            if citation.get("note"):
                for line in textwrap.wrap(
                    f"note: {citation['note']}",
                    width=_WIDTH - 5,
                    subsequent_indent="      ",
                ):
                    print(f"     {line}")

    if show_trace:
        print()
        print("-" * _WIDTH)
        print("REASONING PATH")
        print("-" * _WIDTH)
        for step, entry in enumerate(answer.trace, start=1):
            print(f"{step}. {entry['node']}: {entry['summary']}")

    print()
    print("-" * _WIDTH)
    if answer.answered:
        print(
            f"confidence {answer.confidence:.2f} | "
            f"{answer.verified_count}/{answer.verified_count + answer.unverified_count} "
            f"citations verified | best match {answer.top_similarity:.2f}"
        )
    print(
        f"intent {answer.intent or '-'} | retries {answer.retries} | "
        f"{answer.usage.calls} LLM call(s), "
        f"{answer.usage.prompt_tokens + answer.usage.completion_tokens} tokens"
        + (f" | {answer.usage.fallback_calls} on the fallback model"
           if answer.usage.fallback_calls else "")
    )
    print()


def main() -> int:
    args = parse_args()
    config.configure_logging("DEBUG" if args.verbose else None)

    if args.as_json:
        # The reasoning path is the output in JSON mode; log lines interleaved
        # with it would make the stream unparsable for a caller piping it.
        logging.getLogger().setLevel(logging.WARNING)

    agent = PolicyLensAgent()

    if args.document and args.document not in agent.store.document_names():
        logger.error(
            "Not an indexed document: %s\nIndexed: %s",
            args.document,
            ", ".join(agent.store.document_names()) or "(none)",
        )
        return 1

    answer = agent.ask(args.question, source_document=args.document)

    if args.as_json:
        print(json.dumps(answer.as_dict(), indent=2, ensure_ascii=False))
    else:
        print_answer(answer, show_trace=args.trace)

    # Non-zero only for a pipeline failure. An abstention is a correct outcome,
    # and a script that treats it as an error would make "no evidence" look
    # like a bug every time the system was being honest.
    return 1 if answer.status == "ERROR" else 0


if __name__ == "__main__":
    raise SystemExit(main())
