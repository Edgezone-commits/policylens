"""Verifier node: check each quote against the chunk it claims to come from.

This is the node the project exists for. Everything upstream produces a
plausible answer; this is the only step that asks whether the evidence is real.

Three deliberate choices:

1. The chunk text is re-read from the vector store by chunk_id, not taken from
   the state. The analyzer saw that text and could have echoed it back
   altered; comparing a quote against a copy the analyzer handled would verify
   the analyzer against itself.

2. A cheap deterministic check runs first. RapidFuzz costs microseconds and
   settles the large majority of citations, where the model did copy the text.
   Only a failure escalates to an LLM judge, which costs a call, a few seconds
   of latency, and a slice of the free-tier quota. Judging every citation with
   an LLM would roughly double the cost of every answer to re-confirm what
   string matching already proved.

3. A failed check is reported, never repaired. The node does not edit quotes
   or drop bad citations from the answer -- it marks them. A reader who can see
   "this quote was not found in the cited passage" is better served than one
   shown a tidied answer that hides the fact.

And one correction that a test caught, worth recording because it is the kind
of bug that makes a verification layer worse than none: a fuzzy ratio alone
does NOT catch a changed number or a dropped negation. Swapping "six months"
for "one year" inside a 30-word legal sentence still scores in the nineties,
and deleting a single "not" costs about three points -- so both quotes sailed
through as "verified" while asserting the opposite of the source. Character
similarity is simply the wrong measure for the words that carry the meaning.
Hence citation_divergences() below: the quantities and the negations must MATCH
the span the quote claims to reproduce, not merely resemble it, before a fuzzy
pass counts as verification.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from rapidfuzz import fuzz

from src import config
from src.agent.llm import LLMClient, LLMError, call_record
from src.agent.prompts import JUDGE_SYSTEM, build_judge_prompt
from src.agent.state import AgentState, Citation, trace_entry
from src.retrieval.store import VectorStore

logger = logging.getLogger(__name__)


def normalize(text: str) -> str:
    """Flatten whitespace and unify quote characters for comparison.

    PDF extraction leaves line breaks and runs of spaces mid-sentence, and
    models normalise curly quotes and dashes on the way out. Without this,
    a perfectly faithful quote can score in the eighties purely on invisible
    characters, which would send it to the LLM judge for no reason.
    """
    text = (text or "").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-").replace("…", "...")
    return re.sub(r"\s+", " ", text).strip()


def match_score(quote: str, chunk_text: str) -> float:
    """How well `quote` appears inside `chunk_text`, as 0-100.

    partial_ratio, not ratio: the quote is a SPAN of the chunk, so a correct
    quote of one sentence from a 1000-character chunk scores near 100 on
    partial_ratio and perhaps 20 on whole-string ratio. Using ratio here would
    reject nearly every genuine citation.

    This is also exactly why MIN_CITATION_QUOTE_CHARS exists: partial_ratio is
    generous by design, and a four-word quote clears it against almost any
    text. Short quotes are rejected upstream, in the analyzer, before reaching
    this function.
    """
    quote_n, chunk_n = normalize(quote), normalize(chunk_text)
    if not quote_n or not chunk_n:
        return 0.0
    if quote_n.lower() in chunk_n.lower():
        return 100.0  # exact substring; skip the fuzzy work
    return float(fuzz.partial_ratio(quote_n.lower(), chunk_n.lower()))


# Numbers written as words. Legal text uses both forms, often in the same
# sentence ("eight decimal three three percent"), so digits alone would miss
# most of the quantities in this corpus.
_NUMBER_WORDS = frozenset(
    """
    zero one two three four five six seven eight nine ten eleven twelve
    thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty
    thirty forty fifty sixty seventy eighty ninety hundred thousand million
    half quarter third double triple single
    """.split()
)

# Words whose presence or absence inverts a legal provision. Dropping "not"
# from a clause changes its meaning completely and costs a fuzzy ratio about
# three points, which is nothing.
_NEGATION_WORDS = frozenset(
    "not no never nor neither except unless without exempt prohibited".split()
)

# Digits, decimals and percentages: 6, 8.33, 8.33%, 1,000.
_NUMERIC_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
_WORD_RE = re.compile(r"[a-z]+")


def _critical_tokens(text: str) -> set[str]:
    """The tokens in `text` that carry the quantities and the polarity."""
    lowered = normalize(text).lower()
    tokens = set(_NUMERIC_RE.findall(lowered))
    tokens |= {
        word
        for word in _WORD_RE.findall(lowered)
        if word in _NUMBER_WORDS or word in _NEGATION_WORDS
    }
    return tokens


def aligned_window(quote: str, chunk_text: str) -> str:
    """The span of the passage that the quote best lines up with.

    WHY not compare against the whole chunk: a 1000-character chunk of legal
    prose contains numbers and negations all over it, so "is this number
    anywhere in the chunk?" is far too weak a question -- a quote could borrow
    the "six" from a neighbouring clause and look supported. And the opposite
    check (did the quote DROP a negation?) is impossible against the whole
    chunk, which naturally holds negations the quote never claimed.

    Aligning first makes both checks answerable against the few lines the quote
    actually claims to be reproducing.
    """
    quote_n = normalize(quote).lower()
    chunk_n = normalize(chunk_text).lower()
    if not quote_n or not chunk_n:
        return chunk_n
    if quote_n in chunk_n:
        return quote_n

    try:
        alignment = fuzz.partial_ratio_alignment(quote_n, chunk_n)
    except Exception:  # noqa: BLE001 -- degrade to the whole chunk
        return chunk_n
    if alignment is None:
        return chunk_n
    window = chunk_n[alignment.dest_start : alignment.dest_end]
    return window or chunk_n


def citation_divergences(quote: str, chunk_text: str) -> list[str]:
    """Ways the quote's quantities or polarity differ from its source span.

    This is the check a character ratio cannot make, in both directions:

      * An ALTERED quantity. "six months" -> "one year" is a two-word edit in a
        long sentence: fuzzily a 90-plus match, legally the opposite claim.
      * A DROPPED negation. Deleting "not" from "shall not be deemed to be a
        dismissal" reverses the provision and costs about three points of
        similarity, which is nothing.

    A non-empty result is not a verdict of fabrication -- a faithful quote may
    legitimately render the passage's "eight decimal three three percent" as
    "8.33%", and may quote a sub-clause that leaves a negation behind. It means
    string matching cannot settle this citation, so it goes to the LLM judge
    instead of being waved through.

    Each entry is phrased for a reader, because it ends up in the citation's
    note: a specific "the passage does not say 'one year'" can be checked
    against the source, where a bare "unverified" cannot.
    """
    window = aligned_window(quote, chunk_text)
    quote_tokens = _critical_tokens(quote)
    window_tokens = _critical_tokens(window)

    divergences = [
        f"the passage does not say {token!r}"
        for token in sorted(quote_tokens - window_tokens)
    ]
    # Only negations are checked in this direction. A quote that omits a NUMBER
    # is just quoting a shorter span, which is fine; a quote that omits a "not"
    # is asserting the opposite of its source.
    divergences += [
        f"the passage says {token!r} and the quote omits it"
        for token in sorted((window_tokens & _NEGATION_WORDS) - quote_tokens)
    ]
    return divergences


def build_verifier_node(
    llm: LLMClient, store: VectorStore
) -> Callable[[AgentState], dict]:
    """Create the verifier node bound to an LLM client and a vector store."""

    def verifier_node(state: AgentState) -> dict:
        citations = list(state.get("citations") or [])
        if not citations:
            return {
                "verified_count": 0,
                "unverified_count": 0,
                "verified_ratio": 0.0,
                "trace": [trace_entry("verifier", "No citations to check")],
            }

        checked: list[Citation] = []
        call_records: list[dict] = []
        judged = 0

        for citation in citations:
            result = dict(citation)
            chunk = store.get_chunk(citation["chunk_id"])

            if chunk is None:
                # The chunk id came from a retrieved chunk, so this should not
                # happen -- unless the index changed under us (a re-ingest mid
                # request). Fail closed: unverifiable, not verified.
                result.update(
                    verified=False,
                    method="unknown_chunk",
                    match_score=0.0,
                    note="the cited passage is no longer in the index",
                )
                checked.append(result)  # type: ignore[arg-type]
                continue

            score = match_score(citation["quote"], chunk["text"])
            # An exact substring cannot diverge from itself, so only a fuzzy
            # pass has to be examined for altered numbers or dropped negations.
            divergences = (
                []
                if score == 100.0
                else citation_divergences(citation["quote"], chunk["text"])
            )

            if score >= config.CITATION_FUZZY_MATCH_THRESHOLD and not divergences:
                result.update(
                    verified=True,
                    method="exact" if score == 100.0 else "fuzzy",
                    match_score=round(score, 1),
                    note="",
                )
                checked.append(result)  # type: ignore[arg-type]
                continue

            # String matching cannot settle this one: either the characters
            # diverge, or they match closely while a number or a negation in
            # the quote is absent from the passage. Either way it may still be
            # a faithful paraphrase -- a model writing "8.33%" for the
            # passage's "eight decimal three three percent" -- so ask the judge
            # before calling it fabricated.
            if divergences:
                logger.info(
                    "Citation scored %.0f but diverges on specifics (%s); "
                    "escalating to the LLM judge",
                    score,
                    "; ".join(divergences),
                )
            else:
                logger.info(
                    "Citation scored %.0f (below %d); escalating to the LLM judge",
                    score,
                    config.CITATION_FUZZY_MATCH_THRESHOLD,
                )
            try:
                payload, response = llm.complete_json(
                    build_judge_prompt(citation["quote"], chunk["text"]),
                    system=JUDGE_SYSTEM,
                    simple_task=True,  # a yes/no judgement; no thinking budget
                    required_keys=("supported",),
                    # Short cap: the judge returns a verdict and one sentence.
                    max_tokens=256,
                )
            except LLMError as error:
                # Fail closed. An unreachable judge means we do not know, and
                # "unverified, judge unavailable" is honest where "verified"
                # would be a guess in the system's own favour.
                logger.warning("Citation judge unavailable: %s", error)
                result.update(
                    verified=False,
                    method="judge_unavailable",
                    match_score=round(score, 1),
                    note="; ".join(
                        [*divergences, f"could not be checked: {type(error).__name__}"]
                    ),
                )
                checked.append(result)  # type: ignore[arg-type]
                continue

            judged += 1
            call_records.append(call_record(response, "citation_judge"))
            supported = bool(payload.get("supported", False))
            reason = str(payload.get("reason") or "").strip()
            if not reason:
                reason = (
                    "judged faithful to the passage"
                    if supported
                    else "not found in the cited passage"
                )
            if not supported and divergences:
                # Name the specific divergence. "the passage does not say 'one
                # year'" is checkable by the reader; "not supported" is not.
                reason += " (" + "; ".join(divergences) + ")"
            result.update(
                verified=supported,
                method="llm_judge",
                match_score=round(score, 1),
                note=reason,
            )
            checked.append(result)  # type: ignore[arg-type]

        verified_count = sum(1 for citation in checked if citation["verified"])
        unverified_count = len(checked) - verified_count
        verified_ratio = verified_count / len(checked)

        logger.info(
            "Verified %d/%d citation(s)%s",
            verified_count,
            len(checked),
            f" ({judged} needed the LLM judge)" if judged else "",
        )

        return {
            "citations": checked,
            "verified_count": verified_count,
            "unverified_count": unverified_count,
            "verified_ratio": verified_ratio,
            "llm_calls": call_records,
            "trace": [
                trace_entry(
                    "verifier",
                    f"Verified {verified_count} of {len(checked)} citation(s)"
                    + (f"; {judged} needed the LLM judge" if judged else ""),
                    results=[
                        {
                            "source_document": citation["source_document"],
                            "page_number": citation.get("page_number"),
                            "section": citation.get("section"),
                            "verified": citation["verified"],
                            "method": citation["method"],
                            "match_score": citation["match_score"],
                            "note": citation.get("note", ""),
                        }
                        for citation in checked
                    ],
                )
            ],
        }

    return verifier_node
