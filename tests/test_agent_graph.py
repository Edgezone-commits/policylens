"""Tests for the agent graph's control flow.

These run the REAL graph -- the real nodes, edges and retry loop -- against a
scripted LLM and an in-memory store. WHY that matters: the behaviour worth
testing here is not what the model says, it is what the system does with what
the model says. A fabricated quote must end as an abstention, an out-of-scope
question must never reach retrieval, and the retry must terminate. None of that
depends on a real model, and all of it would be untestable in CI if it did.

No test here makes a network call, so the whole file runs in CI without keys.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src import config
from src.agent.graph import PolicyLensAgent
from src.agent.llm import LLMError, LLMResponse
from src.agent.nodes.verifier import citation_divergences, match_score
from src.agent.prompts import ANALYZER_SYSTEM, JUDGE_SYSTEM, ROUTER_SYSTEM

# A chunk long enough that a real-length quote can be taken from it, and that
# clears MIN_CHUNK_CHARS the way an ingested chunk would.
PROBATION_TEXT = (
    "13. Probation Period: A labour may be kept on probation for a maximum "
    "period of six months from the date of appointment. A labour whose work "
    "during the probation period is not satisfactory may be removed from "
    "employment, and such removal shall not be deemed to be a dismissal."
)
GRATUITY_TEXT = (
    "Gratuity (Rule 23): The employer shall deposit an amount equivalent to "
    "eight decimal three three percent of the basic remuneration of the "
    "labour every month in the Social Security Fund as gratuity."
)

PROBATION_QUOTE = (
    "A labour may be kept on probation for a maximum period of six months "
    "from the date of appointment."
)
GRATUITY_QUOTE = (
    "The employer shall deposit an amount equivalent to eight decimal three "
    "three percent of the basic remuneration"
)


def chunk(
    chunk_id: str,
    text: str,
    *,
    document: str = "Nepal_Labour_Act_2074.pdf",
    page: int = 8,
    section: str = "Section 13",
    similarity: float = 0.71,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "source_document": document,
        "page_number": page,
        "section": section,
        "sections": section,
        "similarity": similarity,
    }


class FakeStore:
    """Minimal stand-in for VectorStore: the four methods the nodes call."""

    def __init__(
        self,
        results: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        self._results = results if results is not None else []
        # `is not None`, not `or`: a test that passes documents=[] is testing
        # the empty-index path and must not be silently given the defaults.
        self._documents = (
            documents
            if documents is not None
            else [
                "Nepal_Labour_Act_2074.pdf",
                "Nepal_Labour_Rules_2075_Summary_PKF.pdf",
            ]
        )
        # Recorded so a test can assert HOW retrieval was called -- that a
        # retry widened top_k, or that a comparison searched each document.
        self.calls: list[dict[str, Any]] = []

    def document_names(self) -> list[str]:
        return list(self._documents)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        source_document: str | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {"query": query, "top_k": top_k, "source_document": source_document}
        )
        results = self._results
        if source_document:
            results = [
                item for item in results if item["source_document"] == source_document
            ]
        return [dict(item) for item in results[: top_k or config.RETRIEVAL_TOP_K]]

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        for item in self._results:
            if item["chunk_id"] == chunk_id:
                return dict(item)
        return None


class ScriptedLLM:
    """An LLMClient whose replies are fixed per prompt role.

    Dispatches on the system prompt, which is how the nodes identify
    themselves, so a test can say "the analyzer returns this, the judge returns
    that" without knowing the prompt text. A role whose script runs out repeats
    its last reply -- the retry loop calls the analyzer more than once, and a
    test about retries should not have to count the calls in advance.
    """

    def __init__(
        self,
        router: list[dict[str, Any]] | None = None,
        analyzer: list[dict[str, Any]] | None = None,
        judge: list[dict[str, Any]] | None = None,
        fail: set[str] | None = None,
    ) -> None:
        self.scripts = {
            "router": router
            or [
                {
                    "intent": "DOCUMENT_QA",
                    "target_documents": [],
                    "search_query": "maximum probation period of worker",
                    "reason": "asks about the Act's probation provision",
                }
            ],
            "analyzer": analyzer or [],
            "judge": judge or [],
        }
        self.fail = fail or set()
        self.calls: list[str] = []

    def _role(self, system: str | None) -> str:
        if system == ROUTER_SYSTEM:
            return "router"
        if system == ANALYZER_SYSTEM:
            return "analyzer"
        if system == JUDGE_SYSTEM:
            return "judge"
        raise AssertionError(f"unrecognised system prompt: {system!r}")

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        simple_task: bool = False,
        required_keys: tuple[str, ...] = (),
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], LLMResponse]:
        role = self._role(system)
        self.calls.append(role)

        if role in self.fail:
            raise LLMError(f"scripted failure for {role}")

        script = self.scripts[role]
        if not script:
            raise AssertionError(f"no scripted {role} reply, but one was requested")
        payload = script[0] if len(script) == 1 else script.pop(0)

        missing = [key for key in required_keys if key not in payload]
        assert not missing, f"scripted {role} reply is missing {missing}"

        return payload, LLMResponse(
            text=json.dumps(payload),
            model="scripted/test-model",
            prompt_tokens=100,
            completion_tokens=50,
        )


def build_agent(store: FakeStore, llm: ScriptedLLM) -> PolicyLensAgent:
    return PolicyLensAgent(store=store, llm=llm)  # type: ignore[arg-type]


def node_sequence(answer) -> list[str]:
    return [entry["node"] for entry in answer.trace]


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_verified_citation_gives_an_answered_status():
    store = FakeStore([chunk("act::p8::c1", PROBATION_TEXT)])
    llm = ScriptedLLM(
        analyzer=[
            {
                "answer": "A worker may be kept on probation for at most six months.",
                "citations": [{"evidence": 1, "quote": PROBATION_QUOTE}],
                "insufficient_evidence": False,
            }
        ]
    )
    answer = build_agent(store, llm).ask("What is the maximum probation period?")

    assert answer.status == "ANSWERED"
    assert answer.answered and not answer.abstained
    assert answer.verified_count == 1 and answer.unverified_count == 0
    assert answer.retries == 0
    # The quote is a verbatim substring, so it must be settled by string
    # matching alone -- escalating it to the judge would be a wasted call.
    assert answer.citations[0]["method"] == "exact"
    assert "judge" not in llm.calls
    # Citations carry the locator a reader needs to check the source themselves.
    assert answer.citations[0]["source_document"] == "Nepal_Labour_Act_2074.pdf"
    assert answer.citations[0]["page_number"] == 8
    assert answer.citations[0]["section"] == "Section 13"
    assert node_sequence(answer) == [
        "router",
        "retriever",
        "analyzer",
        "verifier",
        "responder",
    ]
    # 0.6 * 1.0 verified + 0.4 * 0.71 similarity, no retry penalty.
    assert answer.confidence == pytest.approx(0.884, abs=0.001)
    assert answer.usage.calls == 2  # router + analyzer, and nothing more


def test_router_rewrite_is_what_gets_searched():
    """The embedding model matches wording, so the rewrite must reach the store."""
    store = FakeStore([chunk("act::p8::c1", PROBATION_TEXT)])
    llm = ScriptedLLM(
        router=[
            {
                "intent": "DOCUMENT_QA",
                "target_documents": ["Nepal Labour Act 2074"],  # inexact on purpose
                "search_query": "maximum probation period of worker",
                "reason": "probation",
            }
        ],
        analyzer=[
            {
                "answer": "Six months.",
                "citations": [{"evidence": 1, "quote": PROBATION_QUOTE}],
                "insufficient_evidence": False,
            }
        ],
    )
    build_agent(store, llm).ask("hey how long can my boss keep me on trial?")

    assert store.calls[0]["query"] == "maximum probation period of worker"
    # The router's loose document name resolved to the real filename; passing
    # it through unresolved would have filtered retrieval down to nothing.
    assert store.calls[0]["source_document"] == "Nepal_Labour_Act_2074.pdf"


# ---------------------------------------------------------------------------
# Abstention paths
# ---------------------------------------------------------------------------


def test_out_of_scope_question_never_reaches_retrieval():
    store = FakeStore([chunk("act::p8::c1", PROBATION_TEXT)])
    llm = ScriptedLLM(
        router=[
            {
                "intent": "OUT_OF_SCOPE",
                "target_documents": [],
                "search_query": "",
                "reason": "asks for a cake recipe",
            }
        ]
    )
    answer = build_agent(store, llm).ask("What is the recipe for chocolate cake?")

    assert answer.status == "OUT_OF_SCOPE"
    assert answer.abstained and not answer.answered
    assert answer.confidence == 0.0
    assert store.calls == []  # no embedding work, no analyzer call
    assert llm.calls == ["router"]


def test_nothing_above_the_threshold_abstains_after_one_retry():
    """An empty retrieval is the calibrated threshold doing its job."""
    store = FakeStore([])  # the gate in VectorStore.retrieve filtered everything
    llm = ScriptedLLM()
    answer = build_agent(store, llm).ask("Who won the 2022 FIFA World Cup?")

    assert answer.status == "INSUFFICIENT_EVIDENCE"
    assert answer.citations == []
    assert answer.confidence == 0.0
    # The router rewrote the question, so one retry with the user's original
    # wording is worth a round trip -- and exactly one, not a loop.
    assert answer.retries == 1
    assert node_sequence(answer) == [
        "router",
        "retriever",
        "retry",
        "retriever",
        "responder",
    ]
    assert llm.calls == ["router"]  # no analyzer call without evidence
    # The retry searched the original wording as well as the rewrite.
    assert store.calls[-1]["query"] == "Who won the 2022 FIFA World Cup?"
    assert store.calls[-1]["top_k"] == config.RETRIEVAL_TOP_K_ON_RETRY


def test_no_retry_when_the_rewrite_matched_the_question():
    """With nothing new to search for, a retry would be pure waste."""
    store = FakeStore([])
    llm = ScriptedLLM(
        router=[
            {
                "intent": "DOCUMENT_QA",
                "target_documents": [],
                "search_query": "what is the boiling point of water",
                "reason": "no rewrite needed",
            }
        ]
    )
    answer = build_agent(store, llm).ask("What is the boiling point of water")

    assert answer.status == "INSUFFICIENT_EVIDENCE"
    assert answer.retries == 0
    assert node_sequence(answer) == ["router", "retriever", "responder"]


def test_analyzer_abstention_retries_then_gives_up():
    """Chunks can clear the similarity gate and still be about the wrong thing."""
    store = FakeStore([chunk("act::p8::c1", PROBATION_TEXT, similarity=0.42)])
    llm = ScriptedLLM(
        analyzer=[{"answer": "", "citations": [], "insufficient_evidence": True}]
    )
    answer = build_agent(store, llm).ask("What is the entitlement to sick leave?")

    assert answer.status == "INSUFFICIENT_EVIDENCE"
    assert answer.retries == config.MAX_RETRIES  # capped, not looping
    assert llm.calls.count("analyzer") == config.MAX_RETRIES + 1
    assert node_sequence(answer).count("retry") == config.MAX_RETRIES


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_fabricated_quote_is_not_published_as_an_answer():
    """The failure this whole project exists to prevent."""
    store = FakeStore([chunk("act::p8::c1", PROBATION_TEXT)])
    llm = ScriptedLLM(
        analyzer=[
            {
                "answer": "A worker may be kept on probation for up to one year.",
                "citations": [
                    {
                        "evidence": 1,
                        "quote": (
                            "A labour may be kept on probation for a maximum period "
                            "of one year from the date of appointment."
                        ),
                    }
                ],
                "insufficient_evidence": False,
            }
        ],
        judge=[
            {"supported": False, "reason": "the excerpt says six months, not one year"}
        ],
    )
    answer = build_agent(store, llm).ask("What is the maximum probation period?")

    # "one year" against "six months" is a near-miss on characters, so the
    # fuzzy check alone cannot settle it; the judge is what catches it.
    assert "judge" in llm.calls
    assert answer.status == "INSUFFICIENT_EVIDENCE"
    # The withdrawn answer must not survive anywhere in the result.
    assert "one year" not in answer.answer
    assert answer.citations == []
    assert answer.confidence == 0.0
    assert answer.retries == config.MAX_RETRIES


def test_paraphrase_the_fuzzy_check_rejects_can_still_be_verified_by_the_judge():
    store = FakeStore([chunk("rules::p3::c7", GRATUITY_TEXT, section="Rule 23")])
    llm = ScriptedLLM(
        analyzer=[
            {
                "answer": "The employer deposits 8.33% of basic remuneration monthly.",
                "citations": [
                    {
                        "evidence": 1,
                        "quote": (
                            "The employer shall deposit 8.33 percent of the basic "
                            "remuneration of the labour every month as gratuity."
                        ),
                    }
                ],
                "insufficient_evidence": False,
            }
        ],
        judge=[{"supported": True, "reason": "same figure, written as digits"}],
    )
    answer = build_agent(store, llm).ask("How much gratuity does the employer pay?")

    assert answer.status == "ANSWERED"
    assert answer.citations[0]["method"] == "llm_judge"
    assert answer.citations[0]["verified"] is True
    assert answer.retries == 0


def test_partially_verified_answer_is_published_with_the_failure_marked():
    """One sound citation should not be thrown away because another failed."""
    store = FakeStore(
        [
            chunk("act::p8::c1", PROBATION_TEXT, similarity=0.70),
            chunk("rules::p3::c7", GRATUITY_TEXT, section="Rule 23", similarity=0.55),
        ]
    )
    llm = ScriptedLLM(
        analyzer=[
            {
                "answer": "Probation is six months, and gratuity is 10% of salary.",
                "citations": [
                    {"evidence": 1, "quote": PROBATION_QUOTE},
                    {
                        "evidence": 2,
                        "quote": (
                            "The employer shall deposit ten percent of the basic "
                            "remuneration of the labour every month as gratuity."
                        ),
                    },
                ],
                "insufficient_evidence": False,
            }
        ],
        judge=[
            {"supported": False, "reason": "the excerpt says 8.33 percent, not ten"}
        ],
    )
    answer = build_agent(store, llm).ask("What are the probation and gratuity rules?")

    assert answer.status == "PARTIALLY_VERIFIED"
    assert answer.verified_count == 1 and answer.unverified_count == 1
    # No retry: retrying risks losing the citation that was sound.
    assert answer.retries == 0
    assert "retry" not in node_sequence(answer)
    # The reader is told which claim failed, rather than shown a tidied answer.
    failed = [item for item in answer.citations if not item["verified"]]
    assert failed[0]["note"]
    # Verification is weighted at 0.6, so half-verified roughly halves it.
    assert 0.4 < answer.confidence < 0.65


def test_citation_to_an_excerpt_that_was_never_supplied_is_dropped():
    """A reference the system cannot resolve cannot be verified, so it cannot show."""
    store = FakeStore([chunk("act::p8::c1", PROBATION_TEXT)])
    llm = ScriptedLLM(
        analyzer=[
            {
                "answer": "Probation lasts six months.",
                "citations": [{"evidence": 9, "quote": PROBATION_QUOTE}],
                "insufficient_evidence": False,
            }
        ]
    )
    answer = build_agent(store, llm).ask("What is the maximum probation period?")

    assert answer.status == "INSUFFICIENT_EVIDENCE"
    assert answer.citations == []
    rejections = [entry for entry in answer.trace if entry["detail"].get("rejections")]
    assert rejections, "the dropped citation should be explained in the trace"


def test_quote_too_short_to_be_checked_is_rejected():
    """partial_ratio matches a four-word quote against almost anything."""
    store = FakeStore([chunk("act::p8::c1", PROBATION_TEXT)])
    llm = ScriptedLLM(
        analyzer=[
            {
                "answer": "Probation lasts six months.",
                "citations": [{"evidence": 1, "quote": "probation period"}],
                "insufficient_evidence": False,
            }
        ]
    )
    answer = build_agent(store, llm).ask("What is the maximum probation period?")

    assert answer.status == "INSUFFICIENT_EVIDENCE"
    assert answer.citations == []


# ---------------------------------------------------------------------------
# Comparison routing
# ---------------------------------------------------------------------------


def test_comparison_searches_each_document_separately():
    """A pooled top-5 would let the verbose document crowd out the terse one."""
    store = FakeStore(
        [
            chunk("act::p8::c1", PROBATION_TEXT, similarity=0.70),
            chunk(
                "rules::p3::c7",
                GRATUITY_TEXT,
                document="Nepal_Labour_Rules_2075_Summary_PKF.pdf",
                page=3,
                section="Rule 23",
                similarity=0.52,
            ),
        ]
    )
    llm = ScriptedLLM(
        router=[
            {
                "intent": "COMPARISON",
                "target_documents": [
                    "Nepal_Labour_Act_2074.pdf",
                    "Nepal_Labour_Rules_2075_Summary_PKF.pdf",
                ],
                "search_query": "gratuity contribution rate",
                "reason": "asks how the Act and the Rules differ",
            }
        ],
        analyzer=[
            {
                "answer": "The Rules set the rate; the Act sets the obligation.",
                "citations": [{"evidence": 1, "quote": PROBATION_QUOTE}],
                "insufficient_evidence": False,
            }
        ],
    )
    answer = build_agent(store, llm).ask(
        "How do the Act and the Rules differ on gratuity?"
    )

    assert answer.intent == "COMPARISON"
    searched = {call["source_document"] for call in store.calls}
    assert searched == {
        "Nepal_Labour_Act_2074.pdf",
        "Nepal_Labour_Rules_2075_Summary_PKF.pdf",
    }
    # Both documents made it into the evidence, which is the point.
    assert answer.status == "ANSWERED"


def test_comparison_naming_one_document_falls_back_to_plain_qa():
    store = FakeStore([chunk("act::p8::c1", PROBATION_TEXT)])
    llm = ScriptedLLM(
        router=[
            {
                "intent": "COMPARISON",
                "target_documents": ["Nepal_Labour_Act_2074.pdf"],
                "search_query": "probation period",
                "reason": "mentions a comparison but names one document",
            }
        ],
        analyzer=[
            {
                "answer": "Six months.",
                "citations": [{"evidence": 1, "quote": PROBATION_QUOTE}],
                "insufficient_evidence": False,
            }
        ],
    )
    answer = build_agent(store, llm).ask("Compare the probation rules")

    assert answer.intent == "DOCUMENT_QA"


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


def test_router_outage_degrades_to_searching_the_question_as_asked():
    """A router outage should cost answer quality, not the whole answer."""
    store = FakeStore([chunk("act::p8::c1", PROBATION_TEXT)])
    llm = ScriptedLLM(
        analyzer=[
            {
                "answer": "Six months.",
                "citations": [{"evidence": 1, "quote": PROBATION_QUOTE}],
                "insufficient_evidence": False,
            }
        ],
        fail={"router"},
    )
    answer = build_agent(store, llm).ask("What is the maximum probation period?")

    assert answer.status == "ANSWERED"
    assert store.calls[0]["query"] == "What is the maximum probation period?"


def test_analyzer_outage_is_reported_as_an_error_not_an_abstention():
    """ "I found no evidence" and "I could not run" are different claims."""
    store = FakeStore([chunk("act::p8::c1", PROBATION_TEXT)])
    llm = ScriptedLLM(fail={"analyzer"})
    answer = build_agent(store, llm).ask("What is the maximum probation period?")

    assert answer.status == "ERROR"
    assert not answer.abstained
    assert answer.error


def test_unreachable_judge_fails_closed():
    """Not knowing must never be recorded as verified."""
    store = FakeStore([chunk("act::p8::c1", PROBATION_TEXT)])
    llm = ScriptedLLM(
        analyzer=[
            {
                "answer": "Probation may last up to one year.",
                "citations": [
                    {
                        "evidence": 1,
                        "quote": (
                            "A labour may be kept on probation for a maximum period "
                            "of one year from the date of appointment."
                        ),
                    }
                ],
                "insufficient_evidence": False,
            }
        ],
        fail={"judge"},
    )
    answer = build_agent(store, llm).ask("What is the maximum probation period?")

    assert answer.status == "INSUFFICIENT_EVIDENCE"
    assert answer.verified_count == 0


def test_empty_index_is_an_error_with_an_actionable_message():
    store = FakeStore([], documents=[])
    answer = build_agent(store, ScriptedLLM()).ask("What is the probation period?")

    assert answer.status == "ERROR"
    assert "ingest_docs" in answer.error


def test_empty_question_short_circuits():
    answer = build_agent(FakeStore([]), ScriptedLLM()).ask("   ")

    assert answer.status == "ERROR"
    assert answer.usage.calls == 0


# ---------------------------------------------------------------------------
# The citation checks themselves
# ---------------------------------------------------------------------------


def test_verbatim_quote_scores_100_despite_pdf_whitespace():
    """PDF text is full of mid-sentence line breaks; a faithful quote is not."""
    assert (
        match_score(
            "A labour may be kept on probation for a maximum period of six months",
            PROBATION_TEXT.replace(" ", "\n  ", 6),
        )
        == 100.0
    )


def test_fuzzy_score_alone_does_not_notice_a_changed_number():
    """Why the divergence check exists, stated as a test rather than a comment."""
    altered = PROBATION_QUOTE.replace("six months", "one year")
    # Fuzzily this is a near-perfect match...
    assert match_score(altered, PROBATION_TEXT) > 85
    # ...and legally it is the opposite claim, which only this check sees.
    assert citation_divergences(altered, PROBATION_TEXT)


def test_fuzzy_score_alone_does_not_notice_a_dropped_negation():
    """Deleting one word reverses the provision and costs ~3 points of ratio."""
    inverted = "such removal shall be deemed to be a dismissal"
    assert match_score(inverted, PROBATION_TEXT) > 85
    divergences = citation_divergences(inverted, PROBATION_TEXT)
    assert any("'not'" in item for item in divergences)


def test_faithful_quote_has_no_divergences():
    assert citation_divergences(PROBATION_QUOTE, PROBATION_TEXT) == []


def test_changed_quantity_is_reported_in_the_readers_own_terms():
    divergences = citation_divergences(
        PROBATION_QUOTE.replace("six", "nine"), PROBATION_TEXT
    )
    assert divergences == ["the passage does not say 'nine'"]


def test_digits_for_number_words_escalates_rather_than_passing_silently():
    """A legitimate paraphrase the string check cannot confirm on its own."""
    divergences = citation_divergences(
        "deposit 8.33% of the basic remuneration every month", GRATUITY_TEXT
    )
    assert divergences == ["the passage does not say '8.33%'"]


def test_a_number_elsewhere_in_the_chunk_does_not_vouch_for_the_quote():
    """Alignment is why: the check is against the quoted span, not the chunk."""
    chunk_text = (
        "11. Working Hours: No labour shall be deployed for more than eight "
        "hours a day. 13. Probation Period: A labour may be kept on probation "
        "for a maximum period of six months from the date of appointment."
    )
    # "eight" IS in the chunk, but not in the sentence this quote claims.
    divergences = citation_divergences(
        "A labour may be kept on probation for a maximum period of eight months",
        chunk_text,
    )
    assert divergences == ["the passage does not say 'eight'"]


# ---------------------------------------------------------------------------
# The evaluation's control arm: the graph built without the verifier
# ---------------------------------------------------------------------------
#
# verify=False exists to measure what verification is worth. These tests pin
# the two things that would make that measurement dishonest: an unchecked
# answer must not claim to be checked, and a checked answer that fails must
# still be withdrawn when verification IS on.


def test_without_the_verifier_the_draft_is_published_unchecked():
    """The control arm has to actually answer, or the comparison measures nothing."""
    store = FakeStore([chunk("act::p8::c1", PROBATION_TEXT)])
    llm = ScriptedLLM(
        analyzer=[
            {
                "answer": "A worker may be kept on probation for at most six months.",
                "citations": [{"evidence": 1, "quote": PROBATION_QUOTE}],
                "insufficient_evidence": False,
            }
        ]
    )
    answer = PolicyLensAgent(store=store, llm=llm, verify=False).ask(  # type: ignore[arg-type]
        "What is the maximum probation period?"
    )

    assert answer.status == "ANSWERED"
    assert node_sequence(answer) == ["router", "retriever", "analyzer", "responder"]
    # No judge, no verifier: one fewer stage and no extra calls.
    assert "verifier" not in node_sequence(answer)
    assert answer.usage.calls == 2


def test_an_unchecked_citation_never_claims_to_be_verified():
    """Otherwise the control arm would look like the real thing in every log."""
    store = FakeStore([chunk("act::p8::c1", PROBATION_TEXT)])
    llm = ScriptedLLM(
        analyzer=[
            {
                "answer": "Probation is six months.",
                "citations": [{"evidence": 1, "quote": PROBATION_QUOTE}],
                "insufficient_evidence": False,
            }
        ]
    )
    answer = PolicyLensAgent(store=store, llm=llm, verify=False).ask(  # type: ignore[arg-type]
        "What is the maximum probation period?"
    )

    citation = answer.citations[0]
    assert citation["verified"] is False
    assert citation["method"] == "unchecked"
    assert answer.verified_count == 0


def test_a_fabricated_quote_survives_without_the_verifier_and_is_caught_with_it():
    """The whole claim of the project, as one comparison.

    Same scripted model, same fabricated quote. With verification the answer
    is withdrawn; without it, the fabrication is published. This is the
    measurement the evaluation's ON/OFF arms are reporting.
    """
    fabricated = (
        "A labour may be kept on probation for a maximum period of two years "
        "from the date of appointment."
    )
    draft = {
        "answer": "A worker may be kept on probation for up to two years.",
        "citations": [{"evidence": 1, "quote": fabricated}],
        "insufficient_evidence": False,
    }

    with_verification = PolicyLensAgent(
        store=FakeStore([chunk("act::p8::c1", PROBATION_TEXT)]),
        llm=ScriptedLLM(analyzer=[draft, draft], judge=[{"supported": False}] * 4),  # type: ignore[arg-type]
    ).ask("What is the maximum probation period?")

    without_verification = PolicyLensAgent(
        store=FakeStore([chunk("act::p8::c1", PROBATION_TEXT)]),
        llm=ScriptedLLM(analyzer=[draft]),  # type: ignore[arg-type]
        verify=False,
    ).ask("What is the maximum probation period?")

    assert with_verification.status == "INSUFFICIENT_EVIDENCE"
    assert with_verification.citations == []

    assert without_verification.status == "ANSWERED"
    assert "two years" in without_verification.answer
