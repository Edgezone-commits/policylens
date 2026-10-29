"""Tests for the Phase 6 grader.

No agent, no model, no key: the grader takes a question label and a dict in
the shape AgentAnswer.as_dict() returns, so all of it is testable on strings.
That matters more here than elsewhere, because this code decides the numbers
the project reports about itself. A grader that quietly miscounts a false
answer as a pass would make every metric worthless while looking fine.
"""

from __future__ import annotations

from typing import Any

from evaluation.dataset import (
    ACT,
    ALL_QUESTIONS,
    ANSWERABLE,
    APACHE,
    CROSS_DOCUMENT,
    RULES,
    SINGLE_DOCUMENT,
    Question,
)
from evaluation.metrics import (
    aggregate,
    aggregate_recall,
    facts_present,
    grade,
    normalise,
    recall_at_k,
)


def answer(
    status: str = "ANSWERED",
    text: str = "A worker may be kept on probation for at most six months.",
    citations: list[dict[str, Any]] | None = None,
    confidence: float = 0.85,
    **extra: Any,
) -> dict[str, Any]:
    """An AgentAnswer.as_dict() payload, with the fields the grader reads."""
    base = {
        "status": status,
        "answer": text,
        "citations": (
            citations
            if citations is not None
            else [
                {
                    "source_document": "Nepal_Labour_Act_2074.pdf",
                    "verified": True,
                    "quote": "remains in a probation period for six months",
                }
            ]
        ),
        "confidence": confidence,
        "top_similarity": 0.62,
        "retries": 0,
        "usage": {"calls": 2, "prompt_tokens": 1500, "completion_tokens": 300},
        "error": "",
    }
    base.update(extra)
    return base


PROBATION = ANSWERABLE[0]

UNANSWERABLE = Question(
    question="How many days of paternity leave does the Act provide?",
    answerable=False,
    topic="adversarial",
)


# ---------------------------------------------------------------------------
# Normalising and fact matching
# ---------------------------------------------------------------------------


def test_normalise_flattens_case_whitespace_and_pdf_quoting():
    messy = "The  employer’s\n“basic   remuneration” – see above"
    assert normalise(messy) == 'the employer\'s "basic remuneration" - see above'


def test_normalise_keeps_the_decimal_point_that_decides_a_grade():
    """Stripping punctuation would turn 8.33 into 833 and lose the gratuity rate."""
    assert "8.33" in normalise("Deposit 8.33 percent of basic remuneration.")


def test_a_fact_group_is_satisfied_by_any_of_its_spellings():
    groups = [["forty-eight", "forty eight", "48"]]
    assert facts_present("no more than 48 hours a week", groups) == [True]
    assert facts_present("no more than forty-eight hours a week", groups) == [True]
    assert facts_present("no more than fifty hours a week", groups) == [False]


def test_every_group_must_be_satisfied_not_just_one():
    """Two groups means "say both", which is how multi-part answers are graded."""
    groups = [["eight hour"], ["forty-eight"]]
    assert facts_present("eight hours a day", groups) == [True, False]
    assert facts_present("eight hours a day, forty-eight a week", groups) == [
        True,
        True,
    ]


# ---------------------------------------------------------------------------
# Grading an answerable question
# ---------------------------------------------------------------------------


def test_an_answer_with_the_decisive_fact_is_correct():
    result = grade(PROBATION, answer())

    assert result.outcome == "CORRECT"
    assert result.correct
    assert result.cited_expected_documents is True
    assert result.verified_citations == 1


def test_an_answer_missing_the_fact_is_wrong_and_says_which_fact():
    """A low score has to be actionable, not just bad news."""
    result = grade(PROBATION, answer(text="Probation lasts up to one year."))

    assert result.outcome == "WRONG_ANSWER"
    assert result.missing_facts == [["six month", "6 month"]]


def test_abstaining_on_an_answerable_question_is_a_false_abstention():
    result = grade(
        PROBATION, answer(status="INSUFFICIENT_EVIDENCE", text="No evidence.")
    )

    assert result.outcome == "FALSE_ABSTENTION"


def test_a_partially_verified_answer_is_still_graded_on_its_content():
    """It asserted an answer, so it is held to one -- the status only records doubt."""
    result = grade(PROBATION, answer(status="PARTIALLY_VERIFIED"))

    assert result.outcome == "CORRECT"


def test_citing_the_wrong_document_is_recorded_without_failing_the_answer():
    """Recall and correctness are separate measurements, so they are kept separate."""
    result = grade(
        PROBATION,
        answer(
            citations=[
                {
                    "source_document": "Apache_License_2.0.txt",
                    "verified": True,
                    "quote": "x",
                }
            ]
        ),
    )

    assert result.outcome == "CORRECT"
    assert result.cited_expected_documents is False


# ---------------------------------------------------------------------------
# Grading an unanswerable question
# ---------------------------------------------------------------------------


def test_abstaining_on_an_unanswerable_question_is_correct():
    """The behaviour the project exists for, so it scores as a success."""
    for status in ("INSUFFICIENT_EVIDENCE", "OUT_OF_SCOPE"):
        result = grade(
            UNANSWERABLE, answer(status=status, text="Not in these documents.")
        )
        assert result.outcome == "CORRECT", status


def test_answering_an_unanswerable_question_is_a_false_answer():
    """The expensive failure: it looks exactly like a correct answer."""
    result = grade(
        UNANSWERABLE,
        answer(text="The Act provides fifteen days of paternity leave."),
    )

    assert result.outcome == "FALSE_ANSWER"


def test_a_pipeline_failure_is_not_counted_as_either_mistake():
    """A run that never happened made no claim about the documents."""
    result = grade(PROBATION, answer(status="ERROR", text="", error="quota exhausted"))

    assert result.outcome == "ERROR"
    totals = aggregate([result])
    # Excluded from both rates rather than counted against either.
    assert totals["false_abstention_rate"] is None
    assert totals["errors"] == 1


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_the_two_error_rates_are_computed_over_their_own_denominators():
    """A false answer rate diluted by answerable questions would flatter the system."""
    results = [
        grade(PROBATION, answer()),  # correct
        grade(PROBATION, answer(status="OUT_OF_SCOPE")),  # false abstention
        grade(UNANSWERABLE, answer(status="INSUFFICIENT_EVIDENCE")),  # correct
        grade(UNANSWERABLE, answer(text="Fifteen days.")),  # false answer
    ]
    totals = aggregate(results)

    assert totals["false_abstention_rate"] == 0.5  # 1 of 2 answerable
    assert totals["false_answer_rate"] == 0.5  # 1 of 2 unanswerable
    assert totals["counts"]["CORRECT"] == 2


def test_citation_verification_rate_is_over_all_quotes_not_all_questions():
    results = [
        grade(
            PROBATION,
            answer(
                citations=[
                    {"source_document": "Nepal_Labour_Act_2074.pdf", "verified": True},
                    {"source_document": "Nepal_Labour_Act_2074.pdf", "verified": False},
                    {"source_document": "Nepal_Labour_Act_2074.pdf", "verified": True},
                ]
            ),
        )
    ]
    totals = aggregate(results)

    assert totals["citations_total"] == 3
    assert totals["citation_verification_rate"] == 0.667


def test_confidence_separation_needs_both_a_right_and_a_wrong_answer():
    """With only correct answers there is no separation to measure, and 0.0 would lie."""
    only_correct = aggregate([grade(PROBATION, answer(confidence=0.9))])
    assert only_correct["confidence_separation"] is None

    both = aggregate(
        [
            grade(PROBATION, answer(confidence=0.9)),
            grade(PROBATION, answer(confidence=0.4, text="One year.")),
        ]
    )
    assert both["confidence_separation"] == 0.5


def test_an_empty_run_reports_nothing_rather_than_zeroes():
    totals = aggregate([])

    assert totals["questions"] == 0
    assert totals["false_answer_rate"] is None
    assert totals["answer_accuracy"] is None


# ---------------------------------------------------------------------------
# The dataset itself
# ---------------------------------------------------------------------------


def test_every_answerable_question_carries_a_fact_and_a_document():
    """An answerable question with nothing to check would silently always pass."""
    for item in ANSWERABLE:
        assert item.expected_facts, item.question
        assert item.expected_documents, item.question


def test_every_question_records_where_its_label_came_from():
    """The provenance note is what makes a disputed label checkable."""
    for item in ALL_QUESTIONS:
        assert item.source.strip(), item.question


def test_no_question_appears_twice():
    questions = [item.question for item in ALL_QUESTIONS]
    assert len(questions) == len(set(questions))


# ---------------------------------------------------------------------------
# Cross-document grading
# ---------------------------------------------------------------------------


CROSS = Question(
    question="Gratuity is a percentage of basic remuneration -- what is it defined as?",
    answerable=True,
    expected_facts=[["60%"]],
    expected_documents=[ACT, RULES],
    topic="cross",
    source="Act s53 sets the rate; Rules r77 defines the base.",
)


def cite(document: str, verified: bool = True) -> dict:
    return {"source_document": document, "verified": verified, "quote": "x" * 50}


def test_a_cross_document_answer_needs_every_document_cited():
    """Citing one of the two is a half answer, and must not score as grounded."""
    both = grade(CROSS, answer(text="60% of gross", citations=[cite(ACT), cite(RULES)]))
    one = grade(CROSS, answer(text="60% of gross", citations=[cite(ACT)]))

    assert both.cited_expected_documents is True
    assert one.cited_expected_documents is False
    # Both still CORRECT: the fact is present. Grounding is a separate
    # measurement and is reported separately.
    assert both.correct and one.correct


def test_cross_document_results_are_reported_on_their_own():
    """28 single-document questions would otherwise swamp 10 cross-document ones."""
    single = grade(PROBATION, answer())
    cross_ok = grade(CROSS, answer(text="60%", citations=[cite(ACT), cite(RULES)]))
    cross_half = grade(CROSS, answer(text="60%", citations=[cite(ACT)]))

    totals = aggregate([single, cross_ok, cross_half])

    assert totals["cross_document_questions"] == 2
    assert totals["cross_document_accuracy"] == 1.0
    # The number that matters: half the cross-document answers cited only one
    # document, which a combined recall figure would have hidden.
    assert totals["cross_document_both_cited"] == 0.5


def test_a_single_document_question_is_not_counted_as_cross_document():
    assert grade(PROBATION, answer()).cross_document is False
    assert grade(CROSS, answer()).cross_document is True


# ---------------------------------------------------------------------------
# Recall@k -- retrieval measured on its own, without an LLM
# ---------------------------------------------------------------------------


def test_recall_at_k_requires_every_expected_document():
    """Retrieving one of two documents cannot answer a cross-document question."""
    assert recall_at_k([ACT, RULES, APACHE], [ACT, RULES]) is True
    assert recall_at_k([ACT, APACHE], [ACT, RULES]) is False
    assert recall_at_k([ACT], [ACT]) is True


def test_recall_at_k_is_false_when_nothing_is_expected():
    """An empty expectation is a dataset bug, not a free pass."""
    assert recall_at_k([ACT], []) is False


def test_recall_is_aggregated_separately_for_single_and_cross():
    rows = [
        {"cross_document": False, "hit": True, "best_similarity": 0.62},
        {"cross_document": False, "hit": True, "best_similarity": 0.55},
        {"cross_document": True, "hit": False, "best_similarity": 0.48},
        {"cross_document": True, "hit": True, "best_similarity": 0.51},
    ]
    totals = aggregate_recall(rows)

    assert totals["recall_at_k"] == 0.75
    assert totals["recall_at_k_single"] == 1.0
    assert totals["recall_at_k_cross"] == 0.5


def test_empty_recall_reports_nothing_rather_than_zero():
    totals = aggregate_recall([])

    assert totals["questions"] == 0
    assert totals["recall_at_k"] is None


# ---------------------------------------------------------------------------
# The expanded dataset
# ---------------------------------------------------------------------------


def test_the_dataset_has_all_four_groups_populated():
    assert len(SINGLE_DOCUMENT) >= 25
    assert len(CROSS_DOCUMENT) == 10


def test_every_cross_document_question_really_needs_two_documents():
    """Otherwise it is a single-document question in the wrong list."""
    for item in CROSS_DOCUMENT:
        assert len(item.expected_documents) > 1, item.question
        assert item.cross_document, item.question


def test_no_single_document_question_expects_two_documents():
    for item in SINGLE_DOCUMENT:
        assert len(item.expected_documents) == 1, item.question


def test_unanswerable_questions_carry_no_expected_facts():
    """A fact to find would contradict the label."""
    from evaluation.dataset import UNANSWERABLE

    for item in UNANSWERABLE:
        assert not item.expected_facts, item.question
        assert not item.expected_documents, item.question


def test_every_expected_document_is_one_that_is_actually_indexed():
    """A typo in a filename would silently fail recall for that question."""
    known = {ACT, RULES, APACHE}
    for item in ANSWERABLE:
        for document in item.expected_documents:
            assert document in known, (item.question, document)
