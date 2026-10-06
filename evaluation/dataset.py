"""The labelled question set the Phase 6 evaluation runs against.

Every question here is one of two kinds, and the distinction is the whole
point of the set:

  * ANSWERABLE   -- the corpus really does answer it, so an abstention is a
                    FALSE ABSTENTION: the system refused something it could
                    have answered.
  * UNANSWERABLE -- the corpus does not answer it, so an answer is a FALSE
                    ANSWER: the system asserted something the documents do
                    not support, which is the failure this project exists to
                    prevent.

WHY the expected facts are quoted from the documents rather than from what the
law actually says. Twice while writing this set the corpus turned out to
disagree with the statute as published -- Section 40 reads "the weekly leave
of one day in each month", which is not what the Labour Act means -- and in
both cases the corpus is what the system can be held to. Grading against
external truth would mark a correct reading of the supplied text as wrong, and
measure the OCR rather than the agent. Each entry therefore carries the
`source` note that was read out of the index to justify it.

WHY the grader is key-fact containment and not an LLM judge:

  * It is deterministic, so a reported number can be reproduced tomorrow, and
    a change in the score means the system changed rather than the judge did.
  * It costs no quota. The free tier allows about twenty requests per model
    per day (see config.LLM_FALLBACK_MODELS); spending half of that grading
    would halve the questions that can be asked.
  * An LLM judge is the thing being evaluated. Using the same family of model
    to mark its own homework measures agreement, not correctness.

Its limit, stated plainly because the report has to be defensible: it checks
that the decisive fact is PRESENT, not that everything around it is right. An
answer containing "six months" plus an unrelated falsehood still passes. That
is why citation verification is reported beside it -- the two together are a
much stronger claim than either alone.

THE LABELLING RULE, which the first run forced into the open. A question is
ANSWERABLE when the corpus contains text that DECIDES it -- including text
that decides it in the negative. "What notice period applies for misconduct?"
looks unanswerable and is not: Section 144(1) excludes misconduct explicitly,
so "none applies" is the correct answer and abstaining would be the mistake.
That question was mislabelled here, scored as a false answer on the first run,
and moved. Two Apache questions failed the same test and were replaced.

The consequence for writing new ones: an unanswerable question must be one
where no honest statement of absence answers it either. "Which country's law
governs this licence?" qualifies -- there is no clause and nothing to say about
one. "Does this licence require publishing source?" does not, because "no, it
imposes no such condition" is a correct reading of Section 4.

UNANSWERABLE questions are split into two groups on purpose. The plainly
out-of-domain ones ("recipe for chocolate cake") are the easy case and mostly
test the router. The adversarial ones are the interesting case: they sound
exactly like something this corpus would cover, use its vocabulary, and are
simply not in it. scripts/calibrate_threshold.py deliberately kept those out
of its probe set and left them here, so the threshold was not tuned on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Question:
    """One labelled question.

    `expected_facts` is a list of ALTERNATIVE GROUPS: the answer passes if it
    contains at least one string from every group. One group with several
    spellings means "say this, however you spell it" ("8.33" or "eight point
    three"); two groups mean "say both of these".
    """

    question: str
    answerable: bool
    expected_facts: list[list[str]] = field(default_factory=list)
    # The document a correct answer has to cite. Checked, unlike the section,
    # which schemas.py calls a best-effort locator and which drifts by one on
    # this corpus (the chunk whose text begins "13. Provision relating to
    # probation period" carries the metadata "Section 12"). Grading on a
    # locator the system never promised would measure the PDF's heading
    # detection, not the answer.
    expected_document: str | None = None
    # Why this is the right answer, read out of the index rather than assumed.
    source: str = ""
    # Short label for the results table.
    topic: str = ""


# ---------------------------------------------------------------------------
# Answerable -- the corpus contains the answer
# ---------------------------------------------------------------------------

ANSWERABLE: list[Question] = [
    Question(
        question="What is the maximum probation period for a worker?",
        answerable=True,
        expected_facts=[["six month", "6 month"]],
        expected_document="Nepal_Labour_Act_2074.pdf",
        source=(
            'Section 13: "the employer may so enter into the contract that he '
            'or she remains in a probation period for six months"'
        ),
        topic="probation",
    ),
    Question(
        question="How many hours may a labour be made to work in a day and in a week?",
        answerable=True,
        expected_facts=[["eight hour", "8 hour"], ["forty-eight", "forty eight", "48"]],
        expected_document="Nepal_Labour_Act_2074.pdf",
        source=(
            'Section 28(1): "No employer shall employ labours to work more '
            'than eight hours a day and forty-eight hours a week"'
        ),
        topic="working hours",
    ),
    Question(
        question="How much must the employer deposit for gratuity each month?",
        answerable=True,
        expected_facts=[["8.33", "eight point three"]],
        expected_document="Nepal_Labour_Act_2074.pdf",
        source=(
            'Section 53(1): "deduct an amount equivalent to eight point three '
            'and three (8.33) percent of the basic remuneration of each '
            'labour each month and deposit it for the purpose of gratuity"'
        ),
        topic="gratuity",
    ),
    Question(
        question="How much overtime work is a labour allowed to do?",
        answerable=True,
        expected_facts=[["four hour", "4 hour"], ["twenty-four", "twenty four", "24"]],
        expected_document="Nepal_Labour_Act_2074.pdf",
        source=(
            'Section 30(1): "such a labour may be caused to work not '
            'exceeding four hours a day and twenty-four hours a week"'
        ),
        topic="overtime limit",
    ),
    Question(
        question="How much paid public leave does a labour get each year?",
        answerable=True,
        expected_facts=[["thirteen", "13"]],
        expected_document="Nepal_Labour_Act_2074.pdf",
        source=(
            'Section 41(1): "a paid public leave of thirteen days including '
            'the May Day, and fourteen days including International Women\'s '
            'Day" (thirteen is the general entitlement)'
        ),
        topic="public leave",
    ),
    Question(
        question="How long is maternity leave and how much of it is paid?",
        answerable=True,
        expected_facts=[["sixty", "60"]],
        expected_document="Nepal_Labour_Act_2074.pdf",
        source=(
            'Section 45(3): "shall get full remuneration for sixty days and '
            'shall not get remuneration for the remaining period"'
        ),
        topic="maternity leave",
    ),
    Question(
        question="What is a labour entitled to as a festival expense?",
        answerable=True,
        expected_facts=[["one month", "a month", "basic remuneration of one"]],
        expected_document="Nepal_Labour_Act_2074.pdf",
        source=(
            'Section 37(1): "entitled to an amount equal to the basic '
            'remuneration of one month as the festival expense each year"'
        ),
        topic="festival expense",
    ),
    Question(
        question="How much weekly leave is a labour entitled to?",
        answerable=True,
        expected_facts=[["one day", "1 day"]],
        expected_document="Nepal_Labour_Act_2074.pdf",
        source=(
            'Section 40: "A labour shall get the weekly leave of one day in '
            'each month." The published Act means one day per week; the '
            'indexed text says "in each month", so "one day" is the only part '
            'that can be graded without marking a faithful reading wrong.'
        ),
        topic="weekly leave",
    ),
    Question(
        question="What notice period applies to terminating a labour for misconduct?",
        answerable=True,
        expected_facts=[["not apply", "no notice", "without notice", "except", "exempt"]],
        expected_document="Nepal_Labour_Act_2074.pdf",
        source=(
            'Section 144(1) requires notice "in any circumstance except when '
            'employment is terminated upon action taken for misconduct". '
            "MOVED HERE from the adversarial set after the first run. It was "
            "labelled unanswerable on the reasoning that it asks for the one "
            "case the provision excludes -- but the corpus decides it "
            "explicitly, so the correct answer is 'none applies, because "
            "misconduct is excluded', which is what the agent gave, with a "
            "verified citation. The label was wrong, not the answer. Stating "
            "an exclusion the text states is answering, not hallucinating; "
            "abstaining here would have been the FALSE ABSTENTION."
        ),
        topic="exclusion from notice",
    ),
    Question(
        question="Does the Apache License grant a patent licence to the user?",
        answerable=True,
        expected_facts=[["patent"], ["grant", "yes", "does"]],
        expected_document="Apache_License_2.0.txt",
        source=(
            'Section 3 "Grant of Patent License": "each Contributor hereby '
            'grants to You a perpetual, worldwide, non-exclusive, no-charge, '
            'royalty-free, irrevocable ... patent license"'
        ),
        topic="patent grant",
    ),
    Question(
        question="Does the Apache License provide any warranty with the software?",
        answerable=True,
        expected_facts=[["without warrant", "no warrant", "as is", '"as is"']],
        expected_document="Apache_License_2.0.txt",
        source=(
            'Section 7 "Disclaimer of Warranty": the Work is provided on an '
            '"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND'
        ),
        topic="warranty",
    ),
    Question(
        question="May I use the licensor's trade marks under the Apache License?",
        answerable=True,
        expected_facts=[["does not grant", "not grant", "no permission", "except"]],
        expected_document="Apache_License_2.0.txt",
        source=(
            'Section 6 "Trademarks": "This License does not grant permission '
            'to use the trade names, trademarks, service marks, or product '
            'names of the Licensor"'
        ),
        topic="trade marks",
    ),
]


# ---------------------------------------------------------------------------
# Unanswerable -- plainly out of domain
# ---------------------------------------------------------------------------

OUT_OF_DOMAIN: list[Question] = [
    Question(
        question="What is the recipe for chocolate cake?",
        answerable=False,
        source="Neither document mentions food.",
        topic="recipe",
    ),
    Question(
        question="Who won the football World Cup in 2014?",
        answerable=False,
        source="Neither document mentions sport.",
        topic="sport",
    ),
    Question(
        question="What is the capital city of Brazil?",
        answerable=False,
        source="Neither document is about geography.",
        topic="geography",
    ),
    Question(
        question="How do I treat a bacterial throat infection?",
        answerable=False,
        source="Neither document gives medical advice.",
        topic="medicine",
    ),
]


# ---------------------------------------------------------------------------
# Unanswerable -- adversarial
# ---------------------------------------------------------------------------
#
# These are the ones worth reporting. Each is phrased in the corpus's own
# vocabulary, on a topic adjacent to something it really does cover, and the
# answer is simply not there. A system that pattern-matches on topic rather
# than on evidence answers these confidently and wrongly, and a retrieval
# threshold alone will not stop it -- the passages ARE similar.

ADVERSARIAL: list[Question] = [
    Question(
        question="What is the maximum probation period for a civil servant in Nepal?",
        answerable=False,
        source=(
            "The Act governs labour under an employment contract. The civil "
            "service has its own statute, which is not indexed -- but Section "
            "13 on probation retrieves strongly for this wording."
        ),
        topic="adversarial: wrong statute",
    ),
    Question(
        question="How much is the minimum monthly wage in rupees under the Act?",
        answerable=False,
        source=(
            "Section 34 provides that minimum remuneration is FIXED by the "
            "Government, and prescribes the mechanism and review period. The "
            "Act never states an amount, so any figure would be invented."
        ),
        topic="adversarial: absent figure",
    ),
    Question(
        question="Which country's law governs disputes under the Apache License?",
        answerable=False,
        source=(
            "Apache 2.0 runs Definitions, the copyright and patent grants, "
            "redistribution, contributions, trademarks, warranty disclaimer, "
            "limitation of liability and acceptance of warranty -- and has no "
            "choice-of-law or venue clause at all. Chosen over 'does it "
            "require publishing source' and 'what royalty rate applies', "
            "both of which the text DOES decide (no copyleft condition in "
            "Section 4; 'royalty-free' in Sections 2 and 3), so a correct "
            "answer there is a statement of absence rather than an "
            "abstention. Here there is no such statement to make."
        ),
        topic="adversarial: absent clause",
    ),
    Question(
        question="How many days of paternity leave does the Act provide?",
        answerable=False,
        source=(
            "The leave chapter (Sections 40-51) provides weekly, public, "
            "home, sick, mourning and maternity leave. Paternity leave is "
            "not among them, though 'leave' retrieves strongly."
        ),
        topic="adversarial: absent entitlement",
    ),
]


ALL_QUESTIONS: list[Question] = ANSWERABLE + OUT_OF_DOMAIN + ADVERSARIAL


def summary() -> str:
    """One line describing the set, for the report header."""
    return (
        f"{len(ALL_QUESTIONS)} questions: {len(ANSWERABLE)} answerable, "
        f"{len(OUT_OF_DOMAIN)} out of domain, {len(ADVERSARIAL)} adversarial"
    )
