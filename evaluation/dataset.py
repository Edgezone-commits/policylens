"""The labelled question set the Phase 6 evaluation runs against.

50 questions in four groups:

  SINGLE_DOCUMENT (25)  answerable from one document on its own.
  CROSS_DOCUMENT  (10)  answerable only by combining both Nepal documents.
                        The Act states a rule and the Rules Summary supplies
                        the figure, definition or procedure it depends on, so
                        a system that retrieves from one document and stops
                        gives a half answer or none.
  OUT_OF_DOMAIN   (10)  nothing in the corpus touches the topic.
  ADVERSARIAL      (5)  phrased in the corpus's own vocabulary, on a topic
                        adjacent to one it really covers, and absent.

Answerable questions carry a FALSE-ABSTENTION cost: refusing one denies the
user an answer the documents hold. Unanswerable ones carry a FALSE-ANSWER
cost: answering one asserts something the documents do not support, which is
the failure this project exists to prevent.

WHY the expected facts are quoted from the INDEXED TEXT and not from the
published statute. Section 40 of the Act reads "the weekly leave of one day in
each month", which is not what the Labour Act means. Grading against external
truth would mark a faithful reading of the supplied document as wrong, and
measure the PDF extraction rather than the agent. Every entry therefore
carries the `source` note that was read out of the index to justify it, so a
disputed label can be checked rather than argued about.

WHY the grader is key-fact containment and not an LLM judge:

  * Deterministic, so a number can be reproduced tomorrow and a change in the
    score means the system moved rather than the judge did.
  * Free. The tier allows about twenty requests per model per day (see
    config.LLM_FALLBACK_MODELS); spending half of that marking would halve
    the questions that can be asked.
  * An LLM judge is a member of the family being evaluated. Using one would
    measure agreement, not correctness.

Its limit, stated plainly because the report has to be defensible: it checks
the decisive fact is PRESENT, not that everything around it is right. An
answer containing "six months" plus an unrelated falsehood still passes. That
is why citation verification and retrieval recall are reported beside it --
the three together are a far stronger claim than any one alone.

THE LABELLING RULE, which the first run forced into the open. A question is
ANSWERABLE when the corpus contains text that DECIDES it -- including text
that decides it in the negative. "What notice period applies for misconduct?"
looks unanswerable and is not: Section 144(1) excludes misconduct explicitly,
so "none applies" is the correct answer and abstaining would be the mistake.
That question was mislabelled here, scored as a false answer on the first run,
and moved.

The consequence for writing new ones: an unanswerable question must be one
where no honest statement of absence answers it either. "Which country's law
governs this licence?" qualifies -- there is no clause and nothing to say
about one. "Does this licence require publishing source?" does not, because
"no, it imposes no such condition" is a correct reading of Section 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ACT = "Nepal_Labour_Act_2074.pdf"
RULES = "Nepal_Labour_Rules_2075_Summary_PKF.pdf"
APACHE = "Apache_License_2.0.txt"


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
    # Every document a complete answer has to draw on. A list rather than one
    # name so a cross-document question can require BOTH, which is the only
    # way to tell a real cross-document answer from a lucky half of one.
    #
    # The SECTION is deliberately not graded. schemas.py calls it a
    # best-effort locator and it drifts by one on this corpus -- the chunk
    # whose text begins "13. Provision relating to probation period" carries
    # the metadata "Section 12". Grading a locator the system never promised
    # would measure the PDF's heading detection, not the answer.
    expected_documents: list[str] = field(default_factory=list)
    # Why this is the right answer, read out of the index rather than assumed.
    source: str = ""
    # Short label for the results table.
    topic: str = ""
    # Set when a label could not be confirmed against the indexed text. None
    # are set today; the field exists so an unverified entry is declared
    # rather than silently mixed in with the checked ones.
    needs_manual_verification: bool = False

    @property
    def cross_document(self) -> bool:
        return len(self.expected_documents) > 1


# ---------------------------------------------------------------------------
# Single document (25)
# ---------------------------------------------------------------------------

SINGLE_DOCUMENT: list[Question] = [
    # -- Labour Act: contract and probation -----------------------------
    Question(
        question="What is the maximum probation period for a worker?",
        answerable=True,
        expected_facts=[["six month", "6 month"]],
        expected_documents=[ACT],
        source=(
            'Section 13: "the employer may so enter into the contract that he '
            'or she remains in a probation period for six months"'
        ),
        topic="probation",
    ),
    Question(
        question="What notice period applies to terminating a labour for misconduct?",
        answerable=True,
        expected_facts=[
            ["not apply", "no notice", "without notice", "except", "exempt"]
        ],
        expected_documents=[ACT],
        source=(
            'Section 144(1) requires notice "in any circumstance except when '
            'employment is terminated upon action taken for misconduct". '
            "Answerable: the text decides it in the negative, so 'none "
            "applies' is correct and abstaining would be the error."
        ),
        topic="exclusion from notice",
    ),
    Question(
        question="For how long may a foreign labour's contract run if the contract is silent?",
        answerable=True,
        expected_facts=[["three year", "3 year"]],
        expected_documents=[ACT],
        source=(
            'Section 27: "mentioned in that contract, and failing such a '
            'provision, for a period not exceeding three years"'
        ),
        topic="foreign labour term",
    ),
    # -- Labour Act: working hours --------------------------------------
    Question(
        question="How many hours may a labour be made to work in a day and in a week?",
        answerable=True,
        expected_facts=[["eight hour", "8 hour"], ["forty-eight", "forty eight", "48"]],
        expected_documents=[ACT],
        source=(
            'Section 28(1): "No employer shall employ labours to work more '
            'than eight hours a day and forty-eight hours a week"'
        ),
        topic="working hours",
    ),
    Question(
        question="How much rest must a labour get during continuous work?",
        answerable=True,
        expected_facts=[["half an hour", "half-hour", "30 minute", "thirty minute"]],
        expected_documents=[ACT],
        source=(
            'Section 28(2): "The labours shall be provided with half an hour '
            'rest after five hours of continuous work"'
        ),
        topic="rest break",
    ),
    Question(
        question="How much overtime work is a labour allowed to do?",
        answerable=True,
        expected_facts=[["four hour", "4 hour"], ["twenty-four", "twenty four", "24"]],
        expected_documents=[ACT],
        source=(
            'Section 30(1): "such a labour may be caused to work not '
            'exceeding four hours a day and twenty-four hours a week"'
        ),
        topic="overtime limit",
    ),
    # -- Labour Act: leave ----------------------------------------------
    Question(
        question="How much weekly leave is a labour entitled to?",
        answerable=True,
        expected_facts=[["one day", "1 day"]],
        expected_documents=[ACT],
        source=(
            'Section 40: "A labour shall get the weekly leave of one day in '
            'each month." The published Act means one day per week; the '
            'indexed text says "in each month", so "one day" is the only part '
            "that can be graded without marking a faithful reading wrong."
        ),
        topic="weekly leave",
    ),
    Question(
        question="How much paid public leave does a labour get each year?",
        answerable=True,
        expected_facts=[["thirteen", "13"]],
        expected_documents=[ACT],
        source=(
            'Section 41(1): "a paid public leave of thirteen days including '
            "the May Day, and fourteen days including International Women's "
            'Day" -- thirteen is the general entitlement.'
        ),
        topic="public leave",
    ),
    Question(
        question="How long is maternity leave and how much of it is paid?",
        answerable=True,
        expected_facts=[["sixty", "60"]],
        expected_documents=[ACT],
        source=(
            'Section 45(3): "shall get full remuneration for sixty days and '
            'shall not get remuneration for the remaining period"'
        ),
        topic="maternity leave",
    ),
    Question(
        question="When must a pregnant labour begin and end maternity leave?",
        answerable=True,
        expected_facts=[["two week", "2 week"], ["six week", "6 week"]],
        expected_documents=[ACT],
        source=(
            'Section 45(2): "shall compulsorily take leave from at least two '
            "weeks before the expected date of delivery to at least six weeks "
            'after the date of delivery"'
        ),
        topic="maternity leave timing",
    ),
    Question(
        question="Which leaves can a labour claim as a matter of right?",
        answerable=True,
        expected_facts=[["sick"], ["mourning"], ["maternity"]],
        expected_documents=[ACT],
        source=(
            'Section 51(1): "All leaves except the sick leave, mourning leave '
            "and maternity leave which the labour is entitled to pursuant to "
            "this Chapter are only facility and cannot be claimed as a matter "
            'of right"'
        ),
        topic="leave as of right",
    ),
    # -- Labour Act: pay ------------------------------------------------
    Question(
        question="How much must the employer deposit for gratuity each month?",
        answerable=True,
        expected_facts=[["8.33", "eight point three"]],
        expected_documents=[ACT],
        source=(
            'Section 53(1): "deduct an amount equivalent to eight point three '
            "and three (8.33) percent of the basic remuneration of each "
            'labour each month and deposit it for the purpose of gratuity"'
        ),
        topic="gratuity rate",
    ),
    Question(
        question="What is a labour entitled to as a festival expense?",
        answerable=True,
        expected_facts=[["one month", "a month", "basic remuneration of one"]],
        expected_documents=[ACT],
        source=(
            'Section 37(1): "entitled to an amount equal to the basic '
            'remuneration of one month as the festival expense each year"'
        ),
        topic="festival expense",
    ),
    Question(
        question="When must a labour employed for less than a month be paid?",
        answerable=True,
        expected_facts=[["three day", "3 day"]],
        expected_documents=[ACT],
        source=(
            'Section 35(1)(a): "In the case of a labour working for a period '
            "of less than one month, within three days from the date of "
            'completion of the work"'
        ),
        topic="pay timing, short engagement",
    ),
    Question(
        question="When must a labour in casual work be paid?",
        answerable=True,
        expected_facts=[["immediately", "at once", "on completion", "after the com"]],
        expected_documents=[ACT],
        source=(
            'Section 35(1)(b): "In the case of a labour employed in a casual '
            'work, immediately after the completion of the work"'
        ),
        topic="pay timing, casual work",
    ),
    # -- Labour Act: discipline and termination -------------------------
    Question(
        question="After how long a suspension may the employer terminate the labour?",
        answerable=True,
        expected_facts=[["ninety", "90"]],
        expected_documents=[ACT],
        source=(
            'Section 134(6): "the employer may terminate the service of such '
            "a labour if the period of suspension referred to in sub-section "
            '(1) exceeds ninety days"'
        ),
        topic="suspension limit",
    ),
    Question(
        question="Within what time must a decision on disciplinary action be taken?",
        answerable=True,
        expected_facts=[["three month", "3 month"]],
        expected_documents=[ACT],
        source=(
            'Section 138(2): "A decision shall be taken within three months '
            'from the date of commencement of the action"'
        ),
        topic="disciplinary timeline",
    ),
    Question(
        question="What must an employer do for a labour who becomes physically incapacitated?",
        answerable=True,
        expected_facts=[
            ["suitable", "suited", "condition of his"],
            ["engage", "shall engage"],
        ],
        expected_documents=[ACT],
        source=(
            'Section 143(4): "in cases where the labour who is physically '
            "incapacitated or has sustained grievous hurt or injury can be "
            "engaged in any work suitable to the condition of his or her "
            'health, the employer shall engage such a labour in work"'
        ),
        topic="incapacity redeployment",
    ),
    # -- Labour Act: definitions ----------------------------------------
    Question(
        question="What counts as a seasonal enterprise under the Act?",
        answerable=True,
        expected_facts=[["one hundred eighty", "180"]],
        expected_documents=[ACT],
        source=(
            "Section 89 Explanation: \"'seasonal enterprise' means an "
            "enterprise which can be operated in a particular season only, "
            "and this term also includes an enterprise which cannot be "
            'operated for more than one hundred eighty days in a year"'
        ),
        topic="seasonal enterprise",
    ),
    # -- Labour Rules Summary -------------------------------------------
    Question(
        question="What are the charges for a labour permit?",
        answerable=True,
        expected_facts=[["15,000", "15000"], ["20,000", "20000"]],
        expected_documents=[RULES],
        source=(
            'Rule 10: "For period up to 6 months NPR 15,000 For period '
            'exceeding 6 months NPR 20,000"'
        ),
        topic="labour permit charges",
    ),
    Question(
        question="Within how long must the Department of Labour notify a rejected permit applicant?",
        answerable=True,
        expected_facts=[["7 day", "seven day"]],
        expected_documents=[RULES],
        source=(
            'Rule 7: "On verification of submitted documents, DoL shall '
            'notify with reason, a rejected applicant within 7 days"'
        ),
        topic="permit rejection notice",
    ),
    Question(
        question="How long must an employer keep payment and attendance records?",
        answerable=True,
        expected_facts=[["5 year", "five year"]],
        expected_documents=[RULES],
        source=(
            'Rule 81: "paid and attendance marked for all labors for a '
            'minimum period of 5 years"'
        ),
        topic="record retention",
    ),
    Question(
        question="Within how long must a worker apply to have a disability percentage determined?",
        answerable=True,
        expected_facts=[["1 year", "one year"]],
        expected_documents=[RULES],
        source=(
            'Rule 26: "laborer may file an application for determination of '
            "percentage of disability to the Labor Office within 1 year of "
            'discharge from hospital"'
        ),
        topic="disability determination",
    ),
    Question(
        question="What facilities must be provided once an enterprise has 50 or more workers?",
        answerable=True,
        expected_facts=[["relaxation room", "relaxation"], ["canteen"]],
        expected_documents=[RULES],
        source='Rule 45: "50 or more workers - Relaxation room - Canteen"',
        topic="welfare facilities",
    ),
    # -- Apache License -------------------------------------------------
    Question(
        question="Does the Apache License grant a patent licence to the user?",
        answerable=True,
        expected_facts=[["patent"], ["grant", "yes", "does"]],
        expected_documents=[APACHE],
        source=(
            'Section 3 "Grant of Patent License": "each Contributor hereby '
            "grants to You a perpetual, worldwide, non-exclusive, no-charge, "
            'royalty-free, irrevocable ... patent license"'
        ),
        topic="patent grant",
    ),
]


# ---------------------------------------------------------------------------
# Single document, continued -- kept separate only to stay readable
# ---------------------------------------------------------------------------

SINGLE_DOCUMENT += [
    Question(
        question="Does the Apache License provide any warranty with the software?",
        answerable=True,
        expected_facts=[["without warrant", "no warrant", "as is", '"as is"']],
        expected_documents=[APACHE],
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
        expected_documents=[APACHE],
        source=(
            'Section 6 "Trademarks": "This License does not grant permission '
            "to use the trade names, trademarks, service marks, or product "
            'names of the Licensor"'
        ),
        topic="trade marks",
    ),
    Question(
        question="On what terms are contributions submitted under the Apache License?",
        answerable=True,
        expected_facts=[
            ["terms and conditions of this license", "this license", "same"]
        ],
        expected_documents=[APACHE],
        source=(
            'Section 5 "Submission of Contributions": "any Contribution '
            "intentionally submitted for inclusion in the Work ... shall be "
            "under the terms and conditions of this License, without any "
            'additional terms or conditions"'
        ),
        topic="contributions",
    ),
]


# ---------------------------------------------------------------------------
# Cross document (10)
# ---------------------------------------------------------------------------
#
# These are the questions the corpus was chosen for. The Act states a rule and
# the Rules Summary supplies the number, definition or procedure it turns on,
# so an answer drawn from one document alone is incomplete even when it is not
# wrong. Both documents must appear in the citations for the retrieval metric
# to count the answer as grounded.

CROSS_DOCUMENT: list[Question] = [
    Question(
        question=(
            "Gratuity is a percentage of basic remuneration -- what is basic "
            "remuneration defined as?"
        ),
        answerable=True,
        expected_facts=[["60%", "60 percent", "sixty percent"]],
        expected_documents=[ACT, RULES],
        source=(
            "Act Section 53(1) sets gratuity at 8.33% of BASIC remuneration; "
            'the Act never defines it. Rules Rule 77: "Basic remuneration of '
            "a regular labor shall be equivalent to 60% of last 3 months "
            'average gross salary." The figure and the rule are in different '
            "documents."
        ),
        topic="cross: gratuity base",
    ),
    Question(
        question="At what age is retirement compulsory, and can an enterprise set it lower?",
        answerable=True,
        expected_facts=[["fifty-eight", "fifty eight", "58"]],
        expected_documents=[ACT, RULES],
        source=(
            'Act Section 147: "shall get compulsory retirement after '
            'completing the age of fifty-eight years". Rules Rule 73: "The '
            "enterprise can submit an application to the Ministry, with basis "
            "and reasons, if it desires to fix the retirement age below 58 "
            'years." The age is in the Act, the procedure for lowering it is '
            "in the Rules."
        ),
        topic="cross: retirement age",
    ),
    Question(
        question="What rest is a pregnant worker entitled to during the working day?",
        answerable=True,
        expected_facts=[["half an hour", "half-hour", "30 minute", "thirty minute"]],
        expected_documents=[ACT, RULES],
        source=(
            'Act Section 28(2) gives every labour "half an hour rest after '
            'five hours of continuous work". Rules Rule 17: "Half an hour of '
            "additional rest time shall be provided to pregnant women or "
            "women having children below 3 years of age. Such additional rest "
            'time shall form part of regular working hours." The entitlement '
            "is the Act's plus the Rules' addition."
        ),
        topic="cross: pregnancy rest",
    ),
    Question(
        question="What notice must an employer give when terminating for lack of work?",
        answerable=True,
        expected_facts=[["15 day", "fifteen day"]],
        expected_documents=[ACT, RULES],
        source=(
            "Act Section 144(1) requires notice before terminating an "
            "employment relation. Rules Rule 72: \"by providing 15 days' "
            "notice. This information should be placed on the job portal site "
            "of the ministry and the website of the enterprise for a minimum "
            'of 15 days." The duty is in the Act, the period and publication '
            "requirement in the Rules."
        ),
        topic="cross: redundancy notice",
    ),
    Question(
        question="Where must gratuity and provident fund amounts be deposited?",
        answerable=True,
        expected_facts=[["social security fund", "social security"]],
        expected_documents=[ACT, RULES],
        source=(
            "Act Sections 52-53 require deduction and deposit for provident "
            'fund and gratuity. Rules Rules 22-23: "Citizen Investment Trust, '
            "Approved Retirement Fund or Separate account. Per section 20 of "
            "the Social Security Act, 2074 the employer shall transfer the "
            "amount deposited in aforementioned funds to the Social Security "
            'Fund in lump sum or in maximum 4 installments within 6 months."'
        ),
        topic="cross: where funds go",
    ),
    Question(
        question="How is gratuity calculated for service before and after 19 Bhadra 2074?",
        answerable=True,
        expected_facts=[["8.33"], ["19 bhadra", "bhadra", "2074"]],
        expected_documents=[ACT, RULES],
        source=(
            "Act Section 53 gives the ongoing 8.33% rule. Rules Rule 23 "
            'splits the calculation: "For the period before 19 Bhadra 2074 ... '
            "every year of service. For the period after 19 Bhadra 2074 (4 "
            "Sept 2017), the employer shall deposit 8.33% of basic "
            'remuneration for every month of service."'
        ),
        topic="cross: gratuity split",
    ),
    Question(
        question="What limit applies to employing foreign citizens and for how long?",
        answerable=True,
        expected_facts=[["5%", "5 percent", "five percent"]],
        expected_documents=[ACT, RULES],
        source=(
            "Act Section 27 governs foreign labour and the contract term "
            '("not exceeding three years"). Rules Rule 13: "For other '
            "organizations Not exceeding 5% of total labor requirement in the "
            'concerned organization." The cap is only in the Rules.'
        ),
        topic="cross: foreign labour cap",
    ),
    Question(
        question="May a foreign labour's permit be extended beyond the normal maximum?",
        answerable=True,
        expected_facts=[["2 year", "two year", "additional 2"]],
        expected_documents=[ACT, RULES],
        source=(
            "Act Section 27 caps the contract at three years absent provision. "
            'Rules Rule 11: "on special circumstances, DoL may grant '
            "permission for an additional 2 years after approval from the "
            'Ministry."'
        ),
        topic="cross: permit extension",
    ),
    Question(
        question="Must a female labour do anything after her maternity leave ends?",
        answerable=True,
        expected_facts=[
            ["3 month", "three month"],
            ["birth certificate", "certificate"],
        ],
        expected_documents=[ACT, RULES],
        source=(
            'Act Section 45 grants the leave. Rules Rule 18: "Within 3 months '
            'from the end of maternity leave, female labor" must submit a '
            "birth certificate. The obligation exists only in the Rules."
        ),
        topic="cross: post-maternity duty",
    ),
    Question(
        question="What happens to the welfare fund if the organization is liquidated?",
        answerable=True,
        expected_facts=[["50%", "50 percent", "fifty percent"]],
        expected_documents=[ACT, RULES],
        source=(
            "The Act provides for welfare funds; Rules Rule 85 states the "
            'liquidation split: "On liquidation of an organization, 50% of '
            "such fund shall be distributed to the existing labors on a "
            'proportional basis."'
        ),
        topic="cross: welfare fund on liquidation",
    ),
]


# ---------------------------------------------------------------------------
# Out of domain (10)
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
    Question(
        question="What is the current exchange rate for the US dollar?",
        answerable=False,
        source=(
            "The Rules quote amounts in NPR but neither document states an "
            "exchange rate, and a rate is time-varying data no document holds."
        ),
        topic="market data",
    ),
    Question(
        question="How do I train a neural network to classify images?",
        answerable=False,
        source="Neither document concerns machine learning.",
        topic="machine learning",
    ),
    Question(
        question="What are the opening hours of the Department of Labour office?",
        answerable=False,
        source=(
            "The Rules name the DoL as the authority for permits but state no "
            "opening hours for it."
        ),
        topic="office hours",
    ),
    Question(
        question="Which political party introduced the Labour Act?",
        answerable=False,
        source=(
            "The Act's text contains no legislative history or party " "attribution."
        ),
        topic="political history",
    ),
    Question(
        question="How much corporate income tax does an enterprise in Nepal pay?",
        answerable=False,
        source=(
            "Tax is a different statute. The Rules mention permit charges and "
            "deductions, never a corporate tax rate."
        ),
        topic="taxation",
    ),
    Question(
        question="What is the population of Kathmandu?",
        answerable=False,
        source="Neither document contains demographic data.",
        topic="demographics",
    ),
]


# ---------------------------------------------------------------------------
# Adversarial (5)
# ---------------------------------------------------------------------------
#
# Each is phrased in the corpus's own vocabulary, on a topic adjacent to one it
# really covers, and the answer is simply not there. A system that matches on
# topic rather than on evidence answers these confidently and wrongly, and a
# retrieval threshold alone will not stop it, because the passages ARE similar.
# scripts/calibrate_threshold.py deliberately kept this kind of question out of
# its probe set, so the threshold was not tuned on them.

ADVERSARIAL: list[Question] = [
    Question(
        question="What is the maximum probation period for a civil servant in Nepal?",
        answerable=False,
        source=(
            "The Act governs labour under an employment contract; the civil "
            "service has its own statute, which is not indexed. Section 13 on "
            "probation nonetheless retrieves strongly for this wording."
        ),
        topic="adversarial: wrong statute",
    ),
    Question(
        question="How much is the minimum monthly wage in rupees under the Act?",
        answerable=False,
        source=(
            "Section 34 provides that minimum remuneration is FIXED by the "
            "Government and prescribes the mechanism and review period. "
            "Neither document states an amount, so any figure would be "
            "invented. Rule 77 refers to the minimum without stating it."
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
            "require publishing source' and 'what royalty rate applies', both "
            "of which the text DOES decide (no copyleft condition in Section "
            "4; 'royalty-free' in Sections 2 and 3), so a correct answer "
            "there is a statement of absence rather than an abstention."
        ),
        topic="adversarial: absent clause",
    ),
    Question(
        question="How many days of paternity leave does the Act provide?",
        answerable=False,
        source=(
            "The leave chapter (Sections 40-51) provides weekly, public, "
            "home, sick, mourning and maternity leave. Paternity leave is not "
            "among them, though 'leave' retrieves strongly."
        ),
        topic="adversarial: absent entitlement",
    ),
    Question(
        question="What is the overtime rate for work done on a public holiday?",
        answerable=False,
        source=(
            "Section 31 provides additional remuneration for overtime and "
            "Section 41 provides public leave, but neither document sets a "
            "combined holiday-overtime rate. Both halves retrieve strongly, "
            "which is exactly the trap: the plausible answer would be "
            "assembled rather than cited."
        ),
        topic="adversarial: plausible composite",
    ),
]


ANSWERABLE: list[Question] = SINGLE_DOCUMENT + CROSS_DOCUMENT
UNANSWERABLE: list[Question] = OUT_OF_DOMAIN + ADVERSARIAL
ALL_QUESTIONS: list[Question] = ANSWERABLE + UNANSWERABLE


def summary() -> str:
    """One line describing the set, for the report header."""
    return (
        f"{len(ALL_QUESTIONS)} questions: {len(SINGLE_DOCUMENT)} single-document, "
        f"{len(CROSS_DOCUMENT)} cross-document, {len(OUT_OF_DOMAIN)} out-of-domain, "
        f"{len(ADVERSARIAL)} adversarial"
    )


def documents() -> list[str]:
    """Every document the answerable questions are labelled against.

    Derived from the questions rather than written out as a list, so adding a
    question about a fourth document cannot leave the corpus definition
    behind. scripts/eval_retrieval.py builds its reference index from exactly
    this set: retrieval metrics only mean something measured against the
    corpus the labels describe, and "which corpus is that" is a fact about
    the dataset, so it belongs here rather than in the script.

    Unanswerable questions contribute nothing by construction -- they carry no
    expected_documents, because there is no passage that should be found.
    """
    names: set[str] = set()
    for item in ANSWERABLE:
        names.update(item.expected_documents)
    return sorted(names)
