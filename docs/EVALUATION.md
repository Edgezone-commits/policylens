# Evaluation

Every number here comes from a run recorded in `evaluation/results/`. That
directory is gitignored — it is a build artefact, regenerated with
`python scripts/evaluate.py`. Where a figure is unavailable it says so instead
of printing a zero.

**Headline, stated up front:** the system abstains well (0% false abstention,
6.7% false answer) and grounds every quote it offers (54/54 verified), but two
of its design claims did not survive measurement. The confidence formula does
not discriminate correct answers from wrong ones, and the verification loop
changed nothing on the unanswerable set.

---

## Methodology

### The dataset

53 labelled questions in `evaluation/dataset.py`, as typed Python rather than
JSON so the test suite can check its invariants (every answerable question has
a fact and a document; no unanswerable one has either; no duplicates; every
expected document is actually indexed).

| Group | Count | Cost of getting it wrong |
|---|---|---|
| Single-document | 28 | False abstention |
| Cross-document | 10 | False abstention |
| Out-of-domain | 10 | **False answer** |
| Adversarial | 5 | **False answer** |

A **cross-document** question is one the Act and the Rules Summary only answer
together: the Act sets gratuity at 8.33% of *basic remuneration* and never
defines it, while Rule 77 says basic remuneration is 60% of the last three
months' average gross. Both documents must appear in the citations for the
answer to count as grounded — citing one of two is a half answer.

An **adversarial** question is phrased in the corpus's own vocabulary, on a
topic adjacent to one it really covers, and absent. A retrieval threshold
alone will not stop these, because the passages genuinely are similar.

### Ground truth comes from the indexed text, not the statute

Section 40 of the extracted PDF reads *"the weekly leave of one day in each
month"*, which is not what the Labour Act means. Grading against external
truth would mark a faithful reading of the supplied document as wrong and
measure the PDF extraction rather than the agent. Every label carries the
passage that justifies it, so a disputed one can be checked.

### Grading

Key-fact containment, not an LLM judge: deterministic (a number reproduces
tomorrow), free (a ~20-request daily budget should not be spent marking), and
not circular (an LLM judge is a member of the family under test).

`expected_facts` is a list of alternative groups — the answer passes if it
contains at least one spelling from **every** group. `[["8.33", "eight point
three"]]` means "say the rate, however you spell it"; two groups mean "say
both of these".

**Its limit:** it checks the decisive fact is *present*, not that everything
around it is right. An answer containing "six months" plus an unrelated
falsehood still passes. Citation verification and recall are reported beside
it for exactly that reason.

### Metric definitions

| Metric | Definition | Denominator |
|---|---|---|
| **False abstention rate** | Answerable questions that got `INSUFFICIENT_EVIDENCE` or `OUT_OF_SCOPE` | answerable |
| **False answer rate** | Unanswerable questions that got `ANSWERED` or `PARTIALLY_VERIFIED` | unanswerable |
| **Answer accuracy** | Of answers actually given, those carrying every expected fact | answerable questions that were answered |
| **Citation verification** | Quotes confirmed against the passage cited | all quotes |
| **Citation recall** | Answers citing *every* document the question needs | answered, with an expectation |
| **Recall@5** | Top-5 retrieved chunks include *every* expected document | answerable |
| **Confidence separation** | mean confidence when correct − mean when wrong | needs both to exist |

The two error rates are **never averaged**. A false abstention denies the user
an answer and is visible; a false answer looks exactly like a correct one. A
single accuracy figure would let an improvement in the cheap error hide a
regression in the expensive one. `scripts/evaluate.py` exits non-zero on a
**false answer alone**.

`ERROR` (the pipeline could not run — no key, quota gone) is excluded from
both rates rather than counted as either. A run that never happened made no
claim about the documents.

---

## Results

Run 2026-10-06 against 279 chunks from 3 documents. Primary model exhausted,
so most answers came from `gemini-3.1-flash-lite` via the fallback chain.

### Answerable — 38 questions

| Metric | Result | |
|---|---|---|
| **False abstention rate** | **0%** | 0 of 38 refused |
| **Answer accuracy** | **86.8%** | 33 of 38 |
| **Citation verification** | **100%** | 54 of 54 quotes |
| Citation recall | 78.9% | cited every needed document |
| Cost | 2.0 calls, 10.2s | 88,828 tokens total |

### Unanswerable — 15 questions

| Metric | Result | |
|---|---|---|
| **False answer rate** | **6.7%** | 1 of 15 |
| Correct abstentions | 14 | 8 `OUT_OF_SCOPE`, 6 `INSUFFICIENT_EVIDENCE` |
| Cost | 2.3 calls, 10.9s | |

The eight `OUT_OF_SCOPE` exits are the router doing its job: one LLM call,
mean 5s, no retrieval. The six `INSUFFICIENT_EVIDENCE` exits are the
adversarial and near-domain questions, which reach retrieval and then fail a
later gate.

### Retrieval alone — no LLM, k=5

| Metric | Result |
|---|---|
| Recall@5, all answerable | **92.1%** |
| Recall@5, single-document | **100%** |
| Recall@5, cross-document | **70%** |
| Mean best similarity | 0.614 |

This is measured straight against the vector store, so it costs no quota and
covers all 38 questions rather than the handful a daily limit allows.

**It is a lower bound.** It searches on the question *as asked*, while the live
system searches on the router's rewrite. The gap is real: *"Must a female
labour do anything after her maternity leave ends?"* misses here and was
answered **correctly** end to end, because the rewrite reached a Rules chunk
the raw wording could not.

### Cross-document — 10 questions

| Metric | Result |
|---|---|
| Answer accuracy | 70% |
| **Cited both documents** | **20%** |

### Combined

47 of 53 correct (88.7%), whichever decision was the right one.

---

## Verification loop: ON vs OFF

`scripts/evaluate.py --no-verify` builds the graph without the verifier, so
the analyzer's draft is published unchecked. Run on the 15 unanswerable
questions, which is where verification should matter most.

| | ON | OFF |
|---|---|---|
| False answer rate | 6.7% | **6.7%** |
| Correct | 14 | **14** |
| `ANSWERED` / `INSUFFICIENT_EVIDENCE` / `OUT_OF_SCOPE` | 1 / 6 / 8 | **1 / 6 / 8** |
| Questions whose status changed | — | **0** |
| Citation verification | 100% | 0% (by construction) |
| Mean latency | 10.9s | 11.4s |

**The verification loop changed nothing measurable on this set.** Not one
question of fifteen came out differently.

This is a real negative result and worth stating plainly rather than burying.
The reason is structural: verification is the **third** of three abstention
gates, and on this set the first two caught everything. Eight questions never
reached retrieval (the router classified them `OUT_OF_SCOPE`), and six were
stopped by the relevance threshold or by the analyzer declining to answer from
the passages it got. Only one question produced a draft with a citation at
all — and that citation was a genuine verbatim quote, so verification passed
it in the ON arm exactly as the OFF arm published it unchecked.

**Where verification demonstrably does work** is where a draft contains a
quote that is *not* in the cited passage. That path is covered by a test
rather than by this measurement
(`test_a_fabricated_quote_survives_without_the_verifier_and_is_caught_with_it`):
the same scripted fabrication is withdrawn as `INSUFFICIENT_EVIDENCE` with
verification and published as `ANSWERED` without it.

**What this measurement does not say:** that verification is worthless. It says
this 15-question set does not exercise it, because the cheaper gates fire
first. Measuring it properly needs questions that reliably induce a
*plausible but unsupported quote* — which is a harder dataset to write than an
unanswerable question, and is the obvious next piece of work. The 54/54
verification rate on the answerable set also shows the loop is not costing
anything in false negatives.

An honest reading of the two arms together: **the router and the threshold are
doing most of the abstention work; verification is insurance whose premium is
near zero** (the judge is rarely called, since verbatim quotes settle by
string matching) **and whose payout has not yet been measured at scale.**

---

## Failure analysis

### All 5 wrong answers, in full

**1. "Which leaves can a labour claim as a matter of right?"** — conf 0.84

Said weekly leave and public leave are claimable as of right. Section 51 says
the opposite: *all* leaves **except** sick, mourning and maternity are "only
facility and cannot be claimed as a matter of right". **The answer inverted
the provision.** The most serious of the five — a verified quote supporting a
reversed reading. Verification cannot catch this: the quote is real, and the
error is in the sentence built around it.

**2. "Within what time must a decision on disciplinary action be taken?"** —
conf 0.71

Answered with Section 137's two months for *commencing* action instead of
Section 138(2)'s three months for *deciding*. Two adjacent provisions; it
picked the wrong one. Partly a precision failure, partly an ambiguous
question on my part.

**3. "Gratuity is a percentage of basic remuneration — what is basic
remuneration defined as?"** — conf 0.90

Returned the Act's circular phrasing ("the basic remuneration to be
received") instead of Rule 77's 60% of average gross. **A retrieval failure,
not a synthesis one** — Recall@5 confirms the top five chunks were five Act
chunks and zero Rules chunks. Note the confidence: 0.90, the highest of the
five.

**4. "What notice must an employer give when terminating for lack of
work?"** — conf 0.88

Gave the Act's graduated notice schedule rather than Rule 72's 15 days for
redundancy. Also a Recall@5 miss, same cause.

**5. "What happens to the welfare fund if the organization is
liquidated?"** — conf 0.87

Described proportional distribution but missed Rule 85's 50%. Retrieved the
right document; lost the figure.

### The one false answer

**"What is the overtime rate for work done on a public holiday?"** — conf 0.74

> "For domestic workers, if they are required to work on public holidays,
> they must be compensated with either additional remuneration or compensatory
> leave within 21 days. **The provided documents do not specify a specific
> overtime rate for work done on public holidays for other types of labor.**"

On inspection this is an honest answer. It gives the one adjacent provision
that exists (Rule 83) and then states explicitly that the documents do not
specify a rate. **It hallucinated nothing.**

The grader classifies by **status alone**, so it cannot distinguish this from
an invented figure. By this dataset's own labelling rule — *an unanswerable
question must be one where no honest statement of absence answers it either* —
the question is mislabelled, exactly as two Apache questions were before the
final run.

**It has been left in, and 6.7% reported as the headline anyway.** Editing a
dataset after seeing the result makes every number in it suspect, however good
the reason. The adjudication is recorded here instead, and the fix belongs in
the grader: either require abstention *status* for an unanswerable question
(current behaviour, conservative) or detect an explicit statement of absence
and credit it (more accurate, more complexity).

### A label that was wrong, caught before the final run

The first run scored a false answer on *"What notice period applies to
terminating a labour for misconduct?"* The agent answered that notice does
**not** apply in that case, with a verified citation — which is exactly what
Section 144(1) says. The question had been filed as unanswerable on the
reasoning that it asks for the one case the provision excludes, but a corpus
that states an exclusion **decides** the question, so abstaining would have
been the error.

It moved to the answerable set, and the rule it forced into the open is now
written into `dataset.py`. Two Apache questions failed the same test and were
replaced: "does it require publishing source" and "what royalty rate applies"
are both decided by the text (no copyleft condition in Section 4,
"royalty-free" in Sections 2 and 3). *"Which country's law governs disputes?"*
replaced them — Apache 2.0 has no choice-of-law clause and nothing to say
about one.

---

## The confidence formula does not work

```
0.6 × verified fraction + 0.4 × top similarity − 0.10 × retries
```

| | Mean confidence |
|---|---|
| Correct answers | 0.850 |
| Wrong answers | 0.839 |
| **Separation** | **0.011** |

It does not discriminate. The three wrong answers above with confidence 0.90,
0.88 and 0.87 sit above the mean for *correct* answers.

**Why**, mechanically: both terms are near-constant on this corpus.
`verified_fraction` is 1.0 almost always — 54 of 54 quotes verified — so the
0.6 term contributes 0.6 to nearly every answer. And `top_similarity` ranges
only about 0.50–0.72, so the 0.4 term varies by roughly 0.09 in total. The
formula is dominated by a term that never moves.

That is a more useful finding than the number itself: **confidence built from
verification cannot discriminate in a system where verification almost always
passes.** A signal that does not vary carries no information. Separating right
from wrong would need a signal that correlates with *relevance* rather than
with *groundedness* — the margin between the best and second-best chunk, the
analyzer's own hedging, or agreement across the retrieved set.

The `config.py` comment saying the weights live there so Phase 6 could re-fit
them against measured accuracy was right to anticipate this. Until that work
is done the number is decoration, and the README says so rather than quoting
0.85 as if it meant something.

---

## What the numbers do and do not support

**Supported:**

- The system abstains reliably on out-of-domain and adversarial questions
  (14/15) and never refused an answerable one (0/38).
- It does not fabricate quotes: 54 of 54 verified against source.
- Single-document retrieval is solid (Recall@5 = 100%).
- It is cheap: 2.0 LLM calls and ~10s per question.

**Not supported:**

- That the confidence score is meaningful. Separation 0.011.
- That the verification loop improves outcomes *at the level this set
  measures*. Zero of 15 statuses changed. Only a unit test shows its value.
- That cross-document reasoning works. 70% accuracy and 20% both-cited.
- That answers are *correct* as opposed to *grounded*. 100% verification
  against 86.8% accuracy is the clearest single finding here: **verification
  prevents fabrication, not irrelevance.**

**Known weaknesses in the evaluation itself:**

- 53 questions is small. One question is ~2 percentage points.
- One grader limitation documented above, left uncorrected on purpose.
- The adversarial set does not reliably induce bad *quotes*, which is what
  verification defends against — so the ON/OFF comparison is uninformative
  rather than negative.
- Recall@5 is measured on the raw question, a lower bound on the live
  retriever.
- Single-run numbers at temperature 0.0. Deterministic prompting, but not
  repeated runs, so no variance estimate.

---

## Reproducing

```bash
python scripts/evaluate.py                 # all 53, end to end
python scripts/evaluate.py --only cross    # the 10 cross-document questions
python scripts/evaluate.py --recall-only   # Recall@5, free, no LLM
python scripts/evaluate.py --no-verify     # the control arm
python scripts/evaluate.py --limit 6       # stay inside the daily quota
python scripts/calibrate_threshold.py      # re-fit the relevance threshold
```

The free tier is ~20 requests per model per day and one answer is 2–5 calls,
so a full 53-question run is not always affordable in one sitting. A partial
run is a legitimate result as long as it says it is partial, which the report
header does. Check what is callable first with
`python scripts/check_models.py`.
