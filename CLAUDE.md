# PolicyLens — project state

Agentic question answering over legal and policy documents, where every claim
carries a quote that has been checked against the passage it cites.

**Status: COMPLETE.** 172 tests passing, 53-question evaluation run, README
and three design documents written, Docker and CI in place.

**Last updated: 2026-10-06.** Everything below was verified by running it.
Every number comes from a recorded run in `evaluation/results/`.

---

## 1. State

| Phase | Area | State |
|---|---|---|
| 0 | Scaffold, central config | Done |
| 1 | Ingestion (PDF → chunks → ChromaDB) | Done, 22 tests |
| 2 | Retrieval + calibrated threshold | Done, 15 tests |
| 3 | LangGraph agent, 5 nodes + retry loop | Done, 28 tests |
| 4 | FastAPI | Done, 27 tests |
| 5 | Streamlit frontend | Done, 12 tests |
| 6 | Evaluation, 53 questions | Done, 31 tests |
| 7 | Docker, CI, README, docs/ | Done |

- **172 tests, all passing, none needing an API key** (`pytest -q`, ~60s).
- Index live: 279 chunks across 3 documents.
- `main` is current; branch `feat/frontend` holds the last four commits.

**One thing never verified:** the Docker image has never been built — the
daemon was unavailable on this machine. `docker compose config` validates and
both YAML files parse, but the build (apt layer, embedding-model bake,
cache permissions for the non-root user) is unproven. **CI will be its first
real build.**

---

## 2. Results

All from `evaluation/results/`, 2026-10-06. That directory is gitignored;
regenerate with `python scripts/evaluate.py`.

| Metric | Result | Basis |
|---|---|---|
| **False answer rate** | **6.7%** | 1 of 15 unanswerable |
| **False abstention rate** | **0%** | 0 of 38 answerable |
| Answer accuracy | 86.8% | 33 of 38 |
| Citation verification | 100% | 54 of 54 quotes |
| Citation recall | 78.9% | cited every needed document |
| Recall@5 (no LLM) | 92.1% | 100% single-doc, 70% cross-doc |
| Cross-document accuracy | 70% | only **20%** cited both documents |
| Confidence separation | **0.011** | 0.850 right vs 0.839 wrong |
| Cost | 2.0 calls, 10.2s | 88,828 tokens / 38 questions |
| Overall correct | 88.7% | 47 of 53 |

**Verification ON vs OFF:** identical. 6.7% both arms, 14 correct both arms,
**zero statuses changed** across 15 questions.

---

## 3. The five things to say about this project

Ordered by how much they distinguish it. All five are measured, not claimed.

1. **It abstains.** 0% false abstention and 6.7% false answer. The whole
   design is three independent gates that must all pass before anything is
   asserted.
2. **Quotes are checked against the source, by id** — never against what the
   model echoed back. 54 of 54 verified.
3. **Perfect grounding is not correctness.** 100% verification against 86.8%
   accuracy. Verification prevents fabrication, not irrelevance.
4. **The confidence formula does not work** (0.011 separation), and I can
   explain mechanically why, which is more useful than the number.
5. **The verification loop changed nothing measurable**, and I report that
   rather than the unit test that flatters it.

---

## 4. Fifteen interview questions

**1. What does this do that a RAG tutorial does not?**
It refuses. A nearest-neighbour search always returns something, so a plain
pipeline answers everything — ask the minimum wage in rupees and it invents a
figure from five plausible paragraphs. Three gates here can each end in
`INSUFFICIENT_EVIDENCE`: the calibrated threshold, the analyzer declining, and
verification. Measured: 14 of 15 unanswerable questions refused.

**2. Why LangGraph rather than a chain?**
The retry is a real cycle, entered from three different nodes, carrying a
counter that must increment exactly once per loop or it never terminates, with
three exit conditions. A chain expresses that only as a hand-rolled `while`
around the whole pipeline. Also: every node appends to `state["trace"]`, so
"how did this answer happen" is a response field, not a logging exercise.

**3. How does verification actually work?**
Each citation names a `chunk_id`. The verifier looks that chunk up in the
vector store and re-reads it — so an invented `chunk_id` fails as
`unknown_chunk`. Then RapidFuzz partial match, escalating to an LLM judge only
when string matching cannot settle it. Quotes under 40 characters are rejected
outright, because three or four words score 100 against almost any legal
prose.

**4. Why isn't a high fuzzy score enough?**
Because I tested it and it is not. Swapping "six months" for "one year" in a
30-word clause still scores in the nineties; deleting one "not" costs about
three points. Both passed as verified while asserting the opposite of the
source. So `citation_divergences()` aligns the quote to the span it claims to
reproduce and requires quantities and negations to match in both directions.

**5. How did you pick 0.32?**
Fitted, not guessed. `calibrate_threshold.py` runs two hand-labelled probe
sets against the live index and maximises **balanced accuracy** — plain
accuracy can be gamed by whichever set is larger, and a false answer costs
more than a false abstention. Off-topic questions peak around 0.18, answerable
ones score 0.52–0.71. The probes are deliberately **not** the evaluation
questions, or the metrics would measure the tuning.

**6. Your citation verification is 100% but accuracy is 86.8%. Explain.**
That gap is the most interesting result in the project. Verification proves a
quote is really in the passage cited; it says nothing about whether the
passage answers the question, or whether the sentence built around the quote
reads it correctly. The clearest case: asked which leaves are claimable as of
right, the system cited Section 51 verbatim and asserted the **opposite** of
it. Every check passed. Verification prevents fabrication, not irrelevance.

**7. Does the verification loop improve your metrics?**
No — not at the level this evaluation measures. I ran the control arm
(`--no-verify`) over 15 unanswerable questions and **zero outcomes changed**.
The cause is structural: verification is the third of three gates, and the
router caught 8 and the threshold plus analyzer caught 6, so only one question
produced a citation at all — and that quote was genuine. Its value is shown by
a unit test (the same fabricated quote is withdrawn with verification and
published without it), not by this measurement. Measuring it properly needs a
dataset that reliably induces plausible-but-unsupported quotes, which is
harder to write than an unanswerable question.

**8. Is your confidence score calibrated?**
No, and it is reported as a failure. Mean confidence is 0.850 on correct
answers and 0.839 on wrong ones — separation 0.011. Mechanically:
`verified_fraction` is 1.0 on 54 of 54 quotes, so the 0.6 term is constant,
and `top_similarity` spans only 0.50–0.72, so the 0.4 term varies by 0.09 in
total. The formula is dominated by a term that never moves. **Confidence built
from verification cannot discriminate in a system where verification almost
always passes.**

**9. Why not ask the model for its confidence?**
Self-reported confidence clusters near 0.9 whether the answer is right or
wrong — a number that looks informative and is not. Mine does not work either,
but it is built from facts the system measured, so it can be re-fitted against
outcomes. A self-report cannot.

**10. What is your biggest weakness?**
Cross-document retrieval. 70% accuracy and only **20%** of cross-document
answers cite both documents. Recall@5 says why rather than leaving me to
guess: all three misses retrieved five Act chunks and zero Rules chunks. The
Act has 234 chunks to the Rules Summary's 32, so it dominates a global top-5
by mass. The fix is to retrieve per document and merge, instead of taking one
global top-5.

**11. Why Chroma and not FAISS?**
Metadata filtering (scoping to one document is a `where` clause, not a second
index), persistence by default (an 82-page ingest paid once), and lookup by
id — which is exactly how the verifier re-reads a cited chunk. FAISS is faster
at scale; at 279 chunks retrieval is 0.03s against a 5s LLM call, so that is
irrelevant here.

**12. How do you test an LLM system without an API key?**
All 172 tests run without one. The graph runs against a scripted LLM, so the
real control flow — retry, abstention, failover — is exercised with
deterministic replies. The API runs against a fake agent. Retrieval injects a
fake embedding model where similarity is arithmetic, so a threshold test
asserts the gate's logic rather than MiniLM's opinion. CI has a step that
fails the build if a test starts reading a key from the environment.

**13. Why is an abstention HTTP 200?**
Because it is a correct, reasoned response — the feature working. A 4xx tells
every client the request was malformed; a 5xx makes the project's most
important behaviour read as an outage in every log and dashboard, which
teaches whoever watches it that honesty is a malfunction. Only `ERROR`, where
the pipeline could not run, is 503. There is a test whose only job is to pin
this.

**14. Tell me about a bug you found.**
The evaluation's first run flagged a false answer, and the label was wrong,
not the system. Asked what notice period applies for misconduct, the agent
said none does — exactly what Section 144(1) states. I had filed it as
unanswerable because it asks for the excluded case, but a corpus that states
an exclusion **decides** the question. I moved it and wrote down the rule it
exposed: an unanswerable question must be one where no honest statement of
absence answers it either. Two Apache questions failed that test and were
replaced.

**15. A question you got wrong and left in?**
The one remaining false answer. Asked the overtime rate for a public holiday,
the system gave the one adjacent provision that exists and then stated the
documents do not specify a rate. It hallucinated nothing — the grader
classifies by status alone and cannot tell that from an invented figure. By my
own labelling rule the question is mislabelled. I left it in and reported 6.7%
anyway, because editing a dataset after seeing the result makes every number
in it suspect. The adjudication is in `docs/EVALUATION.md`.

---

## 5. A three-minute live demo

1. **`/health` first (15s).** Shows 279 chunks, 3 documents, the model chain
   and the threshold. Frames the system and proves it is up.
2. **Ask the probation question (30s).** `ANSWERED`, confidence 0.80, one
   citation verified `exact` against Section 13, page 7. Expand the citation
   to show the quote, then the reasoning path.
3. **Ask "how much is the minimum monthly wage in rupees?" (30s).** The money
   shot. `INSUFFICIENT_EVIDENCE` — rendered as a result with its own panel,
   not an error. The Act says the Government fixes it and never states an
   amount, so any figure would be invented.
4. **Ask the gratuity question (45s).** Six citations, all verified, across
   Rule 23 / Section 52 / Section 53. Shows multi-provision synthesis and the
   budget fix that made it possible — it used to return a 503.
5. **Show the evaluation table (45s).** Lead with 0% false abstention and 6.7%
   false answer, then volunteer the 0.011 confidence separation and the
   unchanged ON/OFF comparison before being asked.

**Backup if the quota is gone:** `python scripts/check_models.py` shows
exactly which models are exhausted, and `pytest -q` runs all 172 tests with no
key. `--recall-only` also runs free.

---

## 6. The three weakest points, answered honestly

**1. The confidence score is decoration.**
0.011 separation. Do not defend it — explain why it fails (both terms are
near-constant) and what would fix it: a signal correlating with *relevance*
rather than *groundedness*, such as the margin between the best and
second-best chunk, the analyzer's own hedging, or agreement across the
retrieved set. The weights live in `config.py` precisely so they can be
re-fitted.

**2. The verification loop — the centrepiece — is unproven at scale.**
Zero of 15 outcomes changed. Say so first, then the structural reason (it is
the third gate and the first two fire first), then what it would take to
measure (a dataset inducing bad quotes rather than unanswerable questions).
Do not reach for the unit test as though it settled the question. The
defensible claim is narrow and real: **it costs almost nothing** (the judge is
rarely called, because verbatim quotes settle by string matching) **and it has
never produced a false negative** — 54 of 54 genuine quotes passed.

**3. The evaluation is small and single-run.**
53 questions, so one question is ~2 percentage points. Temperature 0.0 makes
it deterministic but there are no repeated runs, so no variance estimate. The
grader checks the decisive fact is present, not that the rest is right. And
the corpus is two documents plus a licence — nothing has been tested at a
scale where the router's document-selection step would itself have to become
retrieval.

---

## 7. Things to know before changing anything

### The free tier is the binding constraint

**~20 requests per model per day**, charged **per model, not per key**
(`quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `retryDelay`
~18 hours). One answer is 2–5 calls. `config.LLM_FALLBACK_MODELS` chains
sibling models because each name has its own bucket — that is what makes a
53-question run affordable at all.

Run `python scripts/check_models.py` first; it reports how much of the chain
is callable right now, and mirrors the client's own retry quirks so it cannot
report a false failure.

### Two hard-won details

- **A per-day 429 must not be retried like a per-minute one.** Backing off
  2/4/9/17s against an 18-hour reset, on every call, cost one measured
  question 88.5s instead of 14.8s. A daily refusal now marks the model spent
  for the run.
- **The Groq link is `gpt-oss-120b`, not a Llama.** This key reaches no Llama
  chat model at all; the originally configured `llama-3.3-70b-versatile`
  404'd on every call. A fallback that cannot be called is worse than none.

### Conventions

- Comments say **why**, not what. Every non-obvious constant carries its
  reason.
- Type hints everywhere; `logging` in `src/`, never `print` (scripts print
  reports deliberately).
- Tests never need an API key.
- An abstention is a **200**.
- Never commit: `.env`, `data/`, `*.pdf`, `evaluation/results/`,
  `__pycache__`. All in `.gitignore`.
- Routes are `/ask`, `/documents`, `/queries`, `/stats` — **not**
  `/analyze`, `/ingest`, `/history`, which an earlier brief described and
  which do not exist.
- Statuses are `ANSWERED`, `PARTIALLY_VERIFIED`, `INSUFFICIENT_EVIDENCE`,
  `OUT_OF_SCOPE`, `ERROR` — five, not three.

---

## 8. If there is a next session

Highest value first:

1. **Per-document retrieval and merge.** Fixes the 20% both-cited figure, the
   clearest measured weakness, and Recall@5 already localises the cause.
2. **A quote-fidelity dataset** so the verification loop can actually be
   measured — questions that induce plausible-but-unsupported quotes.
3. **Re-fit or drop the confidence formula.** It needs a signal that varies.
4. **Build the Docker image.** Never once built.
5. **Repeat runs for a variance estimate**, once quota allows.
