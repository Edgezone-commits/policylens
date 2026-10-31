# 60-second demo script

For a Loom or a screen recording. Read it roughly as written — it is timed to
fit, and the last fifteen seconds are the part that distinguishes the project,
so do not let the earlier parts run long.

**Before you hit record**

- `python scripts/check_models.py` — confirm the key has quota. One answer
  costs 2–5 calls against about twenty per model per day, and this demo asks
  two questions. A quota failure mid-recording looks like a broken project.
- Start both processes (`make run`) and ask one throwaway question first, so
  the embedding model is loaded and the UI is warm.
- Have the reasoning-path panel collapsed, not expanded. Opening it on camera
  is the beat that shows it exists.
- There is no evaluation dashboard. The last section shows the results table
  in `README.md` — that is the honest artifact, and a reviewer reads it
  faster than a screenshot of a terminal.

---

## 0:00–0:08 — The problem, stated as a failure

> "Ask a normal RAG pipeline what the minimum wage is in a statute that never
> states one, and it will tell you. It retrieves the five nearest paragraphs
> and writes a confident number. That failure is indistinguishable from a
> correct answer — same shape, same citations, same tone. PolicyLens is built
> to make it visible and rare."

*On screen: the UI, idle, with the three indexed documents listed in the
sidebar.*

## 0:08–0:22 — One question it can answer

Type: **What is the maximum probation period for a worker?**

> "One question. The agent classifies it, rewrites it into the Act's own
> vocabulary, retrieves passages above a calibrated threshold, and drafts an
> answer where every claim carries a verbatim quote."

*Let the answer render. Point at the status and the confidence.*

> "Answered. Six months, cited to Section 13, page 7 — and that citation is
> marked verified."

## 0:22–0:34 — Why the citation can be trusted

*Expand the citation, then the reasoning path.*

> "Here is what verification actually means. The quote is not compared against
> what the model echoed back to me. The citation names a chunk id, and the
> verifier goes back to the vector store, re-reads that passage by id, and
> compares. An invented chunk id fails as unknown. A quote that drifted fails
> on the words that carry the meaning — swapping 'six months' for 'one year'
> still scores in the nineties on a fuzzy match, so quantities and negations
> are compared separately."

*On screen: the trace, one line per node — router, retriever, analyzer,
verifier, responder.*

> "And the reasoning path is a field in the API response, not something I dug
> out of a log."

## 0:34–0:46 — The failure case, which is the point

Type: **How much is the minimum monthly wage in rupees?**

> "Now the one that matters. The Act says the Government fixes the minimum
> wage. It never states an amount. So there is no honest answer, and any
> figure would be invented."

*Let it render.*

> "Insufficient evidence — and notice it is a result with its own panel, not
> an error. It is an HTTP 200, because refusing is the feature working. Three
> independent gates can each end here: the relevance threshold, the analyzer
> deciding the passages do not answer the question, and verification."

## 0:46–1:00 — The numbers, including the bad ones

*Switch to the README's results table.*

> "Measured on 53 labelled questions. Zero percent false abstention: it never
> refused a question the documents answer. Six point seven percent false
> answer: one of fifteen. All 54 quotes verified."

> "Two things I will volunteer. The confidence score does not work — 0.850
> when right against 0.839 when wrong, so it does not discriminate, and I can
> explain mechanically why. And the verification loop changed zero outcomes on
> the unanswerable set, because it is the third of three gates and the first
> two fire first. Both of those are in the README, because the interesting
> part of an evaluation is the part that did not come out the way you hoped."

---

## Notes on delivery

**Do not demo the gratuity question.** It produces six citations across three
provisions and is the better engineering story, but it takes 25 seconds to
answer and that is a third of the recording spent watching a spinner. Keep it
for a live conversation where the wait can be talked through.

**Do not apologise for the latency.** Roughly 9 seconds at p50, most of it a
deliberate 4-second throttle between LLM calls to stay inside a free tier. If
asked, that is the answer, and it is a configuration value rather than a
property of the system.

**If the quota dies mid-recording**, stop and say so rather than cutting
around it. `pytest -q` runs all 196 tests with no key, and
`make eval` reproduces the retrieval and latency tables for free — a
recording that shows the fallback honestly is worth more than a second take
that hides it.
