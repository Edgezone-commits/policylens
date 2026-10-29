---
title: PolicyLens
emoji: 📄
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Agentic RAG over policy documents, with every quote verified against its source
---

# PolicyLens

Agentic question answering over legal and policy documents, where every claim
carries a quote that has been checked against the passage it cites.

A LangGraph agent routes the question, retrieves passages above a calibrated
relevance threshold, drafts an answer with verbatim citations, then re-reads
each cited passage **by id** and withdraws the answer if the quotes do not
check out. The point is not to answer more questions than a plain RAG
pipeline — it is to refuse the ones the documents do not support.

**Try the refusal first.** Ask *"how much is the minimum monthly wage in
rupees?"* The indexed Act says the Government fixes the minimum wage and never
states an amount, so the honest answer is that the documents do not contain
one. This Space returns `INSUFFICIENT_EVIDENCE` rather than a plausible
figure.

Then ask something it can answer, such as *"what is the maximum probation
period for a worker?"*, and expand the citation to see the quote, the verdict,
and the agent's reasoning path.

## Corpus

- Nepal Labour Act, 2074 (2017)
- Nepal Labour Rules, 2075 — summary
- Apache License 2.0, as a deliberately unrelated third document

## Notes

This runs on a free Gemini tier, which allows roughly twenty requests per
model per day and charges them per model rather than per key. One answer
costs two to five calls. If every question starts failing, the day's quota is
gone rather than the app being broken.

Source, architecture notes and the measured evaluation:
https://github.com/Edgezone-commits/policylens
