# PolicyLens

Agentic question answering over legal and policy documents, where every claim
carries a quote that has been checked against the passage it cites. A
LangGraph agent routes a question, retrieves passages above a calibrated
relevance threshold, drafts an answer with verbatim citations, then re-reads
each cited passage **by id** and withdraws the answer if the quotes do not
check out. The point is not to answer more questions than a plain RAG
pipeline — it is to refuse the ones the documents do not support, which it
does on 14 of 15 unanswerable questions while answering 33 of 38 answerable
ones.

---

## Why this exists

A conventional RAG pipeline over a statute will answer anything. Ask it the
minimum wage in rupees and it returns five plausible paragraphs of labour law
and a confident figure that appears nowhere in the Act. For a legal document
that is the failure that does real damage, because it is indistinguishable
from a correct answer: same shape, same citations, same confident tone.

PolicyLens is built around making that failure visible and rare. Three
independent gates must all pass before anything is asserted, and a quote is
never trusted because the model produced it.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
  question ───▶ ROUTER ──── OUT_OF_SCOPE ──────────────────┐   │
                    │                                      │   │
                    ▼                                      ▼   │
                RETRIEVER ── nothing above threshold ──▶ RESPONDER ──▶ answer
                    │                                      ▲   ▲
                    ▼                                      │   │
                 ANALYZER ── no answer in passages ────────┤   │
                    │                                      │   │
                    ▼                                      │   │
                 VERIFIER ── nothing verified ─────────────┘   │
                    │                                          │
                    │        ┌──── retry (max 2) ────┐         │
                    │        ▼                       │         │
                    └──▶ all quotes checked ─────────┴─────────┘
```

| Node | Does | Why it exists |
|---|---|---|
| **Router** | Classifies intent, picks documents, rewrites the search query | An out-of-scope question should cost one LLM call, not five. The rewrite matters because embeddings match wording, not intent. |
| **Retriever** | Embeds the query, returns chunks above the threshold | The **only** place that decides whether evidence exists. Without its gate, vector search always "succeeds". |
| **Analyzer** | Drafts the answer with verbatim quotes, or declines | May abstain even with passages in hand — retrieved is not the same as relevant. |
| **Verifier** | Re-reads each cited chunk **by id**, fuzzy-matches, escalates to an LLM judge only when string matching cannot settle it | The heart of the system. A quote is checked against the source, never against what the analyzer echoed back. |
| **Responder** | Assigns the final status and computes confidence | Keeps "I found no evidence" and "I could not run" as different claims. |

Full detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the reasoning
behind each choice in [docs/DECISIONS.md](docs/DECISIONS.md).

### The five statuses

| Status | Meaning | HTTP |
|---|---|---|
| `ANSWERED` | Answered; every citation verified | 200 |
| `PARTIALLY_VERIFIED` | Answered, but a citation could not be confirmed. Shown and labelled, not hidden | 200 |
| `INSUFFICIENT_EVIDENCE` | The documents do not support an answer. **A correct outcome** | 200 |
| `OUT_OF_SCOPE` | Not a question about these documents | 200 |
| `ERROR` | The pipeline itself failed | 503 |

**An abstention is a 200.** A 4xx would tell every client the request was
malformed; a 5xx would make the project's most important behaviour read as an
outage in every log and dashboard. Only a pipeline failure is 5xx. There is a
test whose only job is to hold that line.

---

## Key features

- **Citations verified against the source, not the model.** Each quote is
  re-read from the vector store by `chunk_id`. RapidFuzz settles most cases;
  an LLM judge is called only when string matching cannot.
- **Fuzzy matching alone is not trusted.** Swapping "six months" for "one
  year" in a 30-word clause still scores in the nineties, and deleting one
  "not" costs about three points. Both would pass a ratio check while
  asserting the opposite of the source, so quantities and negations are
  aligned and compared in both directions.
- **Three independent abstention gates** — the calibrated threshold, the
  analyzer's own reading, and verification. Any one failing ends in
  `INSUFFICIENT_EVIDENCE` rather than an uncited answer.
- **Calibrated, not guessed.** The relevance threshold (0.32) is fitted on
  real score distributions by `scripts/calibrate_threshold.py`, maximising
  balanced accuracy rather than plain accuracy.
- **Computed confidence, never self-reported.** Self-reported confidence
  clusters near 0.9 regardless of correctness. (Measured: this formula does
  not work either — see below.)
- **A fallback chain that multiplies a free tier.** The daily quota is charged
  per *model*, so chaining sibling models multiplies what one key can do.
- **172 tests, none needing an API key.** The graph runs against a scripted
  LLM, the API against a fake agent, retrieval against a fake embedder.

---

## Evaluation results

53 labelled questions, every expected fact quoted from the indexed text. Run
on 2026-10-06 against 279 chunks from 3 documents. Raw per-question output is
written to `evaluation/results/` (gitignored — regenerate with
`python scripts/evaluate.py`).

### Honesty — the two error types, never averaged

| Metric | Result | |
|---|---|---|
| **False answer rate** | **6.7%** | 1 of 15 unanswerable questions got an answer |
| **False abstention rate** | **0%** | 0 of 38 answerable questions were refused |

They are reported separately because they do not cost the same. A false
abstention denies the user an answer and they can see it happen. A false
answer looks exactly like a correct one. A single "accuracy" figure would let
an improvement in the cheap error hide a regression in the expensive one, so
`scripts/evaluate.py` exits non-zero on a false answer alone.

### Correctness and grounding

| Metric | Result | Basis |
|---|---|---|
| Answer accuracy | **86.8%** | 33 of 38 answerable |
| Citation verification | **100%** | 54 of 54 quotes confirmed |
| Citation recall | 78.9% | cited every document the answer needed |
| Overall correct | 88.7% | 47 of 53, whichever decision was right |

### Retrieval, measured alone (no LLM, k=5)

| Metric | Result |
|---|---|
| Recall@5, all answerable | **92.1%** |
| Recall@5, single-document | **100%** |
| Recall@5, cross-document | **70%** |
| Mean best similarity | 0.614 |

### Cross-document, broken out

| Metric | Result |
|---|---|
| Questions | 10 |
| Answer accuracy | 70% |
| Cited **both** documents | **20%** |

### Cost

| Metric | Result |
|---|---|
| Mean LLM calls per question | 2.0 |
| Mean latency | 10.2 s |
| Tokens, 38-question run | 88,828 |

### Verification loop: ON vs OFF

| | ON | OFF |
|---|---|---|
| False answer rate | 6.7% | **6.7%** |
| Correct | 14 / 15 | **14 / 15** |
| Statuses changed | — | **0** |

Run on the 15 unanswerable questions. **Nothing changed.** See finding 5.

### Five findings that are not flattering

**1. The confidence formula does not work.** `compute_confidence()` claims
confidence tracks correctness. Measured: **0.850 when right, 0.839 when
wrong — a separation of 0.011.** It does not discriminate. The config comment
inviting Phase 6 to re-fit the weights was right to; until that happens the
number is decoration, and this README will not dress it up.

**2. Perfect grounding is not correctness.** All 54 quotes verified, and 13%
of answers still missed the decisive fact. Verification prevents
*fabrication*, not *irrelevance* — a real quote from a real passage that does
not answer the question passes every check the verifier makes.

**3. Cross-document retrieval is the weak point, and Recall@5 says why.** All
three recall misses are cross-document, and in each the top five chunks are
five Act chunks and zero Rules chunks. The Act has 234 chunks to the Rules
Summary's 32, so it dominates by mass. Two of the five wrong answers are
retrieval failures, not synthesis failures.

**4. The one false answer is arguably the grader's.** Asked the overtime rate
for a public holiday, the system gave the one adjacent provision that exists
and then stated the documents do not specify a rate. It hallucinated nothing.
The grader classifies by status alone, so it cannot tell that from an invented
figure. The question is mislabelled by this dataset's own rule — but it has
been left in and 6.7% reported anyway, because editing a dataset after seeing
the result makes every number in it suspect. Full adjudication in
[docs/EVALUATION.md](docs/EVALUATION.md).

**5. The verification loop did not change a single outcome** on the
unanswerable set. It is the third of three abstention gates, and the router
(8 questions, `OUT_OF_SCOPE`) and the relevance threshold plus the analyzer
(6 questions) caught everything before it was reached. Only one question
produced a citation at all, and that quote was genuine — so verification
passed it exactly as the control arm published it unchecked. This does not
show verification is worthless; it shows this set does not exercise it,
because the cheaper gates fire first. An honest summary: **the router and the
threshold do most of the abstention work, and verification is insurance whose
premium is near zero but whose payout is still unmeasured at scale.**

---

## Tech stack

| Choice | Why |
|---|---|
| **LangGraph** | The retry is a genuine cycle with a counter and three exit conditions; a sequential chain cannot express it without a hand-rolled loop. |
| **LangChain** | Used only for document loading, splitting and the `Document` type — the parts that are plumbing, not orchestration. |
| **ChromaDB** | Persistent on disk, so an 82-page ingest is paid once. Metadata filtering scopes a search to one document without a collection per document. |
| **Sentence Transformers** (`all-MiniLM-L6-v2`) | Local, free and offline. Embedding 279 chunks via an API would cost money and a network dependency for a value that never changes. |
| **Google Gemini via LiteLLM** | Free tier good enough to build on; LiteLLM makes the provider a config string, so the fallback chain costs nothing to express. |
| **FastAPI** | The response model *is* the public contract, and it generates `/docs` for free. Dependency injection is what lets 27 API tests run against a fake agent. |
| **Streamlit** | The UI is a reader for one person, not a product. It talks to the API over HTTP only, never importing the agent. |
| **SQLite** | Every question worth asking of the query log is an aggregate — abstention rate, mean confidence by status — which is one SQL query each and a file scan plus parse against JSONL. |
| **RapidFuzz** | Character-level matching settles most citation checks without an LLM call, which keeps verification affordable on a free tier. |
| **pytest** | 172 tests, no API key required. |

---

## Quickstart

### Local

```bash
git clone <repo> && cd policylens
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then put a real GOOGLE_API_KEY in it
python scripts/check_models.py  # confirms which models your key can reach

python scripts/download_sample_docs.py   # optional sample corpus
python scripts/ingest_docs.py            # PDF/txt -> chunks -> ChromaDB

uvicorn src.api.app:app --port 8000      # terminal 1
streamlit run frontend/app.py            # terminal 2  -> localhost:8501
```

Ask one question without the server:

```bash
python scripts/ask.py "What is the maximum probation period for a worker?" --trace
```

### Docker

```bash
cp .env.example .env            # put a real GOOGLE_API_KEY in it
docker compose up --build       # API on :8000, UI on :8501
docker compose run --rm api python scripts/ingest_docs.py
```

> Built and verified in CI rather than locally — the development machine is
> Windows 11 Home without WSL, so Docker Desktop's Linux engine could not
> start. The `docker` job in CI builds the image, asserts it runs as the
> non-root `policylens` user, and checks the embedding model was baked in so
> the container starts offline. All three pass.

### API by example

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the maximum probation period for a worker?"}'
```

```json
{
  "question": "What is the maximum probation period for a worker?",
  "status": "ANSWERED",
  "answer": "According to the Nepal Labour Act, 2074, an employer may enter into an employment contract with a worker such that they remain in a probation period for a maximum of six months.",
  "citations": [
    {
      "chunk_id": "Nepal_Labour_Act_2074.pdf::p7::c12",
      "quote": "In entering into an employment contract with a labour, the employer may so enter into the contract that he or she remains in a probation period for six months",
      "source_document": "Nepal_Labour_Act_2074.pdf",
      "page_number": 7,
      "section": "Section 13",
      "verified": true,
      "method": "exact",
      "match_score": 100.0,
      "note": ""
    }
  ],
  "confidence": 0.803,
  "intent": "DOCUMENT_QA",
  "top_similarity": 0.5063,
  "verified_count": 1,
  "unverified_count": 0,
  "latency_seconds": 14.8,
  "query_id": 7
}
```

An abstention, which is also a 200:

```json
{
  "status": "INSUFFICIENT_EVIDENCE",
  "answer": "I could not find support for an answer in the indexed documents.",
  "citations": [],
  "confidence": 0.0
}
```

### Endpoints

| Method | Path | Does |
|---|---|---|
| `POST` | `/ask` | Answer a question |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/documents` | Upload and index a PDF or .txt |
| `DELETE` | `/documents/{name}` | Remove a document |
| `GET` | `/health` | Index size, key presence, model chain, threshold |
| `GET` | `/queries` | Recent questions and answers |
| `GET` | `/stats` | Aggregates, including the abstention rate |

Interactive docs at `/docs`.

### Evaluation

```bash
python scripts/evaluate.py                      # all 53 questions
python scripts/evaluate.py --only cross         # cross-document only
python scripts/evaluate.py --recall-only        # Recall@5, no LLM, free
python scripts/evaluate.py --no-verify          # control arm
python scripts/evaluate.py --limit 6            # stay inside the daily quota
```

---

## Limitations

**The confidence score is not calibrated.** Separation between right and
wrong answers is 0.011. Treat it as a placeholder.

**Cross-document answers usually cite one document, not both** (20%). The
larger document dominates the top-5 by chunk count. A fix would retrieve per
document and merge, rather than taking a global top-5.

**Verification catches fabrication, not irrelevance.** A verbatim quote from a
real passage that does not answer the question passes every check.

**The verification loop changed nothing measurable on the unanswerable set.**
Run with `--no-verify`, all 15 questions came out identically: same 6.7% false
answer rate, same 14 correct, **zero statuses changed**. Verification is the
third of three abstention gates, and the router and the relevance threshold
caught every case before it was reached. Its value is demonstrated only by a
unit test — the same fabricated quote is withdrawn with verification and
published without it. Measuring it properly needs a dataset that reliably
induces plausible-but-unsupported *quotes*, which is harder to write than an
unanswerable question and is the obvious next piece of work. Full comparison
in [docs/EVALUATION.md](docs/EVALUATION.md).

**The grader checks that the decisive fact is present**, not that everything
around it is right. An answer containing "six months" plus an unrelated
falsehood still passes. Citation verification and recall are reported beside
it for that reason.

**Section labels are best-effort.** The chunk whose text begins "13. Provision
relating to probation period" carries the metadata "Section 12". Document and
page are reliable; the section locator drifts by one, and the evaluation does
not grade on it.

**The free tier is the binding constraint.** ~20 requests per model per day,
charged per model. One answer is 2–5 calls, so a full 53-question run is not
always affordable in one sitting. `scripts/evaluate.py` records how much of
the set actually ran.

**Text-layer PDFs only.** A scanned PDF extracts nothing; it is reported as
skipped rather than indexed as empty. OCR is not wired in.

**Two documents and a licence is a small corpus.** Nothing here has been
tested at hundreds of documents, where the router's document-selection step
would need to become retrieval in its own right.

---

## Project layout

```
src/ingestion/    loading, cleaning, chunking, indexing
src/retrieval/    embeddings and the ChromaDB store with the relevance gate
src/agent/        the graph, the five nodes, the single LLM call site, prompts
src/api/          FastAPI app and the response schemas that are the contract
src/db/           SQLite query log
frontend/         Streamlit UI (HTTP only)
evaluation/       the 53-question labelled dataset and the grader
scripts/          ingest, ask, calibrate, check models, evaluate
tests/            172 tests, no API key needed
```

---

## References

- Nepal Labour Act, 2074 (2017) and Labour Rules, 2075 — the corpus
- Apache License 2.0 — second corpus document, for cross-document contrast
- [LangGraph](https://langchain-ai.github.io/langgraph/) · [ChromaDB](https://docs.trychroma.com/) · [Sentence Transformers](https://www.sbert.net/) · [LiteLLM](https://docs.litellm.ai/) · [RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/) · [FastAPI](https://fastapi.tiangolo.com/)
- Gemini API rate limits — https://ai.google.dev/gemini-api/docs/rate-limits
