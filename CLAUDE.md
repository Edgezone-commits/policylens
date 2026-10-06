# PolicyLens — project state

Agentic question answering over legal and policy documents, where every claim
carries a quote that has been checked against the passage it cites.

**Audited: 2026-10-06.** Everything below was verified by running it, not read
off a plan. Numbers come from real evaluation runs.

---

## 1. Current state at a glance

| Phase | Area | State |
|---|---|---|
| 0 | Scaffold, central config | **Done** |
| 1 | Ingestion (PDF → chunks → ChromaDB) | **Done** — but **no tests** |
| 2 | Retrieval + calibrated threshold | **Done** — but **no tests** |
| 3 | LangGraph agent, 5 nodes + retry loop | **Done**, 25 tests |
| 4 | FastAPI | **Done**, 27 tests |
| 5 | Streamlit frontend | **Done but UNCOMMITTED** |
| 6 | Evaluation | **Done**, 19 tests — 20 questions, not 50 |
| — | Docker, CI, README, docs/ | **Not started** |

- **120 tests, all passing**, none needing an API key (`pytest -q`, ~54s).
- **Index is live**: 279 chunks across 3 documents.
- **All 15 key modules import cleanly.**
- **FastAPI starts and serves**: verified `uvicorn src.api.app:app --port 8000`,
  `/health` returns `status: ok`.
- **Streamlit runs**: verified end-to-end via `streamlit.testing.v1.AppTest`
  with 0 exceptions, and by driving a real question through it.

### Git

- Branch: `feat/evaluation`. `main` is at `e2a118d` (one commit behind).
- Recent: `306cd06` evaluation · `e2a118d` API layer · `5722c49` token budget
  fix · `99331f4` agent graph · `a072cda` retrieval · `c17111c` ingestion.
- **Uncommitted right now**: `frontend/` (new), `tests/test_frontend.py` (new),
  `src/config.py` (adds `API_BASE_URL`, `API_TIMEOUT_SECONDS`).

---

## 2. Important: the brief does not match the code

Five differences. The code is correct and working; the brief describes
something else. **These need a decision before Steps A–D.**

### 2.1 API route names

| Brief says | Actually is |
|---|---|
| `/analyze` | `POST /ask` |
| `/ingest` | `POST /documents` |
| `/history` | `GET /queries` and `GET /stats` |
| — | `GET /documents`, `DELETE /documents/{name}`, `GET /health` |

The frontend and 27 API tests use the real names. Renaming would touch the
app, the schemas, the tests and the UI.

### 2.2 Status values — there are five, not three

Real: `ANSWERED`, `PARTIALLY_VERIFIED`, `INSUFFICIENT_EVIDENCE`,
`OUT_OF_SCOPE`, `ERROR`.

The brief says `PARTIALLY_ANSWERED` (actual name is `PARTIALLY_VERIFIED`) and
gives a three-colour badge mapping that omits `OUT_OF_SCOPE` and `ERROR`. Those
two matter: `OUT_OF_SCOPE` is a cheap 1-call abstention from the router, and
`ERROR` must **never** look like an abstention — "I could not run" and "I found
no evidence" are different claims. `tests/test_frontend.py` asserts this.

### 2.3 Evaluation: 20 questions in Python, not 50 in JSON

| Brief | Actual |
|---|---|
| `evaluation/test_questions.json` | `evaluation/dataset.py` (typed `Question` dataclass) |
| `evaluation/evaluate.py` | `scripts/evaluate.py` |
| 50 questions: 25 single-doc, 10 cross-doc, 10 unanswerable, 5 adversarial | 20: 12 answerable, 4 out-of-domain, 4 adversarial |

**No cross-document questions exist.** That is the biggest real gap — it is the
one question type the corpus supports (Labour Act + Rules Summary overlap) and
nothing tests it.

### 2.4 Two metrics are missing

Present: false answer rate, false abstention rate, answer accuracy, citation
verification rate, retrieval recall, confidence separation, cost.

Missing:
- **Recall@5** — current recall asks "did the answer cite the expected
  document", not "was the right chunk in the top 5".
- **Verification loop ON vs OFF** — not implemented at all. There is no flag to
  disable verification, so this comparison cannot be run today.

### 2.5 Frontend is done, not partial

`frontend/app.py` already covers upload, question input, colour-coded status,
confidence, per-citation document/page/quote, collapsible reasoning path,
sidebar document list and history. It needs **committing**, and adjusting only
if the brief's naming is adopted.

---

## 3. Verified evaluation results

Full set, run `2026-10-06 12:29` (`evaluation/results/evaluation_20261006_122954.json`):

| Metric | Result |
|---|---|
| Questions | 20 (18 scored, 2 excluded) |
| **False answer rate** | **0%** — 8 unanswerable, none answered |
| **False abstention rate** | **0%** — 12 answerable, none refused |
| Answer accuracy | 100% |
| Citation verification | 100% (16/16 quotes) |
| Retrieval recall | 100% |
| Confidence separation | **n/a** |
| Cost | 2.0 LLM calls, 15.0s per question |

Two caveats that must survive into the README:

1. **The 2 excluded questions** hit an exhausted model chain (`ERROR`), so they
   are excluded rather than counted as mistakes — a run that never happened made
   no claim. Both scored **correct** in a separate adversarial-only run
   (`evaluation_20261006_120926.json`, 4/4).
2. **Confidence separation is unavailable, not good.** `compute_confidence()`
   claims confidence tracks correctness; measuring that needs both correct and
   incorrect answers, and there were no incorrect ones. The formula is
   **unfalsified here, not validated.** Reporting 0.0 would be a lie.

A dataset bug the first run caught, worth keeping in the record: "What notice
period applies to terminating a labour for misconduct?" was labelled
unanswerable and scored as a false answer. The agent was right — Section 144(1)
excludes misconduct explicitly, so "none applies" is the correct answer, with a
verified citation. The label was wrong. It moved to the answerable set, and the
rule is now written down in `dataset.py`: **an unanswerable question must be one
where no honest statement of absence answers it either.** Two Apache questions
failed that test and were replaced.

---

## 4. File inventory

### Source — all working

```
src/config.py              every tuneable, each with a WHY comment
src/ingestion/loader.py    PyPDFLoader + text, per-page Documents
src/ingestion/chunker.py   RecursiveCharacterTextSplitter, chunk_size=1000
src/ingestion/indexer.py   ingest_file / ingest_directory, SUPPORTED_EXTENSIONS
src/retrieval/embeddings.py  all-MiniLM-L6-v2, 384 dims, local
src/retrieval/store.py     ChromaDB wrapper, threshold gate in retrieve()
src/agent/graph.py         StateGraph: 5 nodes, 3 exits, 1 retry cycle
src/agent/llm.py           the ONLY LLM call site: chain, throttle, retries
src/agent/prompts.py       all prompts, commented
src/agent/state.py         AgentState, Citation, RetrievedChunk, trace_entry
src/agent/nodes/*.py       router, retriever, analyzer, verifier, responder
src/api/app.py             create_app(agent, query_log) — injectable for tests
src/api/schemas.py         the public contract; a citation always carries a verdict
src/db/queries.py          SQLite query log, WAL, best-effort writes
evaluation/dataset.py      20 labelled questions, each with provenance
evaluation/metrics.py      the grader — pure, testable on strings
frontend/app.py            Streamlit UI (UNCOMMITTED)
```

### Scripts

`ingest_docs.py` · `ask.py` (`--trace`, `--json`) · `calibrate_threshold.py` ·
`check_models.py` (reports which models the key can actually reach) ·
`evaluate.py` · `download_sample_docs.py`

### Tests — 120 passing

| File | Tests | Covers |
|---|---|---|
| `test_agent_graph.py` | 25 | graph control flow against a scripted LLM |
| `test_api.py` | 27 | every endpoint against a fake agent |
| `test_llm_client.py` | 37 | failover, quota, throttle, truncation, JSON repair |
| `test_evaluation.py` | 19 | the grader, on strings |
| `test_frontend.py` | 12 | UI/API status contract (UNCOMMITTED) |

**Gap: `src/ingestion` and `src/retrieval` have zero tests.**
`tests/fixtures/sample_policy.txt` exists but nothing references it.

### Missing entirely

`README.md` · `docs/ARCHITECTURE.md` · `docs/DECISIONS.md` ·
`docs/EVALUATION.md` · `Dockerfile` · `docker-compose.yml` ·
`.github/workflows/ci.yml` (`.github/` exists, empty) ·
`tests/test_ingestion.py` · `tests/test_retrieval.py`

---

## 5. Things worth knowing before changing anything

### The free tier is the binding constraint

**~20 requests per model per day**, charged **per model, not per key**
(`quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `retryDelay`
~18 hours). One answer is 2–5 calls, so that is ~4–5 questions a day on one
model. `config.LLM_FALLBACK_MODELS` chains sibling models because each name has
its own daily bucket — that is what makes a 20-question evaluation affordable.

**Quota was heavily used on 2026-10-06.** A 50-question set, plus a
verification ON/OFF comparison, is likely **not affordable in one day**. Run
`python scripts/check_models.py` first; it reports how much of the chain is
callable right now.

### The Groq link

`groq/openai/gpt-oss-120b`, not a Llama. This key reaches **no llama chat
model** — `GET /openai/v1/models` lists eleven, of which only the two gpt-oss
sizes and qwen3.8-27b are general chat. The old
`groq/llama-3.3-70b-versatile` 404'd on every call.

### Conventions this codebase holds to

- Comments say **WHY**, not what. Every non-obvious constant carries its reason.
- Type hints everywhere; `logging`, never `print` (in `src/` — scripts print
  reports deliberately).
- Tests never need an API key. The graph runs against a scripted LLM.
- An abstention is a **200**, never a 4xx/5xx. `test_api.py` pins this.
- Never commit: `.env`, `data/`, `*.pdf`, `evaluation/results/`, `__pycache__`.
  All covered by `.gitignore`.

---

## 6. Next step

Awaiting a decision on the five mismatches in section 2 — chiefly whether to
rename the endpoints and convert the dataset to 50-question JSON, or keep what
is built and working.

Then: commit the frontend (A), extend evaluation (B), add
ingestion/retrieval tests + Docker + CI (C), write README and docs/ (D).
