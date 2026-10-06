# Decisions

Why the system is built the way it is, in the form the questions usually get
asked. Where a decision has been measured, the number is here rather than an
assurance.

---

### Why LangGraph and not a sequential chain?

Because the retry is a genuine cycle, not a linear step. It is entered from
three different nodes (the retriever when nothing clears the threshold, the
analyzer when the passages don't answer the question, the verifier when
nothing could be confirmed), it carries a counter that must be incremented
exactly once per loop or the loop never terminates, and it has three exit
conditions. A chain expresses that only as a hand-rolled `while` loop around
the whole pipeline, at which point the graph is being reimplemented badly.

The second reason is the trace. Every node appends to `state["trace"]`, so
"how did this answer happen" is a field on the response rather than a logging
exercise. For a system asking to be trusted, being able to show its work is
part of the product.

What LangGraph is **not** used for: anything else. There is no agent executor,
no tool-calling loop, no memory abstraction. Five nodes and four conditional
edges.

### Why LangChain only for loading and splitting?

`PyPDFLoader`, `RecursiveCharacterTextSplitter` and the `Document` type are
plumbing that works and is not worth rewriting. Orchestration is not — that is
what LangGraph does here, explicitly, where the control flow is readable.

The vector store is wrapped directly against `chromadb` rather than through
LangChain's `VectorStore` interface, because the relevance **gate** is the
most important logic in the system and it needs the raw distance scores. Going
through a `Retriever` abstraction that returns documents without scores would
have meant fighting the abstraction to get back the number the whole design
turns on.

### Why ChromaDB and not FAISS?

Three reasons, in order of how much they mattered:

1. **Metadata filtering.** Scoping a search to one document is
   `where={"source_document": name}`. With FAISS, metadata lives in a
   side-car structure the caller maintains, and per-document scoping means
   either a separate index per document or post-filtering a top-k that may
   contain none of the wanted document.
2. **Persistence is the default.** Ingesting an 82-page Act takes real time
   and nobody should pay it twice. `PersistentClient` writes to disk;
   FAISS needs explicit save/load and a parallel store for the text itself,
   since a FAISS index holds vectors and integer ids, not documents.
3. **Lookup by id.** The verifier re-reads a cited chunk by `chunk_id`.
   Chroma's `get(ids=[...])` is exactly that call.

FAISS is faster at scale. At 279 chunks that is irrelevant — retrieval is
0.03s and the LLM call is 5s.

### Why local embeddings and not an embedding API?

- **Free and offline.** 279 chunks embed once, locally. An API would charge
  for a value that never changes and add a network dependency to ingestion.
- **It is the part that does not need a frontier model.** Retrieval quality
  here is gated by chunking and the threshold far more than by embedding
  sophistication; measured Recall@5 on single-document questions is 100%.
- **It keeps the LLM quota for the LLM.** The free tier is ~20 requests per
  model per day. Spending any of it on embeddings would be absurd.

The cost is a ~90MB model load (~45s cold) and a 384-dimension space that is
weaker on long legal phrasing than a larger model would be. The Dockerfile
bakes the model into the image so a container never pays a network fetch.

### Why a relevance threshold at all?

**Because without it a vector search always succeeds.** Ask about chocolate
cake and the nearest-neighbour search returns the five least-unrelated
paragraphs of labour law — there is no "no results" in a cosine space. The
model then answers from them, with citations, confidently. The threshold is
the only thing that converts "these are the closest passages" into "there is
no evidence here".

It is applied twice: as a **gate** (if the best chunk is below it, return
nothing rather than the best of a bad set) and as a **filter** (once the gate
passes, drop the individually weak chunks, which contribute no evidence but do
give the model extra material to misattribute a citation to).

### Why 0.32, and how was it chosen?

Fitted, not guessed. `scripts/calibrate_threshold.py` runs two hand-labelled
probe sets against the live index and records the top-1 similarity of each:
questions the corpus answers, and questions on topics it never mentions.

It maximises **balanced accuracy** — the mean of the two per-class
accuracies — rather than plain accuracy. Plain accuracy can be gamed by
whichever probe set is larger, and the two error types are not equally cheap:
a false abstention is visibly unhelpful, a false answer is confidently wrong.

Measured separation on this corpus: off-topic questions peak around 0.18,
answerable ones score 0.52–0.71. Mean best similarity across the 38 answerable
evaluation questions is 0.614. There is real signal for the gate to work with.

The calibration probes are deliberately **not** the evaluation questions.
Tuning a parameter on the same questions used to report final metrics would
make those metrics measure the tuning.

### Why a verification loop? Isn't a citation enough?

A citation from an LLM is a claim that a passage says something. Verification
is checking. They are not the same thing, and the gap is where confident
hallucination lives: a quote that reads perfectly and does not appear in the
cited passage.

So each quote is re-read from the vector store **by `chunk_id`** — the source,
not the text the analyzer echoed back. If the `chunk_id` was invented, the
lookup returns nothing and the citation fails as `unknown_chunk`.

**Fuzzy matching alone turned out to be unsafe**, which a test caught before
the evaluation did. Swapping "six months" for "one year" in a 30-word clause
still scores in the nineties; deleting one "not" costs about three points.
Both passed as "verified" while asserting the opposite of the source. So
`citation_divergences()` aligns the quote to the span it claims to reproduce
and requires quantities and negations to match in **both** directions.

Failures are **reported, never repaired**. An answer whose quotes do not check
out is withdrawn; a partially verified one is published with the failing
citation marked and the reason shown. Silently dropping a bad citation would
leave a confident answer with no sign that part of it was unsupported.

**What it is worth, measured honestly:** 54 of 54 quotes verified across the
answerable set, and **13% of answers still missed the decisive fact**.
Verification prevents *fabrication*, not *irrelevance* — a real quote from a
real passage that does not answer the question passes every check it makes. On
the unanswerable set, the control arm (`--no-verify`) shows the other two
gates already catching most cases, because the threshold and the analyzer fire
before verification is reached. See [EVALUATION.md](EVALUATION.md).

### Why chunk_size=1000 and overlap=200?

1000 characters is roughly one or two provisions of this statute — enough that
a chunk usually contains a complete rule, which is what makes a quote from it
self-contained enough to verify. Smaller chunks split a provision from its
qualifying clause, and a quote that omits "except where..." reverses the
meaning while matching the text.

200 overlap (20%) exists so a fact spanning a boundary survives in at least
one chunk. A test pins that.

The cost is that a 1000-character chunk routinely straddles a section
boundary, which is why a chunk records both the section it starts in **and**
every section whose heading falls inside it. The chunk holding the probation
answer begins inside Section 12 and contains the whole of Section 13 — so
labelling it "Section 12" alone would be actively misleading in a citation.
This is also why the section locator is documented as best-effort and the
evaluation does not grade on it.

### Why Gemini, and why a chain of models?

Gemini because the free tier is good enough to build on without a credit card,
and because LiteLLM makes the provider a config string — so this is a
one-line change, not an architectural commitment.

The **chain** is the interesting part. The free-tier daily limit is charged
per **model**, not per key:

```
quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier
quotaValue: 20
retryDelay: 66545s        (~18 hours)
```

One answer is 2–5 calls, so twenty requests is four or five questions a day on
one model — and the evaluation needs dozens. Chaining sibling models
multiplies the budget on the key already in `.env`, which is the only thing
that makes this project's own evaluation runnable for free. Measured: with
`gemini-3.8-flash` exhausted, `gemini-3.5-flash` and `gemini-3.1-flash-lite`
both answered immediately.

Ordered strongest-first, because the chain is a quota and outage fallback — a
later model should give a weaker answer, never a wrong one. Groq sits last as
the one non-Google link, since a Google-wide outage takes every entry above it
at once.

Two things learned the hard way, both now in `config.py` comments: the Groq
key reaches **no** Llama chat model (the `llama-3.3-70b-versatile` originally
configured 404'd on every call, and a fallback that cannot be called is worse
than none — it spends an attempt and the retry budget to reach the same
failure); and a per-**day** 429 must not be retried with backoff like a
per-**minute** one. Backing off 2/4/9/17s against an 18-hour reset, on every
call, cost one measured question **88.5 seconds instead of 14.8**.

### Why SQLite and not a JSONL log?

Because every question worth asking of the log is an aggregate: abstention
rate, mean confidence by status, which questions needed a retry, how the
confidence formula correlates with correctness. Those are one SQL query each
against a table, and a file scan plus parse each against a log file. SQLite is
also already in the standard library.

One wide table, with the citations and trace kept as JSON text. Nothing
queries *inside* them — they are read back whole to redisplay an answer —
while every question that matters is about the scalar columns. Three tables
and two joins would buy nothing and cost every reader of the file five
minutes.

WAL mode so the UI can read the log while the API is writing to it; with the
default journal the reader hits "database is locked" exactly when a question
is being answered. Writes are **best-effort**: the caller has already paid
several LLM calls for a verified answer, and losing it to a full disk would be
absurd, so a logging failure is a warning and the answer is still returned.

### Why separate the API and the frontend?

Because the API response model *is* the public contract, and the discipline
only holds if the UI is a client of it like anything else. The UI reaches the
agent **only over HTTP** and never imports `PolicyLensAgent`. Two consequences
worth having:

- Nothing can be shown in the UI that another client could not also obtain. If
  the UI needs a field, the field goes in the schema.
- The ~90MB embedding model stays in one process. Streamlit re-runs its script
  on every interaction; an imported agent would reload the model constantly.

The cost is a hop and a CORS rule, and that the UI has to handle the API being
down — which it does explicitly, with the command to start it.

### Why is an abstention HTTP 200?

Because it is a successful, correct, reasoned response: the system worked out
that the documents do not support an answer and said so. That is the feature.

A 4xx would tell every client the request was malformed. A 5xx would make the
project's most important behaviour read as an outage in every log, dashboard
and alert that saw it — and would teach anyone monitoring it that honesty is a
malfunction. Only `ERROR`, where the pipeline itself could not run, is 503,
because that is the one case where retrying later might work and the client
did nothing wrong.

`tests/test_api.py` has a test whose only job is to pin this.

### Why compute confidence instead of asking the model?

Self-reported confidence clusters near 0.9 whether the answer is right or
wrong, which adds a number that looks informative and is not. So confidence is
built from signals the system measured: the verified fraction (weighted
highest — an answer whose quotes check out is grounded even if retrieval was
mediocre), the retrieval similarity, and a penalty per retry.

**And measured, it does not work.** Mean confidence is 0.850 on correct
answers and 0.839 on wrong ones — a separation of **0.011**. The formula does
not discriminate. The `config.py` comment saying the weights live there so
Phase 6 could re-fit them against measured accuracy was right to; until that
is done, the number is decoration and the README says so.

Keeping it and reporting the failure is the honest option. Deleting it would
hide that the obvious alternative was tried and measured.

### Why a key-fact grader instead of an LLM judge for the evaluation?

- **Deterministic.** A reported number can be reproduced tomorrow, and a
  change in the score means the system moved rather than the judge did.
- **Free.** Spending half of a ~20-request daily budget on marking would halve
  the questions that can be asked.
- **An LLM judge is a member of the family being evaluated.** Using one would
  measure agreement, not correctness.

Its limit is stated wherever a number from it appears: it checks the decisive
fact is **present**, not that everything around it is right. That is why
citation verification and recall are reported beside it — the three together
are a much stronger claim than any one alone.

### Why are the expected answers quoted from the PDF rather than the real statute?

Because the corpus is what the system can be held to. Section 40 of the
indexed text reads "the weekly leave of one day in each month", which is not
what the Labour Act means. Grading against external truth would mark a
faithful reading of the supplied document as wrong, and measure the PDF
extraction rather than the agent.

Every label carries the passage that justifies it, so a disputed label can be
checked rather than argued about.

### Why report two error rates instead of one accuracy figure?

Because they do not cost the same. A false abstention denies the user an
answer the documents hold, and they can see it happen. A false answer looks
exactly like a correct one — same shape, same citations, same confidence band —
and for a legal document that is the failure that does real damage.

Averaging them would let an improvement in the cheap error disguise a
regression in the expensive one. `scripts/evaluate.py` therefore exits
non-zero on a **false answer alone**, so CI can hold that line while treating
a false abstention as a number to note.
