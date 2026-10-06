"""Central configuration for PolicyLens.

WHY this file exists: every tunable value (model name, chunk size, thresholds,
paths) lives here and nowhere else. Scattering magic numbers across modules
makes experiments impossible to reproduce -- if chunk_size is written in three
files, you can never be sure which one produced a given evaluation result.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root so `python scripts/...` works from anywhere.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# Get a free Gemini key at aistudio.google.com -- no credit card needed.
#
# This is a LiteLLM model string ("<provider>/<model>"). It is the ONLY place
# the model is named, so swapping providers is a one-line change. The agent
# code never mentions Gemini.
#
# gemini-2.5-flash was the original default and had to be changed: Google now
# refuses it for keys created after its retirement ("no longer available to new
# users", HTTP 404), so a fresh clone with a fresh key could not make a single
# call. Anyone reading an older commit and wondering why the model moved: that
# is why. The free tier's model names are not stable, and
# scripts/check_models.py lists what a given key can actually reach.
LLM_MODEL: str = os.getenv("POLICYLENS_MODEL", "gemini/gemini-3.8-flash")

# Models to try, in order, when the one before it fails.
#
# WHY a CHAIN rather than the single fallback this used to be, and why most of
# it is Gemini rather than another provider: the free-tier daily limit is
# charged per MODEL, not per key. The quota the API reports when it refuses is
#   quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue: 20
# -- twenty requests per day, per model, and a retryDelay of about 22 hours.
# Measured on this key: with gemini-3.8-flash exhausted, gemini-3.5-flash and
# gemini-3.1-flash-lite both answered immediately, because each name has its
# own bucket.
#
# That matters beyond a demo. One answer costs 2-5 calls, so twenty requests is
# four or five questions a day on one model -- and the Phase 6 evaluation needs
# a few hundred. Chaining sibling models multiplies the budget with the key
# already in .env, which is the only way this project's own evaluation is
# runnable for free.
#
# Ordered strongest-first: the chain is a quota and outage fallback, so a later
# model should be a weaker answer, never a wrong one. Groq stays at the end as
# the one non-Google link, since a Google-wide outage takes every entry above
# it at once.
#
# The Groq entry is gpt-oss-120b rather than a Llama: this account's key cannot
# reach ANY llama chat model (GET /openai/v1/models lists eleven, of which the
# only general-purpose chat models are the two gpt-oss sizes and qwen3.8-27b),
# so the llama-3.3-70b-versatile that used to sit here answered every call with
# 404 model_not_found. A fallback that cannot be called is worse than none: it
# spends an attempt and the retry budget to arrive at the same failure. Verify
# with scripts/check_models.py, which exists for exactly this.
LLM_FALLBACK_MODELS: list[str] = [
    model.strip()
    for model in os.getenv(
        "POLICYLENS_FALLBACK_MODELS",
        "gemini/gemini-3.5-flash,"
        "gemini/gemini-3.1-flash-lite,"
        "groq/openai/gpt-oss-120b",
    ).split(",")
    if model.strip()
]

# Temperature 0.0 everywhere: this is an extraction-and-citation task, not a
# creative one. We want the same question to give the same answer, which also
# makes the Phase 6 evaluation meaningful rather than noisy.
LLM_TEMPERATURE: float = 0.0

# Cap on response length. Legal answers are short; a high ceiling just invites
# the model to pad, which raises the chance it drifts from the cited text.
LLM_MAX_TOKENS: int = 1024

# The analyzer is the exception, and not because its prose is longer: its
# reply also carries a VERBATIM QUOTE for every citation it makes. The cap
# above sizes the answer; this one has to cover the answer plus five or
# more exact spans copied out of the source, inside JSON, with every quote
# escaped. Measured on "how much gratuity is a worker entitled to", which
# the Act spreads across Rule 23, Section 52 and Section 53: six quotes,
# 8.5k tokens, and a cut-off reply at anything less. A truncated reply is
# not a shorter answer -- it is no answer at all, reported as a 503 -- so
# the ceiling is set where the evidence says, and brevity stays the
# prompt's job rather than the budget's.
LLM_MAX_TOKENS_FOR_CITED_ANSWER: int = 4096

# Extra budget for models that think before replying. WHY separate from the cap
# above: a thinking model counts its reasoning against the SAME output budget
# as the visible reply, so a 1024 cap is not "1024 tokens of answer" -- it is
# 1024 shared with however much the model decides to think. Gemini 3 Flash
# spent nearly all of it reasoning about a cross-document comparison and
# returned a JSON object cut off mid-string, which downstream looks like the
# model replying with garbage rather than like a budget that was too small.
# The answer cap stays where it is (short answers, less drift); thinking gets
# its own headroom on top, and only for the steps that actually think.
LLM_THINKING_TOKEN_HEADROOM: int = 2048

# Gemini 2.5 Flash enables "thinking" by default, which costs latency and
# tokens. The router and verifier are cheap classification tasks that do not
# benefit from it, so we switch it off for them and keep it for the analyzer.
LLM_REASONING_FOR_SIMPLE_TASKS: str = "disable"

# Free tiers limit requests per MINUTE as well as per day. Spacing calls out
# avoids 429s during batch evaluation, where we fire hundreds in a row.
LLM_MIN_SECONDS_BETWEEN_CALLS: float = float(
    os.getenv("POLICYLENS_LLM_THROTTLE_SECONDS", "4.0")
)
LLM_MAX_RETRIES: int = 4  # retries on 429 / transient errors, with backoff


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# all-MiniLM-L6-v2: 384 dimensions, ~90MB, runs on CPU in milliseconds.
# WHY local rather than an embedding API: free forever, no key, no rate limit,
# and ingestion stays fast even for an 82-page Act. Quality is a shade below
# the big hosted models but ample for semantic search over policy text.
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS: int = 384


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# chunk_size=1000 chars balances two opposing pressures:
#   - larger chunks give the LLM more surrounding context (a legal clause often
#     only makes sense alongside its proviso), but
#   - larger chunks dilute the embedding, so retrieval gets less precise.
# 200 overlap (20%) stops a sentence that straddles a boundary from being lost
# to both chunks. See docs/DECISIONS.md for the measured tradeoff.
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# Discard chunks shorter than this. PDF pages that hold only a page number or a
# stray fragment produce 1-5 character chunks; the Labour Rules document yields
# one chunk containing just "3". Such a chunk can never support a citation, but
# it still occupies a slot in the top-5 and its embedding is near-meaningless,
# so it can outrank real content by accident.
MIN_CHUNK_CHARS: int = 50


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
CHROMA_DIR: Path = PROJECT_ROOT / "data" / "chroma_db"

# ONE collection for all documents, with the filename kept in each chunk's
# metadata. WHY not one collection per document: ChromaDB cannot query across
# collections, so a cross-document question would need a manual loop and
# hand-merged scores. With a single collection, a single-document query is a
# metadata filter and a cross-document query is simply the absence of one.
CHROMA_COLLECTION: str = "policylens_docs"

# Chroma defaults to squared-L2 distance. We force cosine so that
# `similarity = 1 - distance` is a real cosine similarity -- without this the
# relevance threshold below would be compared against the wrong quantity and
# would silently do nothing useful.
CHROMA_DISTANCE_METRIC: str = "cosine"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K: int = 5

# WHY a threshold at all: without one, a vector search ALWAYS returns its five
# nearest chunks, even for a question the corpus has never mentioned. The LLM
# then dutifully answers from irrelevant text and we get a confident
# hallucination. The threshold is what lets the agent say "no evidence".
#
# 0.32 is MEASURED, not guessed. scripts/calibrate_threshold.py scores 14
# answerable and 12 off-topic questions against the live index:
#   answerable:   min 0.331, median 0.601, max 0.759
#   off-topic:    min 0.066, median 0.142, max 0.297
# The two sets separate cleanly, and every threshold in 0.30-0.33 classifies
# all 26 probes correctly. We take 0.32, the midpoint of that band, rather
# than the 0.30 end of it: 0.30 sits just 0.003 above the highest off-topic
# score, so a slightly differently worded question would tip into a false
# answer. The midpoint buys margin on both sides at no measured cost.
RETRIEVAL_SIMILARITY_THRESHOLD: float = float(
    os.getenv("POLICYLENS_SIMILARITY_THRESHOLD", "0.32")
)

# On a verification failure the agent retries with a wider net before giving up.
RETRIEVAL_TOP_K_ON_RETRY: int = 10


# ---------------------------------------------------------------------------
# Agent / verification loop
# ---------------------------------------------------------------------------
# Max 2 retries = at most 3 retrieval attempts. WHY cap it: each retry costs
# 2-3 LLM calls, and in practice a query that fails twice is failing because
# the answer is not in the corpus, not because the query was phrased badly.
# An uncapped loop would burn the daily quota on unanswerable questions.
MAX_RETRIES: int = 2

# A citation's quoted text must fuzzy-match the chunk it claims to come from at
# this ratio or better. 85 tolerates whitespace and punctuation noise while
# still catching a fabricated quote. Below this we escalate to the LLM judge.
CITATION_FUZZY_MATCH_THRESHOLD: int = 85

# Shortest quote that can be verified at all. WHY a floor: the fuzzy check is a
# PARTIAL match (does this quote appear somewhere inside the chunk?), and a
# partial match of three or four words succeeds against almost any legal prose
# -- "the employer shall" scores 100 against a chunk that says nothing about
# the question. Such a quote would be stamped "verified" while supporting
# nothing, which is worse than no citation at all. Quotes below this length are
# rejected outright and the analyzer is asked again.
MIN_CITATION_QUOTE_CHARS: int = 40

# Characters of each chunk shown to the analyzer. Chunks are capped at
# CHUNK_SIZE already, so this only bites on an unusually long one; it exists so
# a malformed chunk cannot blow the context window of a 5-chunk prompt.
EVIDENCE_MAX_CHARS_PER_CHUNK: int = 1600

# Confidence weights. WHY a deterministic formula instead of asking the LLM to
# rate its own confidence: self-reported confidence is poorly calibrated and
# clusters near 0.9 regardless of correctness. These observable signals can
# actually be checked against accuracy in Phase 6.
CONFIDENCE_WEIGHT_VERIFIED_RATIO: float = 0.6  # fraction of citations verified
CONFIDENCE_WEIGHT_RETRIEVAL_SCORE: float = 0.4  # top chunk similarity
CONFIDENCE_PENALTY_PER_RETRY: float = 0.10  # each retry signals a shaky answer


# ---------------------------------------------------------------------------
# API / UI
# ---------------------------------------------------------------------------
# Where the Streamlit UI looks for the API. The UI reaches the agent only
# over HTTP, never by importing it: that keeps the API the single contract,
# so anything the UI can show is something another client could get too,
# and it keeps the 100MB embedding model out of the UI process.
API_BASE_URL: str = os.getenv("POLICYLENS_API_URL", "http://127.0.0.1:8000")

# Longer than it looks necessary: one answer is 2-5 LLM calls, each
# throttled, and a cold model chain can spend a minute failing over before
# the first token. A UI that gave up at 30s would report the system as
# broken while it was working.
API_TIMEOUT_SECONDS: float = float(os.getenv("POLICYLENS_API_TIMEOUT", "300"))

# ---------------------------------------------------------------------------
# Storage / logging
# ---------------------------------------------------------------------------
SQLITE_DB_PATH: Path = PROJECT_ROOT / "data" / "policylens.db"
RAW_DOCS_DIR: Path = PROJECT_ROOT / "data" / "raw"

LOG_LEVEL: str = os.getenv("POLICYLENS_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Set up root logging once, for scripts and the API alike.

    Kept here rather than repeated in every entry point so all components log
    in the same format -- which matters when reading the agent reasoning path
    interleaved with retrieval logs.
    """
    logging.basicConfig(
        level=getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )
    # These libraries log every HTTP call at INFO, which drowns out our own.
    for noisy in ("httpx", "LiteLLM", "litellm", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def ensure_directories() -> None:
    """Create the data directories if missing, so a fresh clone just works."""
    for path in (CHROMA_DIR, RAW_DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)
