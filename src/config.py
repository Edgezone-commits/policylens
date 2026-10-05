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
LLM_MODEL: str = os.getenv("POLICYLENS_MODEL", "gemini/gemini-2.5-flash")

# Fallback used only if the primary model errors (e.g. daily quota exhausted).
# The Groq free tier is far more generous per day, which makes it a useful
# safety net during a live demo.
LLM_FALLBACK_MODEL: str = os.getenv(
    "POLICYLENS_FALLBACK_MODEL", "groq/llama-3.3-70b-versatile"
)

# Temperature 0.0 everywhere: this is an extraction-and-citation task, not a
# creative one. We want the same question to give the same answer, which also
# makes the Phase 6 evaluation meaningful rather than noisy.
LLM_TEMPERATURE: float = 0.0

# Cap on response length. Legal answers are short; a high ceiling just invites
# the model to pad, which raises the chance it drifts from the cited text.
LLM_MAX_TOKENS: int = 1024

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

# Confidence weights. WHY a deterministic formula instead of asking the LLM to
# rate its own confidence: self-reported confidence is poorly calibrated and
# clusters near 0.9 regardless of correctness. These observable signals can
# actually be checked against accuracy in Phase 6.
CONFIDENCE_WEIGHT_VERIFIED_RATIO: float = 0.6  # fraction of citations verified
CONFIDENCE_WEIGHT_RETRIEVAL_SCORE: float = 0.4  # top chunk similarity
CONFIDENCE_PENALTY_PER_RETRY: float = 0.10  # each retry signals a shaky answer


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
