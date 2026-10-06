# PolicyLens -- one image, two entrypoints (API and UI).
#
# WHY one image rather than two: the API and the Streamlit UI import the same
# src/ package and the same pinned dependency set. Two Dockerfiles would mean
# two places to keep that set in step, and the image is dominated by torch
# either way, so splitting saves nothing and costs a drift risk.
# docker-compose.yml runs the same image twice with different commands.

FROM python:3.12-slim AS base

# PYTHONDONTWRITEBYTECODE: .pyc files in a read-only-ish container are dead
#   weight and would be written by root into a layer nobody reads.
# PYTHONUNBUFFERED: without it, logs sit in a buffer and `docker logs` shows
#   nothing while the agent is working -- which, given one answer takes 10s of
#   LLM calls, looks exactly like a hang.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build-time only. PyPDF and chromadb pull in packages with C extensions that
# have no wheel for every platform, so a compiler has to be present while pip
# runs -- and must not be present in the final image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Requirements copied alone, before the source, so Docker's layer cache keeps
# the ~2GB dependency install when only application code changes. Copying
# everything first would reinstall torch on every edit.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# -- the embedding model ----------------------------------------------------
#
# Baked into the image at build time rather than downloaded on first run.
# all-MiniLM-L6-v2 is ~90MB from the HuggingFace Hub, and fetching it at
# startup means the first question of every fresh container waits on a network
# call that can be slow, rate-limited for unauthenticated clients, or simply
# unavailable. Baked in, the container starts offline and answers immediately.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}')"

# Non-root. The container writes only to /app/data (the Chroma index and the
# SQLite log), so it needs no privilege beyond that, and an uploaded PDF is
# untrusted input being parsed by a C extension -- not something to hand root.
RUN useradd --create-home --uid 10001 policylens

COPY --chown=policylens:policylens src/ ./src/
COPY --chown=policylens:policylens frontend/ ./frontend/
COPY --chown=policylens:policylens evaluation/ ./evaluation/
COPY --chown=policylens:policylens scripts/ ./scripts/
COPY --chown=policylens:policylens pyproject.toml ./

# Created and owned here so a bind-mounted or named volume inherits a
# writable directory; otherwise the first write fails as the non-root user.
RUN mkdir -p /app/data/chroma_db /app/data/raw \
    && chown -R policylens:policylens /app/data

# The model cache lives in the home directory the build wrote it to, so the
# non-root user must be able to read it.
RUN chown -R policylens:policylens /root 2>/dev/null || true
ENV HF_HOME=/home/policylens/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/policylens/.cache/torch/sentence_transformers
RUN mkdir -p ${HF_HOME} ${SENTENCE_TRANSFORMERS_HOME} \
    && cp -r /root/.cache/huggingface/. ${HF_HOME}/ 2>/dev/null || true \
    && cp -r /root/.cache/torch/sentence_transformers/. \
       ${SENTENCE_TRANSFORMERS_HOME}/ 2>/dev/null || true \
    && chown -R policylens:policylens /home/policylens/.cache

USER policylens

EXPOSE 8000 8501

# Reports readiness the way the application defines it. /health is not a
# liveness ping here: it also reports whether the index is populated and
# whether an LLM key is present, which are the two things that make every
# question fail while the process looks perfectly healthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
