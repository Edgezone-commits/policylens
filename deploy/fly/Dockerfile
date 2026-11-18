# Fly.io deploy -- API only, corpus baked at build time.
#
# WHY A THIRD DOCKERFILE. The root Dockerfile's CMD is the API alone, which
# fits Fly directly -- no Streamlit, no second process. But it does not bake
# the corpus (docker-compose.yml ingests into a *named volume* instead, kept
# across rebuilds on this machine). Fly machines here have no attached
# volume, so a container built from the root Dockerfile would boot with an
# empty index and every question would fail -- a demo that looks broken to
# the one person it exists for. Baking the corpus at build time, the same
# fix already used in deploy/huggingface/Dockerfile, makes a freshly started
# machine answer immediately with no volume required.
#
# KEEP IN SYNC WITH THE ROOT Dockerfile: base image, requirements install,
# non-root user, embedding-model bake step, EXPOSE/HEALTHCHECK/CMD. The only
# addition is the two corpus-ingest RUN lines below.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${EMBEDDING_MODEL}')"

RUN useradd --create-home --uid 10001 policylens

COPY --chown=policylens:policylens src/ ./src/
COPY --chown=policylens:policylens frontend/ ./frontend/
COPY --chown=policylens:policylens evaluation/ ./evaluation/
COPY --chown=policylens:policylens scripts/ ./scripts/
COPY --chown=policylens:policylens pyproject.toml ./

RUN mkdir -p /app/data/chroma_db /app/data/raw \
    && chown -R policylens:policylens /app/data

RUN chown -R policylens:policylens /root 2>/dev/null || true
ENV HF_HOME=/home/policylens/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/policylens/.cache/torch/sentence_transformers
RUN mkdir -p ${HF_HOME} ${SENTENCE_TRANSFORMERS_HOME} \
    && cp -r /root/.cache/huggingface/. ${HF_HOME}/ 2>/dev/null || true \
    && cp -r /root/.cache/torch/sentence_transformers/. \
       ${SENTENCE_TRANSFORMERS_HOME}/ 2>/dev/null || true \
    && chown -R policylens:policylens /home/policylens/.cache

USER policylens

# -- the corpus, baked in -- see the file header for why.
# `|| true`: third-party hosts, so a host outage builds an app with an empty
# index (a working deploy with no documents) rather than no app at all.
RUN python scripts/download_sample_docs.py || true
RUN python scripts/ingest_docs.py || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
