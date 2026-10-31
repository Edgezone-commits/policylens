# Hugging Face Space (Docker SDK) -- both processes in one container.
#
# WHY THIS FILE EXISTS ALONGSIDE THE ROOT Dockerfile, which is the thing a
# reviewer will ask about. A Docker Space builds the Dockerfile at the root of
# the SPACE repo and runs its CMD; it cannot be told to use a different
# command, and the root Dockerfile's CMD is the API alone, because
# docker-compose.yml runs that same image twice with two different commands.
# A Space has one container and needs both processes, so it needs its own CMD,
# so it needs its own Dockerfile. There is no way to express that as a flag.
#
# KEEP IN SYNC WITH THE ROOT Dockerfile: the base image, the requirements
# install, and the embedding-model bake step. The CI job `docker-space` builds
# this file on every push for exactly that reason -- if the two drift into
# incompatibility, the build goes red here rather than at deploy time, which
# is the only honest way to maintain a second Dockerfile.
#
# The deployment steps that go with it are in docs/deploy.md.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face runs the container as uid 1000 and the home directory must be
# writable, because the embedding model's cache, the Chroma index and the
# SQLite log all live under it. Created before pip runs so the model bake
# below writes into the right place first time.
RUN useradd --create-home --uid 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# The app directory is created AND CHOWNED here, before WORKDIR names it.
# WORKDIR creates a missing directory as root, so leaving it to do that gives
# a root-owned directory that the `user` below cannot write to -- and the
# failure surfaces later and somewhere else, as `mkdir: cannot create
# directory 'data': Permission denied` on a line that looks unrelated. The CI
# job for this file caught exactly that.
RUN mkdir -p /home/user/app && chown -R user:user /home/user
WORKDIR /home/user/app

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user:user requirements.txt ./
RUN pip install -r requirements.txt

USER user

# Baked in rather than downloaded on first question: a Space that fetches
# ~90MB from the Hub on its first query looks broken for the thirty seconds
# it takes, to the one person whose opinion the demo exists to change.
ENV HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/torch/sentence_transformers
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user:user src/ ./src/
COPY --chown=user:user frontend/ ./frontend/
COPY --chown=user:user evaluation/ ./evaluation/
COPY --chown=user:user scripts/ ./scripts/
COPY --chown=user:user deploy/huggingface/entrypoint.sh ./entrypoint.sh
COPY --chown=user:user pyproject.toml ./

RUN mkdir -p data/chroma_db data/raw && chmod +x entrypoint.sh

# -- the corpus -------------------------------------------------------------
#
# Downloaded and indexed AT BUILD TIME, not at startup. Two reasons:
#   * Startup. Embedding 279 chunks takes a minute or two, and a Space that
#     spends that before binding its port is a Space that looks dead.
#   * The documents are gitignored (large, and license-restricted in one
#     case), so they are not in the repo to copy -- they have to be fetched.
#
# `|| true` because these are third-party hosts. If a host is down the image
# still builds and the UI says the index is empty, which is a working app with
# no documents; a failed build would be no app at all. The entrypoint retries
# once at startup for the same reason.
RUN python scripts/download_sample_docs.py || true
RUN python scripts/ingest_docs.py || true

# Must match `app_port` in the Space's README front matter.
EXPOSE 7860

CMD ["./entrypoint.sh"]
