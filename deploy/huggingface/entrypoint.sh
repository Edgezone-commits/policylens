#!/usr/bin/env bash
# Start the API and the Streamlit UI in one container, for a Hugging Face
# Space.
#
# The UI reaches the agent over HTTP and never imports it (see
# docs/ARCHITECTURE.md), so even in one container there are two processes. The
# API binds 127.0.0.1 only: it is reachable by the UI next to it and by
# nothing on the internet, because the public surface of a demo should be the
# demo, not an upload endpoint that anyone can post a PDF to.
set -euo pipefail

API_PORT=8000
UI_PORT="${PORT:-7860}"   # Spaces set PORT; 7860 is the Docker SDK default.

if [ -z "${GOOGLE_API_KEY:-}" ] && [ -z "${GROQ_API_KEY:-}" ]; then
  # Not fatal. The app starts, /health reports llm_key_present=false and the
  # UI says so, which is a far better failure than a container that exits
  # before anyone can read why.
  echo "WARNING: no GOOGLE_API_KEY or GROQ_API_KEY in the environment."
  echo "Add one under Settings -> Variables and secrets, then restart."
fi

# The index is built at image build time. If the document hosts were down
# then, try once more now -- a Space restarted a day later may well succeed.
# Backgrounded so it cannot delay the port binding: a Space that has not
# bound its port is reported as failed to start.
python - <<'PY' &
import logging

logging.basicConfig(level=logging.INFO)
try:
    from src.retrieval.store import VectorStore

    store = VectorStore()
    if store.count() == 0:
        logging.info("Index empty; retrying the corpus download and ingest.")
        import subprocess
        import sys

        subprocess.run([sys.executable, "scripts/download_sample_docs.py"], check=False)
        subprocess.run([sys.executable, "scripts/ingest_docs.py"], check=False)
    else:
        logging.info("Index ready: %d chunks.", store.count())
except Exception as error:  # noqa: BLE001 -- must never stop the app starting
    logging.warning("Index check failed, continuing: %s", error)
PY

# Both in the background with a trap, so that if either dies the container
# exits instead of serving a UI whose API is gone -- which presents as a
# working demo that errors on every question.
trap 'kill 0' EXIT INT TERM

uvicorn src.api.app:app --host 127.0.0.1 --port "${API_PORT}" &

streamlit run frontend/app.py \
  --server.port="${UI_PORT}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false &

wait -n
exit $?
