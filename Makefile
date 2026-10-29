# PolicyLens -- the commands this project is actually driven by.
#
#   make            list the targets
#   make install    create .venv and install the pinned dependencies
#   make ingest     build the vector index from data/raw/
#   make run        API on :8000 and the Streamlit UI on :8501
#   make test       the whole suite, no API key needed
#   make eval       the evaluation that costs no quota
#   make docker     build and start both services in containers
#
# WHY a Makefile for a Python project: every command here already existed in
# the README as a line to copy. The difference is that `make test` cannot
# drift from what CI runs the way a README line can, because both read this
# file.
#
# ON WINDOWS: make is not installed by default. Either run the command under
# each target by hand (they are plain one-liners, deliberately), or install
# make with `winget install ezwinports.make` and run it from Git Bash. The
# variables below use forward slashes, which Git Bash and PowerShell both
# accept.

# Overridable so a system Python, a conda env or CI can be used instead:
#   make test PYTHON=python3.12
PYTHON ?= python
VENV ?= .venv

# Extra arguments for `make test`, so CI can ask for timings without the
# Makefile and the workflow drifting into two different pytest invocations:
#   make test PYTEST_ARGS=--durations=10
PYTEST_ARGS ?=

# The venv's interpreter differs by platform: Scripts/ on Windows, bin/
# elsewhere. Resolved here once rather than in each target.
ifeq ($(OS),Windows_NT)
	VENV_PYTHON := $(VENV)/Scripts/python.exe
else
	VENV_PYTHON := $(VENV)/bin/python
endif

# Every target is a command, not a file to build, so all of them are phony.
# Without this, a target would be skipped whenever a directory of the same
# name happened to exist -- `make docs` being the classic way to find out.
.PHONY: help install install-dev ingest run api ui test lint format \
        eval eval-retrieval eval-perf eval-full check-models \
        docker docker-down docker-ingest clean

help:
	@echo "PolicyLens"
	@echo ""
	@echo "  make install        create $(VENV) and install pinned dependencies"
	@echo "  make install-dev    the above plus ruff, black and pre-commit"
	@echo "  make check-models   report which LLMs your key can reach right now"
	@echo "  make ingest         build the index from data/raw/"
	@echo ""
	@echo "  make run            API (:8000) and UI (:8501) together"
	@echo "  make api            API only"
	@echo "  make ui             Streamlit UI only"
	@echo ""
	@echo "  make test           193 tests, no API key required (~25s)"
	@echo "  make lint           ruff and black, in check mode"
	@echo "  make format         ruff --fix and black, writing changes"
	@echo ""
	@echo "  make eval           retrieval and performance metrics (no LLM, free)"
	@echo "  make eval-full      the 53-question end-to-end run (SPENDS QUOTA)"
	@echo ""
	@echo "  make docker         build and start both services"
	@echo "  make docker-ingest  ingest inside the running container"
	@echo "  make docker-down    stop them"

# -- setup -------------------------------------------------------------------

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo ""
	@echo "Next: copy .env.example to .env and put a real GOOGLE_API_KEY in it,"
	@echo "then run 'make ingest'."

install-dev:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt
	$(VENV_PYTHON) -m pre_commit install
	@echo ""
	@echo "pre-commit installed: ruff, black and the file hygiene hooks now"
	@echo "run on every commit."

check-models:
	$(VENV_PYTHON) scripts/check_models.py

ingest:
	$(VENV_PYTHON) scripts/ingest_docs.py

# -- running -----------------------------------------------------------------

api:
	$(VENV_PYTHON) -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

ui:
	$(VENV_PYTHON) -m streamlit run frontend/app.py

# Two processes from one target, so `make run` is a single command for a
# stranger following the README. The API goes to the background and the UI
# holds the terminal, because the UI is the thing to look at and Ctrl-C on it
# should end the session -- hence the trap, which kills the API too rather
# than leaving port 8000 held by an orphan.
run:
	@echo "API on http://127.0.0.1:8000 (docs at /docs), UI on http://localhost:8501"
	@echo "Ctrl-C stops both."
	@trap 'kill 0' EXIT INT TERM; \
	$(VENV_PYTHON) -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 & \
	$(VENV_PYTHON) -m streamlit run frontend/app.py

# -- quality -----------------------------------------------------------------

# VENV_PYTHON is overridable on the command line (make variables given there
# beat the assignment above), which is how CI runs these targets against its
# system Python: `make test VENV_PYTHON=python`. CI running the Makefile
# rather than its own copy of the pytest line is what keeps the two in step.
test:
	$(VENV_PYTHON) -m pytest -m "not llm" -q $(PYTEST_ARGS)

lint:
	$(VENV_PYTHON) -m ruff check src/ tests/ scripts/ evaluation/ frontend/
	$(VENV_PYTHON) -m black --check src/ tests/ scripts/ evaluation/ frontend/

format:
	$(VENV_PYTHON) -m ruff check --fix src/ tests/ scripts/ evaluation/ frontend/
	$(VENV_PYTHON) -m black src/ tests/ scripts/ evaluation/ frontend/

# -- evaluation --------------------------------------------------------------
#
# `make eval` runs only what is FREE, on purpose. The end-to-end run costs
# 2-5 LLM calls per question against a quota of about twenty requests per
# model per day, so a target that spent it silently would be a trap for
# anyone typing `make <tab>`. The expensive one is named eval-full and says
# so.

eval: eval-retrieval eval-perf

eval-retrieval:
	$(VENV_PYTHON) scripts/eval_retrieval.py

eval-perf:
	$(VENV_PYTHON) scripts/eval_perf.py

eval-full:
	@echo "This calls a real LLM 2-5 times per question, for 53 questions."
	@echo "Use 'python scripts/evaluate.py --limit 6' to stay inside a daily quota."
	$(VENV_PYTHON) scripts/evaluate.py

# -- containers --------------------------------------------------------------

docker:
	docker compose up --build

docker-ingest:
	docker compose run --rm api python scripts/ingest_docs.py

docker-down:
	docker compose down

# -- housekeeping ------------------------------------------------------------
#
# Does NOT touch data/: the index takes real time to rebuild and the raw
# documents are not redownloadable in one command. Caches only.
clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "Caches removed. data/ and $(VENV) left alone."
