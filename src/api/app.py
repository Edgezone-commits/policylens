"""FastAPI application.

The API is a thin shell over PolicyLensAgent. It owns three things the agent
should not: when the expensive objects are built, how an upload becomes an
indexed document, and the HTTP status codes.

Two decisions worth stating:

  * The agent is built ONCE, at startup, and shared. The embedding model is
    ~100MB and takes a second or two to load, so building it per request would
    dominate every response and eventually exhaust memory. This is why
    create_app() takes an optional agent: the tests inject a fake and never
    load a model at all.

  * An abstention is HTTP 200. It is a successful, correct response -- the
    system worked out that the documents do not support an answer and said so.
    Returning 404 or 422 for it would tell every client that the request was
    malformed, and would make the project's most important behaviour look like
    an error in every log and dashboard. Only a pipeline failure (status ERROR)
    is a 5xx.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import litellm
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.agent.graph import PolicyLensAgent
from src.api.schemas import (
    AnswerResponse,
    AskRequest,
    DeleteResponse,
    DocumentModel,
    HealthResponse,
    IngestResponse,
    QueryLogEntry,
    StatsResponse,
)
from src.db.queries import QueryLog
from src.ingestion.indexer import SUPPORTED_EXTENSIONS, ingest_file

logger = logging.getLogger(__name__)

# Uploads are capped well below anything that would trouble the ingester; the
# limit exists so a stray multi-gigabyte file cannot fill the disk while being
# streamed. The Nepal Labour Act, 82 pages, is about 1MB.
MAX_UPLOAD_BYTES = 32 * 1024 * 1024

_DESCRIPTION = """
Agentic question answering over legal and policy documents, where every claim
carries a quote that has been checked against the passage it cites.

The status field is the thing to read first:

* **ANSWERED** -- answered, and every citation was confirmed.
* **PARTIALLY_VERIFIED** -- answered, but a citation could not be confirmed.
  It is still shown, labelled, with a note saying why.
* **INSUFFICIENT_EVIDENCE** -- the documents do not support an answer. This is
  a correct outcome and returns 200, not an error.
* **OUT_OF_SCOPE** -- the question is not about the indexed documents.
* **ERROR** -- the pipeline failed (no API key, empty index).
"""


def create_app(
    agent: PolicyLensAgent | None = None, query_log: QueryLog | None = None
) -> FastAPI:
    """Build the application.

    Both dependencies are injectable so the tests can run every endpoint
    against a fake agent and a temporary database -- no model download, no API
    key, no network. An API whose tests need a live LLM does not get tested.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Built here rather than at import time so that importing this module
        # (which the tests and any tooling do) does not load a 100MB model.
        app.state.agent = agent or PolicyLensAgent()
        app.state.query_log = query_log or QueryLog()
        logger.info(
            "API ready: %d chunks indexed, model %s",
            app.state.agent.store.count(),
            config.LLM_MODEL,
        )
        yield

    app = FastAPI(
        title="PolicyLens",
        description=_DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
    )

    # The Streamlit UI is served from a different port, which makes every
    # browser request to this API cross-origin. Permissive because this is a
    # local, single-user tool; a deployed instance would name its origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_agent(request: Request) -> PolicyLensAgent:
        return request.app.state.agent

    def get_log(request: Request) -> QueryLog:
        return request.app.state.query_log

    # -- answering ----------------------------------------------------------

    @app.post(
        "/ask",
        response_model=AnswerResponse,
        summary="Answer a question from the indexed documents",
        tags=["answering"],
    )
    def ask(payload: AskRequest, request: Request) -> AnswerResponse:
        """Put a question to the agent.

        Returns 200 for every outcome the agent can reason its way to,
        including an abstention. 503 only when the pipeline itself could not
        run -- a missing key, an empty index -- because that is the one case
        where retrying later might work and the client did nothing wrong.
        """
        agent = get_agent(request)

        if payload.document and payload.document not in agent.store.document_names():
            # 404 rather than 422: the request was well formed, the named
            # document simply is not here. The message lists what is, because
            # the usual cause is a filename typo.
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Not an indexed document: {payload.document}. "
                    f"Indexed: {', '.join(agent.store.document_names()) or 'none'}"
                ),
            )

        started = time.monotonic()
        answer = agent.ask(payload.question, source_document=payload.document)
        latency = time.monotonic() - started

        query_id = get_log(request).record(answer, latency_seconds=latency)

        if answer.status == "ERROR":
            # Logged first, deliberately: a failed run is exactly what someone
            # will want to look at afterwards.
            raise HTTPException(status_code=503, detail=answer.error or "Agent failed.")

        return AnswerResponse(
            **answer.as_dict(),
            latency_seconds=round(latency, 2),
            query_id=query_id,
        )

    # -- documents ----------------------------------------------------------

    @app.get(
        "/documents",
        response_model=list[DocumentModel],
        summary="List the indexed documents",
        tags=["documents"],
    )
    def list_documents(request: Request) -> list[DocumentModel]:
        return [
            DocumentModel(**document)
            for document in get_agent(request).store.list_documents()
        ]

    @app.post(
        "/documents",
        response_model=IngestResponse,
        summary="Upload and index a PDF or text document",
        tags=["documents"],
    )
    def upload_document(
        request: Request, file: UploadFile = File(...)
    ) -> IngestResponse:
        """Index an uploaded document.

        The upload is streamed to a temporary file and ingested from there,
        then kept in data/raw/ only if it indexed successfully. WHY not read it
        into memory: an 80-page PDF is fine, but the size of an upload is the
        client's choice, not ours. WHY keep it at all: re-indexing after a
        chunking change has to be possible without asking the user to find the
        file again.
        """
        filename = Path(file.filename or "").name  # strip any directory part
        if not filename:
            raise HTTPException(status_code=422, detail="The upload had no filename.")

        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type '{suffix}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                ),
            )

        agent = get_agent(request)
        temporary_directory = Path(tempfile.mkdtemp(prefix="policylens-upload-"))
        temporary_path = temporary_directory / filename

        try:
            written = 0
            with temporary_path.open("wb") as destination:
                while chunk := file.file.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"File is larger than "
                                f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB."
                            ),
                        )
                    destination.write(chunk)

            if written == 0:
                raise HTTPException(status_code=422, detail="The uploaded file is empty.")

            result = ingest_file(temporary_path, agent.store)

            if result.succeeded:
                # Only now is it worth keeping. A scanned PDF that produced no
                # text would otherwise sit in data/raw/ looking indexed.
                config.ensure_directories()
                shutil.copy2(temporary_path, config.RAW_DOCS_DIR / filename)

            return IngestResponse(
                source_document=result.source_document,
                pages=result.pages,
                chunks=result.chunks,
                succeeded=result.succeeded,
                skipped_reason=result.skipped_reason,
                chunks_indexed=agent.store.count(),
            )
        finally:
            file.file.close()
            shutil.rmtree(temporary_directory, ignore_errors=True)

    @app.delete(
        "/documents/{source_document}",
        response_model=DeleteResponse,
        summary="Remove a document from the index",
        tags=["documents"],
    )
    def delete_document(source_document: str, request: Request) -> DeleteResponse:
        agent = get_agent(request)
        if source_document not in agent.store.document_names():
            raise HTTPException(
                status_code=404, detail=f"Not an indexed document: {source_document}"
            )
        removed = agent.store.delete_document(source_document)
        return DeleteResponse(source_document=source_document, chunks_removed=removed)

    # -- introspection ------------------------------------------------------

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Whether the system can answer a question right now",
        tags=["introspection"],
    )
    def health(request: Request) -> HealthResponse:
        agent = get_agent(request)
        documents = agent.store.list_documents()
        chunks = agent.store.count()

        try:
            report = litellm.validate_environment(model=config.LLM_MODEL)
            key_present = bool(report.get("keys_in_environment", False))
        except Exception:  # noqa: BLE001 -- an unknown model string
            key_present = False

        try:
            queries_logged = get_log(request).stats()["total_queries"]
        except Exception:  # noqa: BLE001 -- health must not fail on the log
            queries_logged = 0

        return HealthResponse(
            # Both conditions make a question fail, so neither is "ok".
            status="ok" if (chunks > 0 and key_present) else "degraded",
            chunks_indexed=chunks,
            documents=[DocumentModel(**document) for document in documents],
            model=config.LLM_MODEL,
            fallback_models=list(config.LLM_FALLBACK_MODELS),
            llm_key_present=key_present,
            embedding_model=config.EMBEDDING_MODEL,
            similarity_threshold=config.RETRIEVAL_SIMILARITY_THRESHOLD,
            queries_logged=queries_logged,
        )

    @app.get(
        "/queries",
        response_model=list[QueryLogEntry],
        summary="Recent questions and what the agent answered",
        tags=["introspection"],
    )
    def recent_queries(request: Request, limit: int = 20) -> list[QueryLogEntry]:
        rows = get_log(request).recent(limit=min(max(1, limit), 200))
        return [QueryLogEntry(**row) for row in rows]

    @app.get(
        "/stats",
        response_model=StatsResponse,
        summary="Aggregates over the query log, including the abstention rate",
        tags=["introspection"],
    )
    def stats(request: Request) -> StatsResponse:
        return StatsResponse(**get_log(request).stats())

    return app


def _build_default_app() -> FastAPI:
    """Entry point for `uvicorn src.api.app:app`."""
    config.configure_logging()
    config.ensure_directories()
    return create_app()


# Module-level so uvicorn can find it. The expensive objects are still built in
# the lifespan handler, not here, so importing this module stays cheap.
app: Any = _build_default_app()
