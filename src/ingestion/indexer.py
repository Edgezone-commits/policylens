"""Orchestrates ingestion: load -> clean -> chunk -> embed -> store.

This module is deliberately thin. It owns the *order* of the pipeline and
nothing else; each step's logic lives in its own module. Keeping orchestration
separate from the steps is what makes it possible to test chunking without a
vector store, or retrieval without a PDF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_pdf, load_text_file
from src.retrieval.store import VectorStore

logger = logging.getLogger(__name__)

# Only these extensions are attempted. .txt is included because the Apache
# License 2.0 -- one of our evaluation documents -- ships as plain text.
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


@dataclass
class IngestionResult:
    """What happened when we ingested one document."""

    source_document: str
    pages: int
    chunks: int
    skipped_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.skipped_reason is None and self.chunks > 0


def ingest_file(file_path: str | Path, store: VectorStore) -> IngestionResult:
    """Run the full pipeline for a single file.

    Returns a result object rather than raising on an unusable document,
    because ingesting a folder must not abort on one bad file -- a single
    scanned PDF should not stop the other twenty from indexing.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        logger.warning("Skipping unsupported file type: %s", file_path.name)
        return IngestionResult(file_path.name, 0, 0, f"unsupported type '{suffix}'")

    try:
        pages = load_pdf(file_path) if suffix == ".pdf" else load_text_file(file_path)
    except Exception as exc:  # noqa: BLE001 - a corrupt PDF must not kill the run
        logger.error("Failed to load %s: %s", file_path.name, exc)
        return IngestionResult(file_path.name, 0, 0, f"load error: {exc}")

    if not pages:
        # Almost always a scanned PDF with no text layer.
        return IngestionResult(
            file_path.name, 0, 0, "no extractable text (scanned PDF? needs OCR)"
        )

    chunks = chunk_documents(pages)
    if not chunks:
        return IngestionResult(
            file_path.name, len(pages), 0, "chunking produced nothing"
        )

    stored = store.add_chunks(chunks)
    logger.info(
        "Ingested %s: %d pages -> %d chunks", file_path.name, len(pages), stored
    )
    return IngestionResult(file_path.name, len(pages), stored)


def ingest_directory(
    directory: str | Path, store: VectorStore
) -> list[IngestionResult]:
    """Ingest every supported document in a folder (non-recursive).

    Non-recursive on purpose: data/raw/ is flat, and recursing would risk
    sweeping up unrelated PDFs from nested folders without the user noticing.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not candidates:
        logger.warning(
            "No .pdf or .txt files in %s -- try scripts/download_sample_docs.py",
            directory,
        )
        return []

    logger.info("Found %d document(s) to ingest in %s", len(candidates), directory)
    return [ingest_file(path, store) for path in candidates]
