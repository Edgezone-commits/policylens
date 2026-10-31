"""CLI to index documents into ChromaDB.

    python scripts/ingest_docs.py data/raw/            # a whole folder
    python scripts/ingest_docs.py data/raw/one.pdf     # a single file
    python scripts/ingest_docs.py data/raw/ --reset    # wipe the index first

--reset exists because chunking parameters are an experiment: after changing
chunk_size you must re-index from scratch, or the collection ends up holding a
mix of old and new chunk sizes and your retrieval results mean nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.ingestion.indexer import ingest_directory, ingest_file
from src.retrieval.store import VectorStore

logger = logging.getLogger("ingest_docs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index PDFs or text files into the PolicyLens vector store."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(config.RAW_DOCS_DIR),
        help="File or directory to ingest (default: data/raw/)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all existing chunks before ingesting (needed after a "
        "chunk_size change, so the index is not a mix of two settings)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.configure_logging("DEBUG" if args.verbose else None)
    config.ensure_directories()

    target = Path(args.path)
    if not target.exists():
        logger.error("Path does not exist: %s", target)
        logger.error("Run: python scripts/download_sample_docs.py")
        return 1

    store = VectorStore()

    if args.reset:
        logger.warning("--reset: deleting all %d existing chunks", store.count())
        store.reset()

    results = (
        ingest_directory(target, store)
        if target.is_dir()
        else [ingest_file(target, store)]
    )

    if not results:
        logger.error("Nothing to ingest in %s", target)
        return 1

    print("\n" + "=" * 72)
    print("INGESTION SUMMARY")
    print("=" * 72)
    print(f"{'Document':<42}{'Pages':>8}{'Chunks':>9}")
    print("-" * 72)
    for result in results:
        if result.succeeded:
            print(
                f"{result.source_document[:41]:<42}{result.pages:>8}{result.chunks:>9}"
            )
        else:
            print(f"{result.source_document[:41]:<42}{'--':>8}{'--':>9}")
            print(f"  SKIPPED: {result.skipped_reason}")
    print("-" * 72)

    total_chunks = sum(result.chunks for result in results)
    failures = [result for result in results if not result.succeeded]
    print(f"{'TOTAL':<42}{'':>8}{total_chunks:>9}")
    print(
        f"\nCollection now holds {store.count()} chunks across "
        f"{len(store.list_documents())} document(s)."
    )

    if failures:
        print(f"\n{len(failures)} document(s) could not be indexed (see above).")

    print("\nIndexed documents:")
    for doc in store.list_documents():
        print(f"  - {doc['source_document']} ({doc['chunk_count']} chunks)")
    print()

    # Non-zero only if nothing at all was indexed -- a partial success is still
    # a usable index, and CI should not treat one bad PDF as a build failure.
    return 0 if total_chunks > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
