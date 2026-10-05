"""ChromaDB wrapper: the project's only vector-store interface.

Everything that touches Chroma goes through this class, so the choice of
vector database stays swappable and the rest of the codebase deals in plain
dicts instead of Chroma's column-oriented response format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from src import config
from src.retrieval.embeddings import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)

# Chroma rejects writes above a server-side limit (a few thousand records).
# Long documents can exceed it, so adds are batched.
_MAX_ADD_BATCH = 2000


class VectorStore:
    """Persistent ChromaDB collection holding every indexed document chunk."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir or config.CHROMA_DIR)
        self.collection_name = collection_name or config.CHROMA_COLLECTION
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # PersistentClient writes to disk, so the index survives a restart.
        # WHY this over Chroma's in-memory client or FAISS: ingesting an
        # 82-page Act takes real time, and nobody should pay it twice. Telemetry
        # is off because this is a local, offline-capable tool.
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        self._collection = self._get_or_create_collection()
        self.embeddings = embedding_model or get_embedding_model()
        logger.info(
            "Vector store ready: collection=%s, chunks=%d, path=%s",
            self.collection_name,
            self._collection.count(),
            self.persist_dir,
        )

    def _get_or_create_collection(self) -> Any:
        """Get the collection, forcing cosine distance.

        WHY the explicit metric: Chroma defaults to squared L2. Our relevance
        threshold is defined as a cosine similarity, so leaving the default
        would mean comparing a threshold against a quantity on a different
        scale -- the filter would still "work", but would filter on nonsense.
        Because the metric is fixed when a collection is created, this is not
        something that can be corrected later without re-indexing.

        embedding_function=None: we always pass vectors in ourselves from our
        Sentence Transformers model. Left unset, Chroma would quietly download
        and use its own default embedding model, and documents and queries
        could end up embedded by two different models.
        """
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": config.CHROMA_DISTANCE_METRIC},
            embedding_function=None,
        )

    # -- writing ------------------------------------------------------------

    def add_chunks(self, chunks: list[Document]) -> int:
        """Embed and store chunks. Returns the number stored.

        Uses upsert keyed on the deterministic chunk_id, so re-ingesting a
        document replaces its chunks rather than duplicating them. With plain
        `add`, running the ingest script twice would double every chunk and
        quietly corrupt retrieval (and the evaluation built on it).
        """
        if not chunks:
            logger.warning("add_chunks called with no chunks")
            return 0

        texts = [chunk.page_content for chunk in chunks]
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        metadatas = [dict(chunk.metadata) for chunk in chunks]

        logger.info("Embedding %d chunks...", len(texts))
        vectors = self.embeddings.embed_documents(texts)

        for start in range(0, len(ids), _MAX_ADD_BATCH):
            stop = start + _MAX_ADD_BATCH
            self._collection.upsert(
                ids=ids[start:stop],
                embeddings=vectors[start:stop],
                documents=texts[start:stop],
                metadatas=metadatas[start:stop],
            )

        logger.info(
            "Stored %d chunks; collection now holds %d",
            len(ids),
            self._collection.count(),
        )
        return len(ids)

    # -- reading ------------------------------------------------------------

    def query_by_embedding(
        self,
        embedding: list[float],
        top_k: int | None = None,
        source_document: str | None = None,
    ) -> list[dict[str, Any]]:
        """Nearest-neighbour search for an already-embedded query.

        `source_document` restricts the search to one document via metadata
        filtering -- this is how a single-document question is scoped without
        needing a separate collection per document.

        Returns a list of plain dicts, highest similarity first:
            {chunk_id, text, source_document, page_number, section, similarity}

        Chroma returns a *distance*; with cosine space, similarity is
        `1 - distance`. We convert here, once, so no caller has to remember
        which direction is better.
        """
        top_k = top_k or config.RETRIEVAL_TOP_K

        if self._collection.count() == 0:
            logger.warning("Query against an empty collection -- ingest documents first")
            return []

        where = {"source_document": source_document} if source_document else None

        response = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma nests results one level per query embedding; we sent one.
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results: list[dict[str, Any]] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            results.append(
                {
                    "chunk_id": metadata.get("chunk_id", ""),
                    "text": text,
                    "source_document": metadata.get("source_document", "unknown"),
                    "page_number": metadata.get("page_number"),
                    "section": metadata.get("section"),
                    # Comma-joined list of every section this chunk spans; the
                    # evaluation harness membership-tests against it.
                    "sections": metadata.get("sections", ""),
                    "similarity": round(1.0 - float(distance), 4),
                }
            )
        return results

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Fetch one chunk by id.

        The verifier needs this: a citation names the chunk it came from, and
        verification means re-reading that exact chunk's text rather than
        trusting what the analyzer echoed back.
        """
        response = self._collection.get(
            ids=[chunk_id], include=["documents", "metadatas"]
        )
        documents = response.get("documents") or []
        if not documents:
            return None
        metadata = (response.get("metadatas") or [{}])[0] or {}
        return {
            "chunk_id": chunk_id,
            "text": documents[0],
            "source_document": metadata.get("source_document", "unknown"),
            "page_number": metadata.get("page_number"),
            "section": metadata.get("section"),
            "sections": metadata.get("sections", ""),
        }

    # -- inspection ---------------------------------------------------------

    def list_documents(self) -> list[dict[str, Any]]:
        """List the indexed documents with their chunk counts.

        Used by /health, the Streamlit sidebar, and the router -- the router
        has to be told which documents exist before it can decide whether a
        question is answerable from them at all.
        """
        response = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in response.get("metadatas") or []:
            name = (metadata or {}).get("source_document", "unknown")
            counts[name] = counts.get(name, 0) + 1
        return [
            {"source_document": name, "chunk_count": count}
            for name, count in sorted(counts.items())
        ]

    def document_names(self) -> list[str]:
        """Just the indexed document names."""
        return [doc["source_document"] for doc in self.list_documents()]

    def count(self) -> int:
        """Total number of stored chunks."""
        return self._collection.count()

    # -- maintenance --------------------------------------------------------

    def delete_document(self, source_document: str) -> int:
        """Remove every chunk belonging to one document."""
        before = self._collection.count()
        self._collection.delete(where={"source_document": source_document})
        removed = before - self._collection.count()
        logger.info("Deleted %d chunks for %s", removed, source_document)
        return removed

    def reset(self) -> None:
        """Drop and recreate the collection.

        Used by tests and by re-indexing from scratch after a chunking change.
        """
        self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create_collection()
        logger.info("Reset collection %s", self.collection_name)
