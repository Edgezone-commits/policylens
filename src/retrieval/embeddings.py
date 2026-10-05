"""Sentence Transformers embedding wrapper.

One class, loaded once, used for both document indexing and query embedding.
WHY that matters: documents and queries MUST be embedded by the same model, or
their vectors live in different spaces and cosine similarity between them is
meaningless. Routing both through this one wrapper makes that mistake
impossible to make by accident.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src import config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Thin wrapper over a Sentence Transformers model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL
        # Downloads ~90MB on first use, then reads from the local HF cache.
        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        # sentence-transformers 5.x renamed this method. Support both so the
        # project does not break on either side of that version boundary.
        if hasattr(self._model, "get_embedding_dimension"):
            self.dimensions: int = self._model.get_embedding_dimension()
        else:
            self.dimensions = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model ready (%d dimensions)", self.dimensions)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts at once, for indexing.

        Batched in one call rather than looped per text: the model parallelises
        a batch across CPU cores, which is several times faster for the
        hundreds of chunks a long document produces.
        """
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            # L2-normalised vectors make the cosine distance Chroma computes
            # numerically stable, and keep similarity readable as 1 - distance.
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        Separate from embed_documents only for readability at call sites --
        all-MiniLM-L6-v2 is a symmetric model, so queries and documents are
        encoded identically. (Some models, e.g. the E5 family, require an
        asymmetric "query:" / "passage:" prefix; this one does not. If the
        model in config is ever swapped for such a model, this is the single
        place that would need the prefix added.)
        """
        vector = self._model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str | None = None) -> EmbeddingModel:
    """Return a process-wide singleton embedding model.

    WHY cache it: loading the model takes a second or two and ~100MB of RAM.
    The FastAPI app would otherwise reload it on every request, which would
    dominate response time and eventually exhaust memory.
    """
    return EmbeddingModel(model_name)
