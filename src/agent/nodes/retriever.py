"""Retriever node: gather the evidence, or establish that there is none.

This node is where the agent's honesty comes from. VectorStore.retrieve
applies the calibrated threshold and returns an empty list when even the best
chunk is too far away; this node turns that emptiness into a decision the rest
of the graph can act on, rather than passing weak chunks to the analyzer and
hoping it notices.

Two behaviours are worth reading the code for:

  * A COMPARISON retrieves per document. A single pooled top-5 over two
    documents is routinely lopsided -- the Act is verbose about leave and the
    Rules terse, so all five slots fill from the Act and the "comparison"
    silently becomes a one-sided summary.
  * A retry widens the net AND re-runs the user's original wording alongside
    the router's rewrite. WHY both: if the rewrite was the problem (a term the
    documents do not actually use), repeating it with a larger top_k just
    returns more of the same neighbourhood.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from src import config
from src.agent.state import AgentState, RetrievedChunk, trace_entry
from src.retrieval.store import VectorStore

logger = logging.getLogger(__name__)


def _merge(
    groups: list[list[RetrievedChunk]], limit: int | None = None
) -> list[RetrievedChunk]:
    """Combine result lists, dropping duplicates, best similarity first.

    Chunks repeat across groups whenever two queries hit the same passage,
    which is common on a retry. Deduplicating by chunk_id matters beyond
    tidiness: a duplicated chunk would occupy two evidence slots and let the
    analyzer cite the same text twice as if it were two sources.
    """
    seen: dict[str, RetrievedChunk] = {}
    for group in groups:
        for chunk in group:
            chunk_id = chunk.get("chunk_id") or ""
            existing = seen.get(chunk_id)
            if existing is None or chunk["similarity"] > existing["similarity"]:
                seen[chunk_id] = chunk
    merged = sorted(seen.values(), key=lambda chunk: chunk["similarity"], reverse=True)
    return merged[:limit] if limit else merged


def build_retriever_node(store: VectorStore) -> Callable[[AgentState], dict]:
    """Create the retriever node bound to a vector store."""

    def retriever_node(state: AgentState) -> dict:
        retry_count = state.get("retry_count", 0)
        is_retry = retry_count > 0

        top_k = (
            config.RETRIEVAL_TOP_K_ON_RETRY if is_retry else config.RETRIEVAL_TOP_K
        )
        query = state.get("search_query") or state["question"]
        queries = [query]
        if is_retry and state["question"] not in queries:
            queries.append(state["question"])

        targets = list(state.get("target_documents") or [])
        if state.get("source_document") and state["source_document"] not in targets:
            targets = [state["source_document"]]

        groups: list[list[RetrievedChunk]] = []
        if state.get("intent") == "COMPARISON" and len(targets) > 1:
            # Per document, so each side of the comparison is represented.
            for document in targets:
                for one_query in queries:
                    groups.append(
                        store.retrieve(one_query, top_k=top_k, source_document=document)
                    )
            # Keep every document's share: a pooled cap here would undo the
            # whole point of searching them separately.
            limit = top_k * len(targets)
        else:
            # One target means a metadata filter; none means search everything.
            document = targets[0] if len(targets) == 1 else None
            for one_query in queries:
                groups.append(
                    store.retrieve(one_query, top_k=top_k, source_document=document)
                )
            limit = top_k

        chunks = _merge(groups, limit=limit)

        if not chunks:
            # Every query fell below the threshold. This is the abstention
            # path: no evidence, and the responder will say exactly that.
            logger.info(
                "No chunks above the relevance threshold for %r", query[:80]
            )
            return {
                "chunks": [],
                "top_similarity": 0.0,
                "retrieval_top_k": top_k,
                "trace": [
                    trace_entry(
                        "retriever",
                        "No passage cleared the relevance threshold "
                        f"({config.RETRIEVAL_SIMILARITY_THRESHOLD:.2f})",
                        queries=queries,
                        top_k=top_k,
                        documents=targets or ["(all)"],
                        attempt=retry_count + 1,
                    )
                ],
            }

        top_similarity = chunks[0]["similarity"]
        sources = sorted({chunk["source_document"] for chunk in chunks})
        logger.info(
            "Retrieved %d chunk(s), best similarity %.3f, from %s",
            len(chunks),
            top_similarity,
            ", ".join(sources),
        )

        return {
            "chunks": chunks,
            "top_similarity": top_similarity,
            "retrieval_top_k": top_k,
            "trace": [
                trace_entry(
                    "retriever",
                    f"Retrieved {len(chunks)} passage(s) from {', '.join(sources)} "
                    f"(best similarity {top_similarity:.2f})",
                    queries=queries,
                    top_k=top_k,
                    attempt=retry_count + 1,
                    passages=[
                        {
                            "chunk_id": chunk["chunk_id"],
                            "source_document": chunk["source_document"],
                            "page_number": chunk.get("page_number"),
                            "section": chunk.get("section"),
                            "similarity": chunk["similarity"],
                        }
                        for chunk in chunks
                    ],
                )
            ],
        }

    return retriever_node
