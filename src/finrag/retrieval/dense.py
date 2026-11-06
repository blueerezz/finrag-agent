"""Dense retrieval over the Chroma chunk index."""
from __future__ import annotations

from typing import Any

from finrag.embeddings import embed_texts
from finrag.ingestion.index import load_collection


def _normalize_where(where: dict | None) -> dict | None:
    """Chroma requires a single operator at the top level.

    Accepts the friendly form {"ticker": "NVDA", "section_code": "8"}
    and rewrites it to {"$and": [{"ticker": "NVDA"}, {"section_code": "8"}]}.
    Single-key dicts are passed through unchanged. Dicts that already
    contain a "$" key are assumed to be valid Chroma filters.
    """
    if not where:
        return None
    if any(k.startswith("$") for k in where):
        return where
    if len(where) == 1:
        return where
    return {"$and": [{k: v} for k, v in where.items()]}


def dense_search(
    query: str,
    k: int = 20,
    where: dict | None = None,
    collection=None,
) -> list[dict[str, Any]]:
    """Embed query, retrieve top-k by cosine similarity.

    Filters use the friendly form:
        {"ticker": "NVDA"}                      # single
        {"ticker": "NVDA", "section_code": "8"} # multiple (auto-$and'd)
        {"section_code": {"$in": ["7", "7A"]}}  # already a Chroma op
    """
    coll = collection or load_collection()
    [qvec] = embed_texts([query])

    result = coll.query(
        query_embeddings=[qvec],
        n_results=k,
        where=_normalize_where(where),
        include=["metadatas", "documents", "distances"],
    )

    ids = result["ids"][0]
    metas = result["metadatas"][0]
    docs = result["documents"][0]
    dists = result["distances"][0]

    return [
        {
            "chunk_id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "distance": dists[i],
            "score": 1.0 - dists[i],
        }
        for i in range(len(ids))
    ]


def dense_search_vector(
    qvec: list[float],
    k: int = 20,
    where: dict | None = None,
    collection=None,
) -> list[dict[str, Any]]:
    """Same as dense_search but takes a pre-computed query embedding.

    Used by HyDE: the caller embeds a hypothetical answer, not the query.
    """
    coll = collection or load_collection()
    result = coll.query(
        query_embeddings=[qvec],
        n_results=k,
        where=_normalize_where(where),
        include=["metadatas", "documents", "distances"],
    )
    ids = result["ids"][0]
    metas = result["metadatas"][0]
    docs = result["documents"][0]
    dists = result["distances"][0]
    return [
        {
            "chunk_id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "distance": dists[i],
            "score": 1.0 - dists[i],
        }
        for i in range(len(ids))
    ]
