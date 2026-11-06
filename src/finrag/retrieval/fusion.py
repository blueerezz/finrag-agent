"""Hybrid retrieval: weighted Reciprocal Rank Fusion of dense + sparse.

alpha = weight on dense (0.0 = pure BM25, 1.0 = pure dense).
Each retriever contributes alpha/(k+rank) or (1-alpha)/(k+rank) to a
chunk's fused score. Chunks found by BOTH retrievers naturally rise to
the top — that's the whole point of RRF.

The alpha parameter is the "UI trigger": pass it per-request from an
API endpoint, or set HYBRID_ALPHA=0.6 in .env to change the default.
"""
from __future__ import annotations

from typing import Any

from finrag.config import settings
from finrag.retrieval.dense import dense_search
from finrag.retrieval.sparse import sparse_search

RRF_K = 60  # standard constant from the RRF paper


def fuse(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    alpha: float = 0.7,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Weighted RRF of two ranked lists. alpha ∈ [0,1] = weight on dense."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    sources: dict[str, set[str]] = {}

    for rank, h in enumerate(dense_hits, 1):
        cid = h["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + alpha / (RRF_K + rank)
        by_id[cid] = h
        sources.setdefault(cid, set()).add("dense")

    for rank, h in enumerate(sparse_hits, 1):
        cid = h["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + (1 - alpha) / (RRF_K + rank)
        by_id.setdefault(cid, h)
        sources.setdefault(cid, set()).add("sparse")

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return [
        {**by_id[cid], "score": s, "sources": sorted(sources[cid])}
        for cid, s in ranked
    ]


def hybrid_search(
    query: str,
    k: int = 5,
    pool_size: int | None = None,
    alpha: float | None = None,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    """Dense + BM25 hybrid retrieval with weighted RRF.

    alpha = weight on dense (0.0 = pure BM25, 1.0 = pure dense).
    Defaults come from settings (HYBRID_ALPHA / HYBRID_POOL env vars).
    """
    if alpha is None:
        alpha = settings.hybrid_alpha
    if pool_size is None:
        pool_size = settings.hybrid_pool
    alpha = max(0.0, min(1.0, alpha))

    d_hits = dense_search(query, k=pool_size, where=where)
    s_hits = sparse_search(query, k=pool_size, where=where)
    return fuse(d_hits, s_hits, alpha=alpha, k=k)


def rrf_combine(
    rankings: list[list[dict[str, Any]]],
    k: int = 50,
) -> list[dict[str, Any]]:
    """Combine N ranked lists via unweighted RRF.

    Used for multi-query expansion: each variant's top-N list contributes
    1/(RRF_K + rank) to each chunk's score. Chunks found by multiple
    variants rise to the top.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    sources: dict[str, set[str]] = {}

    for qi, ranking in enumerate(rankings):
        for rank, h in enumerate(ranking, 1):
            cid = h["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            by_id.setdefault(cid, h)
            sources.setdefault(cid, set()).add(f"q{qi}")

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return [
        {**by_id[cid], "score": s, "sources": sorted(sources[cid])}
        for cid, s in ranked
    ]
