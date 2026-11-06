"""High-level ask() pipeline: query → optional HyDE/expansion → hybrid → rerank.

Single entry point so CLI and API behave identically. All advanced features
default OFF (they cost LLM calls and can hurt precision).

Result schema:
    {
        "hits": [...],              # final top-k after rerank
        "features": {"hyde": bool, "expansion": bool, "alpha": float},
        "queries": [original, ...], # what was actually searched
        "hypothetical": str | None, # if HyDE was on
    }
"""
from __future__ import annotations

from typing import Any

from finrag.config import settings
from finrag.retrieval.fusion import hybrid_search, rrf_combine
from finrag.retrieval.hyde import hyde_hybrid_search
from finrag.retrieval.query_transform import llm_expand, transform
from finrag.retrieval.rerank import rerank


def _build_where(
    ticker: str | None,
    section: str | None,
    kind: str | None,
) -> dict | None:
    w: dict = {}
    if ticker:
        w["ticker"] = ticker.upper()
    if section:
        w["section_code"] = section
    if kind:
        w["kind"] = kind
    return w or None


def ask(
    query: str,
    *,
    alpha: float | None = None,
    ticker: str | None = None,
    section: str | None = None,
    kind: str | None = None,
    hyde: bool | None = None,
    expand: bool | None = None,
    k: int = 6,
    pool_size: int = 50,
) -> dict[str, Any]:
    """End-to-end retrieval pipeline. LLM call happens in the caller."""
    alpha = settings.hybrid_alpha if alpha is None else max(0.0, min(1.0, alpha))
    hyde = settings.hyde_enabled if hyde is None else hyde
    expand = settings.query_expansion_enabled if expand is None else expand

    where = _build_where(ticker, section, kind)
    base_q = transform(query, expand=False)  # no rule expansion here — LLM does it

    # Assemble query variants
    queries = [base_q]
    if expand:
        variants = llm_expand(query, n=settings.expansion_variants)
        if variants:
            queries.extend(variants)

    # Retrieve candidates per variant
    rankings: list[list[dict]] = []
    hypothetical: str | None = None
    for q in queries:
        if hyde:
            hits, hyp = hyde_hybrid_search(
                q, k=pool_size, where=where, alpha=alpha, pool_size=pool_size
            )
            if hypothetical is None:
                hypothetical = hyp
        else:
            hits = hybrid_search(q, k=pool_size, where=where, alpha=alpha)
        rankings.append(hits)

    if len(rankings) == 1:
        candidates = rankings[0]
    else:
        candidates = rrf_combine(rankings, k=pool_size)

    # Rerank against the *original user query*, not the transformed one
    final = rerank(query, candidates, k=k)
    if not final and candidates:
        final = candidates[:k]
        for h in final:
            h["low_confidence"] = True
            h.setdefault("rerank_score", None)

    return {
        "hits": final,
        "features": {"hyde": hyde, "expansion": expand, "alpha": alpha},
        "queries": queries,
        "hypothetical": hypothetical,
    }
