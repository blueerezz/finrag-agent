"""Cross-encoder reranker: top-50 candidates → top-5 quality hits.

Uses BAAI/bge-reranker-base via fastembed. Cross-encoders score
(query, chunk) pairs jointly, which is much more precise than
independent embedding. ~1s for 20-50 candidates on CPU.

This is the standard precision-fix for two-stage retrieval:
  hybrid_search(k=50)  →  rerank(k=5)
"""
from __future__ import annotations

from typing import Any

from fastembed.rerank.cross_encoder import TextCrossEncoder

RERANK_MODEL = "BAAI/bge-reranker-base"

_model: TextCrossEncoder | None = None


def get_reranker() -> TextCrossEncoder:
    global _model
    if _model is None:
        # ~1.1 GB first-load; cached in ~/.cache/fastembed after
        _model = TextCrossEncoder(model_name=RERANK_MODEL)
    return _model


import re as _re

_HEADING_RE = _re.compile(r"^(?:ITEM\s+\d+[A-Z]?\.?|PART\s+[IVX]+)[^\n]*\n", _re.M | _re.I)


def _strip_headings(text: str) -> str:
    """Remove 'ITEM 8. FINANCIAL STATEMENTS...' style lines from chunk start.

    BGE-reranker scores the beginning of the text most heavily, so a chunk
    that starts with a section heading gets penalized even when the answer
    (e.g. '| Gross margin | 45% | 48% |') is inside. Strip them before rerank.
    """
    stripped = _HEADING_RE.sub("", text, count=3)  # top 3 headings max
    return stripped.strip() or text  # never return empty


def rerank(
    query: str,
    hits: list[dict[str, Any]],
    k: int = 5,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Rerank candidates. Returns top-k, or [] if none clears min_score."""
    if not hits:
        return []

    doc_texts = [_strip_headings(h["text"]) for h in hits]
    scores = list(get_reranker().rerank(query, doc_texts))

    ranked = sorted(zip(hits, scores), key=lambda x: -x[1])[:k]
    out = [{**h, "rerank_score": float(s), "score": float(s)} for h, s in ranked]

    # Signal "no relevant answer" — caller can fall back to 'I don't know'
    if out and out[0]["rerank_score"] < min_score:
        return []
    return out


def hybrid_rerank_search(
    query: str,
    k: int = 5,
    pool_size: int = 50,
    alpha: float | None = None,
    where: dict | None = None,
    fallback: bool = True,
) -> list[dict[str, Any]]:
    """Full pipeline: hybrid retrieve → rerank → top-k.

    If rerank returns [] (all candidates scored below min_score), and
    fallback=True, return the top-k fused candidates with a
    low_confidence flag. Callers (generation) can decide how to handle.
    """
    from finrag.retrieval.fusion import hybrid_search

    candidates = hybrid_search(query, k=pool_size, where=where, alpha=alpha)
    reranked = rerank(query, candidates, k=k)

    if reranked:
        return reranked

    if fallback and candidates:
        # All candidates scored negative — but we still return them
        # so the LLM can judge. Mark them low_confidence.
        out = candidates[:k]
        for h in out:
            h["low_confidence"] = True
            h["rerank_score"] = None
        return out

    return []
