"""HyDE (Hypothetical Document Embeddings).

Generate a fake 10-K excerpt that *would* answer the query, embed it,
and use that embedding for dense search. Fake answers look structurally
like real filings, so their embedding lands near real chunks.

Cost: 1 extra LLM call per query. Enabled per-request, default off.
"""
from __future__ import annotations

from typing import Any

from finrag.embeddings import embed_texts
from finrag.llm import chat
from finrag.retrieval.dense import dense_search_vector
from finrag.retrieval.fusion import fuse
from finrag.retrieval.sparse import sparse_search

HYDE_SYSTEM = (
    "You write realistic excerpts from SEC 10-K filings. Given a question, "
    "write 2-3 sentences that would answer it, as if excerpted from the "
    "filing itself. Use specific financial terms, dollar amounts, and "
    "period labels (e.g. 'fiscal 2025'). Do not add disclaimers or hedging."
)


def generate_hypothetical(query: str) -> str:
    """Ask the LLM to write a hypothetical 10-K excerpt that answers the query."""
    prompt = f"Question: {query}\n\nHypothetical 10-K excerpt:"
    return chat(HYDE_SYSTEM, prompt, temperature=0.3).strip()


def hyde_hybrid_search(
    query: str,
    k: int = 20,
    where: dict | None = None,
    alpha: float = 0.7,
    pool_size: int = 50,
) -> tuple[list[dict[str, Any]], str]:
    """Hybrid retrieval where the dense side searches the hypothetical answer.

    Returns (hits, hypothetical_text). The caller may want to show the
    hypothetical to the user for transparency / debugging.
    """
    hyp = generate_hypothetical(query)
    [qvec] = embed_texts([hyp])

    d_hits = dense_search_vector(qvec, k=pool_size, where=where)
    # Sparse side uses the *raw query* — BM25 does better with real keywords.
    s_hits = sparse_search(query, k=pool_size, where=where)

    fused = fuse(d_hits, s_hits, alpha=alpha, k=k)
    return fused, hyp
