"""Evaluation metrics for retrieval and generation."""
from typing import Any

def hit_rate_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Did the correct chunk appear in the top-K results? (1.0 = yes, 0.0 = no)"""
    top_k = retrieved_ids[:k]
    for expected in expected_ids:
        if any(expected in rid for rid in top_k):
            return 1.0
    return 0.0

def mrr(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first correct result."""
    for i, rid in enumerate(retrieved_ids):
        if any(expected in rid for expected in expected_ids):
            return 1.0 / (i + 1)
    return 0.0

def citation_accuracy(cited_doc_id: str, cited_lines: str, expected_doc_id: str) -> float:
    """Check if the LLM cited the correct document."""
    if not cited_doc_id or not expected_doc_id:
        return 0.0
    # Check if the base doc matches (e.g., "NVDA_10K_2025-01-26")
    if expected_doc_id in cited_doc_id:
        return 1.0
    return 0.0
