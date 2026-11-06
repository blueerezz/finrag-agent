"""Rewrite natural-language questions into search-friendly queries.

Natural questions like "What was NVDA's revenue in fiscal 2025?"
underperform in BM25 because question words dilute the weight of the
key term. This module strips them and optionally expands synonyms.
"""
import re

QUESTION_WORDS_RE = re.compile(
    r"^(what|what's|which|who|when|where|why|how|did|does|do|is|are|was|were|"
    r"the|a|an|can|could|would|should)\s+",
    re.I,
)
TRAILING_PUNCT_RE = re.compile(r"[?.!]+\s*$")
POSSESSIVE_RE = re.compile(r"'s\b")

FINANCIAL_EXPANSIONS = {
    "revenue": "revenue net revenue total revenue revenues",
    "profit": "profit operating income net income",
    "income": "income operating income net income",
    "assets": "assets total assets consolidated balance sheet",
    "margin": "margin gross margin operating margin",
    "debt": "debt long-term debt borrowings",
}


def rule_based_transform(query: str) -> str:
    q = query.strip()
    prev = None
    while prev != q:
        prev = q
        q = QUESTION_WORDS_RE.sub("", q)
    q = POSSESSIVE_RE.sub("", q)  # "NVDA's" -> "NVDA"
    q = TRAILING_PUNCT_RE.sub("", q).strip()
    return q or query


def expand_financial_terms(query: str) -> str:
    lower = query.lower()
    extras = [v for k, v in FINANCIAL_EXPANSIONS.items() if k in lower]
    return " ".join([query] + extras) if extras else query


def transform(query: str, expand: bool = False) -> str:
    """Rule-based transform + optional expansion.

    Expansion only runs for natural-language questions (queries starting
    with a question word). Search-style queries like "NVDA revenue 2025"
    are already optimized — expanding them dilutes BM25's signal.
    """
    is_question = bool(QUESTION_WORDS_RE.match(query.strip()))
    q = rule_based_transform(query)
    if expand and is_question:
        q = expand_financial_terms(q)
    return q
