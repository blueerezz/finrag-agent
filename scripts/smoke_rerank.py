"""Compare fused vs fused+rerank on the same 10 queries."""
import sys

from finrag.retrieval.fusion import hybrid_search
from finrag.retrieval.rerank import hybrid_rerank_search

QUERIES = [
    ("NVDA total revenue fiscal 2025",   {"ticker": "NVDA",  "section_code": "8"}),
    ("PEP total assets 2021",            {"ticker": "PEP",   "section_code": "8"}),
    ("GOOGL operating income 2024",      {"ticker": "GOOGL", "section_code": "8"}),
    ("META cash flow from operations",   {"ticker": "META",  "section_code": "8"}),
    ("AMD gross margin",                 {"ticker": "AMD",   "section_code": "8"}),
    ("Boeing supply chain risk",         {"ticker": "BA",    "kind": "text"}),
    ("NVDA competition risk",            {"ticker": "NVDA",  "kind": "text"}),
    ("Adobe subscription revenue",       {"ticker": "ADBE",  "section_code": "7"}),
    ("impairment of goodwill",           None),
    ("deferred revenue",                 None),
]


def _line(h: dict) -> str:
    m = h["metadata"]
    text = h["text"][:52].replace("\n", " ")
    score = h.get("rerank_score", h.get("score", 0.0))
    return f"{m['ticker']:5}/{m['section_code'] or '--':2} FY{m['fiscal_year']} " \
           f"s={score:+.3f}  {text!r}"


def main() -> int:
    # Warm up both models on first query to exclude load time from display.
    hybrid_search("warmup", k=1)
    hybrid_rerank_search("warmup", k=1)

    for q, where in QUERIES:
        print(f"\n{'═' * 80}\n{q!r}   where={where}")
        print(f"{'─' * 80}")

        fused = hybrid_search(q, k=3, where=where)
        reranked = hybrid_rerank_search(q, k=3, where=where)

        print("FUSED          :")
        for i, h in enumerate(fused, 1):
            print(f"  {i}. {_line(h)}")
        print("FUSED+RERANKED :")
        for i, h in enumerate(reranked, 1):
            print(f"  {i}. {_line(h)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
