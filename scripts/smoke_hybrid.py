"""Side-by-side dense vs sparse vs fused retrieval on 10 queries.

  uv run python scripts/smoke_hybrid.py                 # default alpha (0.7)
  uv run python scripts/smoke_hybrid.py --alpha 0.5     # custom alpha
  uv run python scripts/smoke_hybrid.py --sweep         # show alpha 0.0/0.25/0.5/0.75/1.0
"""
import argparse
import sys

from finrag.config import settings
from finrag.retrieval.dense import dense_search
from finrag.retrieval.fusion import hybrid_search
from finrag.retrieval.sparse import sparse_search

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


def _row(h: dict, with_src: bool = False) -> str:
    m = h["metadata"]
    snippet = h["text"][:44].replace("\n", " ")
    src = ""
    if with_src and "sources" in h:
        src = f" [{'+'.join(h['sources'])}]"
    return (f"{m['ticker']:5}/{m['section_code'] or '--':2} "
            f"FY{m['fiscal_year']} score={h['score']:.3f}{src}  {snippet!r}")


def show_query(q: str, where, alpha: float) -> None:
    print(f"\n{'═' * 78}")
    print(f"Query: {q!r}   where={where}   α={alpha:.2f}")
    print(f"{'─' * 78}")

    for label, hits, src in (
        ("DENSE ", dense_search(q, k=3, where=where), False),
        ("SPARSE", sparse_search(q, k=3, where=where), False),
        ("FUSED ", hybrid_search(q, k=3, where=where, alpha=alpha), True),
    ):
        print(f"{label}:")
        if not hits:
            print("  (none)")
        else:
            for i, h in enumerate(hits, 1):
                print(f"  {i}. {_row(h, with_src=src)}")


def sweep_query(q: str, where) -> None:
    print(f"\n{q!r}   where={where}")
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        hits = hybrid_search(q, k=3, where=where, alpha=a)
        names = [f"{h['metadata']['ticker']}/{h['metadata']['section_code'] or '--'}"
                 f" FY{h['metadata']['fiscal_year']}" for h in hits]
        label = {0.0: "(sparse only)", 1.0: "(dense only)"}.get(a, "")
        print(f"  α={a:.2f} {label:13} → " + "  |  ".join(names))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=None,
                   help=f"weight on dense (default {settings.hybrid_alpha})")
    p.add_argument("--sweep", action="store_true",
                   help="show top-3 at alpha 0.0/0.25/0.5/0.75/1.0")
    args = p.parse_args()

    alpha = args.alpha if args.alpha is not None else settings.hybrid_alpha

    # Warm the BM25 cache once before timing anything.
    sparse_search("warmup", k=1)

    if args.sweep:
        for q, w in QUERIES:
            sweep_query(q, w)
    else:
        for q, w in QUERIES:
            show_query(q, w, alpha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
