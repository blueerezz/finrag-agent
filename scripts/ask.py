"""End-to-end RAG: question -> hybrid retrieval -> rerank -> LLM -> cited answer.

  uv run python scripts/ask.py "What was NVDA revenue in fiscal 2025?"
  uv run python scripts/ask.py "AMD gross margin" --ticker AMD --alpha 0.7
  uv run python scripts/ask.py "deferred revenue" --k 8
"""
import argparse
import sys

from finrag.generation.citations import (
    extract_citations,
    format_sources,
    validate_citations,
)
from finrag.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from finrag.llm import chat
from finrag.retrieval.query_transform import transform
from finrag.retrieval.rerank import hybrid_rerank_search


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("question", help="the question to answer")
    p.add_argument("--ticker", help="restrict to a ticker (e.g. NVDA)")
    p.add_argument("--section", help="restrict to a section code (e.g. 8)")
    p.add_argument("--kind", choices=["text", "table"], help="restrict by chunk kind")
    p.add_argument("--alpha", type=float, default=0.7,
                   help="dense weight in hybrid (0=sparse, 1=dense); default 0.7")
    p.add_argument("--k", type=int, default=6, help="top-k chunks to send to LLM")
    args = p.parse_args()

    # Build filter dict (same syntax as hybrid_search where=).
    where = {}
    if args.ticker:
        where["ticker"] = args.ticker.upper()
    if args.section:
        where["section_code"] = args.section
    if args.kind:
        where["kind"] = args.kind
    where = where or None

    print(f"\nQuery: {args.question!r}")
    if where:
        print(f"Filter: {where}")
    print(f"Hybrid alpha={args.alpha}, retrieving top {args.k} after rerank\n")

    search_q = transform(args.question, expand=True)
    if search_q != args.question:
        print(f"Transformed query: {search_q!r}\n")

    hits = hybrid_rerank_search(
        search_q, k=args.k, pool_size=50, alpha=args.alpha, where=where
    )

    if not hits:
        print("No candidates retrieved. (Try relaxing filters or rephrasing.)")
        return 1

    low_conf = all(h.get("low_confidence") for h in hits)
    print(f"Retrieved {len(hits)} chunk(s)"
          + ("  ⚠ low confidence" if low_conf else ""))
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        score = h.get("rerank_score")
        s = f"+{score:.2f}" if score is not None else "lc"
        print(f"  {i}. {m['ticker']}/{m['section_code'] or '--'} FY{m['fiscal_year']} "
              f"{s}  {h['text'][:60].replace(chr(10), ' ')!r}")

    print(f"\n{'-' * 78}\nAsking LLM ({len(hits)} chunks as context)...\n")

    user_prompt = build_user_prompt(args.question, hits)
    try:
        answer = chat(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        print(f"LLM call failed: {e}")
        return 2

    print(answer)

    # Validate citations against retrieved hits, drop hallucinations.
    cites = extract_citations(answer)
    valid, hallucinated = validate_citations(cites, hits)

    print(f"\n{'-' * 78}")
    print("Sources:")
    print(format_sources(valid, hits))
    if hallucinated:
        print(f"\n⚠ {len(hallucinated)} hallucinated citation(s) removed:")
        for c in hallucinated:
            print(f"  {c['doc_id']}:{c['line_start']}-{c['line_end']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
