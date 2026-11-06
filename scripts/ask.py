"""CLI: question → pipeline.ask() → LLM → cited answer."""
import argparse
import sys

from finrag.generation.citations import (
    extract_citations,
    format_sources,
    validate_citations,
)
from finrag.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from finrag.llm import chat
from finrag.retrieval.pipeline import ask as pipeline_ask


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--ticker")
    p.add_argument("--section")
    p.add_argument("--kind", choices=["text", "table"])
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--hyde", action="store_true",
                   help="enable HyDE (uses 1 extra LLM call)")
    p.add_argument("--expand", action="store_true",
                   help="enable query expansion (uses 1 extra LLM call)")
    args = p.parse_args()

    result = pipeline_ask(
        args.question, alpha=args.alpha, ticker=args.ticker,
        section=args.section, kind=args.kind,
        hyde=args.hyde, expand=args.expand, k=args.k,
    )

    f = result["features"]
    print(f"\nQuery: {args.question!r}")
    print(f"Features: hyde={f['hyde']} expand={f['expansion']} alpha={f['alpha']}")
    if len(result["queries"]) > 1:
        print("Query variants searched:")
        for i, q in enumerate(result["queries"], 1):
            print(f"  {i}. {q!r}")
    if result["hypothetical"]:
        print(f"\nHyDE hypothetical:\n  {result['hypothetical']}")

    hits = result["hits"]
    if not hits:
        print("\nNo candidates after retrieval. Try relaxing filters.")
        return 1

    lc = all(h.get("low_confidence") for h in hits)
    print(f"\nRetrieved {len(hits)} chunk(s)" + ("  ⚠ low confidence" if lc else ""))
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        s = h.get("rerank_score")
        s_str = f"{s:+.2f}" if s is not None else "lc"
        print(f"  {i}. {m['ticker']}/{m['section_code'] or '--'} "
              f"FY{m['fiscal_year']} {s_str}  "
              f"{h['text'][:55].replace(chr(10), ' ')!r}")

    print(f"\n{'-' * 78}\nAsking LLM...\n")
    try:
        answer = chat(SYSTEM_PROMPT, build_user_prompt(args.question, hits))
    except Exception as e:
        print(f"LLM call failed: {e}")
        return 2

    print(answer)
    cites = extract_citations(answer)
    valid, hall = validate_citations(cites, hits)
    print(f"\n{'-' * 78}\nSources:")
    print(format_sources(valid, hits))
    if hall:
        print(f"\n⚠ {len(hall)} hallucinated citation(s) removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
