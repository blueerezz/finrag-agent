"""Build or inspect the Chroma index over parsed 10-K chunks.

Examples:
  # Full index (idempotent — skips if it already exists)
  uv run python scripts/build_index.py

  # 2-doc smoke test
  uv run python scripts/build_index.py --docs NVDA_10K_2019-01-27 PEP_10K_2021-12-25

  # All filings for 2 tickers
  uv run python scripts/build_index.py --tickers NVDA PEP

  # First 5 docs alphabetically
  uv run python scripts/build_index.py --limit 5

  # Rebuild everything from scratch
  uv run python scripts/build_index.py --force

  # Just report status
  uv run python scripts/build_index.py --stats
"""
import argparse
import sys
import time

from finrag.ingestion.index import build_index, index_stats


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="drop existing collection before building")
    p.add_argument("--stats", action="store_true",
                   help="print index status and exit")
    p.add_argument("--docs", nargs="+", metavar="DOC",
                   help="exact doc_id stems to index")
    p.add_argument("--tickers", nargs="+", metavar="T",
                   help="only index these tickers")
    p.add_argument("--limit", type=int, metavar="N",
                   help="cap number of docs after other filters")
    args = p.parse_args()

    if args.stats:
        s = index_stats()
        print(f"path:   {s['path']}")
        print(f"exists: {s['exists']}")
        print(f"chunks: {s['chunks']}")
        return 0 if s["exists"] else 1

    t0 = time.time()
    n = build_index(
        force=args.force,
        docs=args.docs,
        tickers=args.tickers,
        limit=args.limit,
    )
    print(f"done in {time.time() - t0:.1f}s — {n} chunks in index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
