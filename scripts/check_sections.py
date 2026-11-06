"""Audit chunk sectioning across every parsed 10-K.

Exits 0 if every doc passes, 1 otherwise. Chain it with `&&` so a bad
chunking change breaks the pipeline instead of silently shipping.
"""
import sys

from finrag.config import settings
from finrag.ingestion.chunk import chunk_document, doc_info

MIN_ITEM7 = 5       # MD&A is always substantial
MIN_ITEM8 = 20      # financials are the largest section
MIN_SECTIONS = 8    # at least this many distinct Item headings


def main() -> int:
    paths = sorted((settings.data_dir / "parsed").glob("*/*.md"))
    if not paths:
        print("no parsed files under data/parsed/*/*.md")
        return 1

    print(f"{'doc':28} {'chunks':>6} {'secs':>5} {'Item7':>6} {'Item8':>6}")
    print("-" * 55)

    bad = 0
    for path in paths:
        chunks = chunk_document(path.read_text(encoding="utf-8"), doc_info(path))
        n7 = sum(c.section.startswith("Item 7.") for c in chunks)
        n8 = sum(c.section.startswith("Item 8") for c in chunks)
        n_sec = len({c.section for c in chunks})

        ok = n7 >= MIN_ITEM7 and n8 >= MIN_ITEM8 and n_sec >= MIN_SECTIONS
        flag = "" if ok else "  <-- CHECK"
        bad += not ok
        print(f"{path.stem:28} {len(chunks):6d} {n_sec:5d} {n7:6d} {n8:6d}{flag}")

    print(f"\n{bad} / {len(paths)} docs need attention")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
