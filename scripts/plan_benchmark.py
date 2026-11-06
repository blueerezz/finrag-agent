import json
from collections import Counter

from finrag.config import settings

path = settings.data_dir / "financebench.jsonl"
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
tenk = [r for r in rows if r["doc_name"].upper().endswith("_10K")]

print(f"10-K questions: {len(tenk)} of {len(rows)}")
print("unique 10-K docs:", len({r['doc_name'] for r in tenk}))

print("\nquestions per company (10-K only):")
for company, n in Counter(r["company"] for r in tenk).most_common():
    print(f"  {company}: {n}")

print("\ndocs for the top 8 companies:")
top = [c for c, _ in Counter(r["company"] for r in tenk).most_common(8)]
for company in top:
    docs = sorted({r["doc_name"] for r in tenk if r["company"] == company})
    print(f"  {company}: {docs}")

print("\ndoc-name suffixes (all questions):", Counter(r["doc_name"].split("_")[-1] for r in rows))
