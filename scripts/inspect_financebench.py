import json
from collections import Counter
from pathlib import Path

from finrag.config import settings

path = settings.data_dir / "financebench.jsonl"
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

print("Total questions:", len(rows))
print("\nTop 10 companies in FinanceBench:")
for company, n in Counter(r["company"] for r in rows).most_common(10):
    print(f"  {company}: {n}")

print("\nQuestion types:", Counter(r["question_type"] for r in rows))
print("Reasoning types:", Counter(r["question_reasoning"] for r in rows))

print("\n--- Sample Question ---")
sample = rows[0]
print("Company:", sample["company"])
print("Doc:", sample["doc_name"])
print("Q:", sample["question"])
print("A:", sample["answer"])
print("Justification:", sample["justification"])
print("Evidence:", json.dumps(sample["evidence"], indent=2)[:600])