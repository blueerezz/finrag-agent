import json
from collections import Counter

import httpx

from finrag.config import settings
from finrag.ingestion.edgar import COMPANIES

URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl"

resp = httpx.get(URL, follow_redirects=True, timeout=60)
resp.raise_for_status()
rows = [json.loads(line) for line in resp.text.splitlines() if line.strip()]

settings.data_dir.mkdir(parents=True, exist_ok=True)
(settings.data_dir / "financebench.jsonl").write_text(resp.text)

print("total questions:", len(rows))
print("fields:", list(rows[0].keys()))
print("\nquestions per company:")
for company, n in Counter(r["company"] for r in rows).most_common():
    print(f"  {company}: {n}")
print("\nquestion types:", Counter(r["question_type"] for r in rows))
print("reasoning types:", Counter(r["question_reasoning"] for r in rows))
print("unique docs:", len({r["doc_name"] for r in rows}))

print("\n--- Microsoft / Amazon questions ---")
for r in rows:
    if r["company"] in {"Microsoft", "Amazon"}:
        print(f"  [{r['doc_name']}] {r['question']}")

print("\n--- one example ---")
print("Q:", rows[0]["question"])
print("A:", rows[0]["answer"])
print("doc:", rows[0]["doc_name"])
print("evidence:", json.dumps(rows[0]["evidence"], indent=2)[:900])