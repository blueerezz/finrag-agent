import json
import time
from collections import defaultdict

from finrag.config import settings
from finrag.ingestion.edgar import SLEEP_S, download_filing, list_10k_filings, make_client

# FinanceBench doc-name prefix -> SEC ticker
TICKERS = {
    "AMD": "AMD", "BOEING": "BA", "AMERICANEXPRESS": "AXP", "PEPSICO": "PEP",
    "3M": "MMM", "ADOBE": "ADBE", "AMCOR": "AMCR", "BESTBUY": "BBY",
}

path = settings.data_dir / "financebench.jsonl"
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
rows = [r for r in rows if r["doc_name"].endswith("_10K") and r["doc_name"].split("_")[0] in TICKERS]

wanted = sorted({r["doc_name"] for r in rows})
by_ticker: dict[str, list[tuple[str, int]]] = defaultdict(list)
for doc in wanted:
    prefix, year, _ = doc.split("_")
    by_ticker[TICKERS[prefix]].append((doc, int(year)))

manifest, missing = [], []
with make_client() as client:
    for ticker, docs in by_ticker.items():
        filings = list_10k_filings(client, ticker)
        time.sleep(SLEEP_S)
        for doc_name, year in docs:
            # fiscal year label == calendar year of the period end (verify in the output)
            match = sorted(
                (f for f in filings if int(f["period_end"][:4]) == year),
                key=lambda f: f["filing_date"],
            )
            if not match:
                missing.append(doc_name)
                continue
            f = match[0]
            download_filing(client, f)
            f["doc_name"] = doc_name
            manifest.append(f)
            print(f"{doc_name:28} -> period_end={f['period_end']}  {f['local_path']}")

(settings.data_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))
covered = {m["doc_name"] for m in manifest}
n_q = sum(r["doc_name"] in covered for r in rows)
print(f"\nfound {len(manifest)}/{len(wanted)} docs, covering {n_q}/{len(rows)} questions")
print("missing:", missing)