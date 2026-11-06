import argparse

from finrag.ingestion.edgar import COMPANIES, download_all

parser = argparse.ArgumentParser()
parser.add_argument("--tickers", nargs="+", default=list(COMPANIES))
parser.add_argument("--since", type=int, default=2019, help="earliest period-end year")
args = parser.parse_args()

manifest = download_all(args.tickers, args.since)
print(f"\nDone: {len(manifest)} filings. Manifest at data/manifest.json")