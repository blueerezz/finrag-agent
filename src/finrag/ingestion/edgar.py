"""Download 10-K filings from SEC EDGAR and keep a manifest of what we have."""
import json
import time
from pathlib import Path

import httpx

from finrag.config import settings

# ticker -> CIK (the SEC's company ID, zero-padded to 10 digits)
COMPANIES: dict[str, str] = {
    "MSFT": "0000789019",
    "META": "0001326801",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "NVDA": "0001045810",
}

RAW_DIR = settings.data_dir / "raw"
MANIFEST_PATH = settings.data_dir / "manifest.json"
SLEEP_S = 0.2  # stay far below the SEC's 10 requests/second limit


def make_client() -> httpx.Client:
    if not settings.sec_user_agent:
        raise RuntimeError("Set SEC_USER_AGENT in .env, e.g. 'Your Name you@email.com'")
    return httpx.Client(
        headers={"User-Agent": settings.sec_user_agent},
        timeout=60,
        follow_redirects=True,
    )

def _iter_filings(block: dict):
    """EDGAR returns parallel lists; turn them into one dict per filing."""
    keys = list(block.keys())
    for values in zip(*(block[k] for k in keys)):
        yield dict(zip(keys, values))

_TICKER_CACHE: dict[str, str] | None = None


def resolve_cik(client: httpx.Client, ticker: str) -> str:
    """Look up a company's CIK in the SEC's official ticker list."""
    global _TICKER_CACHE
    if _TICKER_CACHE is None:
        resp = client.get("https://www.sec.gov/files/company_tickers.json")
        resp.raise_for_status()
        _TICKER_CACHE = {v["ticker"]: f"{v['cik_str']:010d}" for v in resp.json().values()}
    return _TICKER_CACHE[ticker]
def list_10k_filings(client: httpx.Client, ticker: str) -> list[dict]:
    """Ask EDGAR for ALL the company's filings (recent + older pages), keep only 10-Ks."""
    cik = COMPANIES.get(ticker) or resolve_cik(client, ticker)
    resp = client.get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    resp.raise_for_status()
    data = resp.json()

    blocks = [data["filings"]["recent"]]
    for extra in data["filings"].get("files", []):  # older filings live in extra pages
        time.sleep(SLEEP_S)
        r = client.get(f"https://data.sec.gov/submissions/{extra['name']}")
        r.raise_for_status()
        blocks.append(r.json())  # extra pages have the lists at top level

    filings = []
    for block in blocks:
        for row in _iter_filings(block):
            if row["form"] != "10-K" or not row["reportDate"]:
                continue
            accession = row["accessionNumber"]
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{accession.replace('-', '')}/{row['primaryDocument']}"
            )
            filings.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "form": row["form"],
                    "accession": accession,
                    "filing_date": row["filingDate"],
                    "period_end": row["reportDate"],
                    "url": url,
                }
            )
    return filings

def download_filing(client: httpx.Client, filing: dict) -> Path:
    suffix = Path(filing["url"]).suffix or ".htm"
    path = RAW_DIR / filing["ticker"] / f"{filing['ticker']}_10K_{filing['period_end']}{suffix}"
    filing["local_path"] = str(path.relative_to(settings.data_dir))
    if path.exists():  # idempotent: re-running costs nothing
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = client.get(filing["url"])
    resp.raise_for_status()
    path.write_bytes(resp.content)
    time.sleep(SLEEP_S)
    return path


def download_all(tickers: list[str], since_year: int) -> list[dict]:
    manifest: list[dict] = []
    with make_client() as client:
        for ticker in tickers:
            filings = list_10k_filings(client, ticker)
            time.sleep(SLEEP_S)
            filings = [f for f in filings if int(f["period_end"][:4]) >= since_year]
            for f in filings:
                download_filing(client, f)
                print(f"{ticker}  period_end={f['period_end']}  -> {f['local_path']}")
                manifest.append(f)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest