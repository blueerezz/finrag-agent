"""Sanity-check dense retrieval across the 13-ticker sample."""
from finrag.retrieval.dense import dense_search

QUERIES = [
    # Financials — should hit Item 8
    ("NVDA total revenue fiscal 2025",        {"ticker": "NVDA", "section_code": "8"}),
    ("PEP total assets 2021",                 {"ticker": "PEP",  "section_code": "8"}),
    ("GOOGL operating income 2024",           {"ticker": "GOOGL","section_code": "8"}),
    ("META cash flow from operations",        {"ticker": "META", "section_code": "8"}),
    ("MSFT revenue by segment",               {"ticker": "MSFT", "section_code": "8"}),
    ("AMZN total assets",                     {"ticker": "AMZN", "section_code": "8"}),
    ("AMD gross margin",                      {"ticker": "AMD",  "section_code": "8"}),
    # Risk factors — should hit Item 1A prose
    ("Google antitrust risk factors",         {"ticker": "GOOGL","kind": "text"}),
    ("Boeing supply chain risk",              {"ticker": "BA",   "kind": "text"}),
    ("NVDA competition risk",                 {"ticker": "NVDA", "kind": "text"}),
    # MD&A
    ("Microsoft revenue growth drivers",      {"ticker": "MSFT", "section_code": "7"}),
    ("Adobe subscription revenue discussion", {"ticker": "ADBE", "section_code": "7"}),
    # Cross-section, no filter
    ("impairment of goodwill",                None),
    ("stock-based compensation expense",      None),
    ("deferred revenue",                      None),
]

for q, where in QUERIES:
    print(f"\n=== {q!r}  where={where} ===")
    hits = dense_search(q, k=3, where=where)
    if not hits:
        print("  (no hits — filter too narrow?)")
        continue
    for h in hits:
        m = h["metadata"]
        text = h["text"][:60].replace("\n", " ")
        print(f"  [{m['ticker']:5} FY{m['fiscal_year']} Item {m['section_code']:3}] "
              f"score={h['score']:.3f}  {text!r}")
