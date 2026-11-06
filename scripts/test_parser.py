import time

from finrag.config import settings
from finrag.ingestion.parse import parse_10k_html

sample = settings.data_dir / "raw" / "MSFT" / "MSFT_10K_2024-06-30.htm"
html = sample.read_text(encoding="utf-8", errors="ignore")

t0 = time.time()
text = parse_10k_html(html)
print(f"parsed in {time.time() - t0:.1f}s: {len(html)/1e6:.2f} MB -> {len(text)/1e6:.2f} MB")

out = settings.data_dir / "parsed" / "MSFT" / f"{sample.stem}.md"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text, encoding="utf-8")

lines = text.splitlines()
table_lines = [l for l in lines if l.startswith("|")]

print("\n=== sanity checks ===")
print("table rows:", len(table_lines))
print("XBRL junk ('us-gaap:' occurrences):", text.count("us-gaap:"))
print("rows with a lone '$' cell:", sum(" | $ | " in l for l in table_lines))
print("lines longer than 200 chars (whole paragraphs):", sum(len(l) > 200 for l in lines))

print("\n=== key financial rows ===")
for kw in ["Total revenue", "Operating income", "Net income"]:
    hits = [l for l in table_lines if l.lower().startswith(f"| {kw.lower()}")]
    print(f"\n[{kw}] {len(hits)} rows. First two:")
    for h in hits[:2]:
        print("  ", h[:200])

print("\n=== a paragraph sample ===")
para = next(l for l in lines if len(l) > 300 and not l.startswith("|"))
print(para[:500])