from finrag.config import settings
from finrag.ingestion.parse import parse_10k_html

raw = settings.data_dir / "raw"
parsed = settings.data_dir / "parsed"

stats = []
for path in sorted(raw.glob("*/*.htm*")):
    out = parsed / path.parent.name / f"{path.stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():  # delete data/parsed to force a re-parse after changing the parser
        html = path.read_text(encoding="utf-8", errors="ignore")
        out.write_text(parse_10k_html(html), encoding="utf-8")
    text = out.read_text(encoding="utf-8")
    table_rows = sum(l.startswith("|") for l in text.splitlines())
    stats.append((path.stem, len(text) / 1e3, table_rows, text.count("us-gaap:")))

print(f"{'file':34} {'KB':>7} {'table rows':>11} {'xbrl junk':>10}")
for name, kb, rows_, junk in stats:
    flag = "  <-- CHECK" if kb < 150 or rows_ < 300 or junk else ""
    print(f"{name:34} {kb:7.0f} {rows_:11d} {junk:10d}{flag}")
print(f"\n{len(stats)} files parsed")