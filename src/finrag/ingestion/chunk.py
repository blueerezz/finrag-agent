"""Section-aware, table-aware chunking of parsed 10-K markdown."""

import re
from dataclasses import asdict, dataclass
from pathlib import Path


ITEM_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors and Executive Officers",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
}

ITEM_RE = re.compile(
    r"^(?:part\s+[ivx]+[\s,.\-–—]*)?item\s+(\d{1,2}[abc]?)\b",
    re.I,
)

NOISE_RE = re.compile(r"^(table of contents|\d{1,3})$", re.I)

# Detect headings incorrectly represented as two-column table rows.
HEADING_ROW_RE = re.compile(
    r"^\|\s*(item\s+\d{1,2}[abc]?\.?)\s*\|\s*([^|]+?)\s*\|\s*$",
    re.I,
)

SEP_ROW_RE = re.compile(r"^\|[\s|\-]+\|$")


def _promote_item_rows(lines: list[str]) -> list[str]:
    """Convert table-based ITEM headings into ordinary text lines.

    Some filings (e.g. AMD 2015) put 'ITEM 7. | TITLE' in a table row.
    One line in, one line out, so line numbers stay valid for citations.
    """
    out = list(lines)

    for i, ln in enumerate(out):
        m = HEADING_ROW_RE.match(ln)

        if m:
            out[i] = f"{m.group(1)} {m.group(2)}"

            if i + 1 < len(out) and SEP_ROW_RE.match(out[i + 1]):
                out[i + 1] = ""

    return out


# Budgets are in characters, including the context line.
# bge-small truncates at 512 tokens: ~4 chars/token for prose,
# ~3 for number-heavy tables.
PROSE_BUDGET = 1800
TABLE_BUDGET = 1400
MIN_CHARS = 60
CAPTION_LINES = 3
MAX_CAPTION_LINE = 160


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ticker: str
    form: str
    fiscal_year: int
    period_end: str
    section: str
    kind: str  # "text" or "table"
    line_start: int  # 1-based line numbers in the parsed .md file
    line_end: int
    context: str
    text: str

    @property
    def embed_text(self) -> str:
        return f"{self.context}\n{self.text}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["embed_text"] = self.embed_text
        return d


def doc_info(path: Path) -> dict:
    """GOOGL_10K_2024-12-31.md -> metadata.

    Fiscal year = calendar year of period end.
    """
    ticker, _form, period_end = path.stem.split("_")

    return {
        "doc_id": path.stem,
        "ticker": ticker,
        "form": "10-K",
        "period_end": period_end,
        "fiscal_year": int(period_end[:4]),
    }


def _context(info: dict, section: str) -> str:
    return (
        f"[{info['ticker']} | 10-K | FY{info['fiscal_year']} "
        f"(period ended {info['period_end']}) | {section}]"
    )


def _table_header_rows(lines: list[str]) -> int:
    """How many leading lines are header: title row, separator,
    and label-less year rows.
    """
    h = (
        2
        if len(lines) > 1
        and set(lines[1].replace("|", "").replace(" ", "")) <= {"-"}
        else 1
    )

    while h < len(lines) and h < 5 and lines[h].startswith("| |"):
        if re.search(r"\d,\d{3}|\$", lines[h]):
            break  # a data row, not a year row
        h += 1

    return h


def _split_table(lines: list[str], caption: list[str], context: str):
    """Yield (first_idx, last_idx, text).

    Every piece repeats caption + header rows.
    """
    h = _table_header_rows(lines)
    fixed = "\n".join(caption + lines[:h])
    limit = TABLE_BUDGET - len(context)

    piece: list[int] = []
    size = len(fixed)

    for idx in range(h, len(lines)):
        if piece and size + len(lines[idx]) + 1 > limit:
            yield (
                piece[0],
                piece[-1],
                fixed + "\n" + "\n".join(lines[i] for i in piece),
            )
            piece, size = [], len(fixed)

        piece.append(idx)
        size += len(lines[idx]) + 1

    if piece:
        yield (
            piece[0],
            piece[-1],
            fixed + "\n" + "\n".join(lines[i] for i in piece),
        )

    elif h == len(lines):
        # Table with only header rows.
        yield 0, len(lines) - 1, fixed


def _split_long(paragraph: str, budget: int) -> list[str]:
    out = []
    cur = ""

    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        if cur and len(cur) + len(sentence) + 1 > budget:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()

    if cur:
        out.append(cur)

    return out

FINANCIAL_MARKERS = (
    "CONSOLIDATED BALANCE SHEETS",
    "CONSOLIDATED STATEMENTS OF INCOME",
    "CONSOLIDATED STATEMENTS OF OPERATIONS",
    "CONSOLIDATED STATEMENTS OF CASH FLOWS",
    "CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME",
    "CONSOLIDATED STATEMENTS OF STOCKHOLDERS",
    "CONSOLIDATED STATEMENTS OF SHAREHOLDERS",
    "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS",
    "Report of Independent Registered Public Accounting Firm",
    "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM",
)


_EXHIBIT_INDEX_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"index\s+to\s+exhibits?"
    r"|exhibits?\s+index"
    r"|exhibit\s+index"
    r"|SIGNATURES?"
    r"|POWER\s+OF\s+ATTORNEY"
    r")\s*[:.]?\s*(?:\n|$)",
    re.I | re.M,
)

_SIGNATURE_RE = re.compile(r"\bBy:\s*\|?\s*/s/", re.I)


_EXHIBIT_INDEX_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"index\s+to\s+exhibits?"
    r"|exhibits?\s+index"
    r"|exhibit\s+index"
    r"|SIGNATURES?"
    r"|POWER\s+OF\s+ATTORNEY"
    r")\s*[:.]?\s*(?:\n|$)",
    re.I | re.M,
)

_SIGNATURE_RE = re.compile(r"\bBy:\s*\|?\s*/s/", re.I)

_FIN_HEADING_RE = re.compile(
    r"^(?:\s*\|?\s*)(?:" + "|".join(re.escape(m) for m in FINANCIAL_MARKERS) + r")(?:\s*\|?\s*)$",
    re.I | re.M,
)


def _relabel_financials(chunks: list["Chunk"], info: dict) -> list["Chunk"]:
    """Move real financial content into Item 8.

    Sources:
      * Item 15 — NVDA/AMZN pattern: statements filed after the TOC.
        Loose marker match is safe (Item 15 is exhibit territory).
      * Item 7  — PEP pattern: statements filed *inside* MD&A.
        We require a standalone uppercase heading match so MD&A prose
        ("...our Consolidated Statements of Income expressed as...")
        isn't stolen.

    A sticky flag stays ON through the notes and flips OFF at
    Exhibit Index / Signatures / Power of Attorney.
    """
    new_section = f"Item 8. {ITEM_NAMES['8']}"
    markers_lower = [m.lower() for m in FINANCIAL_MARKERS]

    in_fin = False

    for c in chunks:
        if c.section.startswith("Item 8"):
            in_fin = True
            continue

        is_item15 = c.section.startswith("Item 15")
        is_item7 = c.section.startswith("Item 7.")  # not 7A

        if not (is_item15 or is_item7):
            in_fin = False
            continue

        if not in_fin:
            if is_item15:
                if any(m in c.text.lower() for m in markers_lower):
                    in_fin = True
            else:  # Item 7 — strict heading match only
                if _FIN_HEADING_RE.search(c.text):
                    in_fin = True

        if in_fin and (
            _EXHIBIT_INDEX_RE.search(c.text)
            or _SIGNATURE_RE.search(c.text)
        ):
            in_fin = False
            continue

        if in_fin:
            c.section = new_section
            c.context = _context(info, new_section)

    return chunks

def chunk_document(text: str, info: dict) -> list[Chunk]:
    # IMPORTANT: Promote table-based headings before processing lines.
    lines = _promote_item_rows(text.splitlines())

    chunks: list[Chunk] = []
    section = "Cover and Table of Contents"

    buf: list[str] = []
    buf_len = 0
    buf_start = buf_end = 0
    buf_section = section

    recent: list[str] = []  # short prose lines just above (caption candidates)

    def emit(kind: str, body: str, start: int, end: int, sec: str) -> None:
        chunks.append(
            Chunk(
                chunk_id=f"{info['doc_id']}#{len(chunks):05d}",
                doc_id=info["doc_id"],
                ticker=info["ticker"],
                form=info["form"],
                fiscal_year=info["fiscal_year"],
                period_end=info["period_end"],
                section=sec,
                kind=kind,
                line_start=start,
                line_end=end,
                context=_context(info, sec),
                text=body,
            )
        )

    def flush() -> None:
        nonlocal buf, buf_len

        if buf and buf_len >= MIN_CHARS:
            emit(
                "text",
                "\n".join(buf),
                buf_start,
                buf_end,
                buf_section,
            )

        buf, buf_len = [], 0

    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if line.startswith("|"):
            # A table: consecutive lines starting with "|".
            j = i

            while j < len(lines) and lines[j].startswith("|"):
                j += 1

            flush()

            ctx = _context(info, section)
            caption = recent[-CAPTION_LINES:]

            for first, last, body in _split_table(
                lines[i:j],
                caption,
                ctx,
            ):
                start = i + (0 if first == 0 else first) + 1
                emit(
                    "table",
                    body,
                    start,
                    i + last + 1,
                    section,
                )

            recent = []
            i = j
            continue

        if NOISE_RE.match(line):
            # Page numbers, repeated "Table of Contents".
            i += 1
            continue

        m = ITEM_RE.match(line)

        if m and len(line) < 140 and m.group(1).upper() in ITEM_NAMES:
            num = m.group(1).upper()
            section = f"Item {num}. {ITEM_NAMES[num]}"

        recent = (
            recent + [line]
            if len(line) <= MAX_CAPTION_LINE
            else []
        )

        if buf and section != buf_section:
            flush()

        if len(line) > PROSE_BUDGET:
            flush()

            for part in _split_long(line, PROSE_BUDGET):
                if len(part) >= MIN_CHARS:
                    emit(
                        "text",
                        part,
                        i + 1,
                        i + 1,
                        section,
                    )

        else:
            if buf and buf_len + len(line) + 1 > PROSE_BUDGET:
                flush()

            if not buf:
                buf_start = i + 1
                buf_section = section

            buf.append(line)
            buf_len += len(line) + 1
            buf_end = i + 1

        i += 1

    flush()

    return _relabel_financials(chunks, info)