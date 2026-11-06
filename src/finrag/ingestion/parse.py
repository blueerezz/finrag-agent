"""Parse raw EDGAR 10-K HTML into clean Markdown text with preserved tables."""
import re
import warnings

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

# EDGAR files are XHTML; lxml's HTML parser handles them fine.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]


def _clean_cells(tr: Tag) -> list[str]:
    """Extract cell texts, gluing together the fragments SEC splits apart."""
    cells: list[str] = []
    for idx, td in enumerate(tr.find_all(["td", "th"])):
        text = re.sub(r"\s+", " ", td.get_text(" ", strip=True)).strip()
        if not text:
            if idx == 0:
                cells.append("")   # keep the empty label column so year headers stay aligned
            continue
        if text in {")", "%", ")%"} and cells and cells[-1]:
            cells[-1] += text                      # "(370" + ")" -> "(370)"
        elif cells and cells[-1] in {"$", "(", "$("}:
            cells[-1] += text                      # "$" + "5,363" -> "$5,363"
        else:
            cells.append(text)
    return cells


def table_to_markdown(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = _clean_cells(tr)
        if any(cells):                             # skip fully empty spacer rows
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n\n" + "\n".join(lines) + "\n\n"

def parse_10k_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "meta", "link", "noscript"]):
        tag.decompose()

    # Hidden inline-XBRL metadata (lives in a display:none block)
    for tag in soup.find_all(lambda t: t.name == "ix:header"):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        if not tag.decomposed:
            tag.decompose()

    for table in soup.find_all("table"):
        table.replace_with(soup.new_string(table_to_markdown(table)))

    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(BLOCK_TAGS):        # newline only at block boundaries
        block.append("\n")

    text = soup.get_text("").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()