from finrag.ingestion.parse import parse_10k_html

HTML = """<html><body><p>Alphabet Inc.</p><table>
<tr><td></td><td>2023</td><td></td><td>2024</td></tr>
<tr><td>Revenues</td><td>$</td><td>307,394</td><td>$</td><td>350,018</td></tr>
</table></body></html>"""


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip("|").split("|")]


def test_year_header_stays_aligned_with_data():
    out = parse_10k_html(HTML)
    lines = [l for l in out.splitlines() if l.startswith("|")]
    assert _cells(lines[0]) == ["", "2023", "2024"]
    assert _cells(lines[2]) == ["Revenues", "$307,394", "$350,018"]
def test_negative_numbers_have_no_inner_spaces():
    html = ("<html><body><table><tr><td>Other</td>"
            "<td><span>(</span><span>3,514</span><span>)</span></td></tr></table></body></html>")
    assert "(3,514)" in parse_10k_html(html)