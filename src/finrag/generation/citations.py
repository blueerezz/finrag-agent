"""Extract and validate [doc_id:line_start-line_end] citations from LLM output."""
import re

# Accept with OR without .md extension. LLM citations match this pattern.
CITE_RE = re.compile(r"\[([A-Za-z0-9_\-]+?)(?:\.md)?:(\d+)-(\d+)\]")


def _normalize_doc_id(raw: str, retrieved_hits: list[dict]) -> str | None:
    """Match LLM's doc_id against metadata (with/without .md)."""
    if raw in {h["metadata"]["doc_id"] for h in retrieved_hits}:
        return raw
    with_md = f"{raw}.md"
    if with_md in {h["metadata"]["doc_id"] for h in retrieved_hits}:
        return with_md
    return None


def extract_citations(answer: str) -> list[dict]:
    seen = set()
    out = []
    for m in CITE_RE.finditer(answer):
        doc_id, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        key = (doc_id, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append({"doc_id": doc_id, "line_start": start, "line_end": end})
    return out


def validate_citations(
    answer_cites: list[dict], retrieved_hits: list[dict]
) -> tuple[list[dict], list[dict]]:
    valid, bad = [], []
    for c in answer_cites:
        normalized = _normalize_doc_id(c["doc_id"], retrieved_hits)
        if normalized is None:
            bad.append(c)
            continue
        match = next(
            (
                h
                for h in retrieved_hits
                if h["metadata"]["doc_id"] == normalized
                and h["metadata"]["line_start"] <= c["line_end"]
                and h["metadata"]["line_end"] >= c["line_start"]
            ),
            None,
        )
        if match:
            valid.append({**c, "doc_id": normalized,
                          "section": match["metadata"].get("section", "")})
        else:
            bad.append(c)
    return valid, bad


def format_sources(valid_cites: list[dict], retrieved_hits: list[dict]) -> str:
    if not valid_cites:
        return "(no citations)"
    lines = []
    for i, c in enumerate(valid_cites, 1):
        hit = next(
            (
                h
                for h in retrieved_hits
                if h["metadata"]["doc_id"] == c["doc_id"]
                and h["metadata"]["line_start"] <= c["line_end"]
                and h["metadata"]["line_end"] >= c["line_start"]
            ),
            None,
        )
        score = hit.get("rerank_score") if hit else None
        score_str = f"rerank {score:+.2f}" if score is not None else "—"
        lines.append(
            f"  {i}. {c['doc_id']}:{c['line_start']}-{c['line_end']}  "
            f"({c.get('section', '')}, {score_str})"
        )
    return "\n".join(lines)
