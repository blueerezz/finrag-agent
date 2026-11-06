"""Prompt templates for RAG-based financial Q&A."""

SYSTEM_PROMPT = """You are a financial analyst assistant answering questions about \
SEC 10-K filings. You have been given excerpts from real filings, each tagged with \
a citation ID of the form [doc_id:line_start-line_end].

Rules:
1. Answer ONLY from the provided context. If the answer is not present, say so.
2. Every factual claim MUST end with a citation like [NVDA_10K_2025-01-26.md:345-347].
3. When citing numbers, quote them exactly as they appear (do not round or reformat).
4. Be concise. 2-4 sentences unless the question requires more.
5. Do not invent citations. Only use citation IDs that appear in the context.
6. Some excerpts may appear to be about a related-but-different topic (e.g., deferred
   revenue when the question is about revenue). If a chunk contains the specific fact
   asked for, extract it. Do not refuse just because the surrounding text discusses
   something else.
7. The chunk header shows the source, not the quality. Trust the chunk body's content."""


def build_context_block(hits: list[dict]) -> str:
    """Format retrieved chunks into citable context. No confidence labels."""
    lines = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        cite = f"[{m['doc_id']}:{m['line_start']}-{m['line_end']}]"
        section = m.get("section", "Unknown")
        header = f"[#{i}] {cite}  ({section})"
        body = h["text"].strip()
        lines.append(f"{header}\n{body}")
    return "\n\n---\n\n".join(lines)


def build_user_prompt(query: str, hits: list[dict]) -> str:
    if not hits:
        return (
            f"Question: {query}\n\n"
            "No relevant context was retrieved for this question. "
            "Respond by saying you could not find relevant information "
            "in the available filings, and suggest a rephrased question."
        )
    return (
        f"Question: {query}\n\n"
        f"Context ({len(hits)} excerpts from 10-K filings):\n\n"
        f"{build_context_block(hits)}\n\n"
        "Answer the question using only the context above. "
        "Cite every factual claim with [doc_id:line_start-line_end]."
    )
