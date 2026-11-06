"""FastAPI service for FinRAG.

Endpoints:
  POST /ask     — full RAG (retrieve → rerank → LLM → cited answer)
  POST /search  — retrieval only (no LLM, no cost)
  GET  /health  — liveness

Run: uv run uvicorn finrag.api.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from finrag.api.schemas import (
    AskRequest, AskResponse, SearchHit, SearchResponse, Source,
)
from finrag.generation.citations import (
    extract_citations, validate_citations,
)
from finrag.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from finrag.llm import chat
from finrag.retrieval.pipeline import ask as pipeline_ask

app = FastAPI(title="FinRAG", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parents[3] / "static"
if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: AskRequest) -> SearchResponse:
    """Retrieval only — no LLM call, no cost."""
    result = pipeline_ask(
        req.question, alpha=req.alpha, ticker=req.ticker,
        section=req.section, kind=req.kind,
        hyde=req.hyde, expand=req.expand, k=req.k,
    )
    hits = [
        SearchHit(
            chunk_id=h["chunk_id"],
            ticker=h["metadata"]["ticker"],
            fiscal_year=h["metadata"]["fiscal_year"],
            section_code=h["metadata"]["section_code"],
            kind=h["metadata"]["kind"],
            line_start=h["metadata"]["line_start"],
            line_end=h["metadata"]["line_end"],
            score=h.get("rerank_score", h.get("score")),
            text=h["text"],
        )
        for h in result["hits"]
    ]
    return SearchResponse(hits=hits, features=result["features"])


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = pipeline_ask(
        req.question, alpha=req.alpha, ticker=req.ticker,
        section=req.section, kind=req.kind,
        hyde=req.hyde, expand=req.expand, k=req.k,
    )
    hits = result["hits"]
    if not hits:
        return AskResponse(
            answer="No relevant context found. Try rephrasing or relaxing filters.",
            sources=[], features=result["features"],
            queries=result["queries"], hypothetical=result["hypothetical"],
            hit_count=0, low_confidence=True,
        )

    answer = chat(SYSTEM_PROMPT, build_user_prompt(req.question, hits))
    cites = extract_citations(answer)
    valid, _ = validate_citations(cites, hits)

    sources = []
    for c in valid:
        hit = next(
            (h for h in hits
             if h["metadata"]["doc_id"] == c["doc_id"]
             and h["metadata"]["line_start"] <= c["line_end"]
             and h["metadata"]["line_end"] >= c["line_start"]),
            None,
        )
        sources.append(Source(
            doc_id=c["doc_id"], line_start=c["line_start"],
            line_end=c["line_end"], section=c.get("section"),
            score=hit.get("rerank_score") if hit else None,
        ))

    return AskResponse(
        answer=answer, sources=sources, features=result["features"],
        queries=result["queries"], hypothetical=result["hypothetical"],
        hit_count=len(hits),
        # Only flag low confidence if the fallback path ran AND no
        # citations survived validation. A correct answer with valid
        # citations overrides the retrieval uncertainty.
        low_confidence=(
            all(h.get("low_confidence") for h in hits) and not sources
        ),
    )
