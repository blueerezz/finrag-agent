"""Pydantic models for the /ask and /search endpoints."""
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    ticker: str | None = None
    section: str | None = None
    kind: str | None = Field(default=None, pattern="^(text|table)$")
    hyde: bool = False
    expand: bool = False
    k: int = Field(default=6, ge=1, le=20)


class Source(BaseModel):
    doc_id: str
    line_start: int
    line_end: int
    section: str | None = None
    score: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    features: dict
    queries: list[str]
    hypothetical: str | None = None
    hit_count: int
    low_confidence: bool


class SearchHit(BaseModel):
    chunk_id: str
    ticker: str
    fiscal_year: int
    section_code: str
    kind: str
    line_start: int
    line_end: int
    score: float | None
    text: str


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    features: dict
