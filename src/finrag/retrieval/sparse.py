"""BM25 sparse retrieval over the same chunks indexed in Chroma.

Corpus is loaded from the Chroma collection once and cached to a pickle
in data/index/ for fast subsequent loads. Cache is invalidated when
chroma.sqlite3 is newer than the cache file (i.e. after a rebuild).

API mirrors dense.py: sparse_search(query, k, where) -> same dict shape.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from finrag.config import settings
from finrag.ingestion.index import load_collection

CACHE_VERSION = 1
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _cache_fresh(cache_path: Path, index_dir: Path) -> bool:
    sqlite = index_dir / "chroma.sqlite3"
    if not (cache_path.exists() and sqlite.exists()):
        return False
    return cache_path.stat().st_mtime > sqlite.stat().st_mtime


def _build_cache(index_dir: Path) -> dict:
    print("building BM25 cache from Chroma (one-time, ~15-30s)...")
    coll = load_collection(index_dir)
    data = coll.get(include=["documents", "metadatas"])
    ids = data["ids"]
    texts = data["documents"]
    metas = data["metadatas"]

    # Tokenize context + body so ticker/section/year are searchable.
    tokens = [
        _tokenize(f"{m.get('context', '')} {t}")
        for m, t in zip(metas, texts)
    ]
    bm25 = BM25Okapi(tokens)

    cache = {"ids": ids, "texts": texts, "metas": metas, "tokens": tokens, "bm25": bm25}
    cache_path = index_dir / f"bm25_v{CACHE_VERSION}.pkl"
    with cache_path.open("wb") as f:
        pickle.dump(cache, f)
    print(f"  cached {len(ids)} chunks -> {cache_path}")
    return cache


_bm25_cache: dict | None = None


def _get_cache(index_dir: Path | None = None) -> dict:
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache

    index_dir = index_dir or (settings.data_dir / "index")
    cache_path = index_dir / f"bm25_v{CACHE_VERSION}.pkl"

    if _cache_fresh(cache_path, index_dir):
        with cache_path.open("rb") as f:
            _bm25_cache = pickle.load(f)
        return _bm25_cache

    _bm25_cache = _build_cache(index_dir)
    return _bm25_cache


def reset_cache() -> None:
    """Force reload from disk / rebuild. Useful in tests."""
    global _bm25_cache
    _bm25_cache = None


def _matches(meta: dict, where: dict | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_matches(meta, sub) for sub in where["$and"])
    if "$or" in where:
        return any(_matches(meta, sub) for sub in where["$or"])
    for key, cond in where.items():
        if key.startswith("$"):
            continue
        if isinstance(cond, dict):
            if "$in" in cond and meta.get(key) not in cond["$in"]:
                return False
            if "$eq" in cond and meta.get(key) != cond["$eq"]:
                return False
            if "$ne" in cond and meta.get(key) == cond["$ne"]:
                return False
        else:
            if meta.get(key) != cond:
                return False
    return True


def sparse_search(
    query: str,
    k: int = 20,
    where: dict | None = None,
    index_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """BM25 keyword search with optional metadata filter.

    Returns list of dicts with same shape as dense_search.
    score is normalized to [0,1] by dividing by max; bm25_raw is the
    raw score so callers can see the real magnitude.
    """
    cache = _get_cache(index_dir)
    bm25 = cache["bm25"]
    ids = cache["ids"]
    texts = cache["texts"]
    metas = cache["metas"]

    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)
    max_score = float(scores.max()) if len(scores) else 0.0
    if max_score <= 0:
        return []

    order = scores.argsort()[::-1]
    out: list[dict[str, Any]] = []
    for idx in order:
        if scores[idx] <= 0:
            break
        if not _matches(metas[idx], where):
            continue
        out.append({
            "chunk_id": ids[idx],
            "text": texts[idx],
            "metadata": metas[idx],
            "score": float(scores[idx]) / max_score,
            "bm25_raw": float(scores[idx]),
        })
        if len(out) >= k:
            break
    return out
