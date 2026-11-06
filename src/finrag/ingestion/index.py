"""Build and persist a Chroma index over chunked 10-K markdown.

Idempotent: build_index() with force=False skips if a full index exists.
Partial builds (docs=, tickers=, or limit=) always run.

Filtering options:
  docs=     exact doc_id stems, e.g. ["NVDA_10K_2019-01-27", ...]
  tickers=  ticker symbols, e.g. ["NVDA", "PEP"]
  limit=    max docs after other filters

The CLI wrapper in scripts/build_index.py exposes these as flags.
"""
from __future__ import annotations

import re
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from finrag.config import settings
from finrag.embeddings import embed_texts
from finrag.ingestion.chunk import Chunk, chunk_document, doc_info

COLLECTION_NAME = "chunks"
BATCH_SIZE = 500

_SECTION_CODE_RE = re.compile(r"Item ([\dA-Z]+)\.")


def _section_code(section: str) -> str:
    m = _SECTION_CODE_RE.match(section)
    return m.group(1) if m else ""


def _chunk_to_record(c: Chunk) -> tuple[str, str, dict, str]:
    meta = {
        "doc_id": c.doc_id,
        "ticker": c.ticker,
        "form": c.form,
        "fiscal_year": c.fiscal_year,
        "period_end": c.period_end,
        "section": c.section,
        "section_code": _section_code(c.section),
        "kind": c.kind,
        "line_start": c.line_start,
        "line_end": c.line_end,
        "context": c.context,
    }
    return c.chunk_id, c.embed_text, meta, c.text


def _client(index_dir: Path) -> chromadb.PersistentClient:
    index_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(index_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def load_collection(index_dir: Path | None = None):
    """Return the Chroma collection. Creates it (empty) if missing."""
    index_dir = index_dir or (settings.data_dir / "index")
    client = _client(index_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_stats(index_dir: Path | None = None) -> dict:
    """Return {'exists': bool, 'chunks': int, 'path': str}. Never builds."""
    index_dir = index_dir or (settings.data_dir / "index")
    sqlite_path = index_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        return {"exists": False, "chunks": 0, "path": str(index_dir)}
    try:
        client = _client(index_dir)
        coll = client.get_collection(COLLECTION_NAME)
        n = coll.count()
        return {"exists": n > 0, "chunks": n, "path": str(index_dir)}
    except Exception:
        return {"exists": False, "chunks": 0, "path": str(index_dir)}


def build_index(
    parsed_dir: Path | None = None,
    index_dir: Path | None = None,
    force: bool = False,
    docs: list[str] | None = None,
    tickers: list[str] | None = None,
    limit: int | None = None,
) -> int:
    """Build the index. Skips if a FULL index exists unless force=True.

    docs / tickers / limit are partial-build filters. When any is set,
    the idempotency gate is bypassed so you can add docs incrementally.

    Returns the chunk count in the collection after the run.
    """
    parsed_dir = parsed_dir or (settings.data_dir / "parsed")
    index_dir = index_dir or (settings.data_dir / "index")

    is_partial = bool(docs) or bool(tickers) or (limit is not None)

    if not force and not is_partial:
        stats = index_stats(index_dir)
        if stats["exists"]:
            print(f"index already built: {stats['chunks']} chunks at {stats['path']}")
            print("  (use --force to rebuild, or a partial filter to extend)")
            return stats["chunks"]

    paths = sorted(parsed_dir.glob("*/*.md"))
    if not paths:
        raise SystemExit(f"no .md files under {parsed_dir}")

    if docs:
        wanted = set(docs)
        paths = [p for p in paths if p.stem in wanted]
        if not paths:
            raise SystemExit(f"no .md files matched docs={docs}")

    if tickers:
        wanted_t = {t.upper() for t in tickers}
        paths = [p for p in paths if p.parent.name.upper() in wanted_t]
        if not paths:
            raise SystemExit(f"no .md files matched tickers={tickers}")

    if limit is not None:
        paths = paths[:limit]

    if not paths:
        raise SystemExit("filters matched 0 docs")

    print(f"building index over {len(paths)} doc(s)")

    client = _client(index_dir)

    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    for path in paths:
        chunks = chunk_document(path.read_text(encoding="utf-8"), doc_info(path))
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            records = [_chunk_to_record(c) for c in batch]
            coll.upsert(
                ids=[r[0] for r in records],
                embeddings=embed_texts([r[1] for r in records]),
                metadatas=[r[2] for r in records],
                documents=[r[3] for r in records],
            )
            total += len(batch)
        print(f"  {path.stem:28} {len(chunks):5d} chunks", flush=True)

    print(f"\nindexed {total} chunks across {len(paths)} docs")
    return total
