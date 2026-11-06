import argparse
from collections import Counter

from finrag.config import settings
from finrag.ingestion.chunk import chunk_document, doc_info

p = argparse.ArgumentParser()
p.add_argument("doc_id", help="e.g. GOOGL_10K_2024-12-31")
p.add_argument("--find", help="print the first chunk containing this text")
args = p.parse_args()

path = next((settings.data_dir / "parsed").glob(f"*/{args.doc_id}.md"))
chunks = chunk_document(path.read_text(encoding="utf-8"), doc_info(path))


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


print(f"{len(chunks)} chunks:", dict(Counter(c.kind for c in chunks)))
print("\nchunks per section:")
for sec, n in Counter(c.section for c in chunks).items():
    print(f"  {n:5d}  {sec}")
for must in ("Item 7.", "Item 8."):
    if not any(c.section.startswith(must) for c in chunks):
        print(f"WARNING: no chunks in {must}  <-- section detection failed")

chars = [len(c.embed_text) for c in chunks]
print(f"\nchars: p50={pct(chars, .5)} p95={pct(chars, .95)} max={max(chars)}")

try:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    tok.no_truncation()  # we want the TRUE length, not the truncated one
    toks = [len(tok.encode(c.embed_text).ids) for c in chunks]
    over = sum(t > 512 for t in toks)
    print(f"tokens: p50={pct(toks, .5)} p95={pct(toks, .95)} max={max(toks)}")
    print(f"chunks over 512 tokens (silently truncated): {over} ({over / len(chunks):.1%})")
except Exception as e:  # network or tokenizer unavailable
    print("(token check skipped:", e, ")")

if args.find:
    hit = next((c for c in chunks if args.find in c.text), None)
    print("\n=== chunk containing", repr(args.find), "===")
    if hit:
        print(f"lines {hit.line_start}-{hit.line_end}, kind={hit.kind}, section={hit.section}\n")
        print(hit.embed_text)
    else:
        print("NOT FOUND")