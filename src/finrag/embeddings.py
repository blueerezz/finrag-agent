"""Sentence embedding via fastembed (BGE-small).

parallel=None runs the ONNX model in the main process — no child
processes, no multiprocessing bootstrapping issues on macOS/Python 3.12.
This is the only mode that works reliably from a plain script without
a __main__ guard, and it uses all ONNX intra-op threads by default.
"""
from fastembed import TextEmbedding

from finrag.config import settings

_model: TextEmbedding | None = None


def get_embedder() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=settings.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    # parallel=None  -> main process, no child processes (the safe default)
    # batch_size=128 -> good balance of memory vs throughput
    vectors = get_embedder().embed(texts, batch_size=128, parallel=None)
    return [v.tolist() for v in vectors]
