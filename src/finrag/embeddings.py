from fastembed import TextEmbedding

from finrag.config import settings

_model: TextEmbedding | None = None


def get_embedder() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=settings.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = get_embedder().embed(texts)
    return [v.tolist() for v in vectors]