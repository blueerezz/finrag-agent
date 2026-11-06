from finrag.config import settings

def test_settings_loaded():
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
