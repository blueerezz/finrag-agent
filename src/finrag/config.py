from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openai_api_key: str
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    sec_user_agent: str = ""
    data_dir: Path = ROOT / "data"
    hybrid_alpha: float = 0.7      # 0.0 = pure sparse, 1.0 = pure dense
    hybrid_pool: int = 20          # top-N from each retriever before fusion
    hyde_enabled: bool = False               # default off — costs 1 LLM call
    query_expansion_enabled: bool = False    # default off — costs 1 LLM call
    expansion_variants: int = 3              # how many rewrites to generate


settings = Settings()