"""OpenAI chat client wrapper (uses settings.openai_*)."""
from openai import OpenAI

from finrag.config import settings

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        _client = OpenAI(**kwargs)
    return _client


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    """One-shot chat call. Returns the assistant's message text."""
    resp = get_client().chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""
