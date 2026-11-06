from openai import OpenAI

from finrag.config import settings


def get_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)