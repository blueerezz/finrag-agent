from finrag.config import settings
from finrag.llm import get_client

client = get_client()
chat = client.chat.completions.create(
    model=settings.llm_model,
    messages=[{"role": "user", "content": "Say hi in 5 languages!."}],
)
print("CHAT OK:", chat.choices[0].message.content)