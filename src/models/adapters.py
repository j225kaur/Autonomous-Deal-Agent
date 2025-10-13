# src/models/adapters.py
from __future__ import annotations
import os

def get_chat_model():
    """
    Returns a minimal chat adapter or None when disabled.
    Supports: OpenAI-compatible APIs (e.g., Groq) via langchain-openai.
      Env:
        CHAT_MODEL=api:openai  (enable) | disabled (default)
        OPENAI_API_KEY=...
        OPENAI_BASE_URL=https://api.groq.com/openai/v1
        OPENAI_MODEL=llama-3.1-8b-instant (or your choice)
    """
    spec = os.getenv("CHAT_MODEL", "disabled").strip().lower()
    if spec in ("", "disabled", "false", "0", "none"):
        return None

    if spec.startswith("api:"):  # OpenAI-compatible (Groq etc.)
        try:
            from langchain_openai import ChatOpenAI
        except Exception:
            return None

        base_url = os.getenv("OPENAI_BASE_URL")  # e.g. https://api.groq.com/openai/v1
        model = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instant")

        # ChatOpenAI reads OPENAI_API_KEY from env. Add base_url if present.
        kwargs = {"model": model, "temperature": 0.2}
        if base_url:
            kwargs["base_url"] = base_url

        llm = ChatOpenAI(**kwargs)

        class _OpenAICompat:
            model_name = f"api:{model}"
            def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
                # langchain-openai ignores max_tokens for some backends; still include for parity
                resp = llm.invoke(prompt)
                return getattr(resp, "content", "") or str(resp)

        return _OpenAICompat()

    # (Optional: add ollama:/llamacpp: support later)
    return None
