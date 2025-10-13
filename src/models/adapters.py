import os

class _NullChat:
    model_name = "disabled"
    def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.2) -> str:
        raise RuntimeError("CHAT_MODEL=disabled (LLM calls are off)")

def get_chat_model():
    """
    Returns a small chat adapter or None when disabled.
    Supported: OpenAI-compatible endpoint (e.g., Groq).
    """
    spec = os.getenv("CHAT_MODEL", "disabled")
    if spec == "disabled":
        return None

    if spec.startswith("api:"):  # OpenAI-compatible (Groq, etc.)
        try:
            from langchain_openai import ChatOpenAI
        except Exception:
            return None
        base_url = os.getenv("OPENAI_BASE_URL")  # e.g. https://api.groq.com/openai/v1
        model = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instant")
        kwargs = {"temperature": 0.2}
        if base_url:
            kwargs["base_url"] = base_url
        llm = ChatOpenAI(model=model, **kwargs)
        class _OpenAICompat:
            model_name = f"api:{model}"
            def generate(self, prompt: str, max_tokens: int = 300, temperature: float = 0.2) -> str:
                # max_tokens/temperature may be ignored by some clients; kept for API parity
                resp = llm.invoke(prompt)
                return resp.content
        return _OpenAICompat()

    # You can add ollama:/llamacpp: branches later.
    return None
