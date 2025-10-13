# src/models/summarizers.py
from __future__ import annotations
import os, re
from typing import List, Optional

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except Exception:  # allow running without ST (CI/offline)
    np = None
    SentenceTransformer = None

# Optional: use the chat model for polishing if requested
try:
    from src.models.adapters import get_chat_model
except Exception:
    def get_chat_model(): return None


def _sent_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    # Simple sentence split; good enough for headlines/short docs
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    # keep reasonable lengths
    return [s.strip() for s in parts if 30 <= len(s) <= 300]


def _mmr_select(embeds: "np.ndarray", k: int = 5, diversity: float = 0.6) -> List[int]:
    """
    Simple Maximal Marginal Relevance over normalized embeddings.
    Returns indices of selected sentences.
    """
    if embeds.shape[0] <= k:
        return list(range(embeds.shape[0]))
    # centroid relevance
    centroid = embeds.mean(axis=0, keepdims=True)
    rel = (embeds @ centroid.T).ravel()  # cosine if normalized

    selected = [int(rel.argmax())]
    candidates = set(range(embeds.shape[0])) - set(selected)

    while len(selected) < k and candidates:
        best_idx, best_score = None, -1e9
        for i in candidates:
            # diversity = penalize similarity to already selected
            sim_to_sel = max((embeds[i] @ embeds[j]) for j in selected)
            score = (1 - diversity) * rel[i] - diversity * sim_to_sel
            if score > best_score:
                best_idx, best_score = i, score
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected


class ExtractiveSTSummarizer:
    """
    Extractive summarizer using sentence-transformers (default: all-MiniLM-L6-v2).
    Env:
      SUMMARY_MODEL=st:auto (default) or st:<model_name>
      SUMMARY_MAX_SENTENCES=5
    """
    def __init__(self, model_name: Optional[str] = None):
        if SentenceTransformer is None or np is None:
            raise RuntimeError("sentence-transformers (and numpy) not installed")
        self.model_name = model_name or os.getenv("SUMMARY_MODEL", "st:auto")
        if self.model_name in ("", "st:auto", "auto"):
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        elif self.model_name.startswith("st:"):
            self.model_name = self.model_name[3:]
        self.model = SentenceTransformer(self.model_name)

    def summarize(self, text: str, max_sentences: Optional[int] = None, max_chars: int = 1200) -> str:
        max_sentences = max_sentences or int(os.getenv("SUMMARY_MAX_SENTENCES", "5"))
        text = (text or "")[:max_chars]
        sents = _sent_split(text)
        if not sents:
            return text.strip()

        embeds = self.model.encode(sents, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        idxs = _mmr_select(embeds, k=min(max_sentences, len(sents)), diversity=0.6)
        idxs = sorted(idxs)  # keep original order for readability
        bullets = [f"• {sents[i]}" for i in idxs]
        return "\n".join(bullets)


class ChatSummarizer:
    """Optional polishing via chat model (Groq/OpenAI-compatible)."""
    def __init__(self):
        self.chat = get_chat_model()

    def summarize(self, text: str, max_sentences: int = 5, max_chars: int = 1200) -> str:
        if not self.chat:
            return text
        prompt = (
            "Rewrite the following into a concise {n}-bullet financial deal brief. "
            "Keep only concrete facts and figures. Output plain text bullets.\n\n---\n{body}"
        ).format(n=max_sentences, body=(text or "")[:max_chars])
        try:
            return (self.chat.generate(prompt, max_tokens=300, temperature=0.2) or "").strip()
        except Exception:
            return text


def get_summarizer():
    """
    Chooses summarizer by env:
      SUMMARY_MODEL=disabled/0/none    -> None (no summarization)
      SUMMARY_MODEL=chat               -> ChatSummarizer (uses get_chat_model)
      SUMMARY_MODEL=st:auto (default)  -> ExtractiveSTSummarizer with all-MiniLM-L6-v2
      SUMMARY_MODEL=st:<model_id>      -> ExtractiveSTSummarizer with given ST model
    """
    spec = os.getenv("SUMMARY_MODEL", "st:auto").strip().lower()
    if spec in ("", "disabled", "0", "none", "false"):
        return None
    if spec == "chat":
        return ChatSummarizer()
    # Any other value -> ST
    try:
        return ExtractiveSTSummarizer()
    except Exception:
        # If ST isn't available for any reason, degrade gracefully
        return None
