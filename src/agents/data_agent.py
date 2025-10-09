"""
DataAgent
- Fetches simple ticker context + mock headlines (replace with real sources later)
- Converts to Documents and upserts into long-term VectorMemory (FAISS)
- Writes a short note to Redis short-term memory
"""
from __future__ import annotations
from typing import Dict, Any, List
from datetime import datetime, timezone

from langchain_core.documents import Document
from agents.base import BaseAgent
from memory.redis_memory import RedisMemory
from memory.vector_memory import VectorMemory
from retriever.ingest import build_documents_from_sources
from utils.io import get_logger

log = get_logger(__name__)

class DataAgent(BaseAgent):
    def __init__(self):
        super().__init__("data_agent")
        self.short = RedisMemory(self.name, max_entries=50)
        self.long = VectorMemory(self.name)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.before_run(state)
        cfg = state.get("config", {})
        tickers: List[str] = cfg.get("tickers", ["AAPL", "MSFT", "NVDA"])

        # Minimal mock ingestion — replace with yfinance/SEC/RSS fetchers.
        news = [{"ticker": t, "title": f"Market update about {t}"} for t in tickers]
        docs: List[Document] = build_documents_from_sources(news)

        # Persist to FAISS (long-term memory)
        self.long.upsert(docs)

        # Update shared state
        state["raw_items"] = {"tickers": tickers, "news_count": len(news)}
        state["documents_added"] = len(docs)

        # Short-term memory note
        self.short.add({
            "ts": datetime.now(timezone.utc).isoformat(),
            "note": f"Ingested {len(docs)} docs for {len(tickers)} tickers."
        })

        self.after_run(state)
        return state
