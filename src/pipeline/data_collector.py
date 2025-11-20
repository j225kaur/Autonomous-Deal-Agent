"""
DataAgent
- Fetches simple ticker context + mock headlines (replace with real sources later)
- Converts to Documents and upserts into long-term VectorMemory (FAISS)
- Writes a short note to Redis short-term memory
"""
from __future__ import annotations
from typing import Dict, Any, List
from datetime import datetime, timezone
import os, traceback
from langchain_core.documents import Document
from src.pipeline.base import Base
from src.storage.stores import FAISSVectorStore, RedisMemoryStore
from src.utils.io import get_logger

from src.retriever.ingest import (
    build_documents_from_sources,
    fetch_yahoo_news,
    fetch_prices_snapshot,
    fetch_sec_filings,
    fetch_price_history,
)
__all__ = ["build_documents_from_sources","fetch_yahoo_news","fetch_prices_snapshot","fetch_sec_filings"]
log = get_logger(__name__)

class DataCollector(Base):
    def __init__(self):
        super().__init__("data_agent")
        self.short = RedisMemoryStore(self.name, max_entries=50)
        self.long = FAISSVectorStore(index_dir=None)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.before_run(state)
        cfg = state.get("config", {})

        tickers: List[str] = cfg.get("tickers", ["AAPL", "MSFT", "NVDA"])
        news_limit: int = cfg.get("news_limit", 15)
        use_sec: bool = bool(cfg.get("use_sec", False))
        ciks: List[str] = cfg.get("ciks", [])
        offline: bool = os.getenv("OFFLINE_MODE", "false").lower() in {"1", "true", "yes"}

        yahoo_news, prices_ctx, sec_items, market_history = [], {}, [], {}
        try:
            if not offline:
                yahoo_news = fetch_yahoo_news(tickers, limit_per_ticker=news_limit)
                prices_ctx = fetch_prices_snapshot(tickers)
                market_history = fetch_price_history(tickers, days=30)
                sec_items = fetch_sec_filings(ciks) if (use_sec and ciks) else []
            else:
                yahoo_news = [
                    {"ticker": tickers[0], "title": f"{tickers[0]} enters definitive agreement to acquire XYZ", "link":"", "publisher":"offline", "published": 0},
                    {"ticker": tickers[1], "title": f"{tickers[1]} announces strategic transaction with ABC", "link":"", "publisher":"offline", "published": 0},
                ]
                prices_ctx = {t: {"last": 100.0, "chg5d": 0.02} for t in tickers}
                market_history = {t: {"close": [100.0]*30, "volume": [1000000]*30} for t in tickers}
        except Exception as e:
            log.error(f"[DataAgent] fetch error: {e}")
            log.debug(traceback.format_exc())

        docs: List[Document] = build_documents_from_sources(
            yahoo_news=yahoo_news,
            prices_ctx=prices_ctx,
            sec_items=sec_items or None,
        )

        if docs:
            self.long.upsert(docs)
        state["ingested_docs"] = [
            {
                "page_content": d.page_content,
               "metadata": dict(d.metadata) if isinstance(d.metadata, dict) else {},
            }
            for d in docs
        ]

        state["raw_items"] = {
            "tickers": tickers,
            "news_count": len(yahoo_news),
            "sec_count": len(sec_items),
            "market_history": market_history,
        }
        state["documents_added"] = len(docs)

        self.short.add({
            "ts": datetime.now(timezone.utc).isoformat(),
            "note": f"Ingested {len(docs)} docs (news={len(yahoo_news)}, sec={len(sec_items)})"
        })

        self.after_run(state)
        return state
