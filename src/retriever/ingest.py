"""
Turn raw items into LangChain Documents.
`news_items` example: [{"ticker":"AAPL","title":"Apple acquires..."}]
"""
from __future__ import annotations
from typing import List, Dict, Any

from langchain_core.documents import Document
from src.data_ingestion.yahoo_sec import (
    build_documents_from_sources as _build_docs,
    fetch_yahoo_news,
    fetch_prices_snapshot,
    fetch_sec_filings,
    fetch_price_history,
)

build_documents_from_sources = _build_docs

__all__ = [
    "build_documents_from_sources",
    "fetch_yahoo_news",
    "fetch_prices_snapshot",
    "fetch_sec_filings",
    "fetch_price_history",
]
