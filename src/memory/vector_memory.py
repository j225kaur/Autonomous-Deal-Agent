"""
Long-term memory wrapper over FAISS.
"""
from __future__ import annotations
from typing import List, Optional
import os

from langchain_core.documents import Document
from src.retriever.store import VectorStore

# Keep a single place for the default index dir
DEFAULT_INDEX_DIR = os.getenv("INDEX_DIR", "embeddings/faiss")


class VectorMemory:
    """
    Thin adapter over VectorStore so existing agents don't need to know
    which backend (BM25 vs FAISS) is in use.

    - RETRIEVER_MODE=bm25  (default; no HF downloads, in-memory)
    - RETRIEVER_MODE=faiss (needs embeddings model; persists to INDEX_DIR)
    """

    def __init__(self, namespace: str = "default", index_dir: Optional[str] = None):
        self.namespace = namespace
        self.index_dir = index_dir or DEFAULT_INDEX_DIR
        self._store = VectorStore(index_dir=self.index_dir)

    def upsert(self, docs: List[Document]) -> int:
        """Add documents to the store (in-memory for BM25, persisted for FAISS)."""
        return self._store.upsert(docs)

    def retriever(self, k: int = 8):
        """
        Return a retriever-like object with .invoke(query) -> List[Document].
        For BM25, this is an in-memory retriever; for FAISS, it proxies to vector store.
        """
        return self._store.retriever(k=k)

    def search(self, query: str, k: int = 8):
        """Optional convenience: directly run a search and get docs."""
        return self._store.search(query, k=k)
