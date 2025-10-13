"""
Long-term memory wrapper over FAISS.
"""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from src.retriever.store import build_or_load_index, as_retriever, DEFAULT_INDEX_DIR

class VectorMemory:
    def __init__(self, namespace: str, index_dir: str = DEFAULT_INDEX_DIR):
        # Namespace allows per-agent isolation: e.g., embeddings/faiss/analysis_agent
        self.path = f"{index_dir}/{namespace}"

    def upsert(self, docs: List[Document]) -> None:
        build_or_load_index(docs, index_dir=self.path)

    def retriever(self, k: int = 12):
        return as_retriever(k=k, index_dir=self.path)
    