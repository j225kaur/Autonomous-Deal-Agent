from typing import List, Dict, Any
from langchain_core.documents import Document
from src.storage.base import VectorStore, ShortTermMemoryStore
from src.retriever.store import VectorStore as LegacyVectorStore
from src.memory.redis_memory import RedisMemory as LegacyRedisMemory

class FAISSVectorStore(VectorStore):
    """
    Wraps the existing VectorStore which handles FAISS (and BM25 fallback).
    """
    def __init__(self, index_dir: str = None):
        self._store = LegacyVectorStore(index_dir=index_dir)

    def upsert(self, documents: List[Document]) -> None:
        self._store.upsert(documents)

    def search(self, query: str, k: int = 10) -> List[Document]:
        return self._store.search(query, k=k)

class RedisMemoryStore(ShortTermMemoryStore):
    """
    Wraps the existing RedisMemory.
    """
    def __init__(self, agent_name: str, max_entries: int = 50):
        self._mem = LegacyRedisMemory(agent_name, max_entries=max_entries)

    def add(self, data: Dict[str, Any]) -> None:
        self._mem.add(data)

    def get(self, limit: int = 10) -> List[Dict[str, Any]]:
        # Legacy get() returns all, we slice if needed
        items = self._mem.get()
        return items[:limit]

class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory store for testing or small scale.
    Uses the same LegacyVectorStore but forces BM25 mode if possible,
    or we can just implement a simple list search.
    """
    def __init__(self):
        # For now, we can reuse LegacyVectorStore in BM25 mode
        # But to be "pure", let's just store docs and do simple search
        self.docs = []

    def upsert(self, documents: List[Document]) -> None:
        self.docs.extend(documents)

    def search(self, query: str, k: int = 10) -> List[Document]:
        # Naive keyword search
        results = []
        q_terms = query.lower().split()
        for d in self.docs:
            content = d.page_content.lower()
            score = sum(1 for t in q_terms if t in content)
            if score > 0:
                results.append((score, d))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:k]]
