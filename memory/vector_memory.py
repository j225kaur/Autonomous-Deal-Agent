"""
Wrapper for long-term memory in FAISS.
Each agent can upsert, search, and retrieve its own embeddings.
"""

from typing import List
from langchain_core.documents import Document
from retriever.store import build_or_load_index, as_retriever, DEFAULT_INDEX_DIR


class VectorMemory:
    def __init__(self, agent_name: str, index_dir: str = DEFAULT_INDEX_DIR):
        self.agent_name = agent_name
        self.index_dir = f"{index_dir}/{agent_name}"

    def upsert(self, docs: List[Document]):
        """Add or update knowledge for long-term recall."""
        build_or_load_index(docs, index_dir=self.index_dir)

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """Retrieve relevant memories."""
        retriever = as_retriever(k=k, index_dir=self.index_dir)
        return retriever.invoke(query)
