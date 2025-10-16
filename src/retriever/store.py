"""
FAISS vector store utilities + embeddings.
Env:
  USE_OPENAI_EMBEDDINGS=true to switch to OpenAI embeddings (requires API key)
  INDEX_DIR=embeddings/faiss
"""
from __future__ import annotations
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers.bm25 import BM25Retriever

# FAISS/HF are optional; import lazily only if used
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    FAISS = None
    HuggingFaceEmbeddings = None


class VectorStore:
    """
    RETRIEVER_MODE:
      - "bm25"  (default, no downloads)
      - "faiss" (requires HuggingFaceEmbeddings; will download a small ST model)
    """
    def __init__(self, index_dir: str | None = None):
        self.mode = os.getenv("RETRIEVER_MODE", "bm25").lower()
        self.index_dir = index_dir or os.getenv("INDEX_DIR", "embeddings/faiss")

        if self.mode == "bm25":
            self._docs: List[Document] = []
            self._bm25: BM25Retriever | None = None
            self.vs = None
            self.emb = None
        else:
            # FAISS path (optional)
            if not (FAISS and HuggingFaceEmbeddings):
                raise RuntimeError("FAISS/HF not available but RETRIEVER_MODE=faiss")
            model_name = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.emb = HuggingFaceEmbeddings(model_name=model_name)
            self._docs = []
            self._bm25 = None
            self.vs = None
            if os.path.isdir(self.index_dir):
                try:
                    self.vs = FAISS.load_local(self.index_dir, self.emb, allow_dangerous_deserialization=True)
                except Exception:
                    self.vs = None

    def upsert(self, docs: List[Document] | None) -> int:
        normalized: List[Document] = []
        for d in docs:
            if isinstance(d, Document):
                normalized.append(d)
            elif isinstance(d, dict):
                page_content = d.get("page_content") or d.get("text") or d.get("content") or json.dumps(d)
                metadata = d.get("metadata") or {}
                normalized.append(Document(page_content=page_content, metadata=metadata))
            elif isinstance(d, str):
                normalized.append(Document(page_content=d, metadata={}))
            else:
                # fallback: stringify unknown types
                normalized.append(Document(page_content=str(d), metadata={}))

        # append to store docs list
        self._docs.extend(normalized)

        # attempt to build/update BM25 retriever; if it fails, log and leave retriever None
        try:
            self._bm25 = BM25Retriever.from_documents(self._docs)
        except Exception as err:
            #logger.warning("BM25Retriever construction failed: %s. Falling back to no-BM25.", err)
            self._bm25 = None

    def retriever(self, k: int = 8):
        if self.mode == "bm25":
            class _R:
                def __init__(self, r: BM25Retriever | None, k: int):
                    self.r = r
                    self.k = k
                def invoke(self, query: str):
                    if not self.r:
                        return []
                    return self.r.get_relevant_documents(query)[: self.k]
            return _R(self._bm25, k)

        # FAISS retriever
        if self.vs is None and os.path.isdir(self.index_dir):
            try:
                self.vs = FAISS.load_local(self.index_dir, self.emb, allow_dangerous_deserialization=True)
            except Exception:
                self.vs = None
        return self.vs.as_retriever(search_kwargs={"k": k}) if self.vs else type("R", (), {"invoke": lambda _self, _q: []})()

    # Optional helper if you call search directly elsewhere
    def search(self, query: str, k: int = 8):
        return self.retriever(k=k).invoke(query)
