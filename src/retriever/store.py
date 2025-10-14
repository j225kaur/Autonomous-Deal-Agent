"""
FAISS vector store utilities + embeddings.
Env:
  USE_OPENAI_EMBEDDINGS=true to switch to OpenAI embeddings (requires API key)
  INDEX_DIR=embeddings/faiss
"""
from __future__ import annotations
from typing import List
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def _emb_model_name() -> str:
    # keep it small/fast
    return os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

class VectorStore:
    def __init__(self, index_dir: str = None):
        self.index_dir = index_dir or os.getenv("INDEX_DIR", "embeddings/faiss")
        self.emb = HuggingFaceEmbeddings(model_name=_emb_model_name())
        self.vs = None
        if os.path.isdir(self.index_dir):
            try:
                self.vs = FAISS.load_local(self.index_dir, self.emb, allow_dangerous_deserialization=True)
            except Exception:
                self.vs = None

    def upsert(self, docs: List):
        if not docs:
            return 0
        if self.vs is None:
            self.vs = FAISS.from_documents(docs, self.emb)
        else:
            self.vs.add_documents(docs)
        self.vs.save_local(self.index_dir)
        return len(docs)

    def search(self, query: str, k: int = 8):
        if self.vs is None:
            # lazy load in case this instance was created before ingest
            if os.path.isdir(self.index_dir):
                try:
                    self.vs = FAISS.load_local(self.index_dir, self.emb, allow_dangerous_deserialization=True)
                except Exception:
                    pass
        if self.vs is None:
            return []
        return self.vs.similarity_search(query, k=k)


USE_OPENAI = os.environ.get("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
DEFAULT_INDEX_DIR = os.environ.get("INDEX_DIR", "embeddings/faiss")

if USE_OPENAI:
    from langchain_openai import OpenAIEmbeddings
    def EMB():
        return OpenAIEmbeddings(model=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
else:
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    def EMB():
        return SentenceTransformerEmbeddings(model_name=os.environ.get("SENTENCE_TX_MODEL", "all-MiniLM-L6-v2"))

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def build_or_load_index(documents: List[Document], index_dir: str = DEFAULT_INDEX_DIR) -> FAISS:
    ensure_dir(index_dir)
    emb = EMB()
    try:
        vs = FAISS.load_local(index_dir, emb)
    except Exception:
        # build a fresh index
        if documents:
            vs = FAISS.from_documents(documents, emb)
        else:
            vs = FAISS.from_texts(["init"], embedding=emb, metadatas=[{"init": True}])
            # optionally delete the init doc after save
    vs.save_local(index_dir)
    return vs

def as_retriever(k: int = 12, index_dir: str = DEFAULT_INDEX_DIR):
    emb = EMB()
    try:
        vs = FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
    except Exception:
        # If the index doesn't exist yet (common on first run or in CI),
        # create a minimal index so callers can still perform searches.
        ensure_dir(index_dir)
        # create a tiny placeholder index
        vs = FAISS.from_texts(["init"], embedding=emb, metadatas=[{"init": True}])
        vs.save_local(index_dir)
    return vs.as_retriever(search_kwargs={"k": k})
