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
        vs = FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
        if documents:
            vs.add_documents(documents)
    except Exception:
        vs = FAISS.from_documents(documents or [Document(page_content="init", metadata={"init": True})], emb)
        if not documents:
            # delete init doc
            try:
                vs.delete(vs.index_to_docstore_id[0])
            except Exception:
                pass
    vs.save_local(index_dir)
    return vs

def as_retriever(k: int = 12, index_dir: str = DEFAULT_INDEX_DIR):
    emb = EMB()
    vs = FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": k})
