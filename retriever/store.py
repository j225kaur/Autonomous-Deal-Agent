# retriever/store.py
"""
Vector store utilities (FAISS) for persisting and retrieving deal-related docs.
Defaults to sentence-transformers embeddings (no API key).
If OPENAI_API_KEY is set + USE_OPENAI_EMBEDDINGS=true, uses OpenAI embeddings.
"""

from __future__ import annotations
from typing import List
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Embeddings backends
USE_OPENAI = os.environ.get("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"

if USE_OPENAI:
    from langchain_openai import OpenAIEmbeddings
    EMB = lambda: OpenAIEmbeddings(model=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
else:
    # SentenceTransformersEmbeddings keeps deps light while avoiding external API calls
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    EMB = lambda: SentenceTransformerEmbeddings(model_name=os.environ.get("SENTENCE_TX_MODEL", "all-MiniLM-L6-v2"))

DEFAULT_INDEX_DIR = os.environ.get("INDEX_DIR", "embeddings/faiss_deals")

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def build_or_load_index(documents: List[Document], index_dir: str = DEFAULT_INDEX_DIR) -> FAISS:
    """
    Creates a FAISS index if not present, else loads and appends documents.
    """
    ensure_dir(index_dir)
    emb = EMB()
    if not documents:
        # Load existing or initialize empty
        try:
            return FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
        except Exception:
            # Create an empty index by adding a dummy doc then deleting it
            vs = FAISS.from_texts(["init"], embedding=emb, metadatas=[{"init": True}])
            vs.delete(vs.index_to_docstore_id[0])
            vs.save_local(index_dir)
            return vs

    # Try loading existing index
    try:
        vs = FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
        vs.add_documents(documents)
    except Exception:
        vs = FAISS.from_documents(documents, emb)

    vs.save_local(index_dir)
    return vs

def as_retriever(k: int = 12, index_dir: str = DEFAULT_INDEX_DIR):
    """
    Return a retriever callable (via FAISS.as_retriever).
    """
    emb = EMB()
    vs = FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": k})
