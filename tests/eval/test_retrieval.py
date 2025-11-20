import pytest
from src.storage.stores import InMemoryVectorStore
from langchain_core.documents import Document

def test_retrieval_precision():
    # Setup
    store = InMemoryVectorStore()
    docs = [
        Document(page_content="Apple acquires AI startup for $200M", metadata={"ticker": "AAPL"}),
        Document(page_content="Microsoft releases new Surface laptop", metadata={"ticker": "MSFT"}),
        Document(page_content="Google in talks to buy Cybersecurity firm", metadata={"ticker": "GOOG"}),
        Document(page_content="Amazon stock falls on earnings miss", metadata={"ticker": "AMZN"}),
    ]
    store.upsert(docs)

    # Test 1: "acquires"
    results = store.search("acquires", k=2)
    assert len(results) >= 1
    assert "Apple" in results[0].page_content

    # Test 2: "buy"
    results = store.search("buy", k=2)
    assert len(results) >= 1
    assert "Google" in results[0].page_content

def test_retrieval_recall():
    store = InMemoryVectorStore()
    docs = [
        Document(page_content="Merger announced between A and B"),
        Document(page_content="A and B sign merger agreement"),
        Document(page_content="C releases new product"),
    ]
    store.upsert(docs)

    results = store.search("merger", k=5)
    # Should find both merger docs
    found = [d.page_content for d in results if "merger" in d.page_content.lower()]
    assert len(found) == 2
