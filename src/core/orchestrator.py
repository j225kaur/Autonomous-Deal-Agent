# src/core/orchestrator.py
"""
LangGraph supervisor: wires DataAgent → Retrieve → AnalysisAgent → ReportAgent.

- BM25 mode (default): uses docs just ingested by DataAgent from state["ingested_docs"]
  so it requires NO HuggingFace downloads.
- FAISS mode: uses VectorStore (embeddings) for similarity search; set RETRIEVER_MODE=faiss.
"""

from __future__ import annotations
from typing import Dict, Any, List
import os
import time
from api.metrics import metrics

from langgraph.graph import StateGraph, END

from src.core.state import GraphState
from src.pipeline.data_collector import DataCollector
from src.pipeline.deal_analyzer import DealAnalyzer
from src.pipeline.report_generator import ReportGenerator
from src.pipeline.report_generator import ReportGenerator
from src.storage.stores import FAISSVectorStore


DEAL_QUERY = (
    "merger OR acquisition OR acquire OR acquiring OR acquired OR buyout OR "
    "takeover OR 'business combination' OR SPAC OR 'tender offer' OR 'definitive agreement' "
    "OR divestiture OR spin-off"
)


def _build_queries(tickers: List[str]) -> List[str]:
    """Simple multi-query: ticker + deal terms."""
    if not tickers:
        return [DEAL_QUERY]
    return [f"{t} {DEAL_QUERY}" for t in tickers]


def retrieve_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populate state['retrieved_docs'] with small, serializable dicts:
      [{"page_content": "...", "metadata": {...}}, ...]

    BM25 mode:
      - Pulls from in-memory docs saved by DataAgent in state['ingested_docs'].
      - Prefers docs flagged as deal-ish (metadata.is_dealish = True).

    FAISS mode:
      - Uses persisted VectorStore similarity_search over the index at INDEX_DIR.
    """
    cfg = state.get("config", {}) or {}
    top_k = int(cfg.get("top_k", 8))
    tickers = cfg.get("tickers", []) or []

    mode = os.getenv("RETRIEVER_MODE", "bm25").lower()
    index_dir = os.getenv("INDEX_DIR", "embeddings/faiss")
    queries = _build_queries(tickers)

    cleaned: List[Dict[str, Any]] = []

    if mode == "bm25":
        ingested = state.get("ingested_docs", []) or []
        if ingested:
            news_like = [
                d for d in ingested
                if d.get("metadata", {}).get("source") in {"yahoo_news", "sec"}
            ]
            dealish = [d for d in ingested if d.get("metadata", {}).get("is_dealish")]
            non_price = [d for d in ingested if d.get("metadata", {}).get("source") != "price_snapshot"]
            selection = dealish or news_like or non_price
            cleaned = selection[:top_k]
        else:
            cleaned = []
    else:
        # FAISS/vector mode
        vs = FAISSVectorStore(index_dir=index_dir)
        retrieved = []
        for q in queries:
            docs = vs.search(q, k=top_k)
            retrieved.extend(docs)

        # de-dup & serialize
        seen = set()
        for d in retrieved:
            meta = getattr(d, "metadata", {}) or {}
            key = (getattr(d, "page_content", "") or "", meta.get("link"))
            if key in seen:
                continue
            seen.add(key)
            cleaned.append({
                "page_content": getattr(d, "page_content", "") or "",
                "metadata": dict(meta) if isinstance(meta, dict) else {},
            })

    state["retrieved_docs"] = cleaned[:top_k] if cleaned else []
    state["retriever_info"] = {"mode": mode, "queries": queries, "hits": len(cleaned)}
    return state


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("data", DataCollector())
    graph.add_node("retrieve", retrieve_step)   # <— inserted retrieve node
    graph.add_node("analysis", DealAnalyzer())
    graph.add_node("report", ReportGenerator())

    # Edges
    graph.set_entry_point("data")
    graph.add_edge("data", "retrieve")
    graph.add_edge("retrieve", "analysis")
    graph.add_edge("analysis", "report")
    graph.add_edge("report", END)
    return graph


def run_once(config: Dict[str, Any]) -> GraphState:
    """
    Run a single pass through the pipeline. `config` supports keys like:
      - tickers (list[str] or comma string)
      - top_k (int), news_limit (int), use_sec (bool), ciks (list[str])
      - enable_checkpointer (bool)
    """
    # normalize comma strings to list
    if isinstance(config.get("tickers"), str):
        config["tickers"] = [t.strip() for t in config["tickers"].split(",") if t.strip()]

    initial: GraphState = {
        "config": config,
        "raw_items": {},
        "documents_added": 0,
        "ingested_docs": [],         # filled by DataAgent
        "retrieved_docs": [],        # filled by retrieve_step
        "findings": {},
        "report": {},
    }

    if config.get("enable_checkpointer"):
        from langgraph.checkpoint.memory import MemorySaver
        app = build_graph().compile(checkpointer=MemorySaver())
    else:
        app = build_graph().compile()

    start_time = time.time()
    metrics.inc("pipeline_runs_total")
    try:
        result = app.invoke(initial)
        duration = time.time() - start_time
        metrics.observe("pipeline_run_duration_seconds", duration)
        return result
    except Exception as e:
        metrics.inc("pipeline_errors_total")
        raise e
