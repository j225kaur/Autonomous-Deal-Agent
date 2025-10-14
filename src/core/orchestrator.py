"""
LangGraph supervisor: wires DataAgent → AnalysisAgent → ReportAgent
"""
from __future__ import annotations
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from src.core.state import GraphState
from src.agents.data_agent import DataAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.report_agent import ReportAgent
import os
from src.retriever.store import VectorStore

DEAL_QUERY = (
    "merger OR acquisition OR acquire OR acquiring OR acquired OR buyout OR "
    "takeover OR 'business combination' OR SPAC OR 'tender offer' OR 'definitive agreement' "
    "OR divestiture OR spin-off"
)

def _build_queries(tickers: list[str]) -> list[str]:
    # simple multi-query: ticker + deal terms
    return [f"{t} {DEAL_QUERY}" for t in tickers]

def retrieve_step(state: dict) -> dict:
    cfg = state.get("config", {})
    top_k = int(cfg.get("top_k", 8))
    tickers = cfg.get("tickers", [])
    index_dir = os.getenv("INDEX_DIR", "embeddings/faiss")

    vs = VectorStore(index_dir=index_dir)
    queries = _build_queries(tickers) or [DEAL_QUERY]

    retrieved = []
    for q in queries:
        docs = vs.search(q, k=top_k)
        retrieved.extend(docs)

    # de-dup a bit by (content, link)
    seen = set()
    cleaned = []
    for d in retrieved:
        meta = getattr(d, "metadata", {})
        key = (getattr(d, "page_content", ""), meta.get("link"))
        if key in seen: 
            continue
        seen.add(key)
        # store small dicts to avoid pydantic serialization issues later
        cleaned.append({
            "page_content": getattr(d, "page_content", ""),
            "metadata": dict(meta) if isinstance(meta, dict) else {},
        })

    state["retrieved_docs"] = cleaned[:top_k] or []
    state["retriever_info"] = {"queries": queries, "hits": len(cleaned)}
    return state


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("data", DataAgent())
    graph.add_node("analysis", AnalysisAgent())
    graph.add_node("report", ReportAgent())

    graph.set_entry_point("data")
    graph.add_edge("data", "analysis")
    graph.add_edge("analysis", "report")
    graph.add_edge("report", END)
    return graph

def run_once(config: Dict[str, Any]) -> GraphState:
    initial: GraphState = {
        "config": config,
        "raw_items": {},
        "documents_added": 0,
        "retrieved_docs": [],
        "findings": {},
        "report": {},
    }
    # By default we compile without a checkpointer to keep runs simple and
    # avoid requiring checkpoint-related config keys (thread_id, checkpoint_ns, ...).
    # If you want to enable checkpointing, set `config['enable_checkpointer']=True`
    # and the graph will be compiled with an in-memory saver.
    if config.get("enable_checkpointer"):
        from langgraph.checkpoint.memory import MemorySaver
        app = build_graph().compile(checkpointer=MemorySaver())
    else:
        app = build_graph().compile()

    return app.invoke(initial)
