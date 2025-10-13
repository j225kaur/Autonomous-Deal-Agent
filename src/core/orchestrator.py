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
