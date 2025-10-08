"""
Supervisor orchestrating memory-aware agents.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any

from agents.data_agent import DataAgent
from agents.analysis_agent import AnalysisAgent
from agents.report_agent import ReportAgent


class Supervisor:
    """Runs the full memory-enabled agent workflow."""

    def __init__(self):
        self.graph = StateGraph(dict)
        self._setup_graph()

    def _setup_graph(self):
        self.graph.add_node("data_agent", DataAgent())
        self.graph.add_node("analysis_agent", AnalysisAgent())
        self.graph.add_node("report_agent", ReportAgent())

        self.graph.set_entry_point("data_agent")
        self.graph.add_edge("data_agent", "analysis_agent")
        self.graph.add_edge("analysis_agent", "report_agent")
        self.graph.add_edge("report_agent", END)

        self.compiled = self.graph.compile(checkpointer=MemorySaver())

    def run_once(self, config: Dict[str, Any]):
        initial_state = {
            "config": config,
            "raw_items": {},
            "documents_added": 0,
            "retrieved_docs": [],
            "findings": {},
            "report": {},
        }
        return self.compiled.invoke(initial_state)
