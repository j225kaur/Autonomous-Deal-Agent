"""
Typed state shared across agents (kept simple for clarity).
"""
from typing import Dict, Any, List, TypedDict

class GraphState(TypedDict):
    config: Dict[str, Any]
    raw_items: Dict[str, Any]
    documents_added: int
    retrieved_docs: List[Dict[str, Any]]
    findings: Dict[str, Any]
    report: Dict[str, Any]
