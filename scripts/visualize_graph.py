"""
Create a DOT of the pipeline and save to data/outputs/workflow.dot
"""
import os
import networkx as nx
from src.utils.io import ensure_dir, write_text

G = nx.DiGraph()
G.add_edges_from([("DataAgent","AnalysisAgent"),("AnalysisAgent","ReportAgent")])
dot = "digraph G { rankdir=LR; DataAgent -> AnalysisAgent; AnalysisAgent -> ReportAgent; }"
ensure_dir("data/outputs")
write_text("data/outputs/workflow.dot", dot)
print("Saved to data/outputs/workflow.dot")
