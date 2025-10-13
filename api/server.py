"""
FastAPI server exposing:
  GET /health
  POST /run_report
  GET /visualize (DOT of workflow)
"""
from __future__ import annotations
from typing import Dict, Any
import networkx as nx
from fastapi import FastAPI, Body
from api.schemas import RunConfig, RunResponse
from src.core.orchestrator import run_once
from src.utils.io import load_config

app = FastAPI(title="Autonomous Deal Intelligence Agent")

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}

@app.post("/run_report", response_model=RunResponse)
def run_report(cfg: RunConfig = Body(default=RunConfig())) -> Dict[str, Any]:
    base_cfg = load_config()
    if cfg.tickers:
        base_cfg["tickers"] = [t.strip() for t in cfg.tickers.split(",")]
    base_cfg["use_sec"] = bool(cfg.use_sec)
    base_cfg["top_k"] = int(cfg.top_k)

    final = run_once(base_cfg)
    return {
        "report": final["report"],
        "findings": final["findings"],
        "documents_added": final["documents_added"],
        "raw_items": final["raw_items"],
    }

@app.get("/visualize")
def visualize() -> Dict[str, str]:
    G = nx.DiGraph()
    G.add_edges_from([("DataAgent","AnalysisAgent"), ("AnalysisAgent","ReportAgent")])
    dot = "digraph G { rankdir=LR; DataAgent -> AnalysisAgent; AnalysisAgent -> ReportAgent; }"
    return {"dot": dot}
