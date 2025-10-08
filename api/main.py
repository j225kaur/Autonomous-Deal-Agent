# api/main.py
"""
FastAPI endpoint to manually trigger the supervisor-run workflow.
"""

from fastapi import FastAPI
from orchestrator.supervisor import Supervisor

app = FastAPI(title="Deal Intelligence Agents")
supervisor = Supervisor()

@app.get("/run_report")
def run_report():
    config = {"tickers": ["AAPL", "MSFT", "GOOGL"], "use_sec": False, "top_k": 20}
    result = supervisor.run_once(config)
    return {"report": result["report"]}


from __future__ import annotations
from typing import Optional, Dict, Any
import os

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import uvicorn

class RunConfig(BaseModel):
    tickers: Optional[str] = Field(default=None, description="Comma-separated tickers")
    use_sec: Optional[bool] = Field(default=False)
    news_limit: Optional[int] = Field(default=15)
    top_k: Optional[int] = Field(default=20)

app = FastAPI(title="Deal Intelligence Agent System")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/run_report")
def run_report(cfg: RunConfig = Body(default=RunConfig())) -> Dict[str, Any]:
    # Build config dict, overriding env defaults if provided
    config = {
        "tickers": (cfg.tickers.split(",") if cfg.tickers else os.environ.get("TICKERS", "AAPL,MSFT,GOOGL").split(",")),
        "use_sec": cfg.use_sec if cfg.use_sec is not None else bool(os.environ.get("USE_SEC", "false").lower() == "true"),
        "news_limit": cfg.news_limit or int(os.environ.get("NEWS_LIMIT", 15)),
        "top_k": cfg.top_k or int(os.environ.get("TOP_K", 20)),
        "index_dir": os.environ.get("INDEX_DIR", "embeddings/faiss_deals"),
    }
    final_state = run_once(config)
    return {
        "report_json": final_state["report"]["json"],
        "report_text": final_state["report"]["text"],
        "raw_ingestion": final_state["raw_items"],
        "documents_added": final_state["documents_added"],
        "retrieved_count": len(final_state["retrieved_docs"]),
    }

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)
