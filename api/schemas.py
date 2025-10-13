from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class RunConfig(BaseModel):
    tickers: Optional[str] = Field(default=None, description="Comma-separated tickers")
    use_sec: Optional[bool] = Field(default=False)
    top_k: Optional[int] = Field(default=20)

class RunResponse(BaseModel):
    report: Dict[str, Any]
    findings: Dict[str, Any]
    documents_added: int
    raw_items: Dict[str, Any]
