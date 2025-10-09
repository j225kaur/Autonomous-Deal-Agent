"""
AnalysisAgent
- Retrieves context from VectorMemory
- Uses a local/API LLM adapter to extract deal trends (JSON-ish text)
- Stores a compact reflection in Redis and appends structured finding to FAISS
"""
from __future__ import annotations
from typing import Dict, Any, List
import json
import re

from langchain_core.documents import Document
from agents.base import BaseAgent
from memory.redis_memory import RedisMemory
from memory.vector_memory import VectorMemory
from models.adapters import get_chat_model

RULES_RX = re.compile(r"(merger|acquisition|buyout|takeover|spin[- ]?off|SPAC)", re.I)

class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("analysis_agent")
        self.short = RedisMemory(self.name, max_entries=50)
        self.long = VectorMemory(self.name)

    def _rule_based(self, texts: List[str]) -> Dict[str, Any]:
        hits = []
        for t in texts:
            if RULES_RX.search(t):
                hits.append(t[:240])
        return {"model": "rule-based", "items": hits}

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.before_run(state)
        cfg = state.get("config", {})
        retr = self.long.retriever(k=cfg.get("top_k", 20))
        docs = retr.invoke("Recent mergers, acquisitions, spin-offs or strategic deals")
        state["retrieved_docs"] = [d.dict() for d in docs]

        # Gather recent short-term context (last few notes)
        recent_notes = self.short.get()
        context_snip = "\n".join([m.get("note","") for m in recent_notes][-3:])
        texts = [d.page_content for d in docs]
        merged = (context_snip + "\n" + "\n".join(texts)).strip()

        # Try LLM via adapters; fallback to rule-based
        chat = get_chat_model()  # decides from ENV/CONFIG
        findings: Dict[str, Any]
        if chat is not None:
            prompt = (
                "You are a deal intelligence analyst.\n"
                "From the context below, extract concise JSON with fields "
                "(ticker/party, deal_type, counterparties, date?, summary, confidence 0-1).\n"
                f"Context:\n{merged}\n\nJSON:"
            )
            try:
                out = chat.generate(prompt, max_tokens=300, temperature=0.2)
                findings = {"model": chat.model_name, "text": out.strip()}
            except Exception as e:
                findings = {"model": "llm-error", "error": str(e), **self._rule_based(texts)}
        else:
            findings = self._rule_based(texts)

        # Write to short-term memory and long-term memory
        self.short.add({"note": json.dumps(findings)[:500]})
        self.long.upsert([Document(page_content=json.dumps(findings))])

        state["findings"] = findings
        self.after_run(state)
        return state
