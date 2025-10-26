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
from pipeline.base import Base
from src.memory.redis_memory import RedisMemory
from src.memory.vector_memory import VectorMemory
from src.models.adapters import get_chat_model


DEAL_SCHEMA_HINT = """Return ONLY JSON:
{
  "deals": [
    {
      "type": "acquisition|merger|divestiture|spin-off|spac|tender|strategic_transaction|other",
      "acquirer": "string|null",
      "target": "string|null",
      "tickers": ["..."],
      "value_usd": "string|null",
      "status": "rumor|agreement|announced|closed|terminated|other",
      "evidence": "short quote/headline",
      "source_link": "url|null"
    }
  ],
  "trend_summary": "2-3 sentences"
}"""

def _format_ctx(retrieved: List[Dict[str, Any]]) -> str:
    lines = []
    for d in retrieved[:8]:
        meta = d.get("metadata", {})
        link = meta.get("link", "")
        src = meta.get("source", "")
        title = (d.get("page_content") or "").strip().replace("\n", " ")[:220]
        lines.append(f"- [{src}] {title} (link={link})")
    return "\n".join(lines) if lines else "(no retrieved docs)"

def analyze_with_llm(query: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    llm = get_chat_model()
    if not llm:
        # keep your rule-based detection here; return structured object
        return {"deals": [], "trend_summary": "LLM disabled; rule-based path."}

    context = _format_ctx(retrieved)
    if context == "(no retrieved docs)":
        # Fallback: still give the model something (global query)
        context = f"- No hits from vector DB for query. Use domain knowledge + deal keywords.\nQuery='{query}'"

    prompt = (
        "You are an M&A analyst. Extract concrete deals and trends from the context. "
        "If uncertain, use null/undisclosed, don't hallucinate.\n\n"
        f"{DEAL_SCHEMA_HINT}\n\n"
        f"Query: {query}\n"
        f"Context:\n{context}\n\n"
        "JSON:"
    )

    raw = llm.generate(prompt, max_tokens=600, temperature=0.1)
    # tolerant JSON parse
    text = (raw or "").strip().strip("`")
    if text.lower().startswith("json"):
        text = text[4:].lstrip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "deals" in data:
            return data
    except Exception:
        pass
    return {"deals": [], "trend_summary": "Model returned unparseable output"}

RULES_RX = re.compile(r"(merger|acquisition|buyout|takeover|spin[- ]?off|SPAC)", re.I)

class DealAnalyzer(Base):
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
        top_k = int(cfg.get("top_k", 20))

        retrieved_serializable = state.get("retrieved_docs") or []
        if not retrieved_serializable:
            retr = self.long.retriever(k=top_k)
            query = "Recent mergers, acquisitions, spin-offs or strategic deals"
            docs = retr.invoke(query)
            for d in docs[:top_k]:
                meta = d.metadata if isinstance(d.metadata, dict) else {}
                retrieved_serializable.append({
                    "page_content": d.page_content,
                    "metadata": {
                        "source": meta.get("source"),
                        "ticker": meta.get("ticker"),
                        "publisher": meta.get("publisher"),
                        "link": meta.get("link"),
                        "published": meta.get("published"),
                        "form": meta.get("form"),
                        "date": meta.get("date"),
                    }
                })
            state["retrieved_docs"] = retrieved_serializable

        recent_notes = self.short.get()
        context_snip = "\n".join([m.get("note","") for m in recent_notes][-3:]).strip()
        retrieved_for_llm = retrieved_serializable
        if context_snip:
            retrieved_for_llm = [{"page_content": context_snip, "metadata": {"source": "memory"}}] + retrieved_serializable

        query = " ".join(cfg.get("tickers", [])) + " M&A deals today"
        llm_out = analyze_with_llm(query, retrieved_for_llm)

        if not llm_out or not isinstance(llm_out, dict) or "deals" not in llm_out:
            # Rule-based fallback
            RULES_RX = re.compile(r"(merger|acquisition|buyout|takeover|spin[- ]?off|SPAC|definitive agreement|tender)", re.I)
            texts = [d["page_content"] for d in retrieved_serializable]
            hits = [t[:220] for t in texts if RULES_RX.search(t or "")]
            llm_out = {
                "deals": [],
                "trend_summary": hits[:5] and " ; ".join(hits[:5]) or "No obvious deal signals.",
                "_fallback": "rule-based"
            }

        chat = get_chat_model()
        model_name = getattr(chat, "model_name", "disabled")
        findings_structured = {"model": model_name, **llm_out}

        compact_note = findings_structured.get("trend_summary") or (
            findings_structured.get("deals") and str(findings_structured["deals"][0])
        ) or ""
        self.short.add({"note": compact_note[:480]})

        try:
            content = json.dumps(findings_structured, default=str)[:4000]
        except Exception:
            content = str(findings_structured)[:4000]
        self.long.upsert([Document(page_content=content, metadata={"source":"analysis","agent":self.name})])

        state["findings"] = findings_structured
        self.after_run(state)
        return state
