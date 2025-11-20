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
from src.pipeline.base import Base
from src.storage.stores import FAISSVectorStore, RedisMemoryStore
from src.models.adapters import get_chat_model
from src.analysis.signal_model import SignalModel


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
      "evidence": "short quote/headline from context",
      "source_link": "url|null"
    }
  ],
  "trend_summary": "2-3 sentences"
}"""

STRICT_SYSTEM_PROMPT = """You are an M&A intelligence analyst. Your job is to extract ONLY concrete, evidence-based deal information.

CRITICAL RULES:
1. If NO deal-related information exists in the context → return {"deals": [], "trend_summary": "No M&A activity detected."}
2. NEVER invent or speculate about deals not mentioned in the context
3. NEVER use phrases like "rumors of", "potential", "various industries" unless those EXACT words appear in a source
4. Evidence MUST be a direct quote or close paraphrase from the context
5. If acquirer/target are unknown → set to null, but there MUST be concrete deal evidence in context
6. Status can only be "rumor" if the word "rumor", "speculation", or "talks" appears in the source
7. NEVER generate multiple identical deals
8. If uncertain or no evidence exists → omit the deal entirely

ACCEPTABLE DEAL EVIDENCE:
✓ "Company X to acquire Company Y for $Z"
✓ "X and Y announce merger"
✓ "Sources say X in talks to buy Y"

UNACCEPTABLE (HALLUCINATION):
✗ "Rumors of potential collaborations"
✗ "Various industries"
✗ Generic speculation not in sources
✗ Deals with no specific companies mentioned
"""

def _format_ctx(retrieved: List[Dict[str, Any]]) -> str:
    lines = []
    for d in retrieved[:8]:
        meta = d.get("metadata", {})
        link = meta.get("link", "")
        src = meta.get("source", "")
        title = (d.get("page_content") or "").strip().replace("\n", " ")[:220]
        lines.append(f"- [{src}] {title} (link={link})")
    return "\n".join(lines) if lines else "(no retrieved docs)"

def _evidence_exists_in_docs(evidence: str, retrieved_docs: List[Dict[str, Any]]) -> bool:
    """Check if evidence text appears in any retrieved document."""
    if not evidence or len(evidence) < 10:
        return False
    evidence_lower = evidence.lower()
    # Remove generic hallucination phrases
    generic_phrases = ["rumors of potential", "various industries", "collaborations and acquisitions"]
    if any(phrase in evidence_lower for phrase in generic_phrases):
        return False
    
    for doc in retrieved_docs:
        content = (doc.get("page_content") or "").lower()
        if len(content) > 10:
            # Check for substantial overlap
            if evidence_lower in content or any(word in content for word in evidence_lower.split() if len(word) > 4):
                return True
    return False

def _validate_deals(deals: List[Dict[str, Any]], retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out hallucinated deals that lack concrete evidence."""
    valid_deals = []
    for deal in deals:
        # Rule 1: Must have either acquirer or target
        acquirer = deal.get("acquirer")
        target = deal.get("target")
        if not acquirer and not target:
            continue
        
        # Rule 2: Evidence must exist in retrieved docs
        evidence = deal.get("evidence", "")
        if not _evidence_exists_in_docs(evidence, retrieved_docs):
            continue
        
        # Rule 3: No generic boilerplate evidence
        evidence_lower = evidence.lower()
        if any(phrase in evidence_lower for phrase in ["rumors of potential", "various industries", "collaborations and acquisitions in"]):
            continue
        
        valid_deals.append(deal)
    
    return valid_deals

def analyze_with_llm(query: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    llm = get_chat_model()
    if not llm:
        return {"deals": [], "trend_summary": "LLM disabled; rule-based path."}

    context = _format_ctx(retrieved)
    
    # CRITICAL: If no docs, return empty immediately - DO NOT hallucinate
    if context == "(no retrieved docs)" or not retrieved:
        return {
            "deals": [],
            "trend_summary": "No documents retrieved. Unable to detect M&A activity."
        }

    prompt = (
        f"{STRICT_SYSTEM_PROMPT}\n\n"
        f"{DEAL_SCHEMA_HINT}\n\n"
        f"Query: {query}\n"
        f"Context:\n{context}\n\n"
        "Analyze the context above and extract deals. If no deals exist, return empty deals array.\n\n"
        "JSON:"
    )

    raw = llm.generate(prompt, max_tokens=600, temperature=0.0)  # temperature=0 for deterministic output
    # tolerant JSON parse
    text = (raw or "").strip().strip("`")
    if text.lower().startswith("json"):
        text = text[4:].lstrip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "deals" in data:
            # VALIDATE: Filter hallucinated deals
            original_count = len(data.get("deals", []))
            data["deals"] = _validate_deals(data.get("deals", []), retrieved)
            
            # If all deals were filtered out, update summary
            if original_count > 0 and len(data["deals"]) == 0:
                data["trend_summary"] = "No verified M&A activity found in sources."
            
            return data
    except Exception:
        pass
    return {"deals": [], "trend_summary": "Model returned unparseable output"}

RULES_RX = re.compile(r"(merger|acquisition|buyout|takeover|spin[- ]?off|SPAC)", re.I)

class DealAnalyzer(Base):
    def __init__(self):
        super().__init__("analysis_agent")
        self.short = RedisMemoryStore(self.name, max_entries=50)
        self.long = FAISSVectorStore(index_dir=None) # uses default
        self.signal_model = SignalModel()

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
            query = "Recent mergers, acquisitions, spin-offs or strategic deals"
            docs = self.long.search(query, k=top_k)
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
        
        # --- Signal Analysis ---
        raw_items = state.get("raw_items", {})
        market_history = raw_items.get("market_history", {})
        tickers = raw_items.get("tickers", [])
        
        signal_scores = {}
        for t in tickers:
            hist = market_history.get(t, {})
            # Collect news text for this ticker from retrieved docs
            ticker_news = [d["page_content"] for d in retrieved_serializable if d["metadata"].get("ticker") == t]
            
            score_obj = self.signal_model.score_ticker(t, hist, ticker_news)
            signal_scores[t] = {
                "score": score_obj.total_score,
                "components": score_obj.components,
                "explanation": score_obj.explanation
            }

        findings_structured = {
            "model": model_name,
            "signal_scores": signal_scores,
            **llm_out
        }

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
