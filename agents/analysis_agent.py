"""
AnalysisAgent with Redis short-term memory and FAISS long-term memory.
"""

import os, re, json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from memory.redis_memory import RedisMemory
from memory.vector_memory import VectorMemory

try:
    from langchain_openai import ChatOpenAI
    HAVE_LLM = True
except Exception:
    HAVE_LLM = False


class AnalysisAgent:
    """Analyzes documents and keeps memory of prior reasoning."""

    def __init__(self):
        self.name = "analysis_agent"
        self.short_memory = RedisMemory(self.name)
        self.long_memory = VectorMemory(self.name)

    def run(self, config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        query = (
            "Recent mergers, acquisitions, or corporate transactions. "
            "Summarize with entities, date, and sentiment."
        )

        # --- Recall from long-term memory (FAISS)
        retrieved = self.long_memory.retrieve(query, k=config.get("top_k", 15))

        # --- Retrieve short-term context (Redis)
        recent_context = self.short_memory.get()
        context_snippet = "\n".join([m["text"] for m in recent_context][-3:])

        # --- Combine with fresh retrieval
        context_docs = [d.page_content for d in retrieved]
        merged_context = context_snippet + "\n" + "\n".join(context_docs)

        # --- Run analysis
        if HAVE_LLM and os.environ.get("OPENAI_API_KEY"):
            llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.3)
            messages = [
                SystemMessage(content="You are a deal-trend analyst."),
                HumanMessage(content=f"Recent context:\n{merged_context}\n\nExtract key M&A insights as JSON."),
            ]
            resp = llm.invoke(messages)
            findings = {"model": "LLM", "text": resp.content.strip()}
        else:
            findings = self.rule_based(merged_context)

        # --- Write to short-term memory
        self.short_memory.add({"text": json.dumps(findings), "timestamp": os.times()})

        # --- Append to long-term memory (persisted knowledge)
        docs = [Document(page_content=json.dumps(findings))]
        self.long_memory.upsert(docs)

        state["findings"] = findings
        return state

    def rule_based(self, text: str):
        rx = re.compile(r"(merger|acquisition|buyout|takeover|SPAC)", re.I)
        matches = rx.findall(text)
        return {"model": "rule-based", "keywords": list(set(matches)), "summary": text[:400]}
