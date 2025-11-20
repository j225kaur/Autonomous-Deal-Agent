"""
ReportAgent
- Produces JSON report + natural language brief
- Uses summarizer adapter (DistilBART/FLAN-T5) to polish text
- Saves JSON and TXT to data/outputs/
"""
# src/pipeline/report_generator.py
from __future__ import annotations
from typing import Dict, Any, List
from pipeline.base import Base
from src.models.summarizer import get_summarizer  # optional

class ReportGenerator(Base):
    def __init__(self):
        super().__init__("report_agent")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.before_run(state)

        findings = state.get("findings", {})
        deals = findings.get("deals") or []
        lines: List[str] = []

        if deals:
            for d in deals[:6]:
                t = (d.get("type") or "?").title()
                acq = d.get("acquirer") or "?"
                tgt = d.get("target") or "?"
                val = d.get("value_usd") or "undisclosed"
                st  = d.get("status") or "other"
                lines.append(f"• {t}: {acq} → {tgt} ({val}, {st})")
        else:
            heads = [
                rd["page_content"]
                for rd in state.get("retrieved_docs", [])
                if rd.get("metadata", {}).get("source") in {"yahoo_news", "sec"}
            ][:5]
            lines = [f"• {h}" for h in heads] or ["• No clear deal signals today."]

        final_text = "\n".join(lines)

        # Optional polishing via summarizer (sentence-transformers) if enabled
        summarizer = get_summarizer()
        if summarizer:
            try:
                final_text = summarizer.summarize(final_text, max_sentences=5)
            except Exception:
                pass

        report = state.setdefault("report", {})
        report.setdefault("json", {})
        report["json"]["findings"] = findings
        report["json"]["summary"] = final_text
        report["text"] = final_text
        
        # Save to file for dashboard
        import os
        import json
        os.makedirs("data/outputs", exist_ok=True)
        with open("data/outputs/latest_report.json", "w") as f:
            json.dump(report["json"], f, indent=2, default=str)

        self.after_run(state)
        return state
