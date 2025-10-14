"""
ReportAgent
- Produces JSON report + natural language brief
- Uses summarizer adapter (DistilBART/FLAN-T5) to polish text
- Saves JSON and TXT to data/outputs/
"""
from __future__ import annotations
from typing import Dict, Any
from datetime import datetime, timezone
import os
import json
from src.utils.io import ensure_dir, write_text, write_json
from src.agents.base_agent import BaseAgent
from src.models.summarizer import get_summarizer


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data/outputs")

class ReportAgent(BaseAgent):
    def __init__(self):
        super().__init__("report_agent")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.before_run(state)
        findings = state.get("findings", {})
        if findings.get("deals"):
            base_text = " • " + "\n • ".join(
                f"{d.get('type','?').title()}: {d.get('acquirer')} → {d.get('target')} "
                f"({d.get('value_usd','undisclosed')}, {d.get('status','other')})"
                for d in findings["deals"][:6]
            )
        else:
            heads = [d["page_content"] for d in state.get("retrieved_docs", [])[:5]]
            base_text = " • " + "\n • ".join(heads) if heads else "No deal signals today."

        raw = state.get("raw_items", {})
        now = datetime.now(timezone.utc).isoformat()

        # Build text; try summarizer to make it punchy
        summarizer = get_summarizer()
        base_text = f"Daily Deal Report ({now})\n\nFindings:\n{json.dumps(findings)[:2000]}"
        text = summarizer.summarize(base_text, max_chars=1200) if summarizer else base_text

        report_json = {
            "generated_at": now,
            "sources": raw,
            "findings": findings,
            "summary": text[:4000],
        }

        ensure_dir(OUTPUT_DIR)
        write_json(os.path.join(OUTPUT_DIR, "latest_report.json"), report_json)
        write_text(os.path.join(OUTPUT_DIR, "latest_report.txt"), text)

        state["report"] = {"json": report_json, "text": text}
        self.after_run(state)
        return state
