"""
ReportAgent:
- Compiles structured JSON + human-readable report.
- Optionally sends via mock email/Slack.
"""

from datetime import datetime, timezone
import os
from typing import Dict, Any


class ReportAgent:
    """Final report generator."""

    def run(self, config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        findings = state.get("findings", {})
        summary = self.generate_text(findings)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "findings": findings,
            "summary": summary,
        }

        if os.environ.get("SEND_EMAIL") == "true":
            self.mock_email(summary)
        if os.environ.get("SEND_SLACK") == "true":
            self.mock_slack(summary)

        state["report"] = report
        return state

    def generate_text(self, findings):
        if "text" in findings:
            return findings["text"]
        items = findings.get("items", [])
        if not items:
            return "No M&A or deal-related events detected today."
        return "Daily Deal Report:\n" + "\n".join(
            f"- {i['source']}: {i['summary']}" for i in items
        )

    def mock_email(self, text):
        print("[EMAIL MOCK] Sending Deal Report:\n", text)

    def mock_slack(self, text):
        print("[SLACK MOCK] Posting Deal Report:\n", text)
