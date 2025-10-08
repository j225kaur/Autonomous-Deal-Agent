# Autonomous Deal Agent System

Three-agent, LangGraph-orchestrated pipeline for M&A/deal trend detection:

- **DataAgent**: Ingests Yahoo Finance news (per ticker) and optional SEC filings, builds `Document`s
- **AnalysisAgent**: Retrieves from FAISS and uses LLM (optional) or rule-based logic to detect deal items
- **ReportAgent**: Emits a JSON + natural language Daily Deal Report, with optional mock email/Slack

## Quick Start

```bash
git clone <this-repo> financial-agent
cd financial-agent

# Python env
python -m venv .venv && source .venv/bin/activate
pip install -r <(cat <<'REQ'
fastapi
uvicorn[standard]
langgraph
langchain
langchain-community
langchain-openai
faiss-cpu
sentence-transformers
yfinance
requests
pandas
numpy
pydantic
REQ
)

# Run API
python -m api.main
