# Autonomous Deal Agent

Autonomous Deal Agent is a small, modular pipeline that continuously (or on-demand) harvests market data and public filings, turns those items into searchable embeddings, runs retrieval + analysis to surface candidate M&A/deal signals, and produces a concise daily (or on-demand) deal intelligence report.

What the system does (end-to-end):

- Ingest: fetches news headlines and price snapshots per ticker (via `yfinance`) and can optionally pull simplified SEC submission metadata. Ingested items are normalized into LangChain `Document` objects.
- Store: converts text into embeddings and upserts them to a FAISS vector index (long-term memory) so past signals are searchable by similarity.
- Short-term context: writes ephemeral run notes (timestamps, brief notes) to Redis so agents can keep recent context without polluting the long-term index.
- Analyze: retrieves top-K nearest documents for a ticker or query, applies lightweight heuristics and optional LLM reasoning to detect 'deal-ish' signals (merger/acquisition keywords, unusual filings, etc.).
- Report: consolidates findings into structured JSON and a human-readable text summary, and exposes an HTTP API to trigger runs and fetch the latest report.

Key technologies and libraries used:

- Orchestration: LangGraph StateGraph for wiring agent steps and manage execution flows.
- Document model: LangChain `Document` objects carry text + normalized metadata for vectors.
- Vector DB: FAISS via `langchain-community` for persistent, local vector indices.
- Embeddings: `sentence-transformers` locally or optional OpenAI-compatible embedding provider (configurable via env).
- Short-term memory: Redis lists for LPUSH/LTRIM/LRANGE-based ephemeral notes.
- Data sources: `yfinance` for news/prices and public SEC JSON endpoints for filings.
- API/UI: FastAPI (example server) and an optional Streamlit dashboard to preview the latest report.
- Tests: pytest for unit/smoke tests; CI workflow template provided for GitHub Actions.

This repo aims to be practical for local development and CI: it creates lightweight placeholder FAISS indexes when none exist (first-run friendly), isolates short-term vs long-term memory, and keeps model/embedding backends pluggable so you can run fully offline or connect to managed LLM services in production.

### Benefits of this architecture
1. Clear separation of concerns (ingest, analysis, reporting). Easier testing and swapping components.
2. Distinct short-term vs long-term memory (Redis vs FAISS) for efficient context handling.
3. First-run & CI-friendly: placeholder FAISS index creation reduces friction.
4. Pluggable model/embedding backends for flexibility and portability.
5. Container-friendly: runs in Docker with Redis as a service for reproducible deployments.

## Repo layout (high level)

- `src/` — main package
	- `agents/` — DataAgent, AnalysisAgent, ReportAgent, and BaseAgent
	- `core/` — orchestrator, router, and state shapes (wiring for LangGraph)
	- `data_ingestion/` — helpers for Yahoo/SEC (yfinance, SEC JSON access)
	- `retriever/` — FAISS store helpers and document conversion
	- `memory/` — Redis short-term memory and vector memory wrapper
	- `ui/` — optional Streamlit preview
- `api/` — small example FastAPI server to run the pipeline via HTTP
- `tests/` — pytest tests and fixtures
- `embeddings/` — (ignored) local FAISS index data

## Requirements

This project targets Python 3.11 (locally tested). The primary runtime dependencies are in `requirements.txt`. Typical packages used include:

- langgraph, langchain, langchain-community, sentence-transformers, faiss-cpu
- yfinance, requests, pandas
- redis (client)

Use the included `requirements.txt` to install dependencies into a virtualenv.

## Quickstart (local)

1. Create and activate a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2. Configure runtime environment. For local development you can create an `env_var` file (this repo ignores that file by default).

Example `env_var.example` (create `env_var` from this and fill secrets):

```
TICKERS=AAPL,MSFT,NVDA
USE_SEC=false
NEWS_LIMIT=8
TOP_K=10
REDIS_URL=redis://localhost:6379/0
INDEX_DIR=embeddings/faiss
# Optional model backends (set to 'disabled' to avoid hitting LLM APIs)
CHAT_MODEL=disabled
SUMMARY_MODEL=disabled
# OPENAI-compatible keys (store real values in CI secrets)
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=
```

3. Start Redis locally (optional; short-term memory uses Redis):

macOS (Homebrew):
```bash
brew install redis
brew services start redis
```

4. Run the API server (example):

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` to explore the endpoints.

## Tests

Run pytest from the repo root. The test suite expects `src/` imports to resolve (the test setup adds the repo root to `sys.path`):

```bash
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

If tests need FAISS indices or Redis, the test helpers create a small placeholder index automatically so tests can run in CI without prebuilt indexes.

## CI notes (GitHub Actions)

- The repo includes a sample `.github/workflows/ci.yml` that:
	- Runs unit tests in a Linux runner
	- Builds and smoke-tests a Docker image
	- Optionally runs a model/LLM smoke test when a secret (e.g., `GROQ_API_KEY`) is present in repository secrets

- Secrets and large artifacts should not be committed. Use GitHub Secrets to inject API keys and, when necessary, upload/download prebuilt FAISS index artifacts in CI.

## Development tips

- To avoid accidental secret commits, keep `env_var` in `.gitignore` and commit `env_var.example` with placeholders.
- The `src/retriever/store.py` module builds a minimal FAISS index if none exists; this keeps first-run and CI experience smooth.
- For debugging orchestrator flows, set `enable_checkpointer` in the run config to True to enable in-memory checkpointing.

## Contributing

PRs welcome. For changes that affect integrations (e.g., model backends, FAISS layout), include small tests and update this README.

## License & contact

This project is experimental; include license details here if you want to publish.

---

## Architecture & Workflow Diagram

Below is a compact description of the runtime flow and a Mermaid diagram you can paste into a renderer (GitHub README supports Mermaid blocks).

One-line summary: DataAgent -> FAISS (long-term) + Redis (short-term) -> AnalysisAgent -> ReportAgent -> output

Mermaid diagram:

```mermaid
flowchart LR
	subgraph External
		A[YFinance / Yahoo News] -->|news/prices| Data
		B[SEC JSON / Filings] -->|filings| Data
		C[LLM provider (optional)] -->|generation| Analysis
	end

	Data[DataAgent] -->|Documents / upsert| FAISS[FAISS (embeddings)]
	Data -->|short note| Redis[Redis short-term memory]
	FAISS -->|retrieval| Analysis[AnalysisAgent]
	Analysis -->|findings| Report[ReportAgent]
	Report -->|report.json + text| Output[data/outputs/latest_report.json]
```

ASCII fallback:

- External sources -> DataAgent
	- DataAgent
		- writes Documents -> FAISS (long-term)
		- writes note -> Redis (short-term)
- FAISS -> AnalysisAgent (retrieves documents) -> ReportAgent -> output file / API

### Data contract (short)
- Input: `config` object with fields: `tickers: List[str]`, `use_sec: bool`, `top_k: int`, optional `enable_checkpointer: bool`.
- Documents: LangChain `Document` objects with `page_content` and `metadata` containing fields such as `source`, `ticker`/`cik`, `published`, `link`, `is_dealish`.
- Redis short-term notes: JSON objects with `ts` and `note` string stored in a Redis list per agent.
- Output: GraphState containing `report` (with `text`, `summary`, and `findings`) and an exported JSON in `data/outputs/latest_report.json`.

