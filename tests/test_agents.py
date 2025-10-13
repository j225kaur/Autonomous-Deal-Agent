from src.core.orchestrator import run_once

def test_pipeline_smoke():
    cfg = {"tickers": ["AAPL","MSFT"], "use_sec": False, "top_k": 5}
    out = run_once(cfg)
    assert "report" in out and "text" in out["report"]
