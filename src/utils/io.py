"""
Config + logging helpers and simple file IO.
"""
from __future__ import annotations
from typing import Dict, Any
import os
import yaml
import json
import logging
import logging.config

def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def setup_logging() -> None:
    if os.path.exists("config/logging.yaml"):
        with open("config/logging.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def write_text(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)

def load_config() -> Dict[str, Any]:
    from dotenv import load_dotenv
    # Try loading from env_var or .env
    load_dotenv("env_var")
    load_dotenv(".env")
    
    cfg = load_yaml("config/config.yaml")
    run = cfg.get("runtime", {})
    ing = cfg.get("ingestion", {})
    # env overrides (optional)
    tickers = os.environ.get("TICKERS")
    if tickers:
        run["tickers"] = [t.strip() for t in tickers.split(",")]
    if "USE_SEC" in os.environ:
        ing["use_sec"] = os.environ["USE_SEC"].lower() == "true"
    if "NEWS_LIMIT" in os.environ:
        ing["news_limit"] = int(os.environ["NEWS_LIMIT"])
    if "CIKS" in os.environ:
        ing["ciks"] = [c.strip() for c in os.environ["CIKS"].split(",")]

    # Flatten the few keys the agents read
    return {
        "tickers": run.get("tickers", ["AAPL","MSFT","NVDA"]),
        "top_k": run.get("top_k", 20),
        "use_sec": ing.get("use_sec", False),
        "news_limit": ing.get("news_limit", 15),
        "ciks": ing.get("ciks", []),
    }
