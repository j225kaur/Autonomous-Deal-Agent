import os
import sys
import argparse
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.orchestrator import run_once

def main():
    parser = argparse.ArgumentParser(description="Run the deal analysis pipeline once.")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers (e.g. AAPL,MSFT)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of docs to retrieve")
    args = parser.parse_args()

    config = {
        "tickers": args.tickers if args.tickers else "AAPL",
        "top_k": args.top_k
    }

    print(f"Running pipeline with config: {config}")
    result = run_once(config)
    
    print("\n--- Findings ---")
    print(json.dumps(result.get("findings", {}), indent=2, default=str))
    
    print("\n--- Report ---")
    print(json.dumps(result.get("report", {}), indent=2, default=str))

if __name__ == "__main__":
    main()
