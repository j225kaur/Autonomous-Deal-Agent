from src.core.orchestrator import run_once
from src.utils.io import load_config
import json

cfg = load_config()
print("Running pipeline...")

try:
    result = run_once(cfg)
    print("Pipeline finished.\n")
    
    # Check signal scores
    findings = result.get("findings", {})
    signal_scores = findings.get("signal_scores", {})
    
    print("Signal Scores:")
    for ticker, scores in signal_scores.items():
        print(f"\n{ticker}:")
        print(f"  Total Score: {scores.get('score')}")
        print(f"  Components: {scores.get('components')}")
        print(f"  Explanation: {scores.get('explanation')}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
