from src.core.orchestrator import run_once
from src.utils.io import load_config
import json

cfg = load_config()
print("Running pipeline with config:", cfg)
try:
    result = run_once(cfg)
    print("Pipeline finished.")
    
    # Check retrieved docs
    docs = result.get("json", {}).get("retrieved_docs", [])
    if docs:
        print(f"Found {len(docs)} retrieved docs.")
        first_doc = docs[0]
        print("First doc content length:", len(first_doc.get("page_content", "")))
        print("First doc content snippet:", first_doc.get("page_content", "")[:100])
    else:
        print("No retrieved docs found.")

except Exception as e:
    print(f"Error: {e}")
