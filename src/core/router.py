"""
Optional policy router: pick different model routes or branches based on config/state.
Currently returns a simple linear path; extend as needed.
"""
from typing import Dict, Any, List

def plan_route(state: Dict[str, Any]) -> List[str]:
    # In the future, return alternate paths based on signal quality / model selection.
    return ["data", "analysis", "report"]
