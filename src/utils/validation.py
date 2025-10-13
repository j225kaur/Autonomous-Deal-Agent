"""
Lightweight JSON validators / schema stubs.
"""
from typing import Any, Dict

def is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())

def ensure_keys(d: Dict[str, Any], keys) -> bool:
    return all(k in d for k in keys)
