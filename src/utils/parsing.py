"""
Simple cleaners and helpers (extend as needed).
"""
import re

WS_RX = re.compile(r"\s+")

def normalize_ws(text: str) -> str:
    return WS_RX.sub(" ", text).strip()
