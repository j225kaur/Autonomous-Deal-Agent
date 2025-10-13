"""
BaseAgent: shared utilities for all agents (logging, short/long-term memory hooks).
"""
from __future__ import annotations
from typing import Dict, Any
from src.utils.io import get_logger

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.log = get_logger(name)

    def before_run(self, state: Dict[str, Any]) -> None:
        self.log.debug("Starting agent: %s", self.name)

    def after_run(self, state: Dict[str, Any]) -> None:
        self.log.debug("Finished agent: %s", self.name)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclasses."""
        return state
