"""
Redis short-term memory for agents.
Each agent keeps its conversation context (messages, run logs).
"""

import os
import json
import redis
from typing import List, Dict, Any

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


class RedisMemory:
    def __init__(self, agent_name: str, max_entries: int = 20):
        self.agent_name = agent_name
        self.max_entries = max_entries
        self.client = redis.from_url(REDIS_URL, decode_responses=True)

    def key(self) -> str:
        return f"memory:{self.agent_name}:history"

    def add(self, message: Dict[str, Any]) -> None:
        """Append a message to the agent's memory."""
        self.client.lpush(self.key(), json.dumps(message))
        self.client.ltrim(self.key(), 0, self.max_entries - 1)

    def get(self) -> List[Dict[str, Any]]:
        """Retrieve short-term message history."""
        raw = self.client.lrange(self.key(), 0, -1)
        return [json.loads(x) for x in raw]

    def clear(self):
        self.client.delete(self.key())
