import os, json, time, redis
from typing import Dict, Any, List

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_MAX_ENTRIES = int(os.environ.get("REDIS_MAX_ENTRIES", "40"))
REDIS_TTL_SECONDS = int(os.environ.get("REDIS_TTL_SECONDS", "86400"))
REDIS_MAX_MSG_BYTES = int(os.environ.get("REDIS_MAX_MSG_BYTES", "4096"))
REDIS_MAX_RETRIES = int(os.environ.get("REDIS_MAX_RETRIES", "3"))

class RedisMemory:
    def __init__(self, agent_name: str, max_entries: int = REDIS_MAX_ENTRIES):
        self.agent_name = agent_name
        self.max_entries = max_entries
        self.client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2.0)

    @property
    def key(self) -> str:
        return f"memory:{self.agent_name}:history"

    def add(self, message: Dict[str, Any]) -> None:
        s = json.dumps(message)
        if len(s.encode("utf-8")) > REDIS_MAX_MSG_BYTES:
            s = s[:REDIS_MAX_MSG_BYTES]  # truncate defensively
        delay = 0.2
        for attempt in range(1, REDIS_MAX_RETRIES + 1):
            try:
                pipe = self.client.pipeline()
                pipe.lpush(self.key, s)
                pipe.ltrim(self.key, 0, self.max_entries - 1)
                pipe.expire(self.key, REDIS_TTL_SECONDS)
                pipe.execute()
                return
            except redis.RedisError:
                if attempt == REDIS_MAX_RETRIES:
                    # Swallow and proceed (don't crash the agent)
                    return
                time.sleep(delay); delay *= 2

    def get(self) -> List[Dict[str, Any]]:
        try:
            raw = self.client.lrange(self.key, 0, -1)
            return [json.loads(x) for x in raw if x]
        except redis.RedisError:
            return []

    def clear(self) -> None:
        try:
            self.client.delete(self.key)
        except redis.RedisError:
            pass
