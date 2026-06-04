"""
Redis-backed session store for per-user conversation history.

Stores:
  - Chat history (list of messages) per session_id
  - Rolling conversation summary per session_id
  - Auto-expires with configurable TTL

Keys:
  rag:history:{session_id}   — JSON list of {role, content} messages
  rag:summary:{session_id}   — plain text rolling summary
"""

import json
import logging
from typing import List, Optional

import redis
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from .config import Config

logger = logging.getLogger(__name__)

_PREFIX = "rag"


def _history_key(session_id: str) -> str:
    return f"{_PREFIX}:history:{session_id}"


def _summary_key(session_id: str) -> str:
    return f"{_PREFIX}:summary:{session_id}"


class SessionStore:
    """Per-user session store backed by Redis."""

    def __init__(self, redis_url: str = None, ttl: int = None):
        self.redis_url = redis_url or Config.REDIS_URL
        self.ttl = ttl or Config.SESSION_TTL
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
            )
        return self._client

    def health_check(self) -> dict:
        """Check Redis connectivity."""
        try:
            self.client.ping()
            return {"status": "connected", "url": self.redis_url}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── History ───────────────────────────────────────────────

    def load_history(self, session_id: str) -> List[BaseMessage]:
        """Load chat history from Redis as LangChain messages."""
        try:
            raw = self.client.get(_history_key(session_id))
            if not raw:
                return []
            messages = json.loads(raw)
            result = []
            for msg in messages:
                if msg["role"] == "human":
                    result.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "ai":
                    result.append(AIMessage(content=msg["content"]))
            return result
        except Exception as e:
            logger.warning(f"Failed to load history for {session_id}: {e}")
            return []

    def save_history(self, session_id: str, messages: List[BaseMessage]) -> None:
        """Save chat history to Redis."""
        try:
            serialized = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    serialized.append({"role": "human", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    serialized.append({"role": "ai", "content": msg.content})
            self.client.set(
                _history_key(session_id),
                json.dumps(serialized),
                ex=self.ttl,
            )
        except Exception as e:
            logger.warning(f"Failed to save history for {session_id}: {e}")

    # ── Summary ───────────────────────────────────────────────

    def load_summary(self, session_id: str) -> str:
        """Load conversation summary from Redis."""
        try:
            return self.client.get(_summary_key(session_id)) or ""
        except Exception as e:
            logger.warning(f"Failed to load summary for {session_id}: {e}")
            return ""

    def save_summary(self, session_id: str, summary: str) -> None:
        """Save conversation summary to Redis."""
        try:
            self.client.set(_summary_key(session_id), summary, ex=self.ttl)
        except Exception as e:
            logger.warning(f"Failed to save summary for {session_id}: {e}")

    # ── Clear ─────────────────────────────────────────────────

    def clear(self, session_id: str) -> None:
        """Clear all session data for a user."""
        try:
            self.client.delete(_history_key(session_id), _summary_key(session_id))
        except Exception as e:
            logger.warning(f"Failed to clear session for {session_id}: {e}")
