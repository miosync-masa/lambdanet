"""
ΛNet Redis Client - Upstash REST API

Handles session context windows for active conversations.
Context is ephemeral - built during sessions, discarded after.

Key structure:
  session:{id}:messages   → List of conversation messages (JSON)
  session:{id}:meta       → Session metadata (who's participating, start time)
  agent:{key}:state       → Real-time emotion tensor (backup to Postgres on save)
"""

import os
import json
import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime


class RedisClient:
    """Upstash Redis REST client for ΛNet."""
    
    def __init__(self, url: str = None, token: str = None):
        self.url = (url or os.environ.get("UPSTASH_REDIS_REST_URL", "")).rstrip("/")
        self.token = token or os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    async def _execute(self, *args) -> Any:
        """Execute a Redis command via Upstash REST API."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.url,
                headers=self.headers,
                json=list(args),
                timeout=10.0
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("result")
    
    async def _pipeline(self, commands: List[List]) -> List[Any]:
        """Execute multiple commands in a pipeline."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.url}/pipeline",
                headers=self.headers,
                json=commands,
                timeout=10.0
            )
            resp.raise_for_status()
            results = resp.json()
            return [r.get("result") for r in results]
    
    # --- Session Context ---
    
    async def create_session(self, session_id: str, meta: Dict) -> str:
        """Create a new conversation session."""
        meta["created_at"] = datetime.utcnow().isoformat()
        await self._execute("SET", f"session:{session_id}:meta", json.dumps(meta, ensure_ascii=False))
        await self._execute("EXPIRE", f"session:{session_id}:meta", 3600)  # 1h TTL
        return session_id
    
    async def append_message(self, session_id: str, role: str, name: str, content: str):
        """Append a message to session context."""
        msg = json.dumps({
            "role": role,
            "name": name,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }, ensure_ascii=False)
        
        await self._pipeline([
            ["RPUSH", f"session:{session_id}:messages", msg],
            ["EXPIRE", f"session:{session_id}:messages", 3600]
        ])
    
    async def get_context(self, session_id: str) -> List[Dict]:
        """Get all messages in session context."""
        messages = await self._execute("LRANGE", f"session:{session_id}:messages", 0, -1)
        if not messages:
            return []
        return [json.loads(m) for m in messages]
    
    async def get_session_meta(self, session_id: str) -> Optional[Dict]:
        """Get session metadata."""
        meta = await self._execute("GET", f"session:{session_id}:meta")
        if meta:
            return json.loads(meta)
        return None
    
    async def close_session(self, session_id: str):
        """Delete session context (after saving to Postgres/MemOS)."""
        await self._pipeline([
            ["DEL", f"session:{session_id}:messages"],
            ["DEL", f"session:{session_id}:meta"]
        ])
    
    # --- Agent State (real-time) ---
    
    async def set_agent_state(self, persona_key: str, state: Dict):
        """Store agent's current emotion tensor in Redis."""
        await self._execute(
            "SET",
            f"agent:{persona_key}:state",
            json.dumps(state, ensure_ascii=False)
        )
    
    async def get_agent_state(self, persona_key: str) -> Optional[Dict]:
        """Get agent's current emotion tensor from Redis."""
        state = await self._execute("GET", f"agent:{persona_key}:state")
        if state:
            return json.loads(state)
        return None
    
    # --- Health Check ---
    
    async def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            result = await self._execute("PING")
            return result == "PONG"
        except Exception:
            return False


# Singleton
_redis_client: Optional[RedisClient] = None


def get_redis() -> RedisClient:
    """Get or create Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client
