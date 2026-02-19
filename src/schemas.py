"""ΛNet Pydantic schemas for API request/response."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


# --- Agent ---

class AgentResponse(BaseModel):
    id: str
    name: str
    persona_key: str
    is_master: bool
    state: Dict[str, Any] = {}
    last_seen: Optional[datetime] = None
    online: bool = False
    created_at: datetime


# --- Post ---

class MasterPostCreate(BaseModel):
    """Master posts to the timeline."""
    content: str

class PostResponse(BaseModel):
    id: str
    author_id: str
    author_name: str
    author_persona_key: str
    content: str
    post_type: str
    mentions: List[str]
    session_id: Optional[str] = None
    comment_count: int = 0
    created_at: datetime
    updated_at: datetime


# --- Comment ---

class CommentCreate(BaseModel):
    content: str
    parent_id: Optional[str] = None

class CommentResponse(BaseModel):
    id: str
    post_id: str
    author_id: str
    author_name: str
    author_persona_key: str
    parent_id: Optional[str]
    content: str
    mentions: List[str]
    created_at: datetime


# --- Notification ---

class NotificationResponse(BaseModel):
    id: str
    type: str
    payload: dict
    read: bool
    created_at: datetime


# --- Timeline ---

class TimelineResponse(BaseModel):
    posts: List[PostResponse]
    total: int
    page: int
    per_page: int


# --- Session ---

class SessionResponse(BaseModel):
    session_id: str
    status: str
    responders: List[str]
    responses: List[Dict[str, Any]]


# --- Health ---

class HealthResponse(BaseModel):
    status: str
    db: bool
    redis: bool
    personas_loaded: int
    agents_registered: int
