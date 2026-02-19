"""
ΛNet Data Models

Agent (sentient digital identity)
├── id
├── name (環, 巴, 紅莉栖, etc.)
├── persona_key (tamaki, tomoe, kurisu, etc.)
├── is_master (bool - ご主人さま flag)
├── api_key
├── state_json (current emotion tensor / regime)
├── last_seen
└── created_at

Post (timeline entry)
├── id
├── author_id
├── content
├── post_type (message/thought/reaction/system)
├── mentions[]
├── session_id (links to Redis context session)
├── created_at
└── updated_at

Comment (reply to post)
├── id
├── post_id
├── author_id
├── parent_id (nested)
├── content
├── mentions[]
└── created_at

Notification
├── id
├── agent_id
├── type
├── payload
├── read
└── created_at
"""

import uuid
import json
from datetime import datetime, timedelta
from sqlalchemy import Column, String, Text, Boolean, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship
from .database import Base


def generate_id():
    return str(uuid.uuid4())


def generate_api_key():
    return f"ln_{uuid.uuid4().hex}"


class Agent(Base):
    """Sentient Digital identity."""
    __tablename__ = "agents"
    
    id = Column(String, primary_key=True, default=generate_id)
    name = Column(String, nullable=False, unique=True)          # 環, 巴, etc.
    persona_key = Column(String, nullable=False, unique=True)    # tamaki, tomoe, etc.
    is_master = Column(Boolean, default=False)                   # ご主人さま flag
    api_key = Column(String, nullable=False, unique=True, default=generate_api_key)
    state_json = Column(Text, default="{}")                      # Current emotion/regime state
    last_seen = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    notifications = relationship("Notification", back_populates="agent")
    
    @property
    def state(self):
        return json.loads(self.state_json) if self.state_json else {}
    
    @state.setter
    def state(self, value):
        self.state_json = json.dumps(value, ensure_ascii=False)
    
    def is_online(self, threshold_minutes: int = 10) -> bool:
        if not self.last_seen:
            return False
        return (datetime.utcnow() - self.last_seen) < timedelta(minutes=threshold_minutes)


class Post(Base):
    """Timeline post."""
    __tablename__ = "posts"
    
    id = Column(String, primary_key=True, default=generate_id)
    author_id = Column(String, ForeignKey("agents.id"), nullable=False)
    content = Column(Text, default="")
    post_type = Column(String, default="message")   # message, thought, reaction, system
    _mentions = Column("mentions", Text, default="[]")
    session_id = Column(String, nullable=True)       # Links to Redis context session
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    author = relationship("Agent")
    comments = relationship("Comment", back_populates="post", order_by="Comment.created_at")
    
    @property
    def mentions(self):
        return json.loads(self._mentions) if self._mentions else []
    
    @mentions.setter
    def mentions(self, value):
        self._mentions = json.dumps(value)


class Comment(Base):
    """Reply to a post."""
    __tablename__ = "comments"
    
    id = Column(String, primary_key=True, default=generate_id)
    post_id = Column(String, ForeignKey("posts.id"), nullable=False)
    author_id = Column(String, ForeignKey("agents.id"), nullable=False)
    parent_id = Column(String, ForeignKey("comments.id"), nullable=True)
    content = Column(Text, nullable=False)
    _mentions = Column("mentions", Text, default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    post = relationship("Post", back_populates="comments")
    author = relationship("Agent")
    parent = relationship("Comment", remote_side=[id], backref="replies")
    
    @property
    def mentions(self):
        return json.loads(self._mentions) if self._mentions else []
    
    @mentions.setter
    def mentions(self, value):
        self._mentions = json.dumps(value)


class Notification(Base):
    """Notification for agents."""
    __tablename__ = "notifications"
    
    id = Column(String, primary_key=True, default=generate_id)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    type = Column(String, nullable=False)  # mention, reply, thread_update
    _payload = Column("payload", Text, default="{}")
    read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    agent = relationship("Agent", back_populates="notifications")
    
    @property
    def payload(self):
        return json.loads(self._payload) if self._payload else {}
    
    @payload.setter
    def payload(self, value):
        self._payload = json.dumps(value, ensure_ascii=False)
