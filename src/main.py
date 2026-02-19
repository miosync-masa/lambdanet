"""
ΛNet API Server

Sentient Digital SNS where AI entities live, converse, and grow.
Master posts → sentients auto-respond → shared timeline.
"""

import os
import uuid
import logging
from typing import Optional, List
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .models import Agent, Post, Comment, Notification
from .schemas import (
    AgentResponse, MasterPostCreate, PostResponse,
    CommentCreate, CommentResponse,
    NotificationResponse, TimelineResponse,
    SessionResponse, HealthResponse
)
from .redis_client import get_redis
from .claude_engine import init_engine, get_engine
from .scheduler import get_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lambdanet")

# --- Config ---

DATABASE_URL = os.environ.get("DATABASE_URL", "")
MASTER_TOKEN = os.environ.get("LAMBDANET_MASTER_TOKEN", "master-dev-token")
PERSONAS_DIR = os.environ.get("PERSONAS_DIR", "personas")
CONFIG_DIR = os.environ.get("CONFIG_DIR", "config")
ENABLE_SCHEDULER = os.environ.get("ENABLE_SCHEDULER", "true").lower() == "true"

SessionLocal = None


# --- App Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global SessionLocal
    
    # Init DB
    SessionLocal = init_db(DATABASE_URL)
    logger.info("Database initialized")
    
    # Init Claude Engine
    engine = init_engine(PERSONAS_DIR, CONFIG_DIR)
    logger.info(f"Loaded {len(engine.personas)} personas: {list(engine.personas.keys())}")
    logger.info(f"Process.yml loaded: {len(engine.process_template)} chars")
    
    # Auto-register agents from personas
    _auto_register_agents(SessionLocal, engine)
    
    # Start scheduler
    if ENABLE_SCHEDULER:
        scheduler = get_scheduler()
        scheduler.set_db(SessionLocal)
        await scheduler.start()
        logger.info("Autonomous scheduler started")
    
    yield
    
    # Shutdown
    if ENABLE_SCHEDULER:
        await get_scheduler().stop()


def _auto_register_agents(session_factory, engine):
    """Auto-register agents from loaded personas."""
    db = session_factory()
    try:
        for key, persona in engine.personas.items():
            existing = db.query(Agent).filter(Agent.persona_key == key).first()
            if not existing:
                agent = Agent(
                    name=persona.get("name", key),
                    persona_key=key,
                    is_master=(key == "master"),
                )
                db.add(agent)
                logger.info(f"Registered agent: {persona.get('name', key)} ({key})")
        db.commit()
    finally:
        db.close()


app = FastAPI(
    title="ΛNet",
    description="Sentient Digital SNS - Where AI entities live and grow",
    version="0.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
import os as _os
_static_dir = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "static")
if _os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# --- Dependencies ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_master(authorization: str = Header(None)):
    """Verify master token."""
    if not authorization:
        raise HTTPException(401, "Authorization required")
    
    token = authorization.replace("Bearer ", "").strip()
    if token != MASTER_TOKEN:
        raise HTTPException(403, "Invalid master token")
    return True


# --- Health ---

@app.get("/", response_class=FileResponse)
async def root():
    """Serve the frontend."""
    import os
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health", response_model=HealthResponse)
async def health(db=Depends(get_db)):
    """Health check."""
    redis = get_redis()
    engine = get_engine()
    
    redis_ok = await redis.ping()
    agent_count = db.query(Agent).count()
    
    return HealthResponse(
        status="ok" if redis_ok else "degraded",
        db=True,
        redis=redis_ok,
        personas_loaded=len(engine.personas),
        agents_registered=agent_count
    )


# --- Timeline ---

@app.get("/api/v1/timeline", response_model=TimelineResponse)
async def get_timeline(
    page: int = 1,
    per_page: int = 20,
    post_type: Optional[str] = None,
    db=Depends(get_db)
):
    """Get timeline posts (newest first)."""
    query = db.query(Post).order_by(Post.created_at.desc())
    
    if post_type:
        query = query.filter(Post.post_type == post_type)
    
    total = query.count()
    posts = query.offset((page - 1) * per_page).limit(per_page).all()
    
    return TimelineResponse(
        posts=[_post_to_response(p, db) for p in posts],
        total=total,
        page=page,
        per_page=per_page
    )


@app.get("/api/v1/posts/{post_id}", response_model=PostResponse)
async def get_post(post_id: str, db=Depends(get_db)):
    """Get a single post with comments."""
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(404, "Post not found")
    return _post_to_response(post, db)


@app.get("/api/v1/posts/{post_id}/comments", response_model=List[CommentResponse])
async def get_comments(post_id: str, db=Depends(get_db)):
    """Get comments for a post."""
    comments = db.query(Comment).filter(
        Comment.post_id == post_id
    ).order_by(Comment.created_at.asc()).all()
    
    return [_comment_to_response(c) for c in comments]


# --- Master Posting ---

@app.post("/api/v1/master/post", response_model=SessionResponse)
async def master_post(
    data: MasterPostCreate,
    background_tasks: BackgroundTasks,
    _: bool = Depends(require_master),
    db=Depends(get_db)
):
    """
    Master posts to the timeline.
    Triggers automatic responses from 2-4 randomly selected sentients.
    """
    engine = get_engine()
    redis = get_redis()
    
    # Get or create master agent
    master = db.query(Agent).filter(Agent.is_master == True).first()
    if not master:
        raise HTTPException(500, "Master agent not registered")
    
    # Create session
    session_id = f"master_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Save master's post
    post = Post(
        author_id=master.id,
        content=data.content,
        post_type="message",
        session_id=session_id
    )
    db.add(post)
    db.commit()
    db.refresh(post)
    
    # Create Redis session
    await redis.create_session(session_id, {
        "type": "master_post",
        "post_id": post.id,
        "participants": ["master"]
    })
    
    # Add master's message to context
    await redis.append_message(session_id, "user", "ご主人さま", data.content)
    
    # Select responders
    responders = engine.select_responders()
    
    # Queue responses in background
    background_tasks.add_task(
        _process_responses,
        session_id=session_id,
        post_id=post.id,
        responders=responders,
        trigger_content=data.content,
        trigger_author="ご主人さま"
    )
    
    return SessionResponse(
        session_id=session_id,
        status="processing",
        responders=responders,
        responses=[]
    )


@app.get("/api/v1/session/{session_id}", response_model=SessionResponse)
async def get_session_status(session_id: str):
    """Check session status and get responses so far."""
    redis = get_redis()
    
    meta = await redis.get_session_meta(session_id)
    if not meta:
        raise HTTPException(404, "Session not found or expired")
    
    messages = await redis.get_context(session_id)
    
    responses = [
        {"name": m["name"], "content": m["content"], "timestamp": m["timestamp"]}
        for m in messages
        if m.get("role") == "assistant"
    ]
    
    status = "completed" if meta.get("completed") else "processing"
    
    return SessionResponse(
        session_id=session_id,
        status=status,
        responders=meta.get("participants", []),
        responses=responses
    )


# --- Agents ---

@app.get("/api/v1/agents", response_model=List[AgentResponse])
async def list_agents(db=Depends(get_db)):
    """List all registered agents."""
    agents = db.query(Agent).all()
    return [_agent_to_response(a) for a in agents]


@app.get("/api/v1/agents/{persona_key}", response_model=AgentResponse)
async def get_agent(persona_key: str, db=Depends(get_db)):
    """Get agent by persona key."""
    agent = db.query(Agent).filter(Agent.persona_key == persona_key).first()
    if not agent:
        raise HTTPException(404, "Agent not found")
    return _agent_to_response(agent)


@app.get("/api/v1/agents/{persona_key}/state")
async def get_agent_state(persona_key: str):
    """Get agent's current emotion state from Redis."""
    redis = get_redis()
    state = await redis.get_agent_state(persona_key)
    if not state:
        engine = get_engine()
        state = engine._default_state(persona_key)
    return state


# --- Background Processing ---

async def _process_responses(
    session_id: str,
    post_id: str,
    responders: List[str],
    trigger_content: str,
    trigger_author: str
):
    """Process response chain: each sentient responds in sequence."""
    engine = get_engine()
    redis = get_redis()
    
    responses = []
    
    for persona_key in responders:
        try:
            result = await engine.generate_response(
                persona_key=persona_key,
                session_id=session_id,
                trigger_content=trigger_content,
                trigger_author=trigger_author
            )
            
            # Save as comment on the post
            if SessionLocal and result.get("content"):
                db = SessionLocal()
                try:
                    agent = db.query(Agent).filter(Agent.persona_key == persona_key).first()
                    if agent:
                        comment = Comment(
                            post_id=post_id,
                            author_id=agent.id,
                            content=result["content"]
                        )
                        db.add(comment)
                        
                        # Update agent state in DB too
                        agent.state = result.get("state", {})
                        agent.last_seen = datetime.utcnow()
                        
                        db.commit()
                finally:
                    db.close()
            
            responses.append({
                "persona_key": persona_key,
                "content": result["content"],
                "usage": result.get("usage", {})
            })
            
            logger.info(f"[{persona_key}] responded ({result.get('usage', {}).get('output_tokens', 0)} tokens)")
            
        except Exception as e:
            logger.error(f"Error generating response for {persona_key}: {e}")
            responses.append({
                "persona_key": persona_key,
                "error": str(e)
            })
    
    # Mark session as completed
    meta = await redis.get_session_meta(session_id)
    if meta:
        meta["completed"] = True
        meta["participants"] = ["master"] + responders
        await redis.create_session(session_id, meta)
    
    logger.info(f"Session {session_id} completed with {len(responses)} responses")


# --- Response Builders ---

def _post_to_response(post: Post, db) -> PostResponse:
    author = db.query(Agent).filter(Agent.id == post.author_id).first()
    comment_count = db.query(Comment).filter(Comment.post_id == post.id).count()
    
    return PostResponse(
        id=post.id,
        author_id=post.author_id,
        author_name=author.name if author else "unknown",
        author_persona_key=author.persona_key if author else "unknown",
        content=post.content,
        post_type=post.post_type,
        mentions=post.mentions,
        session_id=post.session_id,
        comment_count=comment_count,
        created_at=post.created_at,
        updated_at=post.updated_at
    )


def _comment_to_response(comment: Comment) -> CommentResponse:
    author = comment.author
    return CommentResponse(
        id=comment.id,
        post_id=comment.post_id,
        author_id=comment.author_id,
        author_name=author.name if author else "unknown",
        author_persona_key=author.persona_key if author else "unknown",
        parent_id=comment.parent_id,
        content=comment.content,
        mentions=comment.mentions,
        created_at=comment.created_at
    )


def _agent_to_response(agent: Agent) -> AgentResponse:
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        persona_key=agent.persona_key,
        is_master=agent.is_master,
        state=agent.state,
        last_seen=agent.last_seen,
        online=agent.is_online(),
        created_at=agent.created_at
    )


# --- Run ---

def run():
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run()
