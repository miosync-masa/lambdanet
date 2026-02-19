"""
ΛNet Database Setup - PostgreSQL (production) / SQLite (local dev)

When DATABASE_URL points to an unreachable host (e.g. Render internal hostname
from a local machine), automatically falls back to SQLite.
"""

import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger("lambdanet.db")
Base = declarative_base()


def get_engine(database_url: str = None):
    """Create database engine. Falls back to SQLite for local dev."""
    url = database_url if database_url is not None else os.environ.get("DATABASE_URL", "")

    # Empty or not set -> SQLite
    if not url or not url.strip():
        return _sqlite_engine()

    # Render uses postgres:// but SQLAlchemy needs postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # Try PostgreSQL
    try:
        engine = create_engine(url, echo=False, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Connected to PostgreSQL")
        return engine
    except Exception as e:
        logger.warning(f"PostgreSQL connection failed: {e}")
        logger.warning("Falling back to SQLite for local development")
        return _sqlite_engine()


def _sqlite_engine():
    """Create SQLite engine for local development."""
    os.makedirs("data", exist_ok=True)
    url = "sqlite:///data/lambdanet.db"
    logger.info(f"Using SQLite: {url}")
    return create_engine(url, echo=False)


def init_db(database_url: str = None):
    """Initialize database and return session maker."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
