"""
ΛNet Database Setup - PostgreSQL via SQLAlchemy
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()


def get_engine(database_url: str = None):
    """Create database engine for PostgreSQL."""
    url = database_url or os.environ.get(
        "DATABASE_URL",
        "postgresql://localhost/lambdanet"
    )
    # Render uses postgresql:// but SQLAlchemy needs postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    
    return create_engine(url, echo=False, pool_pre_ping=True)


def init_db(database_url: str = None):
    """Initialize database and return session maker."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
