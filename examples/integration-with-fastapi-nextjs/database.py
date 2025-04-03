"""Database connection and models for the QuantaLogic agent server."""
import os
from datetime import datetime
import uuid
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, Column, String, JSON, ARRAY, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from loguru import logger

DATABASE_URL = "postgresql://quantadbu:azerty1234@localhost:5432/quanta_db"
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

engine = create_engine(DATABASE_URL, echo=True)  # Added echo=True for debugging
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Agent(Base):
    __tablename__ = "qagents"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    pid = Column(UUID(as_uuid=True))
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    model_name = Column(String, nullable=False)
    expertise = Column(String, nullable=True)
    project = Column(String, nullable=True)
    agent_mode = Column(String, nullable=True, default="default")
    tags = Column(ARRAY(String), nullable=True)
    tools = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    user_id = Column(String, nullable=True)  # Changed from UUID to String and made nullable
    organization_id = Column(String, nullable=True)  # Changed from UUID to String and made nullable

class QConversation(Base):
    __tablename__ = "qconversations"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    pid = Column(UUID(as_uuid=True))
    project = Column(String, nullable=True)
    title = Column(String, nullable=True)
    description = Column(String, nullable=True)
    model_id = Column(String, nullable=True)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    organization_id = Column(UUID(as_uuid=True), nullable=True)
    agent_id = Column(UUID(as_uuid=True), nullable=True)
    last_message_at = Column(DateTime(timezone=True), server_default=func.now())
    is_public = Column(Boolean, default=False)
    is_archived = Column(Boolean, default=False)
    is_favorite = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # think about adding partion by organization_id, user_id

# Create database tables
def init_db():
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully!")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
