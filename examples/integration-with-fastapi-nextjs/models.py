"""Pydantic models for the QuantaLogic API."""

from typing import Any, Dict, Optional, List

from pydantic import BaseModel

from quantalogic.agent_config import MODEL_NAME
from datetime import datetime


class EventMessage(BaseModel):
    """Event message model for SSE."""

    id: str
    event: str
    task_id: Optional[str] = None
    data: Dict[str, Any]
    timestamp: str

    model_config = {"extra": "forbid"}


class UserValidationRequest(BaseModel):
    """Request model for user validation."""

    validation_id: str
    tool_name: str
    arguments: Dict[str, Any]

    model_config = {"extra": "forbid"}


class UserValidationResponse(BaseModel):
    """Response model for user validation."""

    approved: bool

    model_config = {"extra": "forbid"}


class ToolParameters(BaseModel):
    """Parameters for a tool configuration."""
    connection_string: Optional[str] = None
    model_name: Optional[str] = None


class ToolConfig(BaseModel):
    """Configuration for a single tool."""
    type: str
    parameters: ToolParameters


class AgentConfig(BaseModel):
    """Configuration for creating a new agent."""
    id: str
    name: str
    description: str
    expertise: str
    mode: str = "custom"
    model_name: str
    tools: List[ToolConfig]


class TaskSubmission(BaseModel):
    """Request model for task submission."""

    task: str
    agent_id: str
    model_name: Optional[str] = MODEL_NAME
    max_iterations: Optional[int] = 30
    mode: Optional[str] = "minimal"
    expertise: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None 

    model_config = {"extra": "forbid"}


class TaskStatus(BaseModel):
    """Task status response model."""

    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    total_tokens: Optional[int] = None
    model_name: Optional[str] = None
