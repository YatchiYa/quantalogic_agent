from fastapi import APIRouter
from typing import Dict
from ..agent_server import agent_state, server_state

router = APIRouter(prefix="/api/agent", tags=["health"])

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
