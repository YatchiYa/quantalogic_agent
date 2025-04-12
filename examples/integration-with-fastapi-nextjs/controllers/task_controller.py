import asyncio
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from ..models import TaskSubmission, TaskStatus
from loguru import logger
from ..agent_server import agent_state, server_state

router = APIRouter(prefix="/api/agent", tags=["tasks"])

@router.post("/tasks")
async def submit_task(request: TaskSubmission) -> Dict[str, str]:
    """Submit a new task and return its ID."""
    task_id = await agent_state.submit_task(request)
    # Start task execution in background
    asyncio.create_task(agent_state.execute_task(task_id))
    return {"task_id": task_id}

@router.post("/chat")
async def submit_chat(request: TaskSubmission) -> Dict[str, str]:
    """Submit a new chat and return its ID."""
    chat_id = await agent_state.submit_task(request)
    # Start chat execution in background
    asyncio.create_task(agent_state.execute_chat(chat_id))
    return {"task_id": chat_id}

@router.post("/get_news")
async def submit_chat(request: TaskSubmission) -> Dict[str, str]:
    """Submit a new chat and return its ID."""
    chat_id = await agent_state.submit_task(request)
    # Start chat execution in background
    asyncio.create_task(agent_state.get_news(chat_id))
    return {"task_id": chat_id}

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str) -> TaskStatus:
    """Get the status of a specific task."""
    if task_id not in agent_state.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = agent_state.tasks[task_id]
    return TaskStatus(task_id=task_id, **task)

@router.get("/tasks")
async def list_tasks(status: Optional[str] = None, limit: int = 10, offset: int = 0) -> List[TaskStatus]:
    """List all tasks with optional filtering."""
    tasks = []
    for task_id, task in agent_state.tasks.items():
        if status is None or task["status"] == status:
            tasks.append(TaskStatus(task_id=task_id, **task))

    return tasks[offset : offset + limit]
