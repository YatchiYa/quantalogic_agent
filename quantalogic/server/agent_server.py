#!/usr/bin/env python
"""FastAPI server for the QuantaLogic agent."""

import asyncio
import functools
import json
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from queue import Empty, Queue
from threading import Lock
from typing import Any, AsyncGenerator, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import BaseModel
from rich.console import Console

from quantalogic.agent import Agent
from quantalogic.agent_config import (
    MODEL_NAME,
)
from quantalogic.agent_factory import AgentRegistry, create_agent_for_mode
from quantalogic.console_print_events import console_print_events
from quantalogic.task_runner import configure_logger

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Constants
SHUTDOWN_TIMEOUT = 5.0  # seconds
VALIDATION_TIMEOUT = 30.0  # seconds


def handle_sigterm(signum, frame):
    """Handle SIGTERM signal."""
    logger.debug("Received SIGTERM signal")
    raise SystemExit(0)


signal.signal(signal.SIGTERM, handle_sigterm)


def get_version() -> str:
    """Get the current version of the package."""
    return "QuantaLogic version: 1.0.0"


class ServerState:
    """Global server state management."""

    def __init__(self):
        """Initialize the global server state."""
        self.interrupt_count = 0
        self.force_exit = False
        self.is_shutting_down = False
        self.shutdown_initiated = asyncio.Event()
        self.shutdown_complete = asyncio.Event()
        self.server = None

    async def initiate_shutdown(self, force: bool = False):
        """Initiate the shutdown process."""
        if not self.is_shutting_down or force:
            logger.debug("Initiating server shutdown...")
            self.is_shutting_down = True
            self.force_exit = force
            self.shutdown_initiated.set()
            if force:
                # Force exit immediately
                logger.warning("Forcing immediate shutdown...")
                sys.exit(1)
            await self.shutdown_complete.wait()

    def handle_interrupt(self):
        """Handle interrupt signal."""
        self.interrupt_count += 1
        if self.interrupt_count == 1:
            logger.debug("Graceful shutdown initiated (press Ctrl+C again to force)")
            asyncio.create_task(self.initiate_shutdown(force=False))
        else:
            logger.warning("Forced shutdown initiated...")
            # Use asyncio.create_task to avoid RuntimeError
            asyncio.create_task(self.initiate_shutdown(force=True))


# Models
class EventMessage(BaseModel):
    """Event message model for SSE."""

    id: str
    event: str
    task_id: Optional[str] = None  # Added task_id field
    data: Dict[str, Any]
    timestamp: str

    model_config = {"extra": "forbid"}


class UserValidationRequest(BaseModel):
    """Request model for user validation."""

    question: str
    validation_id: str | None = None

    model_config = {"extra": "forbid"}


class UserValidationResponse(BaseModel):
    """Response model for user validation."""

    response: bool

    model_config = {"extra": "forbid"}


class TaskSubmission(BaseModel):
    """Request model for task submission."""

    task: str
    model_name: Optional[str] = MODEL_NAME
    max_iterations: Optional[int] = 30

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


class AgentState:
    """Manages agent state and event queues."""

    def __init__(self):
        """Initialize the agent state."""
        self.agent = None
        self.agent_registry = AgentRegistry()
        self.event_queues: Dict[str, Dict[str, Queue]] = {}
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.queue_lock = Lock()
        self.client_counter = 0
        self.console = Console()
        self.validation_requests: Dict[str, Dict[str, Any]] = {}
        self.validation_responses: Dict[str, asyncio.Queue] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task_queues: Dict[str, asyncio.Queue] = {}
        self.agent_events = [
            "session_start",
            "session_end",
            "session_add_message",
            "task_solve_start",
            "task_solve_end",
            "task_think_start",
            "task_think_end",
            "task_complete",
            "tool_execution_start",
            "tool_execution_end",
            "tool_execute_validation_start",
            "tool_execute_validation_end",
            "memory_full",
            "memory_compacted",
            "memory_summary",
            "error_max_iterations_reached",
            "error_tool_execution",
            "error_model_response",
        ]

    async def initialize_agent_with_sse_validation(self, model_name: str = MODEL_NAME) -> Agent:
        """Initialize agent with SSE-based user validation.
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            The initialized agent instance
            
        Raises:
            Exception: If agent initialization fails
        """
        try:
            logger.info(f"Initializing agent with model: {model_name}")
            
            if "default" not in self.agent_registry._agents:
                self.agent = self._create_minimal_agent(model_name)
                self._setup_agent_events(self.agent)
                self.agent_registry.register_agent("default", self.agent)
                logger.info("Agent initialized successfully with minimal mode")

            return self.agent_registry.get_agent("default")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}", exc_info=True)
            raise

    def _create_minimal_agent(self, model_name: str) -> Agent:
        """Create a minimal agent with the specified model.
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            The created agent instance
        """
        return create_agent_for_mode(
            mode="minimal",
            model_name=model_name,
            vision_model_name=None,
            no_stream=False
        )

    def _setup_agent_events(self, agent: Agent) -> None:
        """Set up event handlers for the agent.
        
        Args:
            agent: The agent instance to set up events for
        """
        for event in self.agent_events:
            agent.event_emitter.on(event, lambda e, d, event=event: self._handle_event(event, d))
            
    def _handle_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle agent events with rich console output.
        
        Args:
            event_type: Type of the event
            data: Event data
        """
        try:
            event_str = f"[bold]{event_type}[/bold]"
            if isinstance(data, dict):
                if "message" in data:
                    event_str += f": {data['message']}"
                elif "error" in data:
                    event_str += f": [red]{data['error']}[/red]"
                
            self.console.print(event_str)
            self.broadcast_event(event_type, data)
            
        except Exception as e:
            logger.error(f"Error handling event {event_type}: {e}")

    async def execute_task(self, task_id: str) -> None:
        """Execute a task asynchronously.
        
        Args:
            task_id: ID of the task to execute
            
        Raises:
            ValueError: If task is not found
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task_info = self.tasks[task_id]
        task_info["started_at"] = datetime.now().isoformat()
        task_info["status"] = "running"

        try:
            agent = await self.initialize_agent_with_sse_validation(
                task_info.get("request", {}).get("model_name", MODEL_NAME)
            )
            
            task_queue = self.get_task_event_queue(task_id)
            
            # Set up event handling for this task
            async def handle_task_event(event_type: str, data: Dict[str, Any]):
                if task_queue:
                    await task_queue.put({"type": event_type, "data": data})
            
            for event in self.agent_events:
                agent.event_emitter.on(event, handle_task_event)
            
            # Run solve_task in a thread to not block the event loop
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,  # Use default executor
                lambda: agent.solve_task(
                    task=task_info["request"]["task"],
                    max_iterations=task_info["request"].get("max_iterations", 30),
                    streaming=False,
                    clear_memory=True
                )
            )

            self._update_task_success(task_info, result, agent)
            
        except Exception as e:
            self._update_task_failure(task_info, e)
            logger.exception(f"Error executing task {task_id}")
        finally:
            self.remove_task_event_queue(task_id)

    def _update_task_success(self, task_info: Dict[str, Any], result: str, agent: Agent) -> None:
        """Update task info after successful execution.
        
        Args:
            task_info: Task information dictionary
            result: Task execution result
            agent: Agent that executed the task
        """
        task_info["completed_at"] = datetime.now().isoformat()
        task_info["status"] = "completed"
        task_info["result"] = result
        task_info["total_tokens"] = agent.total_tokens if hasattr(agent, "total_tokens") else None
        task_info["model_name"] = self.get_current_model_name()
        
    def _update_task_failure(self, task_info: Dict[str, Any], error: Exception) -> None:
        """Update task info after failed execution.
        
        Args:
            task_info: Task information dictionary
            error: Exception that caused the failure
        """
        task_info["completed_at"] = datetime.now().isoformat()
        task_info["status"] = "failed"
        task_info["error"] = str(error)

    def add_client(self, task_id: Optional[str] = None) -> str:
        """Add a new client and return its ID.

        Ensures unique client-task combination.
        """
        with self.queue_lock:
            # Generate a unique client ID
            client_id = f"client_{self.client_counter}"
            self.client_counter += 1

            # Initialize nested event queue structure
            if client_id not in self.event_queues:
                self.event_queues[client_id] = {}
                self.active_agents[client_id] = {}

            if task_id:
                # Prevent multiple agents for the same client-task combination
                if task_id in self.active_agents[client_id]:
                    raise ValueError(f"An agent already exists for client {client_id} and task {task_id}")

                # Create a specific queue for this client-task combination
                self.event_queues[client_id][task_id] = Queue()
                self.active_agents[client_id][task_id] = {
                    "created_at": datetime.utcnow().isoformat(),
                    "status": "active",
                }
            else:
                # Global client queue
                self.event_queues[client_id] = {"global": Queue()}

            return client_id

    def remove_client(self, client_id: str, task_id: Optional[str] = None):
        """Remove a client's event queue, optionally for a specific task."""
        with self.queue_lock:
            if client_id in self.event_queues:
                if task_id and task_id in self.event_queues[client_id]:
                    # Remove specific task queue for this client
                    del self.event_queues[client_id][task_id]

                    # Remove active agent for this client-task
                    if client_id in self.active_agents and task_id in self.active_agents[client_id]:
                        del self.active_agents[client_id][task_id]
                else:
                    # Remove entire client entry
                    del self.event_queues[client_id]

                    # Remove all active agents for this client
                    if client_id in self.active_agents:
                        del self.active_agents[client_id]

    def broadcast_event(
        self, event_type: str, data: Dict[str, Any], task_id: Optional[str] = None, client_id: Optional[str] = None
    ):
        """Broadcast an event to specific client-task queues or globally.

        Allows optional filtering by client_id and task_id to prevent event leakage.
        """
        event = EventMessage(
            id=str(uuid.uuid4()), event=event_type, task_id=task_id, data=data, timestamp=datetime.utcnow().isoformat()
        )

        with self.queue_lock:
            for curr_client_id, client_queues in self.event_queues.items():
                # Skip if specific client_id is provided and doesn't match
                if client_id and curr_client_id != client_id:
                    continue

                if task_id and task_id in client_queues:
                    # Send to specific task queue
                    client_queues[task_id].put(event)
                elif not task_id and "global" in client_queues:
                    # Send to global queue if no task specified
                    client_queues["global"].put(event)

    def get_current_model_name(self) -> str:
        """Get the current model name safely."""
        if self.agent and self.agent.model:
            return self.agent.model.model
        return MODEL_NAME

    async def cleanup(self):
        """Clean up resources during shutdown."""
        try:
            logger.debug("Cleaning up resources...")
            if server_state.force_exit:
                logger.warning("Forced cleanup - skipping graceful shutdown")
                return

            async with asyncio.timeout(SHUTDOWN_TIMEOUT):
                with self.queue_lock:
                    # Notify all clients
                    self.broadcast_event("server_shutdown", {"message": "Server is shutting down"})
                    # Clear queues
                    self.event_queues.clear()
                    self.validation_requests.clear()
                    self.validation_responses.clear()
                # Clear agent
                self.agent = None
                logger.debug("Cleanup completed")
        except TimeoutError:
            logger.warning(f"Cleanup timed out after {SHUTDOWN_TIMEOUT} seconds")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
        finally:
            self.agent = None
            if server_state.force_exit:
                sys.exit(1)

    async def submit_task(self, task_request: TaskSubmission) -> str:
        """Submit a new task and return its ID."""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "request": task_request.dict(),
        }
        self.task_queues[task_id] = asyncio.Queue()
        return task_id

    async def get_task_event_queue(self, task_id: str) -> Queue:
        """Get or create a task-specific event queue."""
        with self.queue_lock:
            if task_id not in self.task_queues:
                self.task_queues[task_id] = Queue()
            return self.task_queues[task_id]

    def remove_task_event_queue(self, task_id: str):
        """Remove a task-specific event queue."""
        with self.queue_lock:
            if task_id in self.task_queues:
                del self.task_queues[task_id]
                logger.debug(f"Removed event queue for task_id: {task_id}")


# Initialize global states
server_state = ServerState()
agent_state = AgentState()


# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    try:
        # Setup signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(handle_shutdown(s)))
        yield
    finally:
        logger.debug("Shutting down server gracefully...")
        await server_state.initiate_shutdown()
        await agent_state.cleanup()
        server_state.shutdown_complete.set()
        logger.debug("Server shutdown complete")


async def handle_shutdown(sig):
    """Handle shutdown signals."""
    if sig == signal.SIGINT and server_state.interrupt_count >= 1:
        # Force exit on second CTRL+C
        await server_state.initiate_shutdown(force=True)
    else:
        server_state.handle_interrupt()


app = FastAPI(
    title="QuantaLogic API",
    description="AI Agent Server for QuantaLogic",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="quantalogic/server/static"), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="quantalogic/server/templates")


# Middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.debug(
        f"Path: {request.url.path} "
        f"Method: {request.method} "
        f"Time: {process_time:.3f}s "
        f"Status: {response.status_code}"
    )

    return response


@app.post("/validate_response/{validation_id}")
async def submit_validation_response(validation_id: str, response: UserValidationResponse):
    """Submit a validation response."""
    if validation_id not in agent_state.validation_responses:
        raise HTTPException(status_code=404, detail="Validation request not found")

    try:
        response_queue = agent_state.validation_responses[validation_id]
        await response_queue.put(response.response)
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        logger.error(f"Error processing validation response: {e}")
        raise HTTPException(status_code=500, detail="Failed to process validation response")


@app.get("/events")
async def event_stream(request: Request, task_id: Optional[str] = None) -> StreamingResponse:
    """SSE endpoint for streaming agent events."""

    async def event_generator() -> AsyncGenerator[str, None]:
        # Ensure unique client-task combination
        client_id = agent_state.add_client(task_id)
        logger.debug(f"Client {client_id} subscribed to {'task_id: ' + task_id if task_id else 'all events'}")

        try:
            while not server_state.is_shutting_down:
                if await request.is_disconnected():
                    break

                try:
                    # Prioritize task-specific queue if task_id is provided
                    if task_id:
                        event = agent_state.event_queues[client_id][task_id].get_nowait()
                    else:
                        # Fall back to global queue if no task_id
                        event = agent_state.event_queues[client_id]["global"].get_nowait()

                    # Yield the event
                    yield f"event: {event.event}\ndata: {json.dumps(event.dict())}\n\n"

                except Empty:
                    # Send keepalive to maintain connection
                    yield ": keepalive\n\n"
                    await asyncio.sleep(0.1)

                if server_state.is_shutting_down:
                    yield 'event: shutdown\ndata: {"message": "Server shutting down"}\n\n'
                    break

        finally:
            # Clean up the client's event queue
            agent_state.remove_client(client_id, task_id)
            logger.debug(f"Client {client_id} {'unsubscribed from task_id: ' + task_id if task_id else 'disconnected'}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        },
    )


@app.get("/")
async def get_index(request: Request) -> HTMLResponse:
    """Serve the main application page."""
    response = templates.TemplateResponse("index.html", {"request": request})
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.post("/tasks")
async def submit_task(request: TaskSubmission) -> Dict[str, str]:
    """Submit a new task and return its ID."""
    task_id = await agent_state.submit_task(request)
    # Start task execution in background
    asyncio.create_task(agent_state.execute_task(task_id))
    return {"task_id": task_id}


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str) -> TaskStatus:
    """Get the status of a specific task."""
    if task_id not in agent_state.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = agent_state.tasks[task_id]
    return TaskStatus(task_id=task_id, **task)


@app.get("/tasks")
async def list_tasks(status: Optional[str] = None, limit: int = 10, offset: int = 0) -> List[TaskStatus]:
    """List all tasks with optional filtering."""
    tasks = []
    for task_id, task in agent_state.tasks.items():
        if status is None or task["status"] == status:
            tasks.append(TaskStatus(task_id=task_id, **task))

    return tasks[offset : offset + limit]


if __name__ == "__main__":
    config = uvicorn.Config(
        "quantalogic.agent_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        timeout_keep_alive=5,
        access_log=True,
        timeout_graceful_shutdown=5,  # Reduced from 10 to 5 seconds
    )
    server = uvicorn.Server(config)
    server_state.server = server
    try:
        server.run()
    except KeyboardInterrupt:
        logger.debug("Received keyboard interrupt")
        sys.exit(1)
