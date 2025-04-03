#!/usr/bin/env python
"""FastAPI server for the QuantaLogic agent."""

import asyncio
import functools
import json
import os
from pickle import NONE
import shutil
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
from fastapi import FastAPI, File, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import BaseModel
from rich.console import Console

import mimetypes
import urllib.parse
from pathlib import Path

from quantalogic.agent import Agent
from quantalogic.agent_config import (
    MODEL_NAME,
)
from quantalogic.agent_factory import AgentRegistry, create_agent_for_mode
from quantalogic.console_print_events import console_print_events
from quantalogic.task_runner import configure_logger
from .utils import handle_sigterm, get_version
from .app_state import server_state, agent_state
from .models import UPLOAD_DIR, AgentConfig, AnalyzePaperRequest, BookNovelRequest, ConvertRequest, CourseRequest, EventMessage, ImageAnalysisRequest, ImageGenerationRequest, JourneyRequest, LinkedInIntroduceContentRequest, QuizRequest, ToolConfig, ToolParameters, TutorialRequest, UserValidationRequest, UserValidationResponse, TaskSubmission, TaskStatus
from .AgentState import AgentState
from .init_agents import init_agents 
from .middlewares.logger_middleware import log_middleware
from .middlewares.error_middleware import  register_exception_handlers
from .middlewares.authenticate import require_auth
from .controllers import agent_router, file_router, task_router, health_router, validation_router, generation_router
from .database import init_db
# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

signal.signal(signal.SIGTERM, handle_sigterm)

async def load_initial_agents():
    """Load initial agents from configuration."""
    logger.info("Loading initial agents...")
    for agent_dict in init_agents:
        try:
            # Convert tool configurations
            tool_configs = []
            for tool in agent_dict["tools"]:
                tool_configs.append(ToolConfig(
                    type=tool["type"],
                    parameters=ToolParameters(**tool.get("parameters", {}))
                ))
            
            # Create AgentConfig from dictionary
            agent_config = AgentConfig(
                id=agent_dict["id"],
                name=agent_dict["name"],
                description=agent_dict["description"],
                expertise=agent_dict["expertise"],
                model_name=agent_dict["model_name"],
                agent_mode=agent_dict.get("agent_mode", "react"),
                tools=tool_configs
            )
            
            success = await agent_state.create_agent(agent_config)
            if success:
                logger.info(f"Successfully loaded agent: {agent_config.name}")
            else:
                logger.error(f"Failed to load agent: {agent_config.name}")
        except Exception as e:
            logger.error(f"Failed to load agent {agent_dict.get('name', 'unknown')}: {str(e)}")

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    try:
        # Initialize database
        init_db()
        
        # Setup signal handlers
        await load_initial_agents()
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

# Add logger middleware
app.middleware("http")(log_middleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Register exception handlers
register_exception_handlers(app)

# Include routers
app.include_router(agent_router)
app.include_router(file_router)
app.include_router(task_router)
app.include_router(health_router)
app.include_router(validation_router)
app.include_router(generation_router)

# Mount static files
# app.mount("/static", StaticFiles(directory="quantalogic/server/static"), name="static")

# Configure Jinja2 templates
# templates = Jinja2Templates(directory="quantalogic/server/templates")



@app.get("/api/agent/events")
async def event_stream(request: Request, task_id: Optional[str] = None) -> StreamingResponse:
    """SSE endpoint for streaming agent events."""

    async def event_generator() -> AsyncGenerator[str, None]:
        # Ensure unique client-task combination
        client_id = agent_state.add_client(task_id)
        logger.debug(f"Client {client_id} subscribed to {'task_id: ' + task_id if task_id else 'all events'}")

        try:
            while not server_state.is_shutting_down:
                if await request.is_disconnected():
                    logger.debug(f"Client {client_id} disconnected")
                    break

                #logger.info(f"Client {client_id} connected")
                #logger.info("agent_state.event_queues: " + str(agent_state.event_queues))

                try:
                    # Prioritize task-specific queue if task_id is provided
                    if task_id and task_id in agent_state.event_queues[client_id]:
                        event = agent_state.event_queues[client_id][task_id].get_nowait()
                        logger.debug(f"Sending task event to client {client_id}: {event}")
                    else:
                        # Fall back to global queue if no task_id
                        event = agent_state.event_queues[client_id]["global"].get_nowait()
                        logger.debug(f"Sending global event to client {client_id}: {event}")

                    # Format and yield the event
                    event_data = event.dict()
                    event_str = f"event: {event.event}\ndata: {json.dumps(event_data)}\n\n"
                    logger.debug(f"Sending SSE data: {event_str}")
                    yield event_str

                except Empty:
                    # Send keepalive to maintain connection
                    yield ": keepalive\n\n"
                    await asyncio.sleep(0.1)  # Increased sleep time to reduce load

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
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",  # Disable proxy buffering
        },
    )

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



@app.get("/api/agent/")
async def get_index(request: Request) -> HTMLResponse:
    """Serve the main application page."""
    # response = templates.TemplateResponse("index.html", {"request": request})
    # response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    # response.headers["Pragma"] = "no-cache"
    # response.headers["Expires"] = "0"
    # return response
    return HTMLResponse(content="")


if __name__ == "__main__":
    config = uvicorn.Config(
        "quantalogic.agent_server:app",
        host="0.0.0.0",
        port=8002,
        #reload=True,
        log_level="info",
        #timeout_keep_alive=5,
        access_log=True,
        #timeout_graceful_shutdown=10,  # Increased from 5 to 10 seconds
    )
    server = uvicorn.Server(config)
    server_state.server = server
    try:
        server.run()
    except KeyboardInterrupt:
        logger.debug("Received keyboard interrupt")
        sys.exit(1)
