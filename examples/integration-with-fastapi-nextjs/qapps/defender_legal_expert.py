

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
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger
from pydantic import BaseModel
from rich.console import Console

from quantalogic import console_print_token
from quantalogic.agent import Agent
from quantalogic.agent_config import (
    MODEL_NAME,
)
from quantalogic.agent_factory import AgentRegistry, create_agent_for_mode
from quantalogic.create_custom_agent import create_custom_agent
from quantalogic.console_print_events import console_print_events
from quantalogic.memory import AgentMemory, VariableMemory
from quantalogic.task_runner import configure_logger
from ..models import DefendLegalCaseRequest


async def defendLegalCaseModal(
    self,
    task_id: str, 
    request: DefendLegalCaseRequest
    ) -> None:
    """Execute a defend legal case task asynchronously."""
    if task_id not in self.tasks:
        raise ValueError(f"Task {task_id} not found")

    task_info = self.tasks[task_id]
    task_info["started_at"] = datetime.now().isoformat()
    task_info["status"] = "running"

    try:
        logger.info(f"== Starting image analysis: {task_id}") 
        
        # Add the examples directory to Python path
        examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'examples', "integration-with-fastapi-nextjs"))
        sys.path.append(examples_dir)
        
        from flows.legal.case_defender.legal_case_defender import defend_legal_case
        
        # Run tutorial generation in a thread to not block the event loop
        loop = asyncio.get_running_loop()

        # Create event for task completion
        self._handle_event("task_solve_start", {
            "task_id": task_id, 
            "agent_id": "default",
            "message": "Defend legal case started"
        })

        result = await loop.run_in_executor(
            None,
            lambda: asyncio.run(defend_legal_case(  
                case_details=request.case_details,
                analysis_model=request.analysis_model,
                strategy_model=request.strategy_model,
                defense_model=request.defense_model,
                language=request.language,
                document_type=request.document_type,
                custom_metadata_instructions=request.custom_metadata_instructions,
                custom_analysis_instructions=request.custom_analysis_instructions,
                output_dir=request.output_dir,
                task_id=task_id,
                _handle_event=self._handle_event
            ))
        )
        logger.debug(f"================================================================")
        logger.info(f"================================================================")
        logger.info(f"== Talk with document result: {result}")

        self._update_task_success(task_info, result, None) 
        
        # Create event for task completion
        try:
            self._handle_event("task_solve_end", {
                "task_id": task_id, 
                "agent_id": "default",
                "message": "Defend legal case completed",
                "result": result
            })
        except Exception as e:
            logger.error(f"Error sending completion event: {e}")
            # Send a simplified event without the result
            self._handle_event("task_solve_end", {
                "task_id": task_id,
                "agent_id": "default", 
                "message": f"Defend legal case completed (error sending full result: {str(e)})"
            })
        
    except Exception as e:
        self._update_task_failure(task_info, e)
        
        # Create event for task failure with safe error message
        try:
            self._handle_event("error_tool_execution", {
                "task_id": task_id,
                "message": f"Defend legal case failed: {str(e)}"
            })
        except Exception as event_error:
            logger.error(f"Error sending failure event: {event_error}")
        
        logger.exception(f"Error defending legal case {task_id}")
    finally:
        self.remove_task_event_queue(task_id) 
        
        # Create final event with minimal data
        try:
            self._handle_event("task_solve_end", {
                "task_id": task_id,
                "agent_id": "default",
                "message": "Defend legal case completed"
            })
        except Exception as e:
            logger.error(f"Error sending final event: {e}")
