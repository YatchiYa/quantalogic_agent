import asyncio
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

import anyio
from loguru import logger
from pydantic import BaseModel

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from ..service import event_observer
from .legal_embedding_rag import LegalEmbeddingRAG, ResponseMode, LegalContext

# Configure logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Define structured output models
class LegalQueryInput(BaseModel):
    query: str
    context: Optional[LegalContext] = None
    max_sources: int = 10
    min_relevance: float = 0.1
    response_mode: ResponseMode = ResponseMode.ANSWER_WITH_SOURCES

class LegalSearchResult(BaseModel):
    query: str
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Helper function to get template paths
def get_template_path(template_name):
    return os.path.join(TEMPLATES_DIR, template_name)

# Custom Observer for Workflow Events
async def legal_search_observer(event: WorkflowEvent):
    if event.event_type == WorkflowEventType.WORKFLOW_STARTED:
        print(f"\n{'='*50}\n🔍 Starting Legal Search Process 🔍\n{'='*50}")
    elif event.event_type == WorkflowEventType.NODE_STARTED:
        print(f"\n🔄 [{event.node_name}] Starting...")
    elif event.event_type == WorkflowEventType.NODE_COMPLETED:
        if event.node_name == "execute_search":
            print(f"✅ [{event.node_name}] Search completed")
            if event.result:
                preview = str(event.result)[:200] + "..." if len(str(event.result)) > 200 else str(event.result)
                print(f"    Preview:\n    {preview}")
        else:
            print(f"✅ [{event.node_name}] Completed")
    elif event.event_type == WorkflowEventType.WORKFLOW_COMPLETED:
        print(f"\n{'='*50}\n🎉 Legal Search Process Finished 🎉\n{'='*50}")

# Workflow Nodes
@Nodes.define(output=None)
async def initialize_rag_tool(
    persist_dir: str = "./storage/legal_embedding_rag",
    document_paths: Optional[List[str]] = None,
    force_reindex: bool = False
) -> dict:
    """Initialize the Legal RAG tool."""
    logger.info("Initializing Legal RAG tool...")
    tool = LegalEmbeddingRAG(
        persist_dir=persist_dir,
        document_paths=document_paths,
        force_reindex=force_reindex
    )
    return {"tool": tool}

@Nodes.define(output=None)
async def validate_query(query: str) -> dict:
    """Validate the input query."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    return {"validated_query": query.strip()}

@Nodes.define(output=None)
async def prepare_search_context(
    validated_query: str,
    legal_context: Optional[LegalContext] = None
) -> dict:
    """Prepare the search context with query and legal metadata."""
    context = {
        "query": validated_query,
        "timestamp": datetime.now().isoformat(),
        "legal_context": legal_context.dict() if legal_context else {}
    }
    return {"search_context": context}

@Nodes.define(output="search_result")
async def execute_search(
    tool: LegalEmbeddingRAG,
    validated_query: str,
    search_context: dict,
    max_sources: int = 10,
    min_relevance: float = 0.1,
    response_mode: ResponseMode = ResponseMode.ANSWER_WITH_SOURCES
) -> str:
    """Execute the legal search using the RAG tool."""
    logger.info(f"Executing search with mode: {response_mode}")
    return tool.execute(
        query=validated_query,
        max_sources=max_sources,
        min_relevance=min_relevance,
        response_mode=response_mode
    )

@Nodes.define(output=None)
async def process_results(search_result: str) -> dict:
    """Process and format the search results."""
    import json
    try:
        results = json.loads(search_result)
        return {"processed_results": results}
    except Exception as e:
        logger.error(f"Error processing results: {e}")
        return {"processed_results": {"error": str(e)}}

@Nodes.define(output=None)
async def format_output(
    processed_results: dict,
    search_context: dict
) -> dict:
    """Format the final output with results and context."""
    output = {
        "query": search_context["query"],
        "timestamp": search_context["timestamp"],
        "legal_context": search_context.get("legal_context", {}),
        "results": processed_results
    }
    return {"final_output": output}

# Define the Workflow
workflow = (
    Workflow("initialize_rag_tool")
    .then("validate_query")
    .then("prepare_search_context")
    .then("execute_search")
    .then("process_results")
    .then("format_output")
)

def execute_legal_search(
    query: str,
    persist_dir: str = "./storage/legal_embedding_rag",
    document_paths: Optional[List[str]] = None,
    force_reindex: bool = False,
    legal_context: Optional[LegalContext] = None,
    max_sources: int = 10,
    min_relevance: float = 0.1,
    response_mode: ResponseMode = ResponseMode.ANSWER_WITH_SOURCES,
    task_id: str = "default",
    _handle_event: Optional[callable] = None
) -> dict:
    """Execute the legal search workflow."""
    if not query:
        raise ValueError("Query cannot be empty")

    initial_context = {
        "persist_dir": persist_dir,
        "document_paths": document_paths,
        "force_reindex": force_reindex,
        "query": query,
        "legal_context": legal_context,
        "max_sources": max_sources,
        "min_relevance": min_relevance,
        "response_mode": response_mode
    }

    logger.info(f"Starting legal search workflow for query: {query}")
    engine = workflow.build()

    # Add event observers
    if _handle_event:
        bound_observer = lambda event: asyncio.create_task(
            event_observer(event, task_id=task_id, _handle_event=_handle_event)
        )
        engine.add_observer(bound_observer)

    # Add the default observer for console output
    engine.add_observer(legal_search_observer)

    result = anyio.run(engine.run, initial_context)
    logger.info("Legal search workflow completed successfully 🎉")
    return result.get("final_output", {})
