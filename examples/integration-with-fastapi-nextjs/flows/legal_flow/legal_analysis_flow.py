import asyncio
from pathlib import Path
from typing import Annotated, List, Optional, Union
import os
import shutil

import typer
from loguru import logger
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from ..service import event_observer
from quantalogic.tools.rag_tool.legal_embedding_rag import LegalEmbeddingRAG, ResponseMode


# Initialize Typer app and rich console
app = typer.Typer(help="Analyze legal documents using LegalEmbeddingRAG")
console = Console()

# Define Pydantic models for structured data
class LegalQuery(BaseModel):
    query_text: str
    max_sources: int = 5
    min_relevance: float = 0.5
    response_mode: str = ResponseMode.SOURCES_ONLY

class LegalAnalysisResult(BaseModel):
    query: str
    sources: List[str]
    answer: Optional[str]
    context: Optional[str]

# Node: Initialize RAG Tool
@Nodes.define(output="rag_tool")
async def initialize_rag_tool(
    persist_dir: str,
    document_paths: List[str],
    chunk_size: int,
    chunk_overlap: int,
    force_reindex: bool
) -> LegalEmbeddingRAG:
    """Initialize the LegalEmbeddingRAG tool."""
    try:
        # Clean up existing index if force_reindex is True
        if force_reindex and os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        
        tool = LegalEmbeddingRAG(
            persist_dir=persist_dir,
            document_paths=document_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_reindex=force_reindex
        )
        logger.info(f"Initialized RAG tool with {len(document_paths)} documents")
        return tool
    except Exception as e:
        logger.error(f"Error initializing RAG tool: {e}")
        raise

# Node: Validate Query
@Nodes.define(output="validated_query")
async def validate_query(query: str, max_sources: int, min_relevance: float, response_mode: str) -> LegalQuery:
    """Validate and structure the query parameters."""
    try:
        legal_query = LegalQuery(
            query_text=query,
            max_sources=max_sources,
            min_relevance=min_relevance,
            response_mode=response_mode
        )
        logger.info(f"Validated query: {legal_query.query_text}")
        return legal_query
    except Exception as e:
        logger.error(f"Error validating query: {e}")
        raise

# Node: Execute Query
@Nodes.define(output="query_result")
async def execute_query(rag_tool: LegalEmbeddingRAG, validated_query: LegalQuery) -> str:
    """Execute the query using the RAG tool."""
    try:
        result = rag_tool.execute(
            query=validated_query.query_text,
            max_sources=validated_query.max_sources,
            min_relevance=validated_query.min_relevance,
            response_mode=validated_query.response_mode
        )
        logger.info("Query executed successfully")
        return result
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise

# Node: Format Results
@Nodes.define(output="formatted_result")
async def format_results(query_result: str, validated_query: LegalQuery) -> str:
    """Format the results for display."""
    try:
        formatted = f"""# Legal Analysis Results

## Query
{validated_query.query_text}

## Results
{query_result}
"""
        logger.info("Results formatted successfully")
        return formatted
    except Exception as e:
        logger.error(f"Error formatting results: {e}")
        raise

# Create the Workflow
def create_legal_analysis_workflow() -> Workflow:
    """Create a workflow for legal document analysis."""
    wf = Workflow("initialize_rag_tool")
    
    # Add nodes
    wf.node("initialize_rag_tool")
    wf.node("validate_query")
    wf.node("execute_query")
    wf.node("format_results")
    
    # Define transitions
    wf.transitions["initialize_rag_tool"] = [("validate_query", None)]
    wf.transitions["validate_query"] = [("execute_query", None)]
    wf.transitions["execute_query"] = [("format_results", None)]
    
    return wf

# Main execution function
async def analyze_legal_documents(
    document_paths: List[str],
    query: str,
    persist_dir: str = "./storage/legal_embedding_rag",
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    force_reindex: bool = False,
    max_sources: int = 5,
    min_relevance: float = 0.5,
    response_mode: str = ResponseMode.SOURCES_ONLY,
    _handle_event: Optional[callable] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Execute the legal analysis workflow."""
    initial_context = {
        "persist_dir": persist_dir,
        "document_paths": document_paths,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "force_reindex": force_reindex,
        "query": query,
        "max_sources": max_sources,
        "min_relevance": min_relevance,
        "response_mode": response_mode
    }

    try:
        workflow = create_legal_analysis_workflow()
        engine = workflow.build()
    
        # Add the event observer if _handle_event is provided
        if _handle_event:
            # Create a lambda to bind task_id to the observer
            bound_observer = lambda event: asyncio.create_task(
                event_observer(event, task_id=task_id, _handle_event=_handle_event)
            )
            engine.add_observer(bound_observer)

        result = await engine.run(initial_context)
        return result
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise

@app.command()
def analyze(
    document_paths: Annotated[List[str], typer.Argument(help="Paths to legal documents")],
    query: Annotated[str, typer.Argument(help="Query to analyze")],
    persist_dir: Annotated[str, typer.Option(help="Directory to persist embeddings")] = "./storage/legal_embedding_rag",
    chunk_size: Annotated[int, typer.Option(help="Size of text chunks")] = 512,
    chunk_overlap: Annotated[int, typer.Option(help="Overlap between chunks")] = 128,
    force_reindex: Annotated[bool, typer.Option(help="Force reindexing of documents")] = False,
    max_sources: Annotated[int, typer.Option(help="Maximum number of sources to return")] = 5,
    min_relevance: Annotated[float, typer.Option(help="Minimum relevance score")] = 0.5,
    response_mode: Annotated[str, typer.Option(help="Response mode")] = ResponseMode.SOURCES_ONLY
):
    """Analyze legal documents using LegalEmbeddingRAG workflow."""
    try:
        with console.status(f"Processing query: [bold blue]{query}[/]..."):
            result = asyncio.run(analyze_legal_documents(
                document_paths=document_paths,
                query=query,
                persist_dir=persist_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                force_reindex=force_reindex,
                max_sources=max_sources,
                min_relevance=min_relevance,
                response_mode=response_mode
            ))
        
        formatted_result = result["formatted_result"]
        console.print("\n[bold green]Analysis Results:[/]")
        console.print(Panel(Markdown(formatted_result), border_style="blue"))
        
    except Exception as e:
        logger.error(f"Failed to run workflow: {e}")
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    # Test example with predefined parameters
    test_documents = [
        "./docs/folder_test/code_civile.md",
        "./docs/folder_test/code_procedure.md"
    ]
    
    test_queries = [
        "Lois algériennes concernant les ouvertures (fenêtres) donnant sur la propriété voisine",
        "Procédures légales pour l'héritage en Algérie",
        "Droits et obligations des locataires"
    ]
    
    # Test configuration
    test_config = {
        "persist_dir": "./storage/legal_embedding_rag",
        "chunk_size": 512,
        "chunk_overlap": 128,
        "force_reindex": True,  # Force reindex for testing
        "max_sources": 5,
        "min_relevance": 0.5
    }
    
    console.print("[bold blue]Starting Legal Analysis Flow Test[/]\n")
    
    # Test different response modes
    response_modes = [
        ResponseMode.SOURCES_ONLY,
        ResponseMode.CONTEXTUAL_ANSWER,
        ResponseMode.ANSWER_WITH_SOURCES
    ]
    
    try:
        for query in test_queries:
            console.print(f"\n[bold yellow]Testing Query:[/] {query}")
            console.print("[bold cyan]Testing different response modes:[/]")
            
            for mode in response_modes:
                console.print(f"\n[bold green]Response Mode:[/] {mode}")
                try:
                    with console.status(f"Processing with {mode}..."):
                        result = asyncio.run(analyze_legal_documents(
                            document_paths=test_documents,
                            query=query,
                            response_mode=mode,
                            **test_config
                        ))
                    
                    formatted_result = result["formatted_result"]
                    console.print(Panel(Markdown(formatted_result), border_style="blue"))
                    
                except Exception as e:
                    console.print(f"[bold red]Error with {mode}:[/] {str(e)}")
                    continue
                
            console.print("\n" + "="*50 + "\n")
    
    except Exception as e:
        console.print(f"[bold red]Test execution failed:[/] {str(e)}")
        raise typer.Exit(code=1)
    
    console.print("[bold green]Test execution completed![/]")
