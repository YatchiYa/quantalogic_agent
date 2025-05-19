import asyncio
from pathlib import Path
from typing import Annotated, List, Optional, Union, Dict, Any
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
# from ..service import event_observer
from quantalogic.tools.rag_tool.general_rag_tool import ResponseMode 
from quantalogic.tools.rag_tool.legal_rag_tool import AlgerianLegalRagTool


# Initialize Typer app and rich console
app = typer.Typer(help="Analyze InRag documents using AlgerianLegalRagTool")
console = Console()

# Define Pydantic models for structured data
class InRagQuery(BaseModel):
    query_text: str
    max_sources: int = 8
    min_relevance: float = 0.1
    use_llm: bool = True

class InRagAnalysisResult(BaseModel):
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
    force_reindex: bool,
    model_name: str = "text-embedding-3-large"
) -> AlgerianLegalRagTool:
    """Initialize the AlgerianLegalRagTool tool."""
    try:
        # Clean up existing index if force_reindex is True
        if force_reindex and os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        
        # Use home directory if persist_dir is in a problematic location
        home_dir = os.path.expanduser("~")
        if not persist_dir.startswith(home_dir):
            logger.warning(f"Persist directory {persist_dir} might not be writable, using home directory instead")
            persist_dir = os.path.join(home_dir, ".simple_rag_storage")
            
        # Ensure the directory exists with proper permissions
        os.makedirs(persist_dir, exist_ok=True)
        try:
            os.chmod(persist_dir, 0o755)  # rwxr-xr-x
        except Exception as e:
            logger.warning(f"Could not set permissions on persist directory: {e}")
        
        tool = AlgerianLegalRagTool(
            persist_dir=persist_dir,
            document_paths=document_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_reindex=force_reindex,
            model_name=model_name
        )
        logger.info(f"Initialized RAG tool with {len(document_paths)} documents")
        return tool
    except Exception as e:
        logger.error(f"Error initializing RAG tool: {e}")
        raise

# Node: Validate Query
@Nodes.define()
async def validate_query(query: str, max_sources: int, min_relevance: float, use_llm: bool) -> InRagQuery:
    """Validate the query parameters."""
    try:
        # Create and validate the query
        validated_query = InRagQuery(
            query_text=query,
            max_sources=max_sources,
            min_relevance=min_relevance,
            use_llm=use_llm
        )
        logger.info(f"Query validated: {validated_query.query_text}")
        return validated_query
    except Exception as e:
        logger.error(f"Error validating query: {e}")
        raise

# Node: Execute Query
@Nodes.define()
async def execute_query(
    rag_tool: AlgerianLegalRagTool,
    validated_query: InRagQuery
) -> Dict[str, Any]:
    """Execute the query against the RAG tool."""
    try:
        # Execute the query
        result = rag_tool.execute(
            query=validated_query.query_text,
            max_sources=validated_query.max_sources,
            min_relevance=validated_query.min_relevance,
            use_llm=validated_query.use_llm
        )
        
        # Format the response
        response = {
            "query": validated_query.query_text,
            "answer": result["answer"],
            "sources": []
        }
        
        # Process sources to ensure they're in the expected format
        for i, source in enumerate(result["sources"]):
            formatted_source = {
                "text": source["text"],
                "metadata": source["metadata"],
                "score": source["score"]
            }
            response["sources"].append(formatted_source)
        
        # Add summary information
        response["summary"] = {
            "total_sources": len(response["sources"]),
            "use_llm": validated_query.use_llm,
            "min_relevance": validated_query.min_relevance
        }
        
        logger.info(f"Query executed successfully with {len(response['sources'])} sources")
        return response
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return {
            "query": validated_query.query_text,
            "answer": f"Error executing query: {str(e)}",
            "sources": [],
            "summary": {
                "total_sources": 0,
                "use_llm": validated_query.use_llm,
                "min_relevance": validated_query.min_relevance,
                "error": str(e)
            }
        }

# Node: Format Results
@Nodes.define()
async def format_results(query_result: Dict[str, Any], validated_query: InRagQuery) -> str:
    """Format the results for display."""
    try:
        formatted = f"""# InRag Analysis Results

## Query
{validated_query.query_text}

## Answer
{query_result["answer"]}

## Sources
{len(query_result["sources"])} sources found

## Summary
Total sources: {query_result["summary"]["total_sources"]}
Used LLM: {query_result["summary"]["use_llm"]}
Min relevance: {query_result["summary"]["min_relevance"]}
"""
        logger.info("Results formatted successfully")
        return formatted
    except Exception as e:
        logger.error(f"Error formatting results: {e}")
        raise

# Main function to run the workflow
async def analyze_InRag_documents(
    document_paths: List[str],
    query: str,
    persist_dir: str = None,
    chunk_size: int = 4096,
    chunk_overlap: int = 512,
    force_reindex: bool = False,
    max_sources: int = 8,
    min_relevance: float = 0.1,
    use_llm: bool = True,
    model_name: str = "text-embedding-3-large",
    _handle_event: Optional[callable] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Execute the InRag analysis workflow."""
    # Use home directory if persist_dir is None
    if persist_dir is None:
        home_dir = os.path.expanduser("~")
        persist_dir = os.path.join(home_dir, ".simple_rag_storage")
        logger.info(f"Using home directory for storage: {persist_dir}")
    
    # Ensure the persist directory exists and is writable
    os.makedirs(persist_dir, exist_ok=True)
    try:
        os.chmod(persist_dir, 0o755)  # rwxr-xr-x
    except Exception as e:
        logger.warning(f"Could not set permissions on persist directory: {e}")
    
    try:
        # Initialize the RAG tool directly
        rag_tool = AlgerianLegalRagTool(
            persist_dir=persist_dir,
            document_paths=document_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_reindex=force_reindex,
            model_name=model_name
        )
        
        # Create a validated query
        validated_query = InRagQuery(
            query_text=query,
            max_sources=max_sources,
            min_relevance=min_relevance,
            use_llm=use_llm
        )
        
        # Execute the query
        result = rag_tool.execute(
            query=validated_query.query_text,
            max_sources=validated_query.max_sources,
            min_relevance=validated_query.min_relevance,
            use_llm=validated_query.use_llm
        )
        
        # Format the response
        response = {
            "query": validated_query.query_text,
            "answer": result["answer"],
            "sources": result["sources"],
            "summary": {
                "total_sources": len(result["sources"]),
                "use_llm": validated_query.use_llm,
                "min_relevance": validated_query.min_relevance
            }
        }
        
        # Format the results for display
        formatted_result = f"""# InRag Analysis Results

## Query
{validated_query.query_text}

## Answer
{result["answer"]}

## Sources
{len(result["sources"])} sources found

## Summary
Total sources: {len(result["sources"])}
Used LLM: {validated_query.use_llm}
Min relevance: {validated_query.min_relevance}
"""
        
        # Return the result
        return {
            "query": query,
            "answer": result["answer"],
            "sources": result["sources"],
            "summary": response["summary"],
            "formatted_result": formatted_result
        }
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise

@app.command()
def analyze(
    document_paths: Annotated[List[str], typer.Argument(help="Paths to InRag documents")],
    query: Annotated[str, typer.Argument(help="Query to analyze")],
    persist_dir: Annotated[str, typer.Option(help="Directory to persist embeddings")] = None,
    chunk_size: Annotated[int, typer.Option(help="Size of text chunks")] = 4096,
    chunk_overlap: Annotated[int, typer.Option(help="Overlap between chunks")] = 512,
    force_reindex: Annotated[bool, typer.Option(help="Force reindexing of documents")] = False,
    max_sources: Annotated[int, typer.Option(help="Maximum number of sources to return")] = 8,
    min_relevance: Annotated[float, typer.Option(help="Minimum relevance score")] = 0.1,
    use_llm: Annotated[bool, typer.Option(help="Use LLM for answer generation")] = True,
    model_name: Annotated[str, typer.Option(help="OpenAI embedding model name")] = "text-embedding-3-large"
):
    """Analyze InRag documents using AlgerianLegalRagTool workflow."""
    try:
        # Use home directory if persist_dir is None
        if persist_dir is None:
            home_dir = os.path.expanduser("~")
            persist_dir = os.path.join(home_dir, ".simple_rag_storage")
            console.print(f"[bold blue]Using home directory for storage:[/] {persist_dir}")
        
        # Ensure the persist directory exists and is writable
        os.makedirs(persist_dir, exist_ok=True)
        try:
            os.chmod(persist_dir, 0o755)  # rwxr-xr-x
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/] Could not set permissions on persist directory: {e}")
        
        with console.status(f"Processing query: [bold blue]{query}[/]..."):
            result = asyncio.run(analyze_InRag_documents(
                document_paths=document_paths,
                query=query,
                persist_dir=persist_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                force_reindex=force_reindex,
                max_sources=max_sources,
                min_relevance=min_relevance,
                use_llm=use_llm,
                model_name=model_name
            ))
        
        # Display the answer
        console.print("\n[bold green]Answer:[/]")
        console.print(Markdown(result["answer"]))
        
        # Display source information
        console.print(f"\n[bold blue]Sources ({len(result['sources'])}):[/]")
        for i, source in enumerate(result["sources"], 1):
            metadata_str = ", ".join([f"{k}: {v}" for k, v in source["metadata"].items()])
            console.print(f"[bold]Source {i}[/] ({metadata_str}) - Relevance: {source['score']:.2f}")
            
            # Show a preview of the source text (first 100 chars)
            preview = source["text"][:100] + "..." if len(source["text"]) > 100 else source["text"]
            console.print(f"Preview: {preview}\n")
        
        # Display summary information
        console.print("\n[bold blue]Summary:[/]")
        console.print(f"Total sources: {result['summary']['total_sources']}")
        console.print(f"Used LLM: {result['summary']['use_llm']}")
        console.print(f"Min relevance: {result['summary']['min_relevance']}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    # Test example with predefined parameters
    test_documents = [
        "/home/yarab/Bureau/trash_agents_tests/f1/docs/folder_test/code_civile.md"
    ]
    
    test_queries = [
        "What are the main subjects discussed in the document?",
        "What are the key responsibilities of the service provider?",
        "What are the acceptance criteria mentioned in the document?"
    ]
    
    # Test configuration
    test_config = {
        "persist_dir": os.path.join(os.path.expanduser("~"), ".simple_rag_storage"),
        "chunk_size": 4096,
        "chunk_overlap": 512,
        "force_reindex": True,  # Force reindex for testing
        "max_sources": 8,
        "min_relevance": 0.1,
        "use_llm": True,
        "model_name": "text-embedding-3-large"
    }
    
    # Ensure the persist directory exists and is writable
    os.makedirs(test_config["persist_dir"], exist_ok=True)
    try:
        os.chmod(test_config["persist_dir"], 0o755)  # rwxr-xr-x
    except Exception as e:
        console.print(f"[bold yellow]Warning:[/] Could not set permissions on persist directory: {e}")
    
    console.print("[bold blue]Starting Document Analysis Flow Test[/]\n")
    
    try:
        # Initialize the RAG tool directly
        rag_tool = AlgerianLegalRagTool(
            persist_dir=test_config["persist_dir"],
            document_paths=test_documents,
            chunk_size=test_config["chunk_size"],
            chunk_overlap=test_config["chunk_overlap"],
            force_reindex=test_config["force_reindex"],
            model_name=test_config["model_name"]
        )
        
        # Process each test query
        for query in test_queries:
            console.print(f"\n[bold yellow]Testing Query:[/] {query}")
            
            try:
                with console.status(f"Processing query..."):
                    # Execute the query directly
                    result = rag_tool.execute(
                        query=query,
                        max_sources=test_config["max_sources"],
                        min_relevance=test_config["min_relevance"],
                        use_llm=test_config["use_llm"]
                    )
                
                # Display the answer
                console.print("\n[bold green]Answer:[/]")
                console.print(Markdown(result["answer"]))
                
                # Display source information
                console.print(f"\n[bold blue]Sources ({len(result['sources'])}):[/]")
                for i, source in enumerate(result["sources"], 1):
                    metadata_str = ", ".join([f"{k}: {v}" for k, v in source["metadata"].items()])
                    console.print(f"[bold]Source {i}[/] ({metadata_str}) - Relevance: {source['score']:.2f}")
                    
                    # Show a preview of the source text (first 100 chars)
                    preview = source["text"][:100] + "..." if len(source["text"]) > 100 else source["text"]
                    console.print(f"Preview: {preview}\n")
                
            except Exception as e:
                console.print(f"[bold red]Error:[/] {str(e)}")
                continue
            
            console.print("\n" + "="*50 + "\n")
    
    except Exception as e:
        console.print(f"[bold red]Test execution failed:[/] {str(e)}")
        raise typer.Exit(code=1)
    
    console.print("[bold green]Test execution completed![/]")
