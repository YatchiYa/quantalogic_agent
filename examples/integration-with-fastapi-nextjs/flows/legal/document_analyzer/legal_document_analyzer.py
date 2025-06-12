import asyncio
from collections.abc import Callable
import datetime
import os
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Union

import typer
from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
# from ..service import event_observer

# Initialize Typer app and rich console
app = typer.Typer(help="Analyze legal documents using LLMs with customizable templates")
console = Console()

# Default models for different phases
DEFAULT_TEXT_EXTRACTION_MODEL = "openai/gpt-4o-mini"
DEFAULT_ANALYSIS_MODEL = "openai/gpt-4o-mini"
DEFAULT_SUMMARY_MODEL = "openai/gpt-4o-mini"
DEFAULT_LANGUAGE = "french"  # Default language for analysis output

# Templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Helper function to read template files
def read_template(template_name):
    """Read a template file and return its content."""
    template_path = os.path.join(TEMPLATES_DIR, template_name)
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading template {template_name}: {e}")
        raise

# Define Pydantic models for structured output
class LegalParty(BaseModel):
    """Represents a party in a legal document."""
    name: str
    role: str
    description: Optional[str] = None

class LegalClause(BaseModel):
    """Represents a key clause in a legal document."""
    title: str
    content: str
    importance: str = Field(description="High, Medium, or Low")
    potential_issues: Optional[List[str]] = None

class LegalDocumentMetadata(BaseModel):
    """Metadata about the legal document."""
    title: str
    document_type: str
    date: Optional[str] = None
    jurisdiction: Optional[str] = None
    governing_law: Optional[str] = None
    parties: List[LegalParty] = []
    sender: Optional[str] = None
    recipient: Optional[str] = None
    reference_numbers: Optional[List[str]] = None
    laws_mentioned: Optional[List[str]] = None
    key_dates: Optional[Dict[str, str]] = None

class LegalDocumentAnalysis(BaseModel):
    """Complete analysis of a legal document."""
    metadata: LegalDocumentMetadata
    key_clauses: List[LegalClause] = []
    summary: str
    risks: List[str] = []
    recommendations: List[str] = []

# Node: Check File Type
@Nodes.define(output="file_type")
async def check_file_type(file_path: str) -> str:
    """Determine the file type based on its extension."""
    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise ValueError(f"File not found: {file_path}")
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    elif ext in [".txt", ".text"]:
        return "text"
    elif ext == ".md":
        return "markdown"
    elif ext in [".doc", ".docx"]:
        return "word"
    else:
        logger.error(f"Unsupported file type: {ext}")
        raise ValueError(f"Unsupported file type: {ext}. Supported types: .pdf, .txt, .text, .md, .doc, .docx")

# Node: Read Text or Markdown File
@Nodes.define(output="document_content")
async def read_text_or_markdown(file_path: str, file_type: str) -> str:
    """Read content from a text or markdown file."""
    if file_type not in ["text", "markdown"]:
        logger.error(f"Node 'read_text_or_markdown' called with invalid file_type: {file_type}")
        raise ValueError(f"Expected 'text' or 'markdown', got {file_type}")
    try:
        file_path = os.path.expanduser(file_path)
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Read {file_type} content from {file_path}, length: {len(content)} characters")
        return content
    except Exception as e:
        logger.error(f"Error reading {file_type} file {file_path}: {e}")
        raise

# Node: Convert PDF to Text
@Nodes.define(output="document_content")
async def convert_pdf_to_text(file_path: str, model: str) -> str:
    """Convert a PDF to plain text using a vision model."""
    from pyzerox import zerox
    
    file_path = os.path.expanduser(file_path)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise ValueError(f"File not found: {file_path}")
    
    # Read the PDF extraction prompt from template file
    custom_system_prompt = read_template("pdf_extraction_prompt.txt")

    try:
        logger.info(f"Converting PDF to text with model: {model}, file: {file_path}")
        zerox_result = await zerox(
            file_path=file_path,
            model=model,
            system_prompt=custom_system_prompt
        )

        document_content = ""
        if hasattr(zerox_result, 'pages') and zerox_result.pages:
            document_content = "\n\n".join(
                page.content for page in zerox_result.pages
                if hasattr(page, 'content') and page.content
            )
        elif isinstance(zerox_result, str):
            document_content = zerox_result
        elif hasattr(zerox_result, 'text'):
            document_content = zerox_result.text
        else:
            document_content = str(zerox_result)
            logger.warning("Unexpected zerox_result type; converted to string.")

        if not document_content.strip():
            logger.warning("Generated text content is empty.")
            return ""

        logger.info(f"Extracted text content length: {len(document_content)} characters")
        return document_content
    except Exception as e:
        logger.error(f"Error converting PDF to text: {e}")
        raise

# Node: Convert Word to Text
@Nodes.define(output="document_content")
async def convert_word_to_text(file_path: str) -> str:
    """Convert a Word document to plain text."""
    try:
        import docx2txt
        file_path = os.path.expanduser(file_path)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise ValueError(f"File not found: {file_path}")
            
        logger.info(f"Converting Word document to text: {file_path}")
        document_content = docx2txt.process(file_path)
        
        if not document_content.strip():
            logger.warning("Generated text content is empty.")
            return ""
            
        logger.info(f"Extracted text content length: {len(document_content)} characters")
        return document_content
    except ImportError:
        logger.error("docx2txt module not found. Install with 'pip install docx2txt'")
        raise ImportError("docx2txt module not found. Install with 'pip install docx2txt'")
    except Exception as e:
        logger.error(f"Error converting Word document to text: {e}")
        raise

# Node: Save Document Content
@Nodes.define(output="document_file_path")
async def save_document_content(document_content: str, file_path: str) -> str:
    """Save the extracted document content to a file."""
    try:
        file_path_expanded = os.path.expanduser(file_path)
        output_path = Path(file_path_expanded).with_suffix(".extracted.txt")
        with output_path.open("w", encoding="utf-8") as f:
            f.write(document_content)
        logger.info(f"Saved extracted document content to: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving document content: {e}")
        raise

# Node: Extract Document Metadata using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("metadata_extraction_system_prompt.txt"),
    output="document_metadata",
    response_model=LegalDocumentMetadata,
    prompt_template=read_template("metadata_extraction_prompt.txt")
)
async def extract_document_metadata(document_content: str, custom_instructions: str = "") -> LegalDocumentMetadata:
    """Extract metadata from the legal document."""
    pass

# Node: Analyze Document Content using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("document_analysis_system_prompt.txt"),
    output="document_analysis",
    response_model=LegalDocumentAnalysis,
    prompt_template=read_template("document_analysis_prompt.txt")
)
async def analyze_document_content(document_content: str, document_metadata: LegalDocumentMetadata, custom_instructions: str = "") -> LegalDocumentAnalysis:
    """Analyze the legal document content and provide a comprehensive analysis."""
    pass

# Node: Generate Summary Report
@Nodes.llm_node(
    system_prompt=read_template("summary_report_system_prompt.txt"),
    output="summary_report",
    prompt_template=read_template("summary_report_prompt.txt")
)
async def generate_summary_report(document_analysis: LegalDocumentAnalysis, language: str = "french") -> str:
    """Generate a summary report of the legal document analysis."""
    pass

# Node: Save Analysis Report
@Nodes.define(output="report_file_path")
async def save_analysis_report(summary_report: str, file_path: str) -> str:
    """Save the legal document analysis report to a file."""
    try:
        file_path_expanded = os.path.expanduser(file_path)
        output_path = Path(file_path_expanded).with_suffix(".analysis.md")
        with output_path.open("w", encoding="utf-8") as f:
            f.write(summary_report)
        logger.info(f"Saved legal document analysis report to: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving analysis report: {e}")
        raise

# Define the Legal Document Analysis Workflow
def create_legal_document_analysis_workflow() -> Workflow:
    """Create a workflow to analyze legal documents."""
    wf = Workflow("check_file_type")
    
    # Add all nodes with input mappings for dynamic model passing
    wf.node("check_file_type")
    wf.node("convert_pdf_to_text", inputs_mapping={"model": "text_extraction_model"})
    wf.node("convert_word_to_text")
    wf.node("read_text_or_markdown")
    wf.node("save_document_content")
    wf.node("extract_document_metadata", inputs_mapping={"model": "analysis_model"})
    wf.node("analyze_document_content", inputs_mapping={"model": "analysis_model"})
    wf.node("generate_summary_report", inputs_mapping={"model": "summary_model", "language": "language"})
    wf.node("save_analysis_report")
    
    # Define the workflow structure with explicit transitions
    wf.current_node = "check_file_type"
    wf.branch([
        ("convert_pdf_to_text", lambda ctx: ctx["file_type"] == "pdf"),
        ("convert_word_to_text", lambda ctx: ctx["file_type"] == "word"),
        ("read_text_or_markdown", lambda ctx: ctx["file_type"] in ["text", "markdown"])
    ])
    
    # Explicitly set transitions from branches to convergence point
    wf.transitions["convert_pdf_to_text"] = [("save_document_content", None)]
    wf.transitions["convert_word_to_text"] = [("save_document_content", None)]
    wf.transitions["read_text_or_markdown"] = [("save_document_content", None)]
    
    # Define linear sequence after convergence
    wf.transitions["save_document_content"] = [("extract_document_metadata", None)]
    wf.transitions["extract_document_metadata"] = [("analyze_document_content", None)]
    wf.transitions["analyze_document_content"] = [("generate_summary_report", None)]
    wf.transitions["generate_summary_report"] = [("save_analysis_report", None)]
    
    return wf

# Function to Run the Workflow
async def analyze_legal_document(
    file_path: str,
    text_extraction_model: str,
    analysis_model: str,
    summary_model: str,
    language: str = DEFAULT_LANGUAGE,
    custom_metadata_instructions: str = "",
    custom_analysis_instructions: str = "",
    output_dir: Optional[str] = None,
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Execute the workflow with the given file path and models."""
    file_path = os.path.expanduser(file_path)
    if output_dir:
        output_dir = os.path.expanduser(output_dir)
        
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise ValueError(f"File not found: {file_path}")

    # Initial context with model keys for dynamic mapping
    initial_context = {
        "file_path": file_path,
        "text_extraction_model": text_extraction_model,
        "analysis_model": analysis_model,
        "summary_model": summary_model,
        "language": language,  # Language for analysis output
        "output_dir": output_dir if output_dir else str(Path(file_path).parent),
        "custom_instructions": custom_metadata_instructions,  # For metadata extraction
        "custom_analysis_instructions": custom_analysis_instructions  # For document analysis
    }

    try:
        workflow = create_legal_document_analysis_workflow()
        engine = workflow.build()
        
        # Add the event observer if _handle_event is provided
        # if _handle_event:
        #     # Create a lambda to bind task_id to the observer
        #     bound_observer = lambda event: asyncio.create_task(
        #         event_observer(event, task_id=task_id, _handle_event=_handle_event)
        #     )
        #     engine.add_observer(bound_observer)

        result = await engine.run(initial_context)
        
        if "summary_report" not in result or not result["summary_report"]:
            logger.warning("No legal document analysis report generated.")
            raise ValueError("Workflow completed but no analysis report was generated.")
        
        logger.info("Workflow completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise

async def display_results(summary_report: str, document_file_path: str, report_file_path: str):
    """Async helper function to display results with animation."""
    console.print("\n[bold green]Legal Document Analysis Report:[/]")
    console.print(Panel(Markdown(summary_report), border_style="blue"))
    
    console.print(f"[green]✓ Extracted document content saved to:[/] {document_file_path}")
    console.print(f"[green]✓ Analysis report saved to:[/] {report_file_path}")

# @app.command()
# def analyze(
#     file_path: Annotated[str, typer.Argument(help="Path to the legal document (PDF, .txt, .md, .doc, .docx; supports ~ expansion)")],
#     text_extraction_model: Annotated[str, typer.Option(help="LLM model for PDF/Word text extraction")] = DEFAULT_TEXT_EXTRACTION_MODEL,
#     analysis_model: Annotated[str, typer.Option(help="LLM model for document analysis")] = DEFAULT_ANALYSIS_MODEL,
#     summary_model: Annotated[str, typer.Option(help="LLM model for report generation")] = DEFAULT_SUMMARY_MODEL,
#     output_dir: Annotated[Optional[str], typer.Option(help="Directory to save output files (supports ~ expansion)")] = None,
#     custom_metadata_instructions: Annotated[str, typer.Option(help="Custom instructions for metadata extraction")] = "",
#     custom_analysis_instructions: Annotated[str, typer.Option(help="Custom instructions for document analysis")] = "",
# ):
#     """Analyze a legal document using an LLM workflow with customizable templates."""
#     try:
#         with console.status(f"Processing [bold blue]{file_path}[/]..."):
#             result = asyncio.run(analyze_legal_document(
#                 file_path,
#                 text_extraction_model,
#                 analysis_model,
#                 summary_model,
#                 custom_metadata_instructions,
#                 custom_analysis_instructions,
#                 output_dir
#             ))
        
#         summary_report = result["summary_report"]
#         document_file_path = result.get("document_file_path", "Not saved")
#         report_file_path = result.get("report_file_path", "Not saved")
        
#         # Run the async display function
#         asyncio.run(display_results(summary_report, document_file_path, report_file_path))
    
#     except Exception as e:
#         logger.error(f"Failed to run workflow: {e}")
#         console.print(f"[bold red]Error:[/] {str(e)}")
#         raise typer.Exit(code=1)

if __name__ == "__main__":
    # Example usage with direct initialization
    import sys
    from pathlib import Path
    
    # Default values
    default_file = "/home/yarab/Bureau/quantalogic/quantalogic_agent/examples/integration-with-fastapi-nextjs/flows/legal/document_analyzer/nt1.md"
    default_extraction_model = DEFAULT_TEXT_EXTRACTION_MODEL
    default_analysis_model = DEFAULT_ANALYSIS_MODEL
    default_summary_model = DEFAULT_SUMMARY_MODEL
    default_language = DEFAULT_LANGUAGE
    default_output_dir = None
    
    # Check if a file path was provided as an argument
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        default_file = sys.argv[1]
        sys.argv.pop(1)  # Remove the file argument so typer doesn't get confused
    
    # If no file is provided, use a sample file if available
    if not default_file:
        sample_dir = Path(__file__).parent / "samples"
        if sample_dir.exists():
            sample_files = list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.docx"))
            if sample_files:
                default_file = str(sample_files[0])
                logger.info(f"Using sample file: {default_file}")
    
    # Run the app with typer
    if default_file:
        # Direct execution with default values
        try:
            logger.info(f"Analyzing document: {default_file}")
            result = asyncio.run(analyze_legal_document(
                default_file,
                default_extraction_model,
                default_analysis_model,
                default_summary_model,
                language=default_language,
                custom_metadata_instructions="Extraire tous les détails pertinents, y compris les références légales, dates importantes, et toutes les parties mentionnées.",
                custom_analysis_instructions="Fournir une analyse détaillée et structurée avec toutes les informations contenues dans le document.",
                output_dir=default_output_dir
            ))
            
            summary_report = result["summary_report"]
            document_file_path = result.get("document_file_path", "Not saved")
            report_file_path = result.get("report_file_path", "Not saved")
            
            # Display results
            asyncio.run(display_results(summary_report, document_file_path, report_file_path))
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            console.print(f"[bold red]Error:[/] {str(e)}")
            # Fall back to CLI app
            app()
    else:
        # No default file, run the CLI app
        app()
