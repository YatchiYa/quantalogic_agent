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

# Initialize Typer app and rich console
app = typer.Typer(help="Defend legal cases using LLMs with customizable templates")
console = Console()

# Default models for different phases
DEFAULT_ANALYSIS_MODEL = "openai/gpt-4o-mini"
DEFAULT_STRATEGY_MODEL = "openai/gpt-4o-mini"
DEFAULT_DEFENSE_MODEL = "openai/gpt-4o-mini"
DEFAULT_LANGUAGE = "french"  # Default language for output

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
    """Represents a party in a legal case."""
    name: str
    role: str
    description: Optional[str] = None

class LegalArgument(BaseModel):
    """Represents a key legal argument."""
    title: str
    content: str
    strength: str = Field(description="Strong, Medium, or Weak")
    supporting_evidence: Optional[List[str]] = None
    counter_arguments: Optional[List[str]] = None

class LegalCaseMetadata(BaseModel):
    """Metadata about the legal case."""
    title: str
    case_type: str
    date: Optional[str] = None
    jurisdiction: Optional[str] = None
    governing_law: Optional[str] = None
    parties: List[LegalParty] = []
    plaintiff: Optional[str] = None
    defendant: Optional[str] = None
    case_number: Optional[str] = None
    laws_cited: Optional[List[str]] = None
    key_dates: Optional[Dict[str, str]] = None

class LegalCaseAnalysis(BaseModel):
    """Complete analysis of a legal case."""
    metadata: LegalCaseMetadata
    key_arguments: List[LegalArgument] = []
    summary: str
    risks: List[str] = []
    opportunities: List[str] = []

class DefenseStrategy(BaseModel):
    """Strategic defense plan for the case."""
    primary_defense_theory: str
    secondary_defense_theories: List[str] = []
    key_evidence_needed: List[str] = []
    witness_strategy: Optional[str] = None
    settlement_considerations: Optional[str] = None
    timeline_strategy: Optional[str] = None
    recommended_approach: str

# Node: Extract Case Metadata using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("case_metadata_system_prompt.txt"),
    output="case_metadata",
    response_model=LegalCaseMetadata,
    prompt_template=read_template("case_metadata_prompt.txt")
)
async def extract_case_metadata(case_details: str, custom_instructions: str = "") -> LegalCaseMetadata:
    """Extract metadata from the legal case details."""
    pass

# Node: Analyze Case Content using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("case_analysis_system_prompt.txt"),
    output="case_analysis",
    response_model=LegalCaseAnalysis,
    prompt_template=read_template("case_analysis_prompt.txt")
)
async def analyze_case_content(case_details: str, case_metadata: LegalCaseMetadata, custom_instructions: str = "") -> LegalCaseAnalysis:
    """Analyze the legal case content and provide a comprehensive analysis."""
    pass

# Node: Generate Defense Strategy using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("defense_strategy_system_prompt.txt"),
    output="defense_strategy",
    response_model=DefenseStrategy,
    prompt_template=read_template("defense_strategy_prompt.txt")
)
async def generate_defense_strategy(case_analysis: LegalCaseAnalysis, custom_instructions: str = "") -> DefenseStrategy:
    """Generate a comprehensive defense strategy based on case analysis."""
    pass

# Node: Generate Defense Document
@Nodes.llm_node(
    system_prompt=read_template("defense_document_system_prompt.txt"),
    output="defense_document",
    prompt_template=read_template("defense_document_prompt.txt")
)
async def generate_defense_document(case_analysis: LegalCaseAnalysis, defense_strategy: DefenseStrategy, language: str = "french", document_type: str = "legal_brief") -> str:
    """Generate a defense document based on case analysis and strategy."""
    pass

# Node: Save Defense Document
@Nodes.define(output="document_file_path")
async def save_defense_document(defense_document: str, case_details: str, document_type: str, output_dir: str) -> str:
    """Save the defense document to a file."""
    try:
        # Create a filename based on document type and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{document_type}_{timestamp}.md"
        
        # Ensure output directory exists
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the document to file
        with output_path.open("w", encoding="utf-8") as f:
            f.write(defense_document)
        
        logger.info(f"Saved defense document to: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving defense document: {e}")
        raise

# Define the Legal Case Defense Workflow
def create_legal_case_defense_workflow() -> Workflow:
    """Create a workflow to defend legal cases."""
    wf = Workflow("extract_case_metadata")
    
    # Add all nodes with input mappings for dynamic model passing
    wf.node("extract_case_metadata", inputs_mapping={"model": "analysis_model"})
    wf.node("analyze_case_content", inputs_mapping={"model": "analysis_model"})
    wf.node("generate_defense_strategy", inputs_mapping={"model": "strategy_model"})
    wf.node("generate_defense_document", inputs_mapping={"model": "defense_model", "language": "language"})
    wf.node("save_defense_document")
    
    # Define linear sequence
    wf.current_node = "extract_case_metadata"
    wf.transitions["extract_case_metadata"] = [("analyze_case_content", None)]
    wf.transitions["analyze_case_content"] = [("generate_defense_strategy", None)]
    wf.transitions["generate_defense_strategy"] = [("generate_defense_document", None)]
    wf.transitions["generate_defense_document"] = [("save_defense_document", None)]
    
    return wf

# Function to Run the Workflow
async def defend_legal_case(
    case_details: str,
    analysis_model: str,
    strategy_model: str,
    defense_model: str,
    language: str = DEFAULT_LANGUAGE,
    document_type: str = "legal_brief",
    custom_metadata_instructions: str = "",
    custom_analysis_instructions: str = "",
    output_dir: Optional[str] = None,
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Execute the workflow with the given case details and models."""
    if output_dir:
        output_dir = os.path.expanduser(output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initial context with model keys for dynamic mapping
    initial_context = {
        "case_details": case_details,
        "analysis_model": analysis_model,
        "strategy_model": strategy_model,
        "defense_model": defense_model,
        "language": language,  # Language for output
        "document_type": document_type,
        "output_dir": output_dir,
        "custom_instructions": custom_metadata_instructions,  # For metadata extraction
        "custom_analysis_instructions": custom_analysis_instructions  # For case analysis
    }

    try:
        workflow = create_legal_case_defense_workflow()
        engine = workflow.build()
        
        result = await engine.run(initial_context)
        
        if "defense_document" not in result or not result["defense_document"]:
            logger.warning("No defense document generated.")
            raise ValueError("Workflow completed but no defense document was generated.")
        
        logger.info("Workflow completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise

async def display_results(defense_document: str, document_file_path: str):
    """Async helper function to display results with animation."""
    console.print("\n[bold green]Legal Defense Document:[/]")
    console.print(Panel(Markdown(defense_document), border_style="blue"))
    
    console.print(f"[green]✓ Defense document saved to:[/] {document_file_path}")

if __name__ == "__main__":
    # Example usage with direct initialization
    import sys
    
    # Default values
    default_case_details = """
    Case: Smith v. Johnson
    Our client, Johnson, is being sued for breach of contract by Smith.
    Smith claims that Johnson failed to deliver agreed-upon consulting services worth €50,000.
    Johnson maintains that all services were delivered according to the contract specifications.
    The contract was signed on January 15, 2024, with a completion date of May 1, 2024.
    Smith filed the lawsuit on June 10, 2024, claiming damages of €75,000 including lost business opportunities.
    """
    default_analysis_model = DEFAULT_ANALYSIS_MODEL
    default_strategy_model = DEFAULT_STRATEGY_MODEL
    default_defense_model = DEFAULT_DEFENSE_MODEL
    default_language = DEFAULT_LANGUAGE
    default_document_type = "legal_brief"
    default_output_dir = None
    
    # Run with default values
    try:
        logger.info(f"Defending case")
        result = asyncio.run(defend_legal_case(
            default_case_details,
            default_analysis_model,
            default_strategy_model,
            default_defense_model,
            language=default_language,
            document_type=default_document_type,
            custom_metadata_instructions="Extraire tous les détails pertinents, y compris les références légales, dates importantes, et toutes les parties mentionnées.",
            custom_analysis_instructions="Fournir une analyse détaillée et structurée avec toutes les informations contenues dans le document.",
            output_dir=default_output_dir
        ))
        
        defense_document = result["defense_document"]
        document_file_path = result.get("document_file_path", "Not saved")
        
        # Display results
        asyncio.run(display_results(defense_document, document_file_path))
        
    except Exception as e:
        logger.error(f"Error during defense: {e}")
        console.print(f"[bold red]Error:[/] {str(e)}")
