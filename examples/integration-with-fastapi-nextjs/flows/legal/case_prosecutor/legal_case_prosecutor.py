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

from quantalogic.flow.flow import Nodes, Workflow

# Initialize Typer app and rich console
app = typer.Typer(help="Prosecute legal cases using LLMs with customizable templates")
console = Console()

# Default models for different phases
DEFAULT_ANALYSIS_MODEL = "openai/gpt-4o-mini"
DEFAULT_STRATEGY_MODEL = "openai/gpt-4o-mini"
DEFAULT_PROSECUTION_MODEL = "openai/gpt-4o-mini"
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

class Evidence(BaseModel):
    """Represents a piece of evidence."""
    title: str
    description: str
    strength: str = Field(description="Strong, Medium, or Weak")
    source: Optional[str] = None
    challenges: Optional[List[str]] = None

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
    key_evidence: List[Evidence] = []
    summary: str
    strengths: List[str] = []
    weaknesses: List[str] = []
    legal_theories: List[str] = []

class ProsecutionStrategy(BaseModel):
    """Strategic prosecution plan for the case."""
    primary_legal_theory: str
    secondary_legal_theories: List[str] = []
    key_evidence_presentation: List[str] = []
    witness_strategy: Optional[str] = None
    anticipated_defenses: List[str] = []
    burden_of_proof_strategy: str
    recommended_charges: List[str] = []

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

# Node: Generate Prosecution Strategy using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("prosecution_strategy_system_prompt.txt"),
    output="prosecution_strategy",
    response_model=ProsecutionStrategy,
    prompt_template=read_template("prosecution_strategy_prompt.txt")
)
async def generate_prosecution_strategy(case_analysis: LegalCaseAnalysis, custom_instructions: str = "") -> ProsecutionStrategy:
    """Generate a comprehensive prosecution strategy based on case analysis."""
    pass

# Node: Generate Prosecution Document
@Nodes.llm_node(
    system_prompt=read_template("prosecution_document_system_prompt.txt"),
    output="prosecution_document",
    prompt_template=read_template("prosecution_document_prompt.txt")
)
async def generate_prosecution_document(case_analysis: LegalCaseAnalysis, prosecution_strategy: ProsecutionStrategy, language: str = "french", document_type: str = "indictment") -> str:
    """Generate a prosecution document based on case analysis and strategy."""
    pass

# Node: Save Prosecution Document
@Nodes.define(output="document_file_path")
async def save_prosecution_document(prosecution_document: str, case_details: str, document_type: str, output_dir: str) -> str:
    """Save the prosecution document to a file."""
    try:
        # Create a filename based on document type and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{document_type}_{timestamp}.md"
        
        # Ensure output directory exists
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the document to file
        with output_path.open("w", encoding="utf-8") as f:
            f.write(prosecution_document)
        
        logger.info(f"Saved prosecution document to: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving prosecution document: {e}")
        raise

# Define the Legal Case Prosecution Workflow
def create_legal_case_prosecution_workflow() -> Workflow:
    """Create a workflow to prosecute legal cases."""
    wf = Workflow("extract_case_metadata")
    
    # Add all nodes with input mappings for dynamic model passing
    wf.node("extract_case_metadata", inputs_mapping={"model": "analysis_model"})
    wf.node("analyze_case_content", inputs_mapping={"model": "analysis_model"})
    wf.node("generate_prosecution_strategy", inputs_mapping={"model": "strategy_model"})
    wf.node("generate_prosecution_document", inputs_mapping={"model": "prosecution_model", "language": "language"})
    wf.node("save_prosecution_document")
    
    # Define linear sequence
    wf.current_node = "extract_case_metadata"
    wf.transitions["extract_case_metadata"] = [("analyze_case_content", None)]
    wf.transitions["analyze_case_content"] = [("generate_prosecution_strategy", None)]
    wf.transitions["generate_prosecution_strategy"] = [("generate_prosecution_document", None)]
    wf.transitions["generate_prosecution_document"] = [("save_prosecution_document", None)]
    
    return wf

# Function to Run the Workflow
async def prosecute_legal_case(
    case_details: str,
    analysis_model: str,
    strategy_model: str,
    prosecution_model: str,
    language: str = DEFAULT_LANGUAGE,
    document_type: str = "indictment",
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
        "prosecution_model": prosecution_model,
        "language": language,  # Language for output
        "document_type": document_type,
        "output_dir": output_dir,
        "custom_instructions": custom_metadata_instructions,  # For metadata extraction
        "custom_analysis_instructions": custom_analysis_instructions  # For case analysis
    }

    try:
        workflow = create_legal_case_prosecution_workflow()
        engine = workflow.build()
        
        result = await engine.run(initial_context)
        
        if "prosecution_document" not in result or not result["prosecution_document"]:
            logger.warning("No prosecution document generated.")
            raise ValueError("Workflow completed but no prosecution document was generated.")
        
        logger.info("Workflow completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise

async def display_results(prosecution_document: str, document_file_path: str):
    """Async helper function to display results with animation."""
    console.print("\n[bold green]Legal Prosecution Document:[/]")
    console.print(Panel(Markdown(prosecution_document), border_style="blue"))
    
    console.print(f"[green]✓ Prosecution document saved to:[/] {document_file_path}")

if __name__ == "__main__":
    # Example usage with direct initialization
    import sys
    
    # Default values
    default_case_details = """
    Case: State v. Thompson
    The suspect, James Thompson, is accused of embezzling €120,000 from his employer, TechCorp SA.
    Financial records show irregular transfers to his personal account between January and April 2024.
    The company's internal audit discovered the discrepancies on May 15, 2024.
    Thompson had access to the company's financial systems as the Finance Director.
    When confronted, Thompson claimed the transfers were authorized bonuses.
    No documentation supporting these "bonuses" has been found.
    Thompson has worked at the company for 8 years with no prior incidents.
    The company has provided all financial records and system access logs.
    """
    default_analysis_model = DEFAULT_ANALYSIS_MODEL
    default_strategy_model = DEFAULT_STRATEGY_MODEL
    default_prosecution_model = DEFAULT_PROSECUTION_MODEL
    default_language = DEFAULT_LANGUAGE
    default_document_type = "indictment"
    default_output_dir = None
    
    # Run with default values
    try:
        logger.info(f"Prosecuting case")
        result = asyncio.run(prosecute_legal_case(
            default_case_details,
            default_analysis_model,
            default_strategy_model,
            default_prosecution_model,
            language=default_language,
            document_type=default_document_type,
            custom_metadata_instructions="Extraire tous les détails pertinents, y compris les références légales, dates importantes, et toutes les parties mentionnées.",
            custom_analysis_instructions="Fournir une analyse détaillée et structurée avec toutes les informations contenues dans le document.",
            output_dir=default_output_dir
        ))
        
        prosecution_document = result["prosecution_document"]
        document_file_path = result.get("document_file_path", "Not saved")
        
        # Display results
        asyncio.run(display_results(prosecution_document, document_file_path))
        
    except Exception as e:
        logger.error(f"Error during prosecution: {e}")
        console.print(f"[bold red]Error:[/] {str(e)}")
