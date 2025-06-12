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
app = typer.Typer(help="Générer des avis juridiques professionnels en utilisant des LLMs avec des templates personnalisables")
console = Console()

# Default models for different phases
DEFAULT_ANALYSIS_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPINION_MODEL = "openai/gpt-4o-mini"
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
class LegalReference(BaseModel):
    """Represents a legal reference (law, regulation, case law, etc.)."""
    title: str
    type: str = Field(description="Type of reference: law, regulation, case law, doctrine, etc.")
    source: str
    relevance: str = Field(description="High, Medium, or Low")
    description: str

class LegalIssue(BaseModel):
    """Represents a legal issue identified in the query."""
    title: str
    description: str
    applicable_law: List[str] = []
    complexity: str = Field(description="High, Medium, or Low")

class LegalQueryMetadata(BaseModel):
    """Metadata about the legal query."""
    title: str
    query_type: str
    jurisdiction: Optional[str] = None
    area_of_law: List[str] = []
    parties_involved: Optional[List[str]] = None
    key_dates: Optional[Dict[str, str]] = None
    urgency: Optional[str] = None

class LegalQueryAnalysis(BaseModel):
    """Complete analysis of a legal query."""
    metadata: LegalQueryMetadata
    identified_issues: List[LegalIssue] = []
    key_references: List[LegalReference] = []
    summary: str
    complexity_assessment: str = Field(description="Overall complexity: High, Medium, or Low")

class LegalOpinion(BaseModel):
    """Structured legal opinion."""
    main_conclusion: str
    reasoning: str
    legal_basis: List[str] = []
    practical_implications: List[str] = []
    risks_and_limitations: List[str] = []
    recommendations: List[str] = []

# Node: Extract Query Metadata using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("query_metadata_system_prompt.txt"),
    output="query_metadata",
    response_model=LegalQueryMetadata,
    prompt_template=read_template("query_metadata_prompt.txt")
)
async def extract_query_metadata(query_content: str, custom_instructions: str = "") -> LegalQueryMetadata:
    """Extract metadata from the legal query."""
    pass

# Node: Analyze Query Content using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("query_analysis_system_prompt.txt"),
    output="query_analysis",
    response_model=LegalQueryAnalysis,
    prompt_template=read_template("query_analysis_prompt.txt")
)
async def analyze_query_content(query_content: str, query_metadata: LegalQueryMetadata, custom_instructions: str = "") -> LegalQueryAnalysis:
    """Analyze the legal query content and provide a comprehensive analysis."""
    pass

# Node: Generate Legal Opinion using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("legal_opinion_system_prompt.txt"),
    output="legal_opinion",
    response_model=LegalOpinion,
    prompt_template=read_template("legal_opinion_prompt.txt")
)
async def generate_legal_opinion(query_analysis: LegalQueryAnalysis, custom_instructions: str = "") -> LegalOpinion:
    """Generate a structured legal opinion based on query analysis."""
    pass

# Node: Generate Final Opinion Document
@Nodes.llm_node(
    system_prompt=read_template("opinion_document_system_prompt.txt"),
    output="opinion_document",
    prompt_template=read_template("opinion_document_prompt.txt")
)
async def generate_opinion_document(query_analysis: LegalQueryAnalysis, legal_opinion: LegalOpinion, language: str = "french", document_type: str = "avis_juridique") -> str:
    """Generate a formal legal opinion document based on analysis and opinion."""
    pass

# Node: Save Opinion Document
@Nodes.define(output="document_file_path")
async def save_opinion_document(opinion_document: str, query_content: str, document_type: str, output_dir: str) -> str:
    """Save the legal opinion document to a file."""
    try:
        # Create a filename based on document type and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{document_type}_{timestamp}.md"
        
        # Ensure output directory exists
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the document to file
        with output_path.open("w", encoding="utf-8") as f:
            f.write(opinion_document)
        
        logger.info(f"Saved legal opinion document to: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving legal opinion document: {e}")
        raise

# Define the Legal Opinion Generator Workflow
def create_legal_opinion_workflow() -> Workflow:
    """Create a workflow to generate legal opinions."""
    wf = Workflow("extract_query_metadata")
    
    # Add all nodes with input mappings for dynamic model passing
    wf.node("extract_query_metadata", inputs_mapping={"model": "analysis_model"})
    wf.node("analyze_query_content", inputs_mapping={"model": "analysis_model"})
    wf.node("generate_legal_opinion", inputs_mapping={"model": "opinion_model"})
    wf.node("generate_opinion_document", inputs_mapping={"model": "opinion_model", "language": "language"})
    wf.node("save_opinion_document")
    
    # Define linear sequence
    wf.current_node = "extract_query_metadata"
    wf.transitions["extract_query_metadata"] = [("analyze_query_content", None)]
    wf.transitions["analyze_query_content"] = [("generate_legal_opinion", None)]
    wf.transitions["generate_legal_opinion"] = [("generate_opinion_document", None)]
    wf.transitions["generate_opinion_document"] = [("save_opinion_document", None)]
    
    return wf

# Function to Run the Workflow
async def generate_avis_juridique(
    query_content: str,
    analysis_model: str,
    opinion_model: str,
    language: str = DEFAULT_LANGUAGE,
    document_type: str = "avis_juridique",
    custom_metadata_instructions: str = "",
    custom_analysis_instructions: str = "",
    output_dir: Optional[str] = None,
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Execute the workflow with the given query content and models."""
    if output_dir:
        output_dir = os.path.expanduser(output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initial context with model keys for dynamic mapping
    initial_context = {
        "query_content": query_content,
        "analysis_model": analysis_model,
        "opinion_model": opinion_model,
        "language": language,  # Language for output
        "document_type": document_type,
        "output_dir": output_dir,
        "custom_instructions": custom_metadata_instructions,  # For metadata extraction
        "custom_analysis_instructions": custom_analysis_instructions  # For query analysis
    }

    try:
        workflow = create_legal_opinion_workflow()
        engine = workflow.build()
        
        result = await engine.run(initial_context)
        
        if "opinion_document" not in result or not result["opinion_document"]:
            logger.warning("No opinion document generated.")
            raise ValueError("Workflow completed but no opinion document was generated.")
        
        logger.info("Workflow completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise

async def display_results(opinion_document: str, document_file_path: str):
    """Async helper function to display results with animation."""
    console.print("\n[bold green]Avis Juridique:[/]")
    console.print(Panel(Markdown(opinion_document), border_style="blue"))
    
    console.print(f"[green]✓ Document d'avis juridique sauvegardé à:[/] {document_file_path}")

if __name__ == "__main__":
    # Example usage with direct initialization
    import sys
    
    # Default values
    default_query_content = """
    Demande d'avis juridique

    Objet : Validité d'une clause de non-concurrence dans un contrat de travail

    Je suis directeur des ressources humaines d'une entreprise de développement de logiciels basée à Paris. Nous souhaitons inclure une clause de non-concurrence dans nos contrats de travail pour les développeurs seniors. Cette clause interdirait aux employés de travailler pour des entreprises concurrentes pendant une période de 2 ans après leur départ, sur tout le territoire français, sans compensation financière.

    Nous aimerions savoir si cette clause est légale selon le droit du travail français, et si non, quelles modifications devrions-nous apporter pour la rendre valide tout en protégeant au maximum les intérêts de notre entreprise.

    Merci de nous fournir un avis détaillé sur cette question.
    """
    default_analysis_model = DEFAULT_ANALYSIS_MODEL
    default_opinion_model = DEFAULT_OPINION_MODEL
    default_language = DEFAULT_LANGUAGE
    default_document_type = "avis_juridique"
    default_output_dir = None
    
    # Run with default values
    try:
        logger.info(f"Génération d'un avis juridique")
        result = asyncio.run(generate_avis_juridique(
            default_query_content,
            default_analysis_model,
            default_opinion_model,
            language=default_language,
            document_type=default_document_type,
            custom_metadata_instructions="Extraire tous les détails pertinents, y compris le domaine juridique, l'urgence, et les parties concernées.",
            custom_analysis_instructions="Fournir une analyse détaillée et structurée avec toutes les informations contenues dans la demande.",
            output_dir=default_output_dir
        ))
        
        opinion_document = result["opinion_document"]
        document_file_path = result.get("document_file_path", "Not saved")
        
        # Display results
        asyncio.run(display_results(opinion_document, document_file_path))
        
    except Exception as e:
        logger.error(f"Error during legal opinion generation: {e}")
        console.print(f"[bold red]Error:[/] {str(e)}")
