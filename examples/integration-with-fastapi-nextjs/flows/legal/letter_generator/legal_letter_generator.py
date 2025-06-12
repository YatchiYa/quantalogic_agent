import asyncio
from collections.abc import Callable
import datetime
import os
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Union, Literal

import typer
from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from quantalogic.flow.flow import Nodes, Workflow

# Initialize Typer app and rich console
app = typer.Typer(help="Générer des lettres juridiques convaincantes avec références aux lois et preuves")
console = Console()

# Default models
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_LANGUAGE = "french"  # Default language for output
DEFAULT_ROLE = "defense"  # Default role (defense or prosecution)

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
    """A reference to a law, code, or precedent."""
    type: str = Field(description="Type of reference (law, code, precedent, doctrine)")
    reference: str = Field(description="Specific reference (article number, case name, etc.)")
    content: str = Field(description="Relevant text or summary of the reference")
    relevance: str = Field(description="How this reference applies to the case")

class Evidence(BaseModel):
    """A piece of evidence in the case."""
    description: str
    type: str = Field(description="Type of evidence (document, testimony, expert opinion, etc.)")
    source: str
    strength: str = Field(description="Strong, Medium, or Weak")
    relevance: str = Field(description="How this evidence supports the argument")

class LegalArgument(BaseModel):
    """A legal argument with supporting references and evidence."""
    title: str
    content: str
    legal_references: List[LegalReference] = []
    supporting_evidence: List[Evidence] = []
    counter_arguments: List[str] = []
    rebuttal: Optional[str] = None

class CaseParty(BaseModel):
    """A party in the case."""
    name: str
    role: str
    representation: Optional[str] = None
    contact_info: Optional[Dict[str, str]] = None

class CaseMetadata(BaseModel):
    """Metadata about the case."""
    case_title: str
    case_number: Optional[str] = None
    court: str
    jurisdiction: str
    filing_date: Optional[str] = None
    parties: List[CaseParty] = []
    judge: Optional[str] = None
    hearing_date: Optional[str] = None

class LetterMetadata(BaseModel):
    """Metadata about the letter."""
    sender: str
    sender_title: str
    sender_contact: Dict[str, str]
    recipient: str
    recipient_title: Optional[str] = None
    recipient_contact: Dict[str, str]
    date: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))
    subject: str
    reference_numbers: Optional[Dict[str, str]] = None
    letter_type: str = Field(description="Type of legal letter (motion, response, demand, etc.)")

class LetterContent(BaseModel):
    """Content of the legal letter."""
    introduction: str
    summary_of_facts: str
    legal_arguments: List[LegalArgument] = []
    requested_action: str
    conclusion: str
    attachments: Optional[List[str]] = None

class LegalLetter(BaseModel):
    """Complete legal letter with metadata and content."""
    metadata: LetterMetadata
    case_info: CaseMetadata
    content: LetterContent

# Node: Extract Case Metadata using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("case_metadata_system_prompt.txt"),
    output="case_metadata",
    response_model=CaseMetadata,
    prompt_template=read_template("case_metadata_prompt.txt")
)
async def extract_case_metadata(case_content: str, language: str = DEFAULT_LANGUAGE) -> CaseMetadata:
    """Extract metadata from the case."""
    pass

# Node: Generate Letter Metadata using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("letter_metadata_system_prompt.txt"),
    output="letter_metadata",
    response_model=LetterMetadata,
    prompt_template=read_template("letter_metadata_prompt.txt")
)
async def generate_letter_metadata(
    case_metadata: CaseMetadata, 
    case_content: str, 
    role: str = DEFAULT_ROLE,
    language: str = DEFAULT_LANGUAGE
) -> LetterMetadata:
    """Generate metadata for the legal letter."""
    pass

# Node: Generate Letter Content using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("letter_content_system_prompt.txt"),
    output="letter_content",
    response_model=LetterContent,
    prompt_template=read_template("letter_content_prompt.txt")
)
async def generate_letter_content(
    case_metadata: CaseMetadata,
    letter_metadata: LetterMetadata,
    case_content: str,
    role: str = DEFAULT_ROLE,
    language: str = DEFAULT_LANGUAGE
) -> LetterContent:
    """Generate content for the legal letter."""
    pass

# Node: Assemble Complete Letter
@Nodes.define(output="legal_letter")
async def assemble_letter(
    case_metadata: CaseMetadata,
    letter_metadata: LetterMetadata,
    letter_content: LetterContent
) -> LegalLetter:
    """Assemble the complete legal letter."""
    return LegalLetter(
        metadata=letter_metadata,
        case_info=case_metadata,
        content=letter_content
    )

# Node: Format Letter Document
@Nodes.llm_node(
    system_prompt=read_template("format_letter_system_prompt.txt"),
    output="letter_document",
    prompt_template=read_template("format_letter_prompt.txt")
)
async def format_letter_document(
    legal_letter: LegalLetter,
    language: str = DEFAULT_LANGUAGE
) -> str:
    """Format the legal letter into a well-structured document."""
    pass

# Node: Save Letter Document
@Nodes.define(output="document_file_path")
async def save_letter_document(letter_document: str, legal_letter: LegalLetter, output_dir: str) -> str:
    """Save the letter document to a file."""
    try:
        # Create a filename based on letter type, role and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_subject = legal_letter.metadata.subject.replace(" ", "_").lower()[:30]
        role = legal_letter.metadata.letter_type.lower()
        filename = f"{role}_{safe_subject}_{timestamp}.md"
        
        # Ensure output directory exists
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the document to file
        with output_path.open("w", encoding="utf-8") as f:
            f.write(letter_document)
        
        logger.info(f"Saved letter document to: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving letter document: {e}")
        raise

# Define the Legal Letter Generator Workflow
def create_letter_generator_workflow() -> Workflow:
    """Create a workflow to generate legal letters."""
    wf = Workflow("extract_case_metadata")
    
    # Add all nodes with input mappings for dynamic model passing
    wf.node("extract_case_metadata", inputs_mapping={"model": "model", "language": "language"})
    wf.node("generate_letter_metadata", inputs_mapping={"model": "model", "language": "language", "role": "role"})
    wf.node("generate_letter_content", inputs_mapping={"model": "model", "language": "language", "role": "role"})
    wf.node("assemble_letter")
    wf.node("format_letter_document", inputs_mapping={"model": "model", "language": "language"})
    wf.node("save_letter_document")
    
    # Define linear sequence
    wf.current_node = "extract_case_metadata"
    wf.transitions["extract_case_metadata"] = [("generate_letter_metadata", None)]
    wf.transitions["generate_letter_metadata"] = [("generate_letter_content", None)]
    wf.transitions["generate_letter_content"] = [("assemble_letter", None)]
    wf.transitions["assemble_letter"] = [("format_letter_document", None)]
    wf.transitions["format_letter_document"] = [("save_letter_document", None)]
    
    return wf

# Function to Run the Workflow
async def generate_legal_letter(
    case_content: str,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    role: str = DEFAULT_ROLE,
    output_dir: Optional[str] = None,
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Execute the workflow with the given case content and parameters."""
    if output_dir:
        output_dir = os.path.expanduser(output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Validate role
    valid_roles = ["defense", "prosecution"]
    if role not in valid_roles:
        logger.warning(f"Invalid role: {role}. Using default: {DEFAULT_ROLE}")
        role = DEFAULT_ROLE

    # Initial context with model keys for dynamic mapping
    initial_context = {
        "case_content": case_content,
        "model": model,
        "language": language,
        "role": role,
        "output_dir": output_dir
    }

    try:
        workflow = create_letter_generator_workflow()
        engine = workflow.build()
        
        result = await engine.run(initial_context)
        
        if "letter_document" not in result or not result["letter_document"]:
            logger.warning("No letter document generated.")
            raise ValueError("Workflow completed but no letter document was generated.")
        
        logger.info("Workflow completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise

async def display_results(letter_document: str, document_file_path: str, legal_letter: LegalLetter):
    """Async helper function to display results with animation."""
    console.print("\n[bold green]Lettre juridique générée avec succès:[/]")
    
    console.print(f"\n[bold blue]Informations sur la lettre:[/]")
    console.print(f"Type: {legal_letter.metadata.letter_type}")
    console.print(f"Expéditeur: {legal_letter.metadata.sender} ({legal_letter.metadata.sender_title})")
    console.print(f"Destinataire: {legal_letter.metadata.recipient}")
    console.print(f"Sujet: {legal_letter.metadata.subject}")
    
    console.print("\n[bold blue]Arguments juridiques:[/]")
    for i, arg in enumerate(legal_letter.content.legal_arguments, 1):
        console.print(f"[blue]{i}. {arg.title}[/]")
    
    console.print(f"\n[green]✓ Lettre juridique sauvegardée à:[/] {document_file_path}")

@app.command()
def generate(
    case_file: str = typer.Option(..., help="Chemin vers le fichier contenant le cas juridique"),
    model: str = typer.Option(DEFAULT_MODEL, help="Modèle LLM à utiliser"),
    language: str = typer.Option(DEFAULT_LANGUAGE, help="Langue de sortie (french, english, arabic)"),
    role: str = typer.Option(DEFAULT_ROLE, help="Rôle (defense, prosecution)"),
    output_dir: Optional[str] = typer.Option(None, help="Répertoire de sortie pour la lettre")
):
    """Générer une lettre juridique convaincante avec références aux lois et preuves."""
    try:
        # Read the case file
        with open(case_file, 'r', encoding='utf-8') as f:
            case_content = f.read()
        
        console.print(f"[bold]Génération d'une lettre juridique pour le rôle: {role}...[/]")
        
        # Run the workflow
        result = asyncio.run(generate_legal_letter(
            case_content=case_content,
            model=model,
            language=language,
            role=role,
            output_dir=output_dir
        ))
        
        # Display results
        letter_document = result["letter_document"]
        document_file_path = result.get("document_file_path", "Not saved")
        legal_letter = result.get("legal_letter")
        
        asyncio.run(display_results(letter_document, document_file_path, legal_letter))
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
