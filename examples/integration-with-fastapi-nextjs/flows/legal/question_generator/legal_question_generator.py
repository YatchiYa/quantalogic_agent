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
app = typer.Typer(help="Générer des questions détaillées pour RAG juridique")
console = Console()

# Default models
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_LANGUAGE = "french"  # Default language for output
DEFAULT_PERSPECTIVE = "both"  # Default perspective (both, defense, prosecution)
DEFAULT_MIN_QUESTIONS = 10
DEFAULT_MAX_QUESTIONS = 20

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
class LegalFact(BaseModel):
    """A relevant fact from the case."""
    description: str
    relevance: str = Field(description="Why this fact is legally relevant")

class LegalIssue(BaseModel):
    """A legal issue identified in the case."""
    issue: str
    area_of_law: str
    complexity: str = Field(description="High, Medium, or Low")

class CaseMetadata(BaseModel):
    """Metadata about the legal case."""
    title: str
    case_type: str
    jurisdiction: str
    parties: List[str]
    summary: str
    key_facts: List[LegalFact] = []
    legal_issues: List[LegalIssue] = []

class LegalQuestion(BaseModel):
    """A structured legal question for RAG."""
    question: str
    perspective: str = Field(description="defense, prosecution, or both")
    legal_area: str
    relevance: str = Field(description="Why this question is relevant to the case")
    expected_sources: List[str] = Field(description="Types of legal sources that might answer this question")

class QuestionSet(BaseModel):
    """A set of legal questions for RAG."""
    case_metadata: CaseMetadata
    questions: List[LegalQuestion]
    defense_questions_count: int = 0
    prosecution_questions_count: int = 0
    general_questions_count: int = 0

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

# Node: Generate Legal Questions using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("question_generator_system_prompt.txt"),
    output="question_set",
    response_model=QuestionSet,
    prompt_template=read_template("question_generator_prompt.txt")
)
async def generate_legal_questions(
    case_metadata: CaseMetadata, 
    case_content: str, 
    perspective: str = DEFAULT_PERSPECTIVE,
    min_questions: int = DEFAULT_MIN_QUESTIONS,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    language: str = DEFAULT_LANGUAGE
) -> QuestionSet:
    """Generate a set of legal questions for RAG."""
    pass

# Node: Format Questions Document
@Nodes.llm_node(
    system_prompt=read_template("format_questions_system_prompt.txt"),
    output="questions_document",
    prompt_template=read_template("format_questions_prompt.txt")
)
async def format_questions_document(
    question_set: QuestionSet, 
    case_content: str,
    language: str = DEFAULT_LANGUAGE
) -> str:
    """Format the questions into a well-structured document."""
    pass

# Node: Save Questions Document
@Nodes.define(output="document_file_path")
async def save_questions_document(questions_document: str, case_metadata: CaseMetadata, output_dir: str) -> str:
    """Save the questions document to a file."""
    try:
        # Create a filename based on case title and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = case_metadata.title.replace(" ", "_").lower()[:30]
        filename = f"{safe_title}_questions_{timestamp}.md"
        
        # Ensure output directory exists
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the document to file
        with output_path.open("w", encoding="utf-8") as f:
            f.write(questions_document)
        
        logger.info(f"Saved questions document to: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving questions document: {e}")
        raise

# Define the Legal Question Generator Workflow
def create_question_generator_workflow() -> Workflow:
    """Create a workflow to generate legal questions for RAG."""
    wf = Workflow("extract_case_metadata")
    
    # Add all nodes with input mappings for dynamic model passing
    wf.node("extract_case_metadata", inputs_mapping={"model": "model", "language": "language"})
    wf.node("generate_legal_questions", inputs_mapping={
        "model": "model", 
        "language": "language", 
        "perspective": "perspective",
        "min_questions": "min_questions",
        "max_questions": "max_questions"
    })
    wf.node("format_questions_document", inputs_mapping={"model": "model", "language": "language"})
    wf.node("save_questions_document")
    
    # Define linear sequence
    wf.current_node = "extract_case_metadata"
    wf.transitions["extract_case_metadata"] = [("generate_legal_questions", None)]
    wf.transitions["generate_legal_questions"] = [("format_questions_document", None)]
    wf.transitions["format_questions_document"] = [("save_questions_document", None)]
    
    return wf

# Function to Run the Workflow
async def generate_legal_questions_for_rag(
    case_content: str,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    perspective: str = DEFAULT_PERSPECTIVE,
    min_questions: int = DEFAULT_MIN_QUESTIONS,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
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

    # Validate perspective
    valid_perspectives = ["both", "defense", "prosecution"]
    if perspective not in valid_perspectives:
        logger.warning(f"Invalid perspective: {perspective}. Using default: {DEFAULT_PERSPECTIVE}")
        perspective = DEFAULT_PERSPECTIVE

    # Initial context with model keys for dynamic mapping
    initial_context = {
        "case_content": case_content,
        "model": model,
        "language": language,
        "perspective": perspective,
        "min_questions": min_questions,
        "max_questions": max_questions,
        "output_dir": output_dir
    }

    try:
        workflow = create_question_generator_workflow()
        engine = workflow.build()
        
        result = await engine.run(initial_context)
        
        if "questions_document" not in result or not result["questions_document"]:
            logger.warning("No questions document generated.")
            raise ValueError("Workflow completed but no questions document was generated.")
        
        logger.info("Workflow completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise

async def display_results(questions_document: str, document_file_path: str, question_set: QuestionSet):
    """Async helper function to display results with animation."""
    console.print("\n[bold green]Résumé de la génération de questions:[/]")
    
    console.print(f"\n[bold blue]Métadonnées du cas:[/]")
    console.print(f"Titre: {question_set.case_metadata.title}")
    console.print(f"Type: {question_set.case_metadata.case_type}")
    console.print(f"Juridiction: {question_set.case_metadata.jurisdiction}")
    
    console.print("\n[bold blue]Statistiques des questions:[/]")
    console.print(f"Questions pour la défense: {question_set.defense_questions_count}")
    console.print(f"Questions pour l'accusation: {question_set.prosecution_questions_count}")
    console.print(f"Questions générales: {question_set.general_questions_count}")
    console.print(f"Total des questions: {len(question_set.questions)}")
    
    console.print(f"\n[green]✓ Document de questions sauvegardé à:[/] {document_file_path}")

@app.command()
def generate(
    case_file: str = typer.Option(..., help="Chemin vers le fichier contenant le cas juridique"),
    model: str = typer.Option(DEFAULT_MODEL, help="Modèle LLM à utiliser"),
    language: str = typer.Option(DEFAULT_LANGUAGE, help="Langue de sortie (french, english, arabic)"),
    perspective: str = typer.Option(DEFAULT_PERSPECTIVE, help="Perspective (both, defense, prosecution)"),
    min_questions: int = typer.Option(DEFAULT_MIN_QUESTIONS, help="Nombre minimum de questions à générer"),
    max_questions: int = typer.Option(DEFAULT_MAX_QUESTIONS, help="Nombre maximum de questions à générer"),
    output_dir: Optional[str] = typer.Option(None, help="Répertoire de sortie pour le document de questions")
):
    """Générer des questions détaillées pour RAG juridique à partir d'un cas."""
    try:
        # Read the case file
        with open(case_file, 'r', encoding='utf-8') as f:
            case_content = f.read()
        
        console.print(f"[bold]Génération de questions pour RAG juridique...[/]")
        
        # Run the workflow
        result = asyncio.run(generate_legal_questions_for_rag(
            case_content=case_content,
            model=model,
            language=language,
            perspective=perspective,
            min_questions=min_questions,
            max_questions=max_questions,
            output_dir=output_dir
        ))
        
        # Display results
        questions_document = result["questions_document"]
        document_file_path = result.get("document_file_path", "Not saved")
        question_set = result.get("question_set")
        
        asyncio.run(display_results(questions_document, document_file_path, question_set))
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
