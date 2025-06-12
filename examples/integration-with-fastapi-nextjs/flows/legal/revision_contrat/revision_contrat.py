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
from ...service import event_observer

# Initialize Typer app and rich console
app = typer.Typer(help="Réviser des contrats en utilisant des LLMs avec des templates personnalisables")
console = Console()

# Default models for different phases
DEFAULT_ANALYSIS_MODEL = "openai/gpt-4o-mini"
DEFAULT_REVISION_MODEL = "openai/gpt-4o-mini"
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
class ContractClause(BaseModel):
    """Represents a clause in a contract."""
    title: str
    content: str
    issues: List[str] = []
    risk_level: str = Field(description="High, Medium, or Low")
    recommendations: List[str] = []

class ContractParty(BaseModel):
    """Represents a party in a contract."""
    name: str
    role: str
    obligations: Optional[List[str]] = None
    rights: Optional[List[str]] = None

class ContractMetadata(BaseModel):
    """Metadata about the contract."""
    title: str
    contract_type: str
    date: Optional[str] = None
    effective_date: Optional[str] = None
    termination_date: Optional[str] = None
    jurisdiction: Optional[str] = None
    governing_law: Optional[str] = None
    parties: List[ContractParty] = []
    key_dates: Optional[Dict[str, str]] = None

class ContractAnalysis(BaseModel):
    """Complete analysis of a contract."""
    metadata: ContractMetadata
    clauses: List[ContractClause] = []
    summary: str
    overall_risk_assessment: str
    missing_elements: List[str] = []
    ambiguities: List[str] = []
    compliance_issues: List[str] = []

class ContractRevision(BaseModel):
    """Structured contract revision."""
    general_assessment: str
    revised_clauses: List[ContractClause] = []
    suggested_additions: List[str] = []
    suggested_removals: List[str] = []
    compliance_recommendations: List[str] = []
    negotiation_points: List[str] = []

# Node: Extract Contract Metadata using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("contract_metadata_system_prompt.txt"),
    output="contract_metadata",
    response_model=ContractMetadata,
    prompt_template=read_template("contract_metadata_prompt.txt")
)
async def extract_contract_metadata(contract_content: str, custom_instructions: str = "") -> ContractMetadata:
    """Extract metadata from the contract."""
    pass

# Node: Analyze Contract Content using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("contract_analysis_system_prompt.txt"),
    output="contract_analysis",
    response_model=ContractAnalysis,
    prompt_template=read_template("contract_analysis_prompt.txt")
)
async def analyze_contract_content(contract_content: str, contract_metadata: ContractMetadata, custom_instructions: str = "") -> ContractAnalysis:
    """Analyze the contract content and provide a comprehensive analysis."""
    pass

# Node: Generate Contract Revision using Structured LLM
@Nodes.structured_llm_node(
    system_prompt=read_template("contract_revision_system_prompt.txt"),
    output="contract_revision",
    response_model=ContractRevision,
    prompt_template=read_template("contract_revision_prompt.txt")
)
async def generate_contract_revision(contract_analysis: ContractAnalysis, custom_instructions: str = "") -> ContractRevision:
    """Generate a structured contract revision based on contract analysis."""
    pass

# Node: Generate Revised Contract Document
@Nodes.llm_node(
    system_prompt=read_template("revised_contract_system_prompt.txt"),
    output="revised_contract",
    prompt_template=read_template("revised_contract_prompt.txt")
)
async def generate_revised_contract(contract_analysis: ContractAnalysis, contract_revision: ContractRevision, original_contract: str, language: str = "french") -> str:
    """Generate a revised contract document based on analysis and revision."""
    pass

# Node: Save Revised Contract
@Nodes.define(output="document_file_path")
async def save_revised_contract(revised_contract: str, contract_metadata: ContractMetadata, output_dir: str) -> str:
    """Save the revised contract to a file."""
    try:
        # Create a filename based on contract title and timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = contract_metadata.title.replace(" ", "_").lower()[:30]
        filename = f"{safe_title}_revised_{timestamp}.md"
        
        # Ensure output directory exists
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the document to file
        with output_path.open("w", encoding="utf-8") as f:
            f.write(revised_contract)
        
        logger.info(f"Saved revised contract to: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving revised contract: {e}")
        raise

# Define the Contract Revision Workflow
def create_contract_revision_workflow() -> Workflow:
    """Create a workflow to revise contracts."""
    wf = Workflow("extract_contract_metadata")
    
    # Add all nodes with input mappings for dynamic model passing
    wf.node("extract_contract_metadata", inputs_mapping={"model": "analysis_model"})
    wf.node("analyze_contract_content", inputs_mapping={"model": "analysis_model"})
    wf.node("generate_contract_revision", inputs_mapping={"model": "revision_model"})
    wf.node("generate_revised_contract", inputs_mapping={"model": "revision_model", "language": "language", "original_contract": "contract_content"})
    wf.node("save_revised_contract")
    
    # Define linear sequence
    wf.current_node = "extract_contract_metadata"
    wf.transitions["extract_contract_metadata"] = [("analyze_contract_content", None)]
    wf.transitions["analyze_contract_content"] = [("generate_contract_revision", None)]
    wf.transitions["generate_contract_revision"] = [("generate_revised_contract", None)]
    wf.transitions["generate_revised_contract"] = [("save_revised_contract", None)]
    
    return wf

# Function to Run the Workflow
async def revise_contract(
    contract_content: str,
    analysis_model: str,
    revision_model: str,
    language: str = DEFAULT_LANGUAGE,
    custom_metadata_instructions: str = "",
    custom_analysis_instructions: str = "",
    custom_revision_instructions: str = "",
    output_dir: Optional[str] = None,
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Execute the workflow with the given contract content and models."""
    if output_dir:
        output_dir = os.path.expanduser(output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initial context with model keys for dynamic mapping
    initial_context = {
        "contract_content": contract_content,
        "analysis_model": analysis_model,
        "revision_model": revision_model,
        "language": language,  # Language for output
        "output_dir": output_dir,
        "custom_instructions": custom_metadata_instructions,  # For metadata extraction
        "custom_analysis_instructions": custom_analysis_instructions,  # For contract analysis
        "custom_revision_instructions": custom_revision_instructions  # For contract revision
    }

    try:
        workflow = create_contract_revision_workflow()
        engine = workflow.build()
        
        
        # Add the event observer if _handle_event is provided
        if _handle_event:
            # Create a lambda to bind task_id to the observer
            bound_observer = lambda event: asyncio.create_task(
                event_observer(event, task_id=task_id, _handle_event=_handle_event)
            )
            engine.add_observer(bound_observer)
        
        result = await engine.run(initial_context)
        
        if "revised_contract" not in result or not result["revised_contract"]:
            logger.warning("No revised contract generated.")
            raise ValueError("Workflow completed but no revised contract was generated.")
        
        logger.info("Workflow completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise

async def display_results(revised_contract: str, document_file_path: str, contract_revision: ContractRevision):
    """Async helper function to display results with animation."""
    console.print("\n[bold green]Résumé de la révision du contrat:[/]")
    console.print(Panel(contract_revision.general_assessment, border_style="blue"))
    
    console.print("\n[bold green]Points clés de la révision:[/]")
    if contract_revision.negotiation_points:
        for i, point in enumerate(contract_revision.negotiation_points, 1):
            console.print(f"[blue]{i}.[/] {point}")
    
    console.print(f"\n[green]✓ Contrat révisé sauvegardé à:[/] {document_file_path}")

if __name__ == "__main__":
    # Example usage with direct initialization
    import sys
    
    # Default values
    default_contract_content = """
    CONTRAT DE PRESTATION DE SERVICES
    
    ENTRE LES SOUSSIGNÉS:
    
    La société TechSolutions SAS, au capital de 50.000 euros, dont le siège social est situé au 42 Avenue des Champs-Élysées, 75008 Paris, immatriculée au RCS de Paris sous le numéro 123 456 789, représentée par M. Jean Dupont, en sa qualité de Président, dûment habilité aux fins des présentes,
    
    Ci-après dénommée "le Prestataire",
    
    ET
    
    La société ClientCorp SA, au capital de 100.000 euros, dont le siège social est situé au 15 Rue de la Paix, 75002 Paris, immatriculée au RCS de Paris sous le numéro 987 654 321, représentée par Mme Marie Martin, en sa qualité de Directrice Générale, dûment habilitée aux fins des présentes,
    
    Ci-après dénommée "le Client",
    
    IL A ÉTÉ CONVENU CE QUI SUIT:
    
    Article 1 - Objet du contrat
    Le Prestataire s'engage à fournir au Client des prestations de développement informatique telles que décrites en Annexe 1.
    
    Article 2 - Durée
    Le présent contrat est conclu pour une durée de 12 mois à compter de sa signature. Il pourra être renouvelé par tacite reconduction pour des périodes successives de 12 mois.
    
    Article 3 - Prix et modalités de paiement
    Le prix des prestations est fixé à 10.000 euros HT par mois.
    Le paiement sera effectué par virement bancaire dans un délai de 30 jours à compter de la date d'émission de la facture.
    
    Article 4 - Obligations du Prestataire
    Le Prestataire s'engage à:
    - Exécuter les prestations conformément aux règles de l'art
    - Respecter les délais convenus
    - Affecter du personnel qualifié à la réalisation des prestations
    
    Article 5 - Obligations du Client
    Le Client s'engage à:
    - Fournir au Prestataire toutes les informations nécessaires à la bonne exécution des prestations
    - Payer le prix convenu dans les délais impartis
    
    Article 6 - Propriété intellectuelle
    Les droits de propriété intellectuelle sur les développements réalisés par le Prestataire restent la propriété du Prestataire.
    
    Article 7 - Confidentialité
    Les parties s'engagent à maintenir confidentielles les informations échangées dans le cadre du présent contrat.
    
    Article 8 - Responsabilité
    La responsabilité du Prestataire est limitée au montant des sommes perçues au titre du présent contrat.
    
    Article 9 - Résiliation
    Le contrat pourra être résilié par l'une ou l'autre des parties en cas de manquement grave de l'autre partie à ses obligations, après mise en demeure restée sans effet pendant 30 jours.
    
    Article 10 - Loi applicable et juridiction compétente
    Le présent contrat est soumis au droit français. Tout litige relatif à son interprétation ou à son exécution relèvera de la compétence exclusive du Tribunal de Commerce de Paris.
    
    Fait à Paris, le [DATE]
    
    Pour le Prestataire                                Pour le Client
    Jean Dupont                                        Marie Martin
    Président                                          Directrice Générale
    """
    default_analysis_model = DEFAULT_ANALYSIS_MODEL
    default_revision_model = DEFAULT_REVISION_MODEL
    default_language = DEFAULT_LANGUAGE
    default_output_dir = None
    
    # Run with default values
    try:
        logger.info(f"Révision du contrat en cours")
        result = asyncio.run(revise_contract(
            default_contract_content,
            default_analysis_model,
            default_revision_model,
            language=default_language,
            custom_metadata_instructions="Extraire tous les détails pertinents du contrat, y compris les parties, dates clés et juridiction.",
            custom_analysis_instructions="Analyser en détail toutes les clauses du contrat, identifier les risques et les ambiguïtés.",
            custom_revision_instructions="Proposer des améliorations pour toutes les clauses à risque et suggérer des ajouts pour les éléments manquants.",
            output_dir=default_output_dir
        ))
        
        revised_contract = result["revised_contract"]
        document_file_path = result.get("document_file_path", "Not saved")
        contract_revision = result.get("contract_revision")
        
        # Display results
        asyncio.run(display_results(revised_contract, document_file_path, contract_revision))
        
    except Exception as e:
        logger.error(f"Error during contract revision: {e}")
        console.print(f"[bold red]Error:[/] {str(e)}")
