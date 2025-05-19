import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Union

import pandas as pd
import typer
from loguru import logger
from pydantic import BaseModel
from pyzerox import zerox
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from ..service import event_observer

# Initialize Typer app and rich console
app = typer.Typer(help="Extract structured data from documents and output to CSV format")
console = Console()

# Default models for different phases
DEFAULT_ANALYSIS_MODEL = "openai/gpt-4o-mini"

# Define Pydantic models for structured output
class InvoiceEntry(BaseModel):
    contract_number: str
    matricule: str
    color: str
    quantity: float
    amount_ht: float
    page_number: int

class InvoiceData(BaseModel):
    entries: List[InvoiceEntry]
    total_amount_ht: float

class DocumentContext(BaseModel):
    """Model for document context data"""
    context_id: str
    content: str
    metadata: Optional[Dict] = None

# Node: Extract Invoice Data using Structured LLM
@Nodes.structured_llm_node(
    system_prompt="You are an AI assistant specialized in extracting structured data from invoice documents.",
    output="invoice_data",
    response_model=InvoiceData,
    prompt_template="""En tant que spécialiste de l'extraction de données précises, analyse le document fourni qui présente une liste de factures et effectue les tâches suivantes :

EXIGENCES D'EXTRACTION : 
Traite toutes les pages du document :
Pour chaque Contrat du document, extraie toutes les quantités livrées (colonne Qté Liv.) avec les informations suivantes : 
- Extraie le Numéro de contrat 
- Extraie le matricule
- Identifier la spécification de couleur pour le format A4 (e.g. noir ou Couleur) 
- Collecte toutes les quantités du format de ligne A4 (Information colonne Qté Liv.) et les Montants HT associés (colonne Montant H.T.)
- Fournis le Numéro de page du document pdf en lecture (e.g. page 1 pour 1 / 56, page 2 pour 2 / 56))
Attention, Les informations peuvent etre réparties sur 2 pages successives

FORMATAGE DE LA SORTIE : 
Ecris un tableau avec six colonnes "N° Contrat", "Matricule", "Couleur", "Quantity", "Montant HT", "Numéro de page" 
| N° Contrat | Matricule | Couleur  | Quantity | Montant HT | Numéro de Page |

Chaque entrée sur une nouvelle ligne 
Maintenir l'ordre original des données par ordre croissant des pages 
Fournir le tableau complet pour la lecture de toutes les pages du document

La dernière ligne du tableau propose le cumul du Montant H.T. de toutes les lignes du tableau

Voici le contenu du document à analyser :

{{document_content}}

Extrait les données selon les exigences et retourne-les au format structuré.
"""
)
async def extract_invoice_data(document_content: str) -> InvoiceData:
    """Extract structured invoice data from document content."""
    pass

# Node: Convert Invoice Data to DataFrame
@Nodes.define(output="dataframe")
async def convert_to_dataframe(invoice_data: InvoiceData) -> pd.DataFrame:
    """Convert the structured invoice data to a pandas DataFrame."""
    try:
        # Create DataFrame from invoice entries
        entries = [entry.dict() for entry in invoice_data.entries]
        df = pd.DataFrame(entries)
        
        # Add a total row
        total_row = {
            'contract_number': 'TOTAL',
            'matricule': '',
            'color': '',
            'quantity': df['quantity'].sum(),
            'amount_ht': invoice_data.total_amount_ht,
            'page_number': None
        }
        
        # Don't append the total row to the DataFrame as it will be handled separately
        # when saving to CSV or displaying
        
        logger.info(f"Created DataFrame with {len(df)} rows of invoice data")
        return df
    except Exception as e:
        logger.error(f"Error converting invoice data to DataFrame: {e}")
        raise

# Node: Save DataFrame to Output Format
@Nodes.define(output="output_file_path")
async def save_to_format(dataframe: pd.DataFrame, invoice_data: InvoiceData, output_dir: str, context_id: str, output_format: str) -> str:
    """Save the DataFrame to the specified output format (csv, json)."""
    try:
        output_dir_expanded = os.path.expanduser(output_dir)
        os.makedirs(output_dir_expanded, exist_ok=True)
        
        if output_format.lower() == "csv":
            output_path = Path(output_dir_expanded) / f"{context_id}.results.csv"
            
            # Save the main data
            dataframe.to_csv(output_path, index=False)
            
            # Append the total row to the CSV file
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(f"TOTAL,,{dataframe['quantity'].sum()},{invoice_data.total_amount_ht},,\n")
            
            logger.info(f"Saved invoice data to CSV: {output_path}")
            
        elif output_format.lower() == "json":
            output_path = Path(output_dir_expanded) / f"{context_id}.results.json"
            
            # Convert to JSON structure with entries and total
            json_data = {
                "entries": dataframe.to_dict(orient="records"),
                "total": {
                    "quantity": float(dataframe['quantity'].sum()),
                    "amount_ht": float(invoice_data.total_amount_ht)
                }
            }
            
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved invoice data to JSON: {output_path}")
            
        else:
            raise ValueError(f"Unsupported output format: {output_format}. Supported formats: csv, json")
            
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving DataFrame to {output_format}: {e}")
        raise

# Define the Workflow for Context Processing
def create_context_analysis_workflow() -> Workflow:
    """Create a workflow to extract structured data from document contexts and output to the specified format."""
    wf = Workflow("extract_invoice_data")
    
    # Add all nodes with input mappings for dynamic model passing
    wf.node("extract_invoice_data", inputs_mapping={"model": "analysis_model"})
    wf.node("convert_to_dataframe")
    wf.node("save_to_format")
    
    # Define the workflow structure
    wf.current_node = "extract_invoice_data"
    wf.transitions["extract_invoice_data"] = [("convert_to_dataframe", None)]
    wf.transitions["convert_to_dataframe"] = [("save_to_format", None)]
    
    return wf

# Function to Run the Workflow with Context
async def analyze_document_context(
    document_context: DocumentContext,
    analysis_model: str,
    output_dir: str,
    output_format: str = "csv",
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Execute the workflow with the given document context."""
    output_dir_expanded = os.path.expanduser(output_dir)
    os.makedirs(output_dir_expanded, exist_ok=True)
    
    # Validate output format
    if output_format.lower() not in ["csv", "json"]:
        logger.warning(f"Unsupported output format: {output_format}. Defaulting to CSV.")
        output_format = "csv"

    # Initial context with model keys for dynamic mapping
    initial_context = {
        "document_content": document_context.content,
        "context_id": document_context.context_id,
        "analysis_model": analysis_model,
        "output_dir": output_dir_expanded,
        "output_format": output_format
    }

    try:
        workflow = create_context_analysis_workflow()
        engine = workflow.build()

        # Add the event observer if _handle_event is provided
        if _handle_event:
            # Create a lambda to bind task_id to the observer
            bound_observer = lambda event: asyncio.create_task(
                event_observer(event, task_id=task_id, _handle_event=_handle_event)
            )
            engine.add_observer(bound_observer)

        result = await engine.run(initial_context)
        
        if "dataframe" not in result or result["dataframe"].empty:
            logger.warning("No invoice data extracted.")
            raise ValueError("Workflow completed but no invoice data was extracted.")
        
        logger.info(f"Workflow completed successfully for context {document_context.context_id}")
        return result
    except Exception as e:
        logger.error(f"Error during workflow execution for context {document_context.context_id}: {e}")
        raise

async def display_results(dataframe: pd.DataFrame, invoice_data: InvoiceData, output_file_path: str, context_id: str, output_format: str = "csv"):
    """Async helper function to display results with animation."""
    # Create a rich table from the DataFrame
    table = Table(title=f"Extracted Invoice Data - Context: {context_id}")
    
    # Add columns
    table.add_column("N° Contrat", style="cyan")
    table.add_column("Matricule", style="green")
    table.add_column("Couleur", style="yellow")
    table.add_column("Quantity", style="blue")
    table.add_column("Montant HT", style="magenta")
    table.add_column("Numéro de Page", style="white")
    
    # Add rows from DataFrame
    for _, row in dataframe.iterrows():
        table.add_row(
            str(row['contract_number']),
            str(row['matricule']),
            str(row['color']),
            str(row['quantity']),
            str(row['amount_ht']),
            str(row['page_number'])
        )
    
    # Add total row
    table.add_row(
        "TOTAL", "", "", 
        str(dataframe['quantity'].sum()), 
        str(invoice_data.total_amount_ht), 
        "",
        style="bold"
    )
    
    console.print(f"\n[bold green]Extracted Invoice Data for Context {context_id}:[/]")
    console.print(table)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        progress.add_task("[cyan]Processing results...", total=None)
        await asyncio.sleep(1)  # Simulate some processing time for effect
    
    console.print(f"[green]✓ {output_format.upper()} file saved to:[/] {output_file_path}")

@app.command()
def analyze_context(
    context_content: Annotated[str, typer.Argument(help="Content of the document to analyze")],
    analysis_model: Annotated[str, typer.Option(help="LLM model for data extraction and analysis")] = DEFAULT_ANALYSIS_MODEL,
    output_format: Annotated[str, typer.Option(help="Output format (csv, json)")] = "csv",
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None,
    output_dir: Annotated[str, typer.Option(help="Directory to save output files (supports ~ expansion)")] = "~/output"
):
    """Extract structured data from document context and output to the specified format."""
    try:
        # Generate a timestamp-based context ID
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        context_id = f"context_{timestamp}"
        
        document_context = DocumentContext(
            context_id=context_id,
            content=context_content
        )
        
        with console.status(f"Processing document content..."):
            result = asyncio.run(analyze_document_context(
                document_context,
                analysis_model,
                output_dir,
                output_format=output_format,
                _handle_event=_handle_event,
                task_id=task_id
            ))
        
        dataframe = result["dataframe"]
        invoice_data = result["invoice_data"]
        output_file_path = result["output_file_path"]
        
        # Run the async display function
        asyncio.run(display_results(dataframe, invoice_data, output_file_path, context_id, output_format))
    
    except Exception as e:
        logger.error(f"Failed to run workflow: {e}")
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)

@app.command()
def batch_analyze_contexts(
    contexts_file: Annotated[str, typer.Argument(help="Path to JSON file containing contexts (list of DocumentContext objects)")],
    analysis_model: Annotated[str, typer.Option(help="LLM model for data extraction and analysis")] = DEFAULT_ANALYSIS_MODEL,
    output_format: Annotated[str, typer.Option(help="Output format (csv, json)")] = "csv",
    output_dir: Annotated[str, typer.Option(help="Directory to save output files (supports ~ expansion)")] = "~/output",
    combine_results: Annotated[bool, typer.Option(help="Combine all results into a single file")] = False,
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None,
):
    """Batch process multiple document contexts and extract structured data to CSV format."""
    try:
        import json
        
        # Load contexts from file
        contexts_file_expanded = os.path.expanduser(contexts_file)
        with open(contexts_file_expanded, 'r', encoding='utf-8') as f:
            contexts_data = json.load(f)
        
        # Convert to DocumentContext objects
        contexts = [DocumentContext(**ctx) for ctx in contexts_data]
        
        all_dataframes = []
        all_results = []
        
        for context in contexts:
            with console.status(f"Processing context [bold blue]{context.context_id}[/]..."):
                result = asyncio.run(analyze_document_context(
                    context,
                    analysis_model,
                    output_dir,
                    output_format=output_format,
                    _handle_event=_handle_event,
                    task_id=task_id
                ))
                
                all_results.append(result)
                all_dataframes.append(result["dataframe"])
                
                # Display individual results
                asyncio.run(display_results(
                    result["dataframe"], 
                    result["invoice_data"], 
                    result["output_file_path"],
                    context.context_id,
                    output_format
                ))
        
        # Combine results if requested
        if combine_results and all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            total_amount = combined_df['amount_ht'].sum()
            total_quantity = combined_df['quantity'].sum()
            
            # Save combined results
            output_dir_expanded = os.path.expanduser(output_dir)
            
            if output_format.lower() == "csv":
                output_path = os.path.join(output_dir_expanded, "combined_results.csv")
                combined_df.to_csv(output_path, index=False)
                
                # Append total row
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(f"TOTAL,,{total_quantity},{total_amount},,\n")
            
            elif output_format.lower() == "json":
                output_path = os.path.join(output_dir_expanded, "combined_results.json")
                
                # Convert to JSON structure with entries and total
                json_data = {
                    "entries": combined_df.to_dict(orient="records"),
                    "total": {
                        "quantity": float(total_quantity),
                        "amount_ht": float(total_amount)
                    }
                }
                
                # Save as JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            console.print(f"\n[bold green]✓ Combined results saved to:[/] {output_path}")
    
    except Exception as e:
        logger.error(f"Failed to run batch workflow: {e}")
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)

def create_contexts_file(contexts: List[Dict], output_file: str = "contexts.json"):
    """Helper function to create a contexts file from a list of context dictionaries."""
    import json
    
    output_file_expanded = os.path.expanduser(output_file)
    with open(output_file_expanded, 'w', encoding='utf-8') as f:
        json.dump(contexts, f, indent=2)
    
    console.print(f"[green]✓ Contexts file created at:[/] {output_file_expanded}")
    return output_file_expanded

@app.command()
def create_context_file(
    output_file: Annotated[str, typer.Argument(help="Path to save the contexts file")] = "contexts.json"
):
    """Create a sample contexts file that can be edited and used with batch_analyze_contexts."""
    sample_contexts = [
        {
            "context_id": "context1",
            "content": "Sample invoice content 1...",
            "metadata": {"source": "manual_entry"}
        },
        {
            "context_id": "context2",
            "content": "Sample invoice content 2...",
            "metadata": {"source": "manual_entry"}
        }
    ]
    
    file_path = create_contexts_file(sample_contexts, output_file)
    console.print(f"[bold green]Sample contexts file created. Edit it with your actual contexts and then run:[/]")
    console.print(f"python document_analysis_flow_with_context.py batch-analyze-contexts {file_path}")

if __name__ == "__main__":
    # Test with sample context
    import sys
    
    if len(sys.argv) > 1:
        # If arguments are provided, use the CLI interface
        app()
    else:
        # No arguments provided, run with default test context
        print("Running with sample context...")
        
        # Sample context from the example provided
        sample_context = """### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Contrat n° 003503 du 25/10/2017
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
34 RUE DES CAYENNES
78700 CONFLANS STE HONORINE
RICOH MP301SP
Matricule : W907P601853
Compteur NB
Estimation au 31/03/2025 : 49428
Dernier facturé le 31/12/2024 : 48362
.

000005 FA4N Format A4 noir 1066 0.00731 7.79 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
64 RUE DES HAUTES RAYES
78700 CONFLANS STE HONORINE
RICOH MP301SP
Matricule : W907P601861
Compteur NB
Estimation au 31/03/2025 : 61792
Dernier facturé le 31/12/2024 : 59664
.

000005 FA4N Format A4 noir 2128 0.00731 15.56 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 1 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS"""
        
        # Create a document context
        document_context = DocumentContext(
            context_id="sample_invoice",
            content=sample_context
        )
        
        # Run the analysis with the sample context
        try:
            print(f"Processing context {document_context.context_id}...")
            result = asyncio.run(analyze_document_context(
                document_context,
                DEFAULT_ANALYSIS_MODEL,
                "~/output",
                output_format="csv"
            ))
            
            # Display the results
            dataframe = result["dataframe"]
            invoice_data = result["invoice_data"]
            output_file_path = result["output_file_path"]
            
            # Print summary before detailed display
            print(f"\nExtracted {len(dataframe)} invoice entries")
            print(f"Total amount: {invoice_data.total_amount_ht}")
            print(f"Results saved to: {output_file_path}\n")
            
            # Display detailed results
            asyncio.run(display_results(dataframe, invoice_data, output_file_path, document_context.context_id, "csv"))
            
            print("\nTest completed successfully!")
            
            # Show how to create a contexts file for batch processing
            print("\nTo process multiple contexts, you can create a contexts file:")
            print("python document_analysis_flow_with_context.py create-context-file")
            
        except Exception as e:
            print(f"\n[ERROR] Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
