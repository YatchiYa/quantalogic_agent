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
# from ..service import event_observer

# Initialize Typer app and rich console
app = typer.Typer(help="Extract structured data from documents and output to CSV format")
console = Console()

# Default models for different phases
DEFAULT_TEXT_EXTRACTION_MODEL = "openai/gpt-4o"
DEFAULT_ANALYSIS_MODEL = "openai/gpt-4o-mini" # "anthropic/claude-3-7-sonnet-20250219"

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
    elif ext in [".txt", ".csv", ".xlsx", ".xls"]:
        return ext[1:]  # Remove the dot
    else:
        logger.error(f"Unsupported file type: {ext}")
        raise ValueError(f"Unsupported file type: {ext}")

# Node: Convert PDF to Text
@Nodes.define(output="document_content")
async def convert_pdf_to_text(
    file_path: str,
    model: str,
    custom_system_prompt: Optional[str] = None,
    output_dir: Optional[str] = None,
    select_pages: Optional[Union[int, List[int]]] = None
) -> str:
    """Convert a PDF to text using a vision model."""
    file_path = os.path.expanduser(file_path)
    if output_dir:
        output_dir = os.path.expanduser(output_dir)
        
    if not file_path:
        logger.error("File path is required")
        raise ValueError("File path is required")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise ValueError(f"File not found: {file_path}")
    if not file_path.lower().endswith(".pdf"):
        logger.error(f"File must be a PDF: {file_path}")
        raise ValueError(f"File must be a PDF: {file_path}")

    if custom_system_prompt is None:
        custom_system_prompt = (
            "Extract the text content from this PDF document. "
            "Preserve the structure, tables, and formatting as much as possible. "
            "For tables, maintain the alignment of columns and rows. "
            "Include all numerical data, especially quantities, amounts, and identifiers. "
            "Return the content in a well-structured format that preserves the relationships between data points."
        )

    try:
        logger.info(f"Calling zerox with model: {model}, file: {file_path}")
        zerox_result = await zerox(
            file_path=file_path,
            model=model,
            system_prompt=custom_system_prompt,
            output_dir=output_dir,
            select_pages=select_pages
        )

        text_content = ""
        if hasattr(zerox_result, 'pages') and zerox_result.pages:
            text_content = "\n\n".join(
                page.content for page in zerox_result.pages
                if hasattr(page, 'content') and page.content
            )
        elif isinstance(zerox_result, str):
            text_content = zerox_result
        elif hasattr(zerox_result, 'text'):
            text_content = zerox_result.text
        else:
            text_content = str(zerox_result)
            logger.warning("Unexpected zerox_result type; converted to string.")

        if not text_content.strip():
            logger.warning("Extracted text content is empty.")
            return ""

        logger.info(f"Extracted text content length: {len(text_content)} characters")
        return text_content
    except Exception as e:
        logger.error(f"Error converting PDF to text: {e}")
        raise

# Node: Read Text or CSV File
@Nodes.define(output="document_content")
async def read_text_or_csv_file(file_path: str, file_type: str) -> str:
    """Read content from a text or CSV file."""
    if file_type not in ["txt", "csv"]:
        logger.error(f"Node 'read_text_or_csv_file' called with invalid file_type: {file_type}")
        raise ValueError(f"Expected 'txt' or 'csv', got {file_type}")
    try:
        file_path = os.path.expanduser(file_path)
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Read {file_type} content from {file_path}, length: {len(content)} characters")
        return content
    except Exception as e:
        logger.error(f"Error reading {file_type} file {file_path}: {e}")
        raise

# Node: Read Excel File
@Nodes.define(output="document_content")
async def read_excel_file(file_path: str, file_type: str) -> str:
    """Read content from an Excel file and convert to text representation."""
    if file_type not in ["xlsx", "xls"]:
        logger.error(f"Node 'read_excel_file' called with invalid file_type: {file_type}")
        raise ValueError(f"Expected 'xlsx' or 'xls', got {file_type}")
    try:
        file_path = os.path.expanduser(file_path)
        df = pd.read_excel(file_path)
        # Convert DataFrame to string representation
        content = df.to_string(index=False)
        logger.info(f"Read Excel content from {file_path}, shape: {df.shape}")
        return content
    except Exception as e:
        logger.error(f"Error reading Excel file {file_path}: {e}")
        raise

# Node: Save Extracted Content
@Nodes.define(output="content_file_path")
async def save_extracted_content(document_content: str, file_path: str) -> str:
    """Save the extracted document content to a file."""
    try:
        file_path_expanded = os.path.expanduser(file_path)
        output_path = Path(file_path_expanded).with_suffix(".extracted.txt")
        with output_path.open("w", encoding="utf-8") as f:
            f.write(document_content)
        logger.info(f"Saved extracted content to: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving extracted content: {e}")
        raise

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

# Node: Save DataFrame to CSV
@Nodes.define(output="csv_file_path")
async def save_to_csv(dataframe: pd.DataFrame, invoice_data: InvoiceData, file_path: str) -> str:
    """Save the DataFrame to a CSV file."""
    try:
        file_path_expanded = os.path.expanduser(file_path)
        output_path = Path(file_path_expanded).with_suffix(".results.csv")
        
        # Save the main data
        dataframe.to_csv(output_path, index=False)
        
        # Append the total row to the CSV file
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"TOTAL,,{dataframe['quantity'].sum()},{invoice_data.total_amount_ht},,\n")
        
        logger.info(f"Saved invoice data to CSV: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving DataFrame to CSV: {e}")
        raise

# Define the Workflow
def create_document_analysis_workflow() -> Workflow:
    """Create a workflow to extract structured data from documents and output to CSV."""
    wf = Workflow("check_file_type")
    
    # Add all nodes with input mappings for dynamic model passing
    wf.node("check_file_type")
    wf.node("convert_pdf_to_text", inputs_mapping={"model": "text_extraction_model"})
    wf.node("read_text_or_csv_file")
    wf.node("read_excel_file")
    wf.node("save_extracted_content")
    wf.node("extract_invoice_data", inputs_mapping={"model": "analysis_model"})
    wf.node("convert_to_dataframe")
    wf.node("save_to_csv")
    
    # Define the workflow structure with explicit transitions to prevent loops
    wf.current_node = "check_file_type"
    wf.branch([
        ("convert_pdf_to_text", lambda ctx: ctx["file_type"] == "pdf"),
        ("read_text_or_csv_file", lambda ctx: ctx["file_type"] in ["txt", "csv"]),
        ("read_excel_file", lambda ctx: ctx["file_type"] in ["xlsx", "xls"])
    ])
    
    # Explicitly set transitions from branches to convergence point
    wf.transitions["convert_pdf_to_text"] = [("save_extracted_content", None)]
    wf.transitions["read_text_or_csv_file"] = [("save_extracted_content", None)]
    wf.transitions["read_excel_file"] = [("save_extracted_content", None)]
    
    # Define linear sequence after convergence
    wf.transitions["save_extracted_content"] = [("extract_invoice_data", None)]
    wf.transitions["extract_invoice_data"] = [("convert_to_dataframe", None)]
    wf.transitions["convert_to_dataframe"] = [("save_to_csv", None)]
    
    return wf

# Function to Run the Workflow
async def analyze_document(
    file_path: str,
    text_extraction_model: str,
    analysis_model: str,
    output_dir: Optional[str] = None,
    custom_instructions: Optional[str] = None,
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
        "output_dir": output_dir if output_dir else str(Path(file_path).parent),
        "custom_instructions": custom_instructions
    }

    try:
        workflow = create_document_analysis_workflow()
        engine = workflow.build()

        # Add the event observer if _handle_event is provided
        # if _handle_event:
        #     # Create a lambda to bind task_id to the observer
        #     bound_observer = lambda event: asyncio.create_task(
        #         event_observer(event, task_id=task_id, _handle_event=_handle_event)
        #     )
        #     engine.add_observer(bound_observer)

        result = await engine.run(initial_context)
        
        if "dataframe" not in result or result["dataframe"].empty:
            logger.warning("No invoice data extracted.")
            raise ValueError("Workflow completed but no invoice data was extracted.")
        
        logger.info("Workflow completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise

async def display_results(dataframe: pd.DataFrame, invoice_data: InvoiceData, csv_file_path: str):
    """Async helper function to display results with animation."""
    # Create a rich table from the DataFrame
    table = Table(title="Extracted Invoice Data")
    
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
    
    console.print("\n[bold green]Extracted Invoice Data:[/]")
    console.print(table)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        progress.add_task("[cyan]Processing results...", total=None)
        await asyncio.sleep(1)  # Simulate some processing time for effect
    
    console.print(f"[green]✓ CSV file saved to:[/] {csv_file_path}")

@app.command()
def analyze(
    file_path: Annotated[str, typer.Argument(help="Path to the document file (PDF, .txt, .csv, .xlsx, .xls; supports ~ expansion)")],
    text_extraction_model: Annotated[str, typer.Option(help="LLM model for document text extraction")] = DEFAULT_TEXT_EXTRACTION_MODEL,
    analysis_model: Annotated[str, typer.Option(help="LLM model for data extraction and analysis")] = DEFAULT_ANALYSIS_MODEL,
    output_dir: Annotated[Optional[str], typer.Option(help="Directory to save output files (supports ~ expansion)")] = None,
    custom_instructions: Annotated[Optional[str], typer.Option(help="Custom instructions for data extraction")] = None
):
    """Extract structured data from a document and output to CSV format."""
    try:
        with console.status(f"Processing [bold blue]{file_path}[/]..."):
            result = asyncio.run(analyze_document(
                file_path,
                text_extraction_model,
                analysis_model,
                output_dir,
                custom_instructions
            ))
        
        dataframe = result["dataframe"]
        invoice_data = result["invoice_data"]
        csv_file_path = result["csv_file_path"]
        
        # Run the async display function
        asyncio.run(display_results(dataframe, invoice_data, csv_file_path))
    
    except Exception as e:
        logger.error(f"Failed to run workflow: {e}")
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)

@app.command()
def batch_analyze(
    file_paths: Annotated[List[str], typer.Argument(help="Paths to document files (PDF, .txt, .csv, .xlsx, .xls; supports ~ expansion)")],
    text_extraction_model: Annotated[str, typer.Option(help="LLM model for document text extraction")] = DEFAULT_TEXT_EXTRACTION_MODEL,
    analysis_model: Annotated[str, typer.Option(help="LLM model for data extraction and analysis")] = DEFAULT_ANALYSIS_MODEL,
    output_dir: Annotated[Optional[str], typer.Option(help="Directory to save output files (supports ~ expansion)")] = None,
    custom_instructions: Annotated[Optional[str], typer.Option(help="Custom instructions for data extraction")] = None,
    combine_results: Annotated[bool, typer.Option(help="Combine all results into a single CSV file")] = False
):
    """Batch process multiple documents and extract structured data to CSV format."""
    try:
        all_dataframes = []
        all_results = []
        
        for file_path in file_paths:
            with console.status(f"Processing [bold blue]{file_path}[/]..."):
                result = asyncio.run(analyze_document(
                    file_path,
                    text_extraction_model,
                    analysis_model,
                    output_dir,
                    custom_instructions
                ))
                
                all_results.append(result)
                all_dataframes.append(result["dataframe"])
                
                # Display individual results
                asyncio.run(display_results(
                    result["dataframe"], 
                    result["invoice_data"], 
                    result["csv_file_path"]
                ))
        
        # Combine results if requested
        if combine_results and all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            total_amount = combined_df['amount_ht'].sum()
            
            # Save combined results
            if output_dir:
                output_path = os.path.join(os.path.expanduser(output_dir), "combined_results.csv")
            else:
                output_path = "combined_results.csv"
                
            combined_df.to_csv(output_path, index=False)
            
            # Append total row
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(f"TOTAL,,{combined_df['quantity'].sum()},{total_amount},,\n")
            
            console.print(f"\n[bold green]✓ Combined results saved to:[/] {output_path}")
    
    except Exception as e:
        logger.error(f"Failed to run batch workflow: {e}")
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    # Test with specific file for quick testing
    import sys
    
    if len(sys.argv) > 1:
        # If arguments are provided, use the CLI interface
        app()
    else:
        # No arguments provided, run with default test file
        print("Running with default test file...")
        test_file = "/home/yarab/Téléchargements/Facture_infos a extraire.pdf"
        
        # Check if the test file exists
        if not os.path.exists(test_file):
            print(f"[ERROR] Test file not found: {test_file}")
            print("Please run with a specific file path:")
            print("python document_analysis_flow.py analyze <file_path>")
            sys.exit(1)
            
        # Run the analysis with the test file
        try:
            print(f"Processing {test_file}...")
            result = asyncio.run(analyze_document(
                test_file,
                DEFAULT_TEXT_EXTRACTION_MODEL,
                DEFAULT_ANALYSIS_MODEL
            ))
            
            # Display the results
            dataframe = result["dataframe"]
            invoice_data = result["invoice_data"]
            csv_file_path = result["csv_file_path"]
            
            # Print summary before detailed display
            print(f"\nExtracted {len(dataframe)} invoice entries")
            print(f"Total amount: {invoice_data.total_amount_ht}")
            print(f"Results saved to: {csv_file_path}\n")
            
            # Display detailed results
            asyncio.run(display_results(dataframe, invoice_data, csv_file_path))
            
            print("\nTest completed successfully!")
        except Exception as e:
            print(f"\n[ERROR] Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)