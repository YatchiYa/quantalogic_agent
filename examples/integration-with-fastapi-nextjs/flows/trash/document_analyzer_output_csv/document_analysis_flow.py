#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru>=0.7.2",
#     "litellm>=1.0.0",
#     "pydantic>=2.0.0",
#     "asyncio",
#     "jinja2>=3.1.0",
#     "quantalogic",
#     "instructor>=0.5.2",
#     "rich>=13.0.0",
#     "pandas>=2.0.0",
#     "pypdf>=3.15.1",
#     "openpyxl>=3.1.2",
#     "pyzerox>=0.4.0",
# ]
# ///

import asyncio
import os
import json
import csv
from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
from datetime import datetime

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, validator
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pyzerox import zerox

from quantalogic.flow.flow import Nodes, Workflow

console = Console()

# Constants
MAX_TOKENS = {
    "document_analysis": 4000,  # Higher token limit for document analysis
    "data_extraction": 4000,    # Higher token limit for data extraction
    "data_validation": 2000     # Token limit for validation
}

OUTPUT_FORMATS = ["csv", "json", "pandas", "excel", "markdown"]

# Define Pydantic models for structured output
class DocumentMetadata(BaseModel):
    """Metadata about the document being analyzed."""
    filename: str = Field(description="Name of the document file")
    file_path: str = Field(description="Full path to the document file")
    page_count: int = Field(description="Total number of pages in the document")
    file_type: str = Field(description="Type of document (PDF, DOCX, etc.)")
    file_size_bytes: int = Field(description="Size of the file in bytes")
    analysis_timestamp: str = Field(description="Timestamp when analysis was performed")

class DocumentAnalysisResult(BaseModel):
    """Structured result of document analysis."""
    # This is a flexible model that can hold any extracted data
    # The actual structure will depend on the extraction instructions
    rows: List[Dict[str, Any]] = Field(default_factory=list, description="Rows of extracted data")
    summary: Optional[Dict[str, Any]] = Field(default=None, description="Summary information")
    
    # Simple model config without problematic fields for Gemini API
    model_config = {
        "extra": "allow"  # Allow extra fields to be included
    }
    
    def model_dump(self, exclude_none: bool = True, **kwargs):
        """Convert to dictionary for serialization."""
        return super().model_dump(exclude_none=exclude_none, **kwargs)

class ExtractedData(BaseModel):
    """Base model for extracted data from documents."""
    raw_data: Dict[str, Any] = Field(description="Raw extracted data in dictionary format")
    
    @validator('raw_data')
    def validate_raw_data(cls, v):
        """Ensure raw_data is not empty."""
        if not v:
            raise ValueError("Extracted data cannot be empty")
        return v
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the extracted data to a pandas DataFrame."""
        try:
            # Handle different data structures
            if isinstance(self.raw_data, dict):
                if all(isinstance(v, list) for v in self.raw_data.values()):
                    # Dictionary of lists with equal lengths
                    return pd.DataFrame(self.raw_data)
                elif "rows" in self.raw_data and isinstance(self.raw_data["rows"], list):
                    # Data has a "rows" key with list of records
                    return pd.DataFrame(self.raw_data["rows"])
                else:
                    # Single record as dictionary
                    return pd.DataFrame([self.raw_data])
            elif isinstance(self.raw_data, list):
                # List of dictionaries (records)
                return pd.DataFrame(self.raw_data)
            else:
                raise ValueError(f"Unsupported data structure: {type(self.raw_data)}")
        except Exception as e:
            logger.error(f"Error converting to DataFrame: {e}")
            # Fallback: try to normalize the data structure
            return pd.json_normalize(self.raw_data)
    
    def to_csv(self, output_path: str) -> str:
        """Save the extracted data as CSV and return the file path."""
        df = self.to_dataframe()
        output_file = output_path
        df.to_csv(output_file, index=False)
        return output_file
    
    def to_excel(self, output_path: str) -> str:
        """Save the extracted data as Excel and return the file path."""
        df = self.to_dataframe()
        output_file = output_path
        df.to_excel(output_file, index=False)
        return output_file
    
    def to_json(self, output_path: str) -> str:
        """Save the extracted data as JSON and return the file path."""
        output_file = output_path
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.raw_data, f, ensure_ascii=False, indent=2)
        return output_file
    
    def to_markdown(self) -> str:
        """Convert the extracted data to a markdown table."""
        df = self.to_dataframe()
        return df.to_markdown(index=False)

class AnalysisResult(BaseModel):
    """Result of document analysis with extracted data and metadata."""
    metadata: DocumentMetadata = Field(description="Metadata about the analyzed document")
    extracted_data: ExtractedData = Field(description="Structured data extracted from the document")
    output_files: Dict[str, str] = Field(default_factory=dict, description="Paths to output files in different formats")

# Node: Check File Type
@Nodes.define(output="file_type")
async def check_file_type(document_path: str) -> str:
    """Determine the file type based on its extension."""
    document_path = os.path.expanduser(document_path)
    if not os.path.exists(document_path):
        logger.error(f"Document not found: {document_path}")
        raise FileNotFoundError(f"Document not found: {document_path}")
    
    ext = Path(document_path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    elif ext == ".txt":
        return "text"
    elif ext == ".csv":
        return "csv"
    elif ext == ".xlsx" or ext == ".xls":
        return "excel"
    elif ext == ".md":
        return "markdown"
    else:
        logger.error(f"Unsupported file type: {ext}")
        raise ValueError(f"Unsupported file type: {ext}")

# Node: Read Text, CSV, or Excel File
@Nodes.define(output="document_content")
async def read_text_or_tabular(document_path: str, file_type: str) -> Dict[str, Any]:
    """Read content from a text, CSV, or Excel file."""
    if file_type not in ["text", "csv", "excel", "markdown"]:
        logger.error(f"Node 'read_text_or_tabular' called with invalid file_type: {file_type}")
        raise ValueError(f"Expected 'text', 'csv', 'excel', or 'markdown', got {file_type}")
    
    try:
        document_path = os.path.expanduser(document_path)
        file_path = Path(document_path)
        file_stat = file_path.stat()
        
        content = ""
        pages = []
        
        if file_type in ["text", "markdown"]:
            with open(document_path, encoding="utf-8") as f:
                content = f.read()
            pages = [content]  # Single page for text files
        elif file_type == "csv":
            df = pd.read_csv(document_path)
            content = df.to_string()
            pages = [content]  # Single page for CSV files
        elif file_type == "excel":
            df = pd.read_excel(document_path)
            content = df.to_string()
            pages = [content]  # Single page for Excel files
        
        # Create metadata
        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path.absolute()),
            "page_count": len(pages),
            "file_type": file_type,
            "file_size_bytes": file_stat.st_size,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Read {file_type} content from {document_path}, length: {len(content)} characters")
        
        return {
            "content": content,
            "metadata": metadata,
            "pages": pages
        }
    
    except Exception as e:
        logger.error(f"Error reading {file_type} file {document_path}: {e}")
        raise

# Node: Convert PDF to Text using Zerox
@Nodes.define(output="document_content")
async def convert_pdf_to_text(document_path: str, model: str) -> Dict[str, Any]:
    """Convert a PDF to text using Zerox and a vision model."""
    logger.info(f"Converting PDF to text: {document_path} using model: {model}")
    
    try:
        document_path = os.path.expanduser(document_path)
        file_path = Path(document_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        # Get file metadata
        file_stat = file_path.stat()
        file_type = "pdf"
        
        # Define system prompt for PDF extraction
        custom_system_prompt = (
            "Extract all text content from the PDF page, preserving structure and formatting. "
            "Pay special attention to tables, lists, and structured data. "
            "For tables, maintain column alignment and row structure. "
            "For forms and invoices, capture all field names and their values. "
            "Ensure all numerical data is extracted precisely, especially amounts, dates, and quantities."
        )
        
        # Use Zerox to extract text from PDF
        logger.info(f"Calling zerox with model: {model}, file: {document_path}")
        zerox_result = await zerox(
            file_path=document_path,
            model=model,
            system_prompt=custom_system_prompt
        )
        
        # Process the result
        content = ""
        pages = []
        
        if hasattr(zerox_result, 'pages') and zerox_result.pages:
            # Extract content from each page
            for page in zerox_result.pages:
                if hasattr(page, 'content') and page.content:
                    pages.append(page.content)
            content = "\n\n".join(pages)
        elif isinstance(zerox_result, str):
            content = zerox_result
            pages = [content]
        elif hasattr(zerox_result, 'text'):
            content = zerox_result.text
            pages = [content]
        else:
            content = str(zerox_result)
            pages = [content]
            logger.warning("Unexpected zerox_result type; converted to string.")
        
        # Get page count
        page_count = len(pages)
        
        # Create metadata
        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path.absolute()),
            "page_count": page_count,
            "file_type": file_type,
            "file_size_bytes": file_stat.st_size,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Extracted text content length: {len(content)} characters from {page_count} pages")
        
        # Return document content and metadata
        return {
            "content": content,
            "metadata": metadata,
            "pages": pages
        }
    
    except Exception as e:
        logger.error(f"Error converting PDF to text: {e}")
        raise

# Node: Analyze Document
@Nodes.structured_llm_node(
    system_prompt="""You are an expert document analyst specializing in data extraction.
    Your task is to analyze documents and extract structured data according to specific instructions.
    Focus on accuracy, completeness, and proper formatting of the extracted data.
    You must follow the extraction instructions precisely and maintain the exact structure requested.
    When data spans multiple pages, ensure you track and merge it correctly.
    Your output must be structured in a way that can be easily converted to tabular formats like CSV or DataFrame.
    Always return your response as a valid JSON object with the structure matching the DocumentAnalysisResult model.
    The DocumentAnalysisResult model has a 'rows' field which should contain an array of objects, where each object represents a row with named properties for each column.
    It also has an optional 'summary' field for any summary information.""",
    output="analysis_result",
    response_model=DocumentAnalysisResult,
    max_tokens=MAX_TOKENS["document_analysis"],
    prompt_template="""
Analyze the following document and extract structured data according to these instructions:

DOCUMENT CONTENT:
{{document_content.content}}

EXTRACTION INSTRUCTIONS:
{{instruction}}

DOCUMENT METADATA:
- Filename: {{document_content.metadata.filename}}
- Page Count: {{document_content.metadata.page_count}}
- File Type: {{document_content.metadata.file_type}}

REQUIREMENTS:
1. Extract ALL data matching the instructions
2. Maintain the exact structure specified in the instructions
3. Ensure data is properly formatted for conversion to {{output_format}}
4. If data spans multiple pages, track and merge it correctly
5. For tabular data, preserve column names exactly as specified

Your output MUST be valid JSON that can be parsed into the requested format.
If the instruction requests a specific table structure, ensure your output can be converted to that structure.

For tables, structure your response as an array of objects in the 'rows' field, where each object represents a row with named properties for each column.
"""
)
async def analyze_document(
    document_content: Dict[str, Any],
    instruction: str,
    output_format: str,
    model: str
) -> DocumentAnalysisResult:
    """Analyze the document and extract structured data according to instructions."""
    logger.debug(f"analyze_document called with model: {model}")
    pass  # Implementation provided by the structured_llm_node decorator

class ValidationResult(BaseModel):
    """Result of data validation."""
    status: str = Field(description="Overall validation status (PASS/FAIL)")
    issues: List[str] = Field(default_factory=list, description="List of issues found during validation")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for fixing issues")
    confidence_score: int = Field(description="Confidence score (0-100%) in the extraction quality")
    additional_notes: Optional[str] = Field(default=None, description="Any additional notes from the validator")

# Node: Validate Extracted Data
@Nodes.structured_llm_node(
    system_prompt="""You are an expert data validator specializing in document extraction.
    Your task is to validate extracted data against the original extraction instructions.
    Focus on identifying missing data, format issues, or inconsistencies.
    You must be thorough and precise in your validation.
    Always return your response as a valid JSON object with the structure matching the ValidationResult model.
    The ValidationResult model has fields for status, issues, suggestions, confidence_score, and additional_notes.""",
    output="validation_result",
    response_model=ValidationResult,
    max_tokens=MAX_TOKENS["data_validation"],
    prompt_template="""
Validate the following extracted data against the original extraction instructions:

EXTRACTION INSTRUCTIONS:
{{instruction}}

EXTRACTED DATA:
```json
{{extracted_data}}
```

VALIDATION REQUIREMENTS:
1. Check if ALL required data points were extracted
2. Verify the structure matches what was requested
3. Identify any missing or potentially incorrect data
4. Validate data types and formats
5. Check for consistency across all extracted items

Provide a validation report with:
1. Overall validation status (PASS/FAIL)
2. List of any issues found
3. Suggestions for fixing issues (if any)
4. Confidence score (0-100%) in the extraction quality
"""
)
async def validate_extracted_data(
    instruction: str,
    extracted_data: Dict[str, Any],
    model: str
) -> ValidationResult:
    """Validate the extracted data against the original instructions."""
    logger.debug(f"validate_extracted_data called with model: {model}")
    pass  # Implementation provided by the structured_llm_node decorator

# Node: Format Output
@Nodes.define(output="formatted_result")
async def format_output(
    document_content: Dict[str, Any],
    analysis_result: Dict[str, Any],
    validation_result: Dict[str, Any],
    output_format: str,
    output_dir: str
) -> AnalysisResult:
    """Format the extracted data into the requested output format."""
    logger.info(f"Formatting output as {output_format}")
    
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata object
        metadata = DocumentMetadata(**document_content["metadata"])
        
        # Create extracted data object - convert from DocumentAnalysisResult to dict
        if isinstance(analysis_result, DocumentAnalysisResult):
            extracted_data = ExtractedData(raw_data=analysis_result.model_dump())
        else:
            extracted_data = ExtractedData(raw_data=analysis_result)
        
        # Generate base filename from original document
        base_filename = Path(metadata.filename).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = f"{base_filename}_analysis_{timestamp}"
        
        # Create result object
        result = AnalysisResult(
            metadata=metadata,
            extracted_data=extracted_data,
            output_files={}
        )
        
        # Generate requested output format
        if output_format == "csv" or output_format == "all":
            csv_path = str(output_path / f"{output_base}.csv")
            result.output_files["csv"] = extracted_data.to_csv(csv_path)
            logger.info(f"CSV output saved to: {csv_path}")
        
        if output_format == "excel" or output_format == "all":
            excel_path = str(output_path / f"{output_base}.xlsx")
            result.output_files["excel"] = extracted_data.to_excel(excel_path)
            logger.info(f"Excel output saved to: {excel_path}")
        
        if output_format == "json" or output_format == "all":
            json_path = str(output_path / f"{output_base}.json")
            result.output_files["json"] = extracted_data.to_json(json_path)
            logger.info(f"JSON output saved to: {json_path}")
        
        if output_format == "markdown" or output_format == "all":
            # For markdown, we just store the content in memory
            result.output_files["markdown"] = "in_memory"
            logger.info("Markdown table generated (in memory)")
        
        # For pandas, we don't save a file but note that it's available
        if output_format == "pandas" or output_format == "all":
            result.output_files["pandas"] = "in_memory"
            logger.info("Pandas DataFrame generated (in memory)")
        
        # Add validation information
        if validation_result:
            validation_path = str(output_path / f"{output_base}_validation.json")
            with open(validation_path, 'w', encoding='utf-8') as f:
                if isinstance(validation_result, ValidationResult):
                    json.dump(validation_result.model_dump(), f, ensure_ascii=False, indent=2)
                else:
                    json.dump(validation_result, f, ensure_ascii=False, indent=2)
            result.output_files["validation"] = validation_path
            logger.info(f"Validation report saved to: {validation_path}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error formatting output: {e}")
        raise

# Create the workflow
def create_document_analysis_workflow(include_validation: bool = True) -> Workflow:
    """Create a workflow for document analysis and data extraction."""
    # Create a workflow with branching based on file type
    wf = Workflow("check_file_type")
    
    # Add all nodes
    wf.node("check_file_type")
    wf.node("convert_pdf_to_text", inputs_mapping={"model": "extraction_model"})
    wf.node("read_text_or_tabular")
    wf.node("analyze_document", inputs_mapping={"model": "analysis_model"})
    
    if include_validation:
        wf.node("validate_extracted_data", inputs_mapping={
            "model": "validation_model",
            "extracted_data": lambda ctx: ctx["analysis_result"]
        })
    
    wf.node("format_output")
    
    # Define the workflow structure with branching based on file type
    wf.current_node = "check_file_type"
    wf.branch([
        ("convert_pdf_to_text", lambda ctx: ctx["file_type"] == "pdf"),
        ("read_text_or_tabular", lambda ctx: ctx["file_type"] in ["text", "csv", "excel", "markdown"])
    ])
    
    # Define transitions from branches to analyze_document
    wf.transitions["convert_pdf_to_text"] = [("analyze_document", None)]
    wf.transitions["read_text_or_tabular"] = [("analyze_document", None)]
    
    # Define remaining transitions
    if include_validation:
        wf.transitions["analyze_document"] = [("validate_extracted_data", None)]
        wf.transitions["validate_extracted_data"] = [("format_output", None)]
    else:
        wf.transitions["analyze_document"] = [("format_output", None)]
    
    return wf

async def analyze_document_flow(
    document_path: str,
    instruction: str,
    output_format: str = "csv",
    output_dir: str = "./output",
    include_validation: bool = True,
    extraction_model: str = "gemini/gemini-2.0-flash",
    analysis_model: str = "gemini/gemini-2.0-flash",
    validation_model: str = "gemini/gemini-2.0-flash",
    task_id: str = "default",
    _handle_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
) -> AnalysisResult:
    """Analyze a document and extract structured data according to instructions."""
    
    # Validate inputs
    if not document_path or not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found: {document_path}")
    
    if not instruction or instruction.strip() == "":
        raise ValueError("Instruction cannot be empty")
    
    if output_format not in OUTPUT_FORMATS and output_format != "all":
        raise ValueError(f"Output format must be one of {OUTPUT_FORMATS} or 'all'")
    
    logger.info(f"Starting document analysis for: {document_path}")
    
    initial_context = {
        "document_path": document_path,
        "instruction": instruction,
        "output_format": output_format,
        "output_dir": output_dir,
        "extraction_model": extraction_model,
        "analysis_model": analysis_model,
        "validation_model": validation_model
    }
    
    try:
        workflow = create_document_analysis_workflow(include_validation)
        engine = workflow.build()
        
        # Add the event observer if _handle_event is provided
        if _handle_event:
            # Create a lambda to bind task_id to the observer
            from quantalogic.service import event_observer
            bound_observer = lambda event: asyncio.create_task(
                event_observer(event, task_id=task_id, _handle_event=_handle_event)
            )
            engine.add_observer(bound_observer)
        
        result = await engine.run(initial_context)
        
        if not isinstance(result.get("formatted_result"), AnalysisResult):
            raise ValueError("Workflow did not produce a valid analysis result")
        
        logger.info("Document analysis completed successfully")
        return result["formatted_result"]
    
    except Exception as e:
        logger.error(f"Error analyzing document: {e}")
        raise

def display_analysis_result(result: AnalysisResult) -> None:
    """Display the analysis result in a user-friendly format."""
    # Display metadata
    console.print("\n[bold blue]Document Analysis Results[/]")
    
    metadata_table = Table(title="Document Metadata")
    metadata_table.add_column("Property", style="cyan")
    metadata_table.add_column("Value", style="green")
    
    for field, value in result.metadata.model_dump().items():
        metadata_table.add_row(field.replace('_', ' ').title(), str(value))
    
    console.print(metadata_table)
    
    # Display output files
    if result.output_files:
        files_table = Table(title="Output Files")
        files_table.add_column("Format", style="cyan")
        files_table.add_column("Path", style="green")
        
        for format_type, path in result.output_files.items():
            if path != "in_memory":
                files_table.add_row(format_type.upper(), path)
            else:
                files_table.add_row(format_type.upper(), "(Available in memory)")
        
        console.print(files_table)
    
    # Display data preview as markdown
    console.print("\n[bold blue]Data Preview:[/]")
    markdown_preview = result.extracted_data.to_markdown()
    console.print(Panel(Markdown(markdown_preview), title="Extracted Data"))


# Main function for direct testing
async def main():
    """Test function for the document analysis workflow with direct parameters."""
    # Example parameters
    document_path = "/home/yarab/Téléchargements/Facture_infos a extraire.pdf"  # Use the specified PDF file
    instruction = """
    Extract the following information from the document:
    1. Invoice Number (Numéro de facture)
    2. Invoice Date (Date de facture)
    3. Vendor Name (Nom du vendeur)
    4. Vendor Address (Adresse du vendeur)
    5. Customer Name (Nom du client)
    6. Customer Address (Adresse du client)
    7. Line Items (with Item Description, Quantity, Unit Price, and Total)
    8. Subtotal (Sous-total)
    9. Tax Amount (Montant de la taxe)
    10. Total Amount Due (Montant total dû)
    
    Structure the data as a table with the following columns:
    - invoice_number
    - invoice_date
    - vendor_name
    - vendor_address
    - customer_name
    - customer_address
    - items (as a JSON array of objects with description, quantity, unit_price, total)
    - subtotal
    - tax
    - total_amount
    """
    output_format = "all"
    output_dir = "./output"
    
    # Display parameters
    console.print(Panel.fit(
        "[bold green]Document Analysis Flow[/]\n\n"
        f"Document: [cyan]{document_path}[/]\n"
        f"Output Format: [cyan]{output_format}[/]\n"
        f"Output Directory: [cyan]{output_dir}[/]",
        title="Parameters",
        border_style="blue"
    ))
    
    # Display instruction
    console.print(Panel(
        Markdown(instruction),
        title="Extraction Instructions",
        border_style="green"
    ))
    
    # Run the workflow with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Analyzing document...", total=None)
        try:
            result = await analyze_document_flow(
                document_path=document_path,
                instruction=instruction,
                output_format=output_format,
                output_dir=output_dir,
                include_validation=True,
                extraction_model="gemini/gemini-2.0-flash",
                analysis_model="gemini/gemini-2.0-flash",
                validation_model="gemini/gemini-2.0-flash"
            )
            
            # Display the result
            display_analysis_result(result)
            
            # If we have a markdown representation, display it
            if "markdown" in result.output_files:
                markdown_content = result.extracted_data.to_markdown()
                console.print("\n[bold blue]Extracted Data (Markdown Table)[/]")
                console.print(Panel(Markdown(markdown_content), border_style="green"))
            
            console.print("\n[bold green]✓ Document analysis completed successfully![/]")
            
        except Exception as e:
            console.print(f"\n[bold red]✗ Error: {e}")
            logger.error(f"Error in main: {e}")
            raise

# Alternative main function with different example
async def main_invoice_example():
    """Alternative test function with invoice example."""
    # Example document path - replace with your actual document path
    document_path = "/home/yarab/Téléchargements/Facture_infos a extraire.pdf"
    
    # Example extraction instruction for invoice data
    instruction = """
    Extract all invoice information including:
    - Invoice number
    - Date
    - Customer information
    - Line items (product, quantity, price)
    - Total amount
    
    Format the output as a table with columns:
    | Invoice_Number | Date | Customer | Product | Quantity | Price | Total |
    """
    
    # Configuration parameters
    output_format = "csv"  # Generate CSV output
    output_dir = "./output"  # Output directory
    
    try:
        # Check if document exists
        if not os.path.exists(document_path):
            console.print(f"[bold yellow]Warning:[/] Document not found: {document_path}")
            console.print("[bold yellow]Please update the document_path variable in the main function.[/]")
            return
        
        # Run document analysis
        console.print("\n[bold blue]Running invoice analysis...[/]")
        
        result = await analyze_document_flow(
            document_path=document_path,
            instruction=instruction,
            output_format=output_format,
            output_dir=output_dir,
            include_validation=True
        )
        
        # Display results
        display_analysis_result(result)
        
        console.print("[bold green]✓ Invoice analysis completed successfully[/]")
        
    except Exception as e:
        console.print(f"[bold red]Error during invoice analysis:[/] {str(e)}")
        raise

if __name__ == "__main__":
    # Choose which main function to run
    asyncio.run(main())
    # Uncomment to run the invoice example instead
    # asyncio.run(main_invoice_example())
