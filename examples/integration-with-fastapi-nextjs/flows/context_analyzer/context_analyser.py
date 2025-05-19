## context analyzer

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
#     "typer>=0.9.0",
#     "rich>=13.0.0",
#     "pyperclip>=1.8.2",
#     "pandas>=2.0.0",
# ]
# ///

import asyncio
from collections.abc import Callable
import datetime
import os
import json
import csv
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
# from ..service import event_observer

console = Console()

# Global variable to store the last generated formats
_last_generated_formats = {
    'csv': '',
    'json': '',
    'markdown': '',
    'text': ''
}

# # Constants for token limits
# MAX_TOKENS = {
#     "analysis": 4000,  # For detailed context analysis
#     "structured_output": 8000,  # For comprehensive structured results
#     "format_conversion": 4000  # For converting to different formats
# }

# Define Pydantic models for structured output
class ContextAnalysisResult(BaseModel):
    """Analysis of the context based on instructions."""
    key_findings: List[str] = Field(description="Key findings from the analysis")
    extracted_data_json: str = Field(description="JSON string of structured data extracted from the context")
    confidence_score: float = Field(description="Confidence score for the analysis results", ge=0.0, le=1.0)
    metadata_source: str = Field(description="Source of the analyzed data")
    metadata_page_count: str = Field(description="Number of pages in the document")

class StructuredOutput(BaseModel):
    """Structured output based on the analysis and requested format."""
    content: str = Field(description="Formatted content based on requested output format")
    format_type: str = Field(description="Type of format (e.g., csv, json, table, text)")

class FileOutput(BaseModel):
    """Output file details."""
    file_path: str = Field(description="Path to the output file")
    file_type: str = Field(description="Type of the file (e.g., csv, json, txt)")
    file_size: int = Field(description="Size of the file in bytes")

class AnalysisOutput(BaseModel):
    """Final output of the context analysis."""
    content: str = Field(description="Formatted content of the analysis in the primary requested format")
    format_type: str = Field(description="Primary format type of the output")
    file_path: Optional[str] = Field(None, description="Path to the output file if saved")
    summary: str = Field(description="Brief summary of the analysis results")
    execution_time: float = Field(description="Time taken to execute the analysis in seconds")
    csv_content: Optional[str] = Field(None, description="CSV formatted content")
    json_content: Optional[str] = Field(None, description="JSON formatted content")
    markdown_content: Optional[str] = Field(None, description="Markdown formatted content")
    text_content: Optional[str] = Field(None, description="Plain text formatted content")

# Node: Analyze Context
@Nodes.structured_llm_node(
    system_prompt="""You are an expert data analyst specializing in context analysis and information extraction.
    Your task is to analyze the provided context based on specific instructions and extract relevant information.
    You must be precise, thorough, and follow the instructions exactly as given.
    Focus on extracting only the information requested in the instructions, nothing more or less.
    Maintain the exact format, structure, and organization as specified in the instructions.""",
    output="analysis_result",
    response_model=ContextAnalysisResult,
    # max_tokens=MAX_TOKENS["analysis"],
    prompt_template="""
## INSTRUCTIONS
{{instructions}}

## CONTEXT
{{context}}

## TASK
Analyze the above context according to the instructions provided.

1. Extract all required information with high precision
2. Organize the data exactly as specified in the instructions
3. Ensure all data points are correctly identified and categorized
4. Maintain the original formatting where required
5. Include confidence scores for extracted information

Provide your analysis as a structured result with the following components:
- Key findings: List the main insights or data points extracted
- Extracted data: Provide the structured data extracted from the context
- Confidence score: Rate your overall confidence in the extraction accuracy
- Metadata: Include any additional relevant information about the analysis
"""
)
async def analyze_context(context: str, instructions: str, model: str) -> ContextAnalysisResult:
    """Analyze the context based on the provided instructions."""
    logger.debug(f"analyze_context called with model: {model}")
    pass

# Node: Generate Structured Output
@Nodes.structured_llm_node(
    system_prompt="""You are an expert in data formatting and presentation.
    Your task is to take analyzed data and format it according to the specified output format.
    You must adhere strictly to the requested format and ensure all data is properly structured.
    Focus on clarity, accuracy, and proper organization of the information.
    Ensure that the output is machine-readable if a structured format is requested.""",
    output="structured_output",
    response_model=StructuredOutput,
    # max_tokens=MAX_TOKENS["structured_output"],
    prompt_template="""
## ANALYSIS RESULT
{{analysis_result}}

## OUTPUT FORMAT
{{output_format}}

## TASK
Format the analysis results according to the specified output format.

For the requested format ({{output_format}}):
- Ensure the output strictly follows the format requirements
- Include all relevant data from the analysis results
- Organize the information logically and clearly
- Maintain proper structure for machine readability if applicable
- Validate that all data is correctly formatted

Provide your output as a structured result with the following components:
- content: The formatted content based on the requested format
- format_type: The type of format used (e.g., csv, json, table, text)
"""
)
async def generate_structured_output(
    analysis_result: ContextAnalysisResult,
    output_format: str,
    model: str
) -> StructuredOutput:
    """Generate structured output in the requested format and store additional formats in memory."""
    logger.debug(f"generate_structured_output called with model: {model}")
    
    # Generate the primary output format using the LLM
    try:
        # Create a context dictionary for the LLM node
        context = {
            "analysis_result": analysis_result,
            "output_format": output_format,
            "model": model
        }
        # The LLM node will handle the generation of the primary format
        result = await Nodes.structured_llm_node.run(context)
        
        # Store the primary format result
        primary_content = result.content
        primary_format = result.format_type.lower()
        
        # Generate other formats programmatically based on the primary format
        # We'll store these in global variables or attach them to the output later
        try:
            # Extract data from the analysis result as a fallback
            extracted_data = json.loads(analysis_result.extracted_data_json)
            
            # Generate JSON format (if primary format is not JSON)
            if primary_format != 'json':
                if primary_format == 'csv':
                    json_content = convert_csv_to_json(primary_content)
                else:
                    json_content = json.dumps(extracted_data, indent=2)
            else:
                json_content = primary_content
                
            # Generate CSV format (if primary format is not CSV)
            if primary_format != 'csv':
                if primary_format == 'json':
                    csv_content = convert_json_to_csv(primary_content)
                else:
                    csv_content = convert_json_to_csv(json_content)
            else:
                csv_content = primary_content
                
            # Generate Markdown format (if primary format is not Markdown/Table)
            if primary_format not in ['markdown', 'table']:
                markdown_content = convert_to_markdown(json_content)
            else:
                markdown_content = primary_content
                
            # Generate Text format (if primary format is not Text)
            if primary_format != 'text':
                text_content = convert_to_text(json_content)
            else:
                text_content = primary_content
                
            # Store these formats in global variables for later use
            global _last_generated_formats
            _last_generated_formats = {
                'csv': csv_content,
                'json': json_content,
                'markdown': markdown_content,
                'text': text_content
            }
            
        except Exception as conversion_error:
            logger.warning(f"Error generating additional formats: {conversion_error}")
            # Continue with just the primary format
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating structured output: {e}")
        
        # Create a basic fallback output if the LLM fails
        try:
            # Try to extract data from the analysis result
            extracted_data = json.loads(analysis_result.extracted_data_json)
            
            # Create a simple text representation as fallback
            fallback_content = f"Analysis Results:\n"
            for key, value in extracted_data.items():
                if isinstance(value, list):
                    fallback_content += f"\n{key}:\n"
                    for item in value:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                fallback_content += f"  {k}: {v}\n"
                        else:
                            fallback_content += f"  {item}\n"
                else:
                    fallback_content += f"{key}: {value}\n"
            
            return StructuredOutput(
                content=fallback_content,
                format_type="text"
            )
            
        except Exception as fallback_error:
            logger.error(f"Fallback output generation failed: {fallback_error}")
            
            # Last resort fallback
            error_message = f"Error generating output: {str(e)}. Fallback also failed: {str(fallback_error)}"
            return StructuredOutput(
                content=error_message,
                format_type="text"
            )

# Helper functions for the chunking process
async def save_output_to_file_direct(structured_output: StructuredOutput) -> Optional[FileOutput]:
    """Direct version of save_output_to_file for use in chunking process."""
    try:
        output_dir = Path("analysis_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Generate a filename based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine file type and extension
        file_type = structured_output.format_type.lower()
        if file_type == "csv":
            file_ext = ".csv"
        elif file_type == "json":
            file_ext = ".json"
        elif file_type in ["table", "markdown"]:
            file_ext = ".md"
        else:
            file_ext = ".txt"
        
        file_path = output_dir / f"analysis_{timestamp}{file_ext}"
        
        # Write content to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(structured_output.content)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        logger.info(f"Output saved to file: {file_path}")
        
        return FileOutput(
            file_path=str(file_path),
            file_type=file_type,
            file_size=file_size
        )
    
    except Exception as e:
        logger.error(f"Error saving output to file: {e}")
        return None

async def compile_final_output_direct(
    analysis_result: ContextAnalysisResult,
    structured_output: StructuredOutput,
    file_output: Optional[FileOutput],
    execution_start_time: float
) -> AnalysisOutput:
    """Direct version of compile_final_output for use in chunking process."""
    try:
        # Calculate execution time
        execution_time = asyncio.get_event_loop().time() - execution_start_time
        
        # Generate a brief summary
        summary = f"Analysis completed with {analysis_result.confidence_score:.2f} confidence score. "
        file_path = None
        if file_output:
            summary += f"Output saved to {file_output.file_path}."
            file_path = file_output.file_path
        else:
            summary += "No file output was generated."
        
        # Get the additional formats from the global variable
        global _last_generated_formats
        csv_content = _last_generated_formats.get('csv', '')
        json_content = _last_generated_formats.get('json', '')
        markdown_content = _last_generated_formats.get('markdown', '')
        text_content = _last_generated_formats.get('text', '')
        
        # Create the final output with all formats
        output = AnalysisOutput(
            content=structured_output.content,
            format_type=structured_output.format_type,
            file_path=file_path,
            summary=summary,
            execution_time=execution_time,
            csv_content=csv_content,
            json_content=json_content,
            markdown_content=markdown_content,
            text_content=text_content
        )
        
        logger.info("Compiled final analysis output with all formats successfully")
        return output
    
    except Exception as e:
        logger.error(f"Error compiling final output: {e}")
        raise

# Helper functions for format conversion
def convert_csv_to_json(csv_content: str) -> str:
    """Convert CSV content to JSON format."""
    try:
        lines = csv_content.strip().split('\n')
        if not lines:
            return '[]'
            
        headers = [h.strip() for h in lines[0].split(',')]
        result = []
        
        for line in lines[1:]:
            values = line.split(',')
            row = {}
            for i, header in enumerate(headers):
                if i < len(values):
                    row[header] = values[i].strip()
            result.append(row)
            
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error converting CSV to JSON: {e}")
        return '[]'

def convert_json_to_csv(json_content: str) -> str:
    """Convert JSON content to CSV format."""
    try:
        data = json.loads(json_content)
        if not data:
            return ''
            
        if isinstance(data, dict):
            data = [data]
            
        # Get all unique keys
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
            
        headers = sorted(list(all_keys))
        csv_lines = [','.join(headers)]
        
        for item in data:
            row = [str(item.get(key, '')) for key in headers]
            csv_lines.append(','.join(row))
            
        return '\n'.join(csv_lines)
    except Exception as e:
        logger.error(f"Error converting JSON to CSV: {e}")
        return ''

def convert_to_markdown(content: str) -> str:
    """Convert content to Markdown format."""
    try:
        # Try to parse as JSON first
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                data = [data]
                
            if not data:
                return ''
                
            # Get all unique keys
            all_keys = set()
            for item in data:
                all_keys.update(item.keys())
                
            headers = sorted(list(all_keys))
            
            # Create markdown table
            md_lines = ['| ' + ' | '.join(headers) + ' |']
            md_lines.append('| ' + ' | '.join(['---' for _ in headers]) + ' |')
            
            for item in data:
                row = [str(item.get(key, '')) for key in headers]
                md_lines.append('| ' + ' | '.join(row) + ' |')
                
            return '\n'.join(md_lines)
        except json.JSONDecodeError:
            # Try to parse as CSV
            lines = content.strip().split('\n')
            if not lines:
                return ''
                
            headers = [h.strip() for h in lines[0].split(',')]
            
            # Create markdown table
            md_lines = ['| ' + ' | '.join(headers) + ' |']
            md_lines.append('| ' + ' | '.join(['---' for _ in headers]) + ' |')
            
            for line in lines[1:]:
                values = [v.strip() for v in line.split(',')]
                md_lines.append('| ' + ' | '.join(values) + ' |')
                
            return '\n'.join(md_lines)
    except Exception as e:
        logger.error(f"Error converting to Markdown: {e}")
        return ''

def convert_to_text(content: str) -> str:
    """Convert content to plain text format."""
    try:
        # Try to parse as JSON
        try:
            data = json.loads(content)
            text_lines = []
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    text_lines.append(f"Item {i+1}:")
                    for key, value in item.items():
                        text_lines.append(f"  {key}: {value}")
                    text_lines.append("")
            elif isinstance(data, dict):
                for key, value in data.items():
                    text_lines.append(f"{key}: {value}")
                    
            return '\n'.join(text_lines)
        except json.JSONDecodeError:
            # If it's not JSON, return as is (might be CSV or markdown)
            return content
    except Exception as e:
        logger.error(f"Error converting to text: {e}")
        return content

# Node: Save Output to File
@Nodes.define(output="file_output")
async def save_output_to_file(structured_output: StructuredOutput) -> Optional[FileOutput]:
    """Save the structured output to a file if requested."""
    try:
        output_dir = Path("analysis_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Generate a filename based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine file type and extension
        file_type = structured_output.format_type.lower()
        if file_type == "csv":
            file_ext = ".csv"
        elif file_type == "json":
            file_ext = ".json"
        elif file_type in ["table", "markdown"]:
            file_ext = ".md"
        else:
            file_ext = ".txt"
        
        file_path = output_dir / f"analysis_{timestamp}{file_ext}"
        
        # Write content to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(structured_output.content)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        logger.info(f"Output saved to file: {file_path}")
        
        return FileOutput(
            file_path=str(file_path),
            file_type=file_type,
            file_size=file_size
        )
    
    except Exception as e:
        logger.error(f"Error saving output to file: {e}")
        return None

# Node: Compile Final Output
@Nodes.define(output="analysis_output")
async def compile_final_output(
    analysis_result: ContextAnalysisResult,
    structured_output: StructuredOutput,
    file_output: Optional[FileOutput],
    execution_start_time: float
) -> AnalysisOutput:
    """Compile the final output of the context analysis with all output formats."""
    try:
        # Calculate execution time
        execution_time = asyncio.get_event_loop().time() - execution_start_time
        
        # Generate a brief summary
        summary = f"Analysis completed with {analysis_result.confidence_score:.2f} confidence score. "
        file_path = None
        if file_output:
            summary += f"Output saved to {file_output.file_path}."
            file_path = file_output.file_path
        else:
            summary += "No file output was generated."
        
        # Get the additional formats from the global variable
        global _last_generated_formats
        csv_content = _last_generated_formats.get('csv', '')
        json_content = _last_generated_formats.get('json', '')
        markdown_content = _last_generated_formats.get('markdown', '')
        text_content = _last_generated_formats.get('text', '')
        
        # Create the final output with all formats
        output = AnalysisOutput(
            content=structured_output.content,
            format_type=structured_output.format_type,
            file_path=file_path,
            summary=summary,
            execution_time=execution_time,
            csv_content=csv_content,
            json_content=json_content,
            markdown_content=markdown_content,
            text_content=text_content
        )
        
        logger.info("Compiled final analysis output with all formats successfully")
        return output
    
    except Exception as e:
        logger.error(f"Error compiling final output: {e}")
        raise

# Create the workflow
def create_context_analysis_workflow(save_to_file: bool = True) -> Workflow:
    """Create a workflow for context analysis."""
    if save_to_file:
        workflow = (
            Workflow("analyze_context")
            .then("generate_structured_output")
            .then("save_output_to_file")
            .then("compile_final_output")
        )
    else:
        workflow = (
            Workflow("analyze_context")
            .then("generate_structured_output")
            .then("compile_final_output")
        )
    
    # Add input mappings
    workflow.node_input_mappings = {
        "analyze_context": {
            "model": "analysis_model"
        },
        "generate_structured_output": {
            "model": "output_model"
        },
        "compile_final_output": {
            "execution_start_time": "execution_start_time"
        }
    }
    
    return workflow

async def process_large_context(
    context: str,
    instructions: str,
    output_format: str,
    save_to_file: bool,
    analysis_model: str,
    output_model: str,
    execution_start_time: float,
    chunk_size: int
) -> AnalysisOutput:
    """
    Process a large context by breaking it into manageable chunks.
    
    This function handles large documents by:
    1. Splitting the context into chunks of appropriate size
    2. Processing each chunk separately
    3. Aggregating the results into a single coherent output
    """
    logger.info(f"Processing large context of {len(context)} chars in chunks of {chunk_size} chars")
    
    # Split the context into chunks
    chunks = []
    for i in range(0, len(context), chunk_size):
        chunk = context[i:i + chunk_size]
        chunks.append(chunk)
    
    logger.info(f"Split context into {len(chunks)} chunks")
    
    # Process each chunk
    chunk_results = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Create context for this chunk
        chunk_context = {
            "context": chunk,
            "instructions": instructions,
            "output_format": "json",  # Use JSON for intermediate results for easier merging
            "analysis_model": analysis_model,
            "output_model": output_model,
            "execution_start_time": execution_start_time
        }
        
        try:
            # Process this chunk
            workflow = create_context_analysis_workflow(False)  # Don't save intermediate chunks to file
            engine = workflow.build()
            result = await engine.run(chunk_context)
            
            if not isinstance(result.get("analysis_output"), AnalysisOutput):
                logger.warning(f"Chunk {i+1} did not produce a valid analysis output")
                continue
                
            # Extract the structured data from this chunk
            chunk_results.append(result["analysis_output"])
            
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {e}")
            # Continue with other chunks even if one fails
    
    # Merge the results from all chunks
    if not chunk_results:
        raise ValueError("No valid results were produced from any chunks")
    
    # Extract and merge the JSON data from all chunks
    merged_data = []
    key_findings = []
    confidence_scores = []
    
    for result in chunk_results:
        try:
            # The content might already be a dict or a JSON string
            try:
                content_data = json.loads(result.content)
            except (json.JSONDecodeError, TypeError):
                # If it's not a valid JSON string, it might already be a dict
                if isinstance(result.content, dict):
                    content_data = result.content
                else:
                    # If we can't parse it, skip this chunk
                    logger.warning(f"Could not parse content as JSON: {result.content[:100]}...")
                    continue
            
            # Extract the extracted_data_json field
            extracted_data_json = content_data.get("extracted_data_json", "[]")
            
            # Parse the extracted_data_json field
            try:
                if isinstance(extracted_data_json, str):
                    chunk_data = json.loads(extracted_data_json)
                else:
                    # If it's already a dict or list, use it directly
                    chunk_data = extracted_data_json
            except json.JSONDecodeError:
                logger.warning(f"Could not parse extracted_data_json: {extracted_data_json[:100]}...")
                chunk_data = []
            
            # Add the data to our merged data
            if isinstance(chunk_data, list):
                merged_data.extend(chunk_data)
            elif isinstance(chunk_data, dict):
                merged_data.append(chunk_data)
                
            # Extract key findings and confidence scores
            chunk_findings = content_data.get("key_findings", [])
            key_findings.extend(chunk_findings)
            
            chunk_confidence = content_data.get("confidence_score", 0)
            confidence_scores.append(chunk_confidence)
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error extracting data from chunk result: {e}")
    
    # Calculate average confidence score
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
    
    # Create a merged analysis result
    merged_analysis_result = ContextAnalysisResult(
        key_findings=key_findings[:10],  # Limit to top findings
        extracted_data_json=json.dumps(merged_data),
        confidence_score=avg_confidence,
        metadata_source="Multiple chunks",
        metadata_page_count=str(len(chunks))
    )
    
    # Generate final structured output in the requested format
    try:
        # Create context for final output generation
        final_context = {
            "analysis_result": merged_analysis_result,
            "output_format": output_format,
            "output_model": output_model
        }
        
        # Generate structured output
        structured_output = await generate_structured_output(**final_context)
        
        # Save to file if requested
        file_output = None
        if save_to_file:
            # Use the direct version of save_output_to_file
            file_output = await save_output_to_file_direct(structured_output)
        
        # Compile final output - use the direct version
        final_output = await compile_final_output_direct(
            analysis_result=merged_analysis_result,
            structured_output=structured_output,
            file_output=file_output,
            execution_start_time=execution_start_time
        )
        
        logger.info("Successfully processed and merged large context")
        return final_output
        
    except Exception as e:
        logger.error(f"Error generating final output from merged chunks: {e}")
        raise

async def analyze_context_with_instructions(
    context: str,
    instructions: str,
    output_format: str = "text",
    save_to_file: bool = True,
    analysis_model: str = "gemini/gemini-2.0-flash",
    output_model: str = "gemini/gemini-2.0-flash",
    task_id: str = "default",
    chunk_size: int = 10000,  # Size of each context chunk
    _handle_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
) -> AnalysisOutput:
    """Analyze context based on instructions and generate structured output."""
    
    # Ensure inputs are not empty
    if not context or context.strip() == "":
        raise ValueError("Context cannot be empty")
    
    if not instructions or instructions.strip() == "":
        raise ValueError("Instructions cannot be empty")
    
    # Clean up inputs
    context = context.strip()
    instructions = instructions.strip()
    output_format = output_format.strip().lower()
    
    logger.info(f"Processing context analysis with format: {output_format}")
    
    # Record start time for execution timing
    execution_start_time = asyncio.get_event_loop().time()
    
    # Check if we need to chunk the context (if it's very large)
    if len(context) > chunk_size:
        logger.info(f"Large context detected ({len(context)} chars), processing in chunks")
        return await process_large_context(
            context=context,
            instructions=instructions,
            output_format=output_format,
            save_to_file=save_to_file,
            analysis_model=analysis_model,
            output_model=output_model,
            execution_start_time=execution_start_time,
            chunk_size=chunk_size
        )
    
    # For smaller contexts, process normally
    initial_context = {
        "context": context,
        "instructions": instructions,
        "output_format": output_format,
        "analysis_model": analysis_model,
        "output_model": output_model,
        "execution_start_time": execution_start_time
    }
    
    logger.info(f"Starting context analysis {'with' if save_to_file else 'without'} file output")
    
    try:
        workflow = create_context_analysis_workflow(save_to_file)
        engine = workflow.build()
        
        # Add the event observer if _handle_event is provided
        # if _handle_event:
        #     # Create a lambda to bind task_id to the observer
        #     bound_observer = lambda event: asyncio.create_task(
        #         event_observer(event, task_id=task_id, _handle_event=_handle_event)
        #     )
        #     engine.add_observer(bound_observer)

        result = await engine.run(initial_context)
        
        if not isinstance(result.get("analysis_output"), AnalysisOutput):
            raise ValueError("Workflow did not produce a valid analysis output")
        
        logger.info("Context analysis completed successfully")
        return result["analysis_output"]
        
    except Exception as e:
        logger.error(f"Error analyzing context: {e}")
        raise

# Add main function for testing
async def main():
    """Test function for the context analysis workflow with chunking and multi-format output."""
    # Test context and instructions
    test_context = """
    ### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

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
    """
    
    # Create a larger context by duplicating the test context multiple times to test chunking
    large_test_context = test_context * 5  # Duplicate 5 times to create a larger document
    
    test_instructions = """
    En tant que spécialiste de l'extraction de données précises, analyse le document fourni qui présente une liste de factures et effectue les tâches suivantes :

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
    """
    
    try:
        # Test with regular context first
        console.print("\n[bold blue]Testing with regular context...[/]")
        analysis_regular = await analyze_context_with_instructions(
            context=test_context,
            instructions=test_instructions,
            output_format="json",  # Request JSON as primary format
            save_to_file=True,
            chunk_size=10000  # Small chunk size to force chunking even for small context
        )
        console.print("[bold green]✓ Regular context analysis completed[/]")
        
        # Display all output formats from the same analysis
        console.print("\n[bold cyan]Demonstrating multi-format output from a single analysis:[/]")
        
        # Display the primary format first
        console.print(Panel(Markdown(f"```\n{analysis_regular.content}\n```"), 
                           title=f"Primary Output Format: {analysis_regular.format_type.upper()}", 
                           border_style="red"))
        
        # Display JSON output if available
        if analysis_regular.json_content and analysis_regular.format_type.lower() != 'json':
            console.print(Panel(Markdown(f"```json\n{analysis_regular.json_content}\n```"), 
                               title="JSON Output", 
                               border_style="green"))
        
        # Display CSV output if available
        if analysis_regular.csv_content and analysis_regular.format_type.lower() != 'csv':
            console.print(Panel(Markdown(f"```\n{analysis_regular.csv_content}\n```"), 
                               title="CSV Output", 
                               border_style="blue"))
        
        # Display Markdown output if available
        if analysis_regular.markdown_content and analysis_regular.format_type.lower() not in ['markdown', 'table']:
            console.print(Panel(Markdown(analysis_regular.markdown_content), 
                               title="Markdown Table Output", 
                               border_style="yellow"))
        
        # Display Text output if available
        if analysis_regular.text_content and analysis_regular.format_type.lower() != 'text':
            console.print(Panel(analysis_regular.text_content, 
                               title="Plain Text Output", 
                               border_style="magenta"))
        
        # Now test with large context to demonstrate chunking
        console.print("\n[bold blue]Testing with large context (chunking)...[/]")
        console.print(f"Context size: {len(large_test_context)} characters")
        
        analysis_large = await analyze_context_with_instructions(
            context=large_test_context,
            instructions=test_instructions,
            output_format="table",  # Request table/markdown as primary format
            save_to_file=True,
            chunk_size=1000  # Small chunk size to force chunking
        )
        
        console.print("[bold green]✓ Large context analysis with chunking completed[/]")
        console.print(f"[bold]Execution time:[/] {analysis_large.execution_time:.2f} seconds")
        
        # Display the primary output format (table/markdown)
        console.print(Panel(Markdown(analysis_large.content), 
                           title="Table Output from Chunked Analysis", 
                           border_style="red"))
        
        # Summary of features demonstrated
        console.print("\n[bold green]✓ Features successfully demonstrated:[/]")
        console.print("  • [bold]Chunking mechanism[/] for large contexts")
        console.print("  • [bold]Multi-format output[/] (JSON, CSV, Markdown, Text) from a single analysis")
        console.print("  • [bold]Error handling[/] with fallback mechanisms")
        console.print("  • [bold]Comprehensive output model[/] with all formats accessible")
        
    except Exception as e:
        console.print(f"[bold red]Error during testing:[/] {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())