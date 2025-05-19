#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru",
#     "litellm",
#     "pydantic>=2.0",
#     "anyio",
#     "quantalogic>=0.35",
#     "jinja2",
#     "typer",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "plotly",
#     "instructor"
# ]
# ///

import os
import glob
from typing import Dict, List, Optional, Any
from pathlib import Path

import anyio
import typer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from pydantic import BaseModel, Field

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType

# Configure logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Define structured output models
class ColumnAnalysis(BaseModel):
    name: str
    data_type: str
    description: str
    statistics: Dict[str, Any] = {}
    visualization_suggestions: List[str] = []

class Visualization(BaseModel):
    title: str
    description: str
    chart_type: str
    columns: List[str]
    parameters: Dict[str, Any] = {}

class CSVAnalysis(BaseModel):
    file_name: str
    row_count: int
    column_count: int
    columns: List[ColumnAnalysis]
    correlations: Dict[str, Any] = {}
    visualization_suggestions: List[Visualization] = []
    insights: List[str] = []

class MultiCSVAnalysis(BaseModel):
    files: List[CSVAnalysis]
    cross_file_insights: List[str] = []
    cross_file_visualizations: List[Visualization] = []

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Create output directories
def create_output_dirs(base_dir: str) -> Dict[str, str]:
    """Create output directories for visualizations and reports."""
    dirs = {
        "visualizations": os.path.join(base_dir, "visualizations"),
        "reports": os.path.join(base_dir, "reports"),
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

# Helper function to get template paths
def get_template_path(template_name: str) -> str:
    return os.path.join(TEMPLATES_DIR, template_name)

# Create template files if they don't exist
def ensure_templates_exist():
    """Create template files if they don't exist."""
    templates = {
        "system_analyze_csv.j2": """You are a data analysis expert specializing in exploratory data analysis.
Your task is to analyze CSV data and provide insights about the structure, content, and potential visualizations.
Focus on identifying patterns, correlations, and interesting aspects of the data that would be valuable to visualize.
""",
        
        "prompt_analyze_csv.j2": """Analyze the following CSV file: {{ file_name }}

Here's a summary of the data:
{{ data_summary }}

Sample data (first 5 rows):
{{ sample_data }}

Data types:
{{ data_types }}

Basic statistics:
{{ basic_stats }}

Please provide:
1. A detailed analysis of each column (data type, description, key statistics)
2. Suggestions for meaningful visualizations with specific chart types
3. Insights about patterns, outliers, or interesting aspects of the data
4. Correlations between columns that might be worth exploring
""",
        
        "system_suggest_visualizations.j2": """You are a data visualization expert.
Your task is to suggest the most insightful and relevant visualizations for the given dataset(s).
Focus on visualizations that reveal patterns, trends, distributions, correlations, or comparisons.
For each visualization, provide a clear title, description, chart type, and the columns to use.
""",
        
        "prompt_suggest_visualizations.j2": """Based on the analysis of the CSV file(s), suggest visualizations that would provide valuable insights.

{% if is_multi_file %}
Files analyzed:
{% for file in files %}
- {{ file.file_name }} ({{ file.row_count }} rows, {{ file.column_count }} columns)
{% endfor %}

Consider both individual file visualizations and cross-file visualizations where appropriate.
{% else %}
File analyzed: {{ analysis.file_name }} ({{ analysis.row_count }} rows, {{ analysis.column_count }} columns)

Columns:
{% for column in analysis.columns %}
- {{ column.name }} ({{ column.data_type }}): {{ column.description }}
{% endfor %}

Correlations:
{{ correlations }}
{% endif %}

For each visualization, provide:
1. A clear title
2. A description of what insight the visualization aims to provide
3. The chart type (bar, line, scatter, histogram, heatmap, etc.)
4. The columns to use
5. Any specific parameters (e.g., color by, size by, facet by)
""",
        
        "system_generate_insights.j2": """You are a data science expert specializing in extracting insights from data visualizations.
Your task is to analyze the generated visualizations and provide meaningful insights about the data.
Focus on patterns, trends, anomalies, and relationships that are revealed by the visualizations.
""",
        
        "prompt_generate_insights.j2": """Based on the visualizations generated for the CSV file(s), provide insights about the data.

{% if is_multi_file %}
Files analyzed:
{% for file in files %}
- {{ file.file_name }}
{% endfor %}
{% else %}
File analyzed: {{ analysis.file_name }}
{% endif %}

Visualizations generated:
{% for viz in visualizations %}
{{ loop.index }}. {{ viz.title }} ({{ viz.chart_type }})
   Description: {{ viz.description }}
   Columns used: {{ viz.columns|join(', ') }}
{% endfor %}

Please provide:
1. Key insights from each visualization
2. Overall patterns or trends observed across multiple visualizations
3. Surprising or unexpected findings
4. Recommendations for further analysis or data collection
{% if is_multi_file %}
5. Cross-file insights and relationships
{% endif %}
"""
    }
    
    for name, content in templates.items():
        template_path = get_template_path(name)
        if not os.path.exists(template_path):
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Created template: {name}")

# Custom Observer for Workflow Events
async def csv_explorer_progress_observer(event: WorkflowEvent):
    if event.event_type == WorkflowEventType.WORKFLOW_STARTED:
        print(f"\n{'='*50}\n🚀 Starting CSV Exploration 🚀\n{'='*50}")
    elif event.event_type == WorkflowEventType.NODE_STARTED:
        print(f"\n🔄 [{event.node_name}] Starting...")
    elif event.event_type == WorkflowEventType.NODE_COMPLETED:
        if event.node_name == "generate_visualization":
            viz_title = event.result.get("title", "Visualization")
            print(f"✅ [{event.node_name}] Created: {viz_title}")
        elif event.node_name == "analyze_csv_file":
            # Handle string output from LLM node
            if isinstance(event.result, str):
                # Extract file name from context
                file_name = event.context.get("file_analysis", {}).get("csv_file", {}).get("file_name", "CSV file")
                print(f"✅ [{event.node_name}] Analyzed: {file_name}")
            else:
                # Handle structured output if available
                try:
                    file_name = event.result.file_name
                    print(f"✅ [{event.node_name}] Analyzed: {file_name}")
                except AttributeError:
                    print(f"✅ [{event.node_name}] Completed")
        elif event.node_name == "parse_csv_analysis":
            # Handle parsed analysis
            file_name = event.result.get("file_name", "CSV file")
            print(f"✅ [{event.node_name}] Parsed analysis for: {file_name}")
        elif event.node_name == "suggest_visualizations":
            # Handle string output from LLM node
            viz_count = 3  # Default number of visualizations
            print(f"✅ [{event.node_name}] Generated {viz_count} visualization suggestions")
        else:
            print(f"✅ [{event.node_name}] Completed")
    elif event.event_type == WorkflowEventType.WORKFLOW_COMPLETED:
        print(f"\n{'='*50}\n🎉 CSV Exploration Finished 🎉\n{'='*50}")
    elif event.event_type == WorkflowEventType.TRANSITION_EVALUATED:
        logger.debug(f"Transition evaluated: {event.transition_from} -> {event.transition_to}")

# Workflow Nodes
@Nodes.define(output=None)
async def read_csv_files(csv_paths: List[str]) -> dict:
    """Read CSV files and return their contents."""
    csv_files = []
    
    # Handle glob patterns in paths
    expanded_paths = []
    for path in csv_paths:
        path = os.path.expanduser(path)
        if '*' in path:
            expanded_paths.extend(glob.glob(path))
        else:
            expanded_paths.append(path)
    
    # Read each CSV file
    for path in expanded_paths:
        try:
            df = pd.read_csv(path)
            file_name = os.path.basename(path)
            csv_files.append({
                "path": path,
                "file_name": file_name,
                "dataframe": df,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
            })
            logger.info(f"Read CSV file: {file_name} ({len(df)} rows, {len(df.columns)} columns)")
        except Exception as e:
            logger.error(f"Error reading CSV file {path}: {str(e)}")
    
    if not csv_files:
        raise ValueError("No valid CSV files found")
    
    return {
        "csv_files": csv_files,
        "file_count": len(csv_files),
        "is_multi_file": len(csv_files) > 1
    }

@Nodes.define(output=None)
async def setup_output_directories(output_dir: str) -> dict:
    """Set up output directories for visualizations and reports."""
    output_dirs = create_output_dirs(output_dir)
    logger.info(f"Created output directories in {output_dir}")
    return {"output_dirs": output_dirs}

@Nodes.define(output=None)
async def prepare_templates() -> None:
    """Ensure that all required templates exist."""
    ensure_templates_exist()
    logger.info("Prepared templates")

@Nodes.llm_node(
    system_prompt_file=get_template_path("system_analyze_csv.j2"),
    output="csv_analysis",
    prompt_file=get_template_path("prompt_analyze_csv.j2"),
    temperature=0.2,
)
async def analyze_csv_file(model: str, file_name: str, data_summary: str, sample_data: str, 
                          data_types: str, basic_stats: str) -> str:
    """Analyze a single CSV file using LLM."""
    logger.debug(f"Analyzing CSV file: {file_name}")
    pass

@Nodes.define(output="file_analysis")
async def process_single_file(csv_file: Dict, model: str) -> Dict:
    """Process a single CSV file and prepare it for LLM analysis."""
    df = csv_file["dataframe"]
    
    # Prepare data for LLM
    data_summary = f"Shape: {df.shape[0]} rows, {df.shape[1]} columns"
    sample_data = df.head(5).to_string()
    data_types = df.dtypes.to_string()
    
    # Calculate basic statistics safely
    try:
        basic_stats = df.describe(include='all').to_string()
    except Exception as e:
        basic_stats = f"Error calculating statistics: {str(e)}"
    
    # Calculate correlations for numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        try:
            correlations = numeric_df.corr().to_dict()
            csv_file["correlations"] = correlations
        except Exception as e:
            logger.warning(f"Error calculating correlations: {str(e)}")
            csv_file["correlations"] = {}
    else:
        csv_file["correlations"] = {}
    
    # Call LLM to analyze the file
    analysis_inputs = {
        "model": model,
        "file_name": csv_file["file_name"],
        "data_summary": data_summary,
        "sample_data": sample_data,
        "data_types": data_types,
        "basic_stats": basic_stats
    }
    
    return {
        "csv_file": csv_file,
        "analysis_inputs": analysis_inputs
    }

@Nodes.llm_node(
    system_prompt_file=get_template_path("system_suggest_visualizations.j2"),
    output="visualization_suggestions",
    prompt_file=get_template_path("prompt_suggest_visualizations.j2"),
    temperature=0.3,
)
async def suggest_visualizations(model: str, is_multi_file: bool, analysis: Optional[str] = None, 
                               files: Optional[List[str]] = None, correlations: Optional[str] = None) -> str:
    """Suggest visualizations for the CSV data using LLM."""
    logger.debug("Suggesting visualizations")
    pass

@Nodes.define(output="parsed_analysis")
async def parse_csv_analysis(csv_analysis: str) -> Dict:
    """Parse the CSV analysis from LLM output to a structured format."""
    try:
        import json
        import re
        
        # Try to extract JSON from the string
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', csv_analysis)
        if json_match:
            analysis_json = json_match.group(1)
        else:
            # Look for anything that might be JSON
            json_match = re.search(r'({[\s\S]*})', csv_analysis)
            if json_match:
                analysis_json = json_match.group(1)
            else:
                analysis_json = csv_analysis
                
        analysis = json.loads(analysis_json)
        logger.info("Successfully parsed CSV analysis from LLM output")
        return analysis
        
    except Exception as e:
        logger.warning(f"Failed to parse CSV analysis JSON: {str(e)}. Creating basic structure.")
        # Create a basic structure if parsing fails
        return {
            "file_name": "csv_file",
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "correlations": {},
            "visualization_suggestions": [],
            "insights": []
        }

@Nodes.define(output="visualization_result")
async def generate_visualization(visualization: str, csv_files: List[Dict], 
                               output_dirs: Dict[str, str], is_multi_file: bool) -> Dict:
    """Generate a visualization based on the suggestion."""
    viz_dir = output_dirs["visualizations"]
    
    # Parse visualization from string to get structured data
    try:
        # Try to parse as JSON
        import json
        viz_data = json.loads(visualization)
        
        # Extract visualization details
        title = viz_data.get('title', 'Visualization')
        description = viz_data.get('description', 'Auto-generated visualization')
        chart_type = viz_data.get('chart_type', 'bar')
        columns = viz_data.get('columns', [])
        
    except Exception as e:
        logger.warning(f"Failed to parse visualization JSON: {str(e)}. Using default values.")
        title = "Default Visualization"
        description = "Auto-generated visualization"
        chart_type = "bar"
        columns = []
    
    # Determine which dataframe(s) to use
    dfs = {}
    if is_multi_file:
        # For cross-file visualizations, we might need to merge or process multiple dataframes
        for file in csv_files:
            dfs[file["file_name"]] = file["dataframe"]
    else:
        # For single file, just use the first dataframe
        dfs = {"data": csv_files[0]["dataframe"]}
    
    # Create a unique filename for the visualization
    safe_title = title.replace(" ", "_").replace("/", "_").lower()
    file_path = os.path.join(viz_dir, f"{safe_title}.png")
    
    try:
        # Set up the figure
        plt.figure(figsize=(12, 8))
        
        # Get the first dataframe for simple charts
        df = list(dfs.values())[0]
        
        # Make sure we have columns to work with
        if not columns and len(df.columns) > 0:
            columns = [df.columns[0]]
        
        # Check if columns exist in the dataframe
        valid_columns = [col for col in columns if col in df.columns]
        if not valid_columns and len(df.columns) > 0:
            valid_columns = [df.columns[0]]
        
        # Generate the visualization based on chart type
        if chart_type.lower() in ["bar", "barplot"]:
            if valid_columns:
                col = valid_columns[0]
                # Get top 10 values for bar chart to avoid overcrowding
                value_counts = df[col].value_counts().head(10)
                value_counts.plot(kind='bar')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
            else:
                df.iloc[:, 0].value_counts().head(10).plot(kind='bar')
                
        elif chart_type.lower() in ["line", "lineplot"]:
            if len(valid_columns) >= 2:
                x_col, y_col = valid_columns[:2]
                df.plot(x=x_col, y=y_col, kind='line')
            elif valid_columns:
                # If only one column, plot its values over index
                df[valid_columns[0]].plot(kind='line')
            else:
                df.iloc[:, 0].plot(kind='line')
                
        elif chart_type.lower() in ["scatter", "scatterplot"]:
            if len(valid_columns) >= 2:
                x_col, y_col = valid_columns[:2]
                plt.scatter(df[x_col], df[y_col])
                plt.xlabel(x_col)
                plt.ylabel(y_col)
            else:
                # Fallback to histogram if not enough columns for scatter
                if valid_columns:
                    df[valid_columns[0]].hist()
                else:
                    df.iloc[:, 0].hist()
                
        elif chart_type.lower() in ["hist", "histogram"]:
            if valid_columns:
                df[valid_columns[0]].hist(bins=20)
                plt.xlabel(valid_columns[0])
            else:
                df.iloc[:, 0].hist(bins=20)
            
        elif chart_type.lower() in ["box", "boxplot"]:
            if valid_columns:
                df[valid_columns].boxplot()
            else:
                df.boxplot()
                
        elif chart_type.lower() in ["heatmap", "correlation"]:
            # Get numeric columns for correlation
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                corr = numeric_df.corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=.5)
            else:
                # Fallback if no numeric columns
                plt.text(0.5, 0.5, "No numeric columns for correlation", 
                         horizontalalignment='center', verticalalignment='center')
            
        elif chart_type.lower() in ["pie", "piechart"]:
            if valid_columns:
                # Get top 5 values for pie chart to avoid too many slices
                value_counts = df[valid_columns[0]].value_counts().head(5)
                value_counts.plot(kind='pie', autopct='%1.1f%%')
            else:
                df.iloc[:, 0].value_counts().head(5).plot(kind='pie', autopct='%1.1f%%')
                
        elif chart_type.lower() == "table":
            # Create a table visualization
            plt.axis('off')
            if valid_columns:
                table_data = df[valid_columns].head(10)
            else:
                table_data = df.head(10)
                
            table = plt.table(
                cellText=table_data.values,
                colLabels=table_data.columns,
                cellLoc='center',
                loc='center',
                bbox=[0.2, 0.2, 0.7, 0.6]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            
        else:
            # Default to a simple plot for unsupported chart types
            logger.warning(f"Unsupported chart type: {chart_type}. Using default plot.")
            if valid_columns:
                df[valid_columns[0]].plot()
            else:
                df.iloc[:, 0].plot()
        
        # Set title and labels
        plt.title(title)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(file_path)
        plt.close()
        
        logger.info(f"Generated visualization: {title}")
        
        return {
            "title": title,
            "description": description,
            "chart_type": chart_type,
            "columns": columns,
            "file_path": file_path,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error generating visualization '{title}': {str(e)}")
        return {
            "title": title,
            "description": description,
            "error": str(e),
            "success": False
        }

@Nodes.llm_node(
    system_prompt_file=get_template_path("system_generate_insights.j2"),
    output="insights",
    prompt_file=get_template_path("prompt_generate_insights.j2"),
    temperature=0.3,
)
async def generate_insights(model: str, is_multi_file: bool, visualizations: List[Dict], 
                          analysis: Optional[str] = None, files: Optional[List[str]] = None) -> str:
    """Generate insights based on the visualizations using LLM."""
    logger.debug("Generating insights from visualizations")
    pass

@Nodes.define(output="report_path")
async def generate_report(csv_files: List[Dict], visualizations: List[Dict] = None, insights: str = None, 
                        output_dirs: Dict[str, str] = None, is_multi_file: bool = False) -> str:
    """Generate a comprehensive report with visualizations and insights."""
    # Handle missing parameters with defaults
    if visualizations is None:
        visualizations = []
    
    if insights is None:
        insights = "No detailed insights available for this dataset."
    
    if output_dirs is None:
        output_dirs = {"reports": "./csv_explorer_output/reports"}
        os.makedirs(output_dirs["reports"], exist_ok=True)
        
    report_dir = output_dirs["reports"]
    
    # Create report filename
    if is_multi_file:
        report_filename = "multi_csv_analysis_report.html"
    else:
        file_name = os.path.splitext(csv_files[0]["file_name"])[0]
        report_filename = f"{file_name}_analysis_report.html"
    
    report_path = os.path.join(report_dir, report_filename)
    
    # Build HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CSV Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .file-info {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .visualization {{ margin-bottom: 30px; }}
            .visualization img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            .insights {{ background-color: #f0f7ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CSV Analysis Report</h1>
    """
    
    # Add file information
    html_content += "<h2>Files Analyzed</h2>"
    html_content += "<div class='file-info'>"
    for file in csv_files:
        html_content += f"""
        <h3>{file['file_name']}</h3>
        <p>Rows: {file['row_count']} | Columns: {file['column_count']}</p>
        <table>
            <tr>
                <th>Column Name</th>
                <th>Data Type</th>
            </tr>
        """
        for col, dtype in file['dataframe'].dtypes.items():
            html_content += f"<tr><td>{col}</td><td>{dtype}</td></tr>"
        html_content += "</table>"
    html_content += "</div>"
    
    # Add visualizations
    html_content += "<h2>Visualizations</h2>"
    successful_visualizations = [v for v in visualizations if v.get("success", False)]
    
    if successful_visualizations:
        for viz in successful_visualizations:
            rel_path = os.path.relpath(viz["file_path"], report_dir)
            html_content += f"""
            <div class="visualization">
                <h3>{viz['title']}</h3>
                <p>{viz['description']}</p>
                <img src="{rel_path}" alt="{viz['title']}">
            </div>
            """
    else:
        html_content += "<p>No visualizations were successfully generated.</p>"
    
    # Add insights
    html_content += "<h2>Insights</h2>"
    html_content += f"<div class='insights'>{insights.replace('\n', '<br>')}</div>"
    
    # Close HTML
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Generated report: {report_path}")
    return report_path

@Nodes.define(output=None)
async def display_summary(report_path: str, visualizations: List[Dict] = None) -> None:
    """Display a summary of the analysis."""
    if visualizations is None:
        visualizations = []
        
    successful_viz = [v for v in visualizations if v.get("success", False)]
    failed_viz = [v for v in visualizations if not v.get("success", False)]
    
    print("\n" + "="*50)
    print("📊 CSV Exploration Summary 📊")
    print("="*50)
    print(f"Total visualizations generated: {len(successful_viz)}")
    if failed_viz:
        print(f"Failed visualizations: {len(failed_viz)}")
    print(f"Report generated at: {report_path}")
    print("="*50)
    print("To view the report, open the HTML file in your browser.")
    print("="*50)

@Nodes.define(output=None)
async def initialize_visualization_loop() -> Dict:
    """Initialize the visualization loop with empty results."""
    logger.debug("Initializing visualization loop")
    return {"current_viz_index": 0, "visualization_results": []}

@Nodes.define(output=None)
async def process_visualization(visualization_suggestions: str, current_viz_index: int, csv_files: List[Dict]) -> Dict:
    """Process the current visualization in the loop."""
    # Instead of trying to parse LLM suggestions, generate standard visualizations based on data types
    df = csv_files[0]["dataframe"]
    
    # Get column names and types
    columns = list(df.columns)
    if not columns:
        return {"current_visualization": '{"title": "No columns found", "chart_type": "none", "columns": []}'}
    
    # Define standard visualization types based on column index
    standard_viz_types = [
        {"title": "Column Distribution", "chart_type": "bar", "columns": [columns[min(0, len(columns)-1)]]},
        {"title": "Value Counts", "chart_type": "pie", "columns": [columns[min(0, len(columns)-1)]]},
        {"title": "Data Overview", "chart_type": "table", "columns": columns[:min(5, len(columns))]}
    ]
    
    # Select visualization based on index
    if current_viz_index < len(standard_viz_types):
        viz_config = standard_viz_types[current_viz_index]
    else:
        # Fallback to a default visualization
        viz_config = {"title": f"Column {current_viz_index} Analysis", "chart_type": "bar", "columns": [columns[0]]}
    
    # Add column-specific visualizations based on data types
    if current_viz_index < len(columns):
        col = columns[current_viz_index]
        dtype = str(df[col].dtype)
        
        if "int" in dtype or "float" in dtype:
            viz_config = {
                "title": f"Distribution of {col}",
                "chart_type": "histogram",
                "columns": [col],
                "description": f"Histogram showing the distribution of values in column '{col}'"
            }
        elif "object" in dtype or "string" in dtype:
            viz_config = {
                "title": f"Top Values in {col}",
                "chart_type": "bar",
                "columns": [col],
                "description": f"Bar chart showing the most frequent values in column '{col}'"
            }
        elif "date" in dtype or "time" in dtype:
            viz_config = {
                "title": f"Timeline of {col}",
                "chart_type": "line",
                "columns": [col],
                "description": f"Line chart showing the timeline of values in column '{col}'"
            }
        elif "bool" in dtype:
            viz_config = {
                "title": f"Distribution of {col}",
                "chart_type": "pie",
                "columns": [col],
                "description": f"Pie chart showing the distribution of boolean values in column '{col}'"
            }
    
    # Convert to JSON string
    import json
    current_visualization = json.dumps(viz_config)
    
    logger.debug(f"Processing visualization {current_viz_index + 1}: {viz_config['title']}")
    return {"current_visualization": current_visualization}

@Nodes.define(output=None)
async def update_visualization_results(visualization_results: List[Dict], visualization_result: Dict, current_viz_index: int) -> Dict:
    """Update the visualization results and increment the index."""
    updated_results = visualization_results + [visualization_result]
    next_index = current_viz_index + 1
    logger.debug(f"Updated visualization results. Processed {next_index} visualizations so far.")
    return {
        "visualization_results": updated_results,
        "current_viz_index": next_index
    }


# Define the Workflow with explicit transitions
workflow = (
    Workflow("read_csv_files")
    .add_observer(csv_explorer_progress_observer)
    .then("setup_output_directories")
    .then("prepare_templates")
    .branch([
        # Single file path
        ("process_single_file", lambda ctx: not ctx.get("is_multi_file", False)),
        # Multi-file path (to be implemented)
        ("process_single_file", lambda ctx: ctx.get("is_multi_file", False))
    ])
    
    # Process single file
    .node("process_single_file", inputs_mapping={
        "csv_file": lambda ctx: ctx["csv_files"][0],
        "model": "model"
    })
    .then("analyze_csv_file")
    .node("analyze_csv_file", inputs_mapping={
        "model": "model",
        "file_name": lambda ctx: ctx["file_analysis"]["csv_file"]["file_name"],
        "data_summary": lambda ctx: ctx["file_analysis"]["analysis_inputs"]["data_summary"],
        "sample_data": lambda ctx: ctx["file_analysis"]["analysis_inputs"]["sample_data"],
        "data_types": lambda ctx: ctx["file_analysis"]["analysis_inputs"]["data_types"],
        "basic_stats": lambda ctx: ctx["file_analysis"]["analysis_inputs"]["basic_stats"]
    })
    .then("parse_csv_analysis")
    .node("parse_csv_analysis", inputs_mapping={
        "csv_analysis": "csv_analysis"
    })
    .then("initialize_visualization_loop")
    .node("initialize_visualization_loop")
    .then("process_visualization")
    .node("process_visualization", inputs_mapping={
        "visualization_suggestions": "visualization_suggestions",
        "current_viz_index": "current_viz_index",
        "csv_files": "csv_files"
    })
    .then("generate_visualization")
    .node("generate_visualization", inputs_mapping={
        "visualization": "current_visualization",
        "csv_files": "csv_files",
        "output_dirs": "output_dirs",
        "is_multi_file": "is_multi_file"
    })
    .then("update_visualization_results")
    .node("update_visualization_results", inputs_mapping={
        "visualization_results": "visualization_results",
        "visualization_result": "visualization_result",
        "current_viz_index": "current_viz_index"
    })
    .branch([
        ("process_visualization", lambda ctx: ctx["current_viz_index"] < 3),  # Limit to 3 visualizations for simplicity
        ("generate_insights", lambda ctx: ctx["current_viz_index"] >= 3)
    ])
    
    # Generate insights and report
    .node("generate_insights", inputs_mapping={
        "model": "model",
        "is_multi_file": "is_multi_file",
        "visualizations": "visualization_results",
        "analysis": "csv_analysis"
    })
    .then("generate_report")
    .node("generate_report", inputs_mapping={
        "csv_files": "csv_files",
        "visualizations": "visualization_results",
        "insights": "insights",
        "output_dirs": "output_dirs",
        "is_multi_file": "is_multi_file"
    })
    .then("display_summary")
)


# CLI with Typer
app = typer.Typer()

@app.command()
def explore_csv(
    csv_paths: List[str] = typer.Argument(..., help="Paths to CSV files (can include glob patterns)"),
    output_dir: str = typer.Option("./csv_explorer_output", help="Output directory for visualizations and reports"),
    model: str = typer.Option("gemini/gemini-2.0-flash", help="LLM model to use"),
):
    """
    Explore CSV files with data visualization and LLM-powered analysis.
    
    Example usage:
    python csv_explorer.py data.csv --output-dir ./output --model gemini/gemini-2.0-flash
    python csv_explorer.py "data/*.csv" --output-dir ./output
    """
    initial_context = {
        "csv_paths": csv_paths,
        "output_dir": output_dir,
        "model": model,
    }
    logger.info(f"Starting CSV exploration for {csv_paths}")
    engine = workflow.build()
    result = anyio.run(engine.run, initial_context)
    logger.info("CSV exploration completed successfully 🎉")

def main(csv_paths: List[str], output_dir: str = "./csv_explorer_output", 
         model: str = "gemini/gemini-2.0-flash") -> None:
    """
    Run CSV exploration directly without using the Typer CLI.
    
    Args:
        csv_paths: List of paths to CSV files (can include glob patterns)
        output_dir: Output directory for visualizations and reports
        model: LLM model to use
    
    Example usage:
        from csv_explorer import main
        main(["data.csv"], "./output")
        main(["data/*.csv"])
    """
    initial_context = {
        "csv_paths": csv_paths,
        "output_dir": output_dir,
        "model": model,
    }
    logger.info(f"Starting CSV exploration for {csv_paths}")
    engine = workflow.build()
    result = anyio.run(engine.run, initial_context)
    logger.info("CSV exploration completed successfully 🎉")
    return result


if __name__ == "__main__":
    main(["/home/yarab/Téléchargements/liste-podcasts-groupe-bpce.csv"])
