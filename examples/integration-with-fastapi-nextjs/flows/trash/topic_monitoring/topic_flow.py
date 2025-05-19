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
#     "feedparser>=6.0.0",
#     "beautifulsoup4>=4.12.0",
#     "aiohttp>=3.9.0",
#     "pandas>=2.0.0"
# ]
# ///

import asyncio
from collections.abc import Callable
import datetime
import os
from typing import List, Optional, Dict, Any, Set
from pydantic import BaseModel, Field

from loguru import logger
from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def get_template_path(template_name: str) -> str:
    """Get the full path to a template file."""
    return os.path.join(TEMPLATES_DIR, template_name)

# Data Models
class ThemeDefinition(BaseModel):
    """Definition of the monitoring theme"""
    main_topic: str = Field(description="Main topic or theme to monitor")
    subtopics: List[str] = Field(description="Related subtopics")
    keywords: List[str] = Field(description="Key terms and phrases to track")
    excluded_terms: List[str] = Field(description="Terms to exclude from monitoring")
    time_scope: str = Field(description="Time scope for monitoring (e.g., 'last week', 'last month')")
    languages: List[str] = Field(description="Languages to monitor")

class InformationSource(BaseModel):
    """Information source details"""
    source_type: str = Field(description="Type of source (website, RSS, social media, etc.)")
    url: str = Field(description="Source URL")
    reliability_score: float = Field(description="Source reliability score (0-1)")
    update_frequency: str = Field(description="Update frequency of the source")
    last_checked: datetime.datetime = Field(description="Last check timestamp")
    metadata: Dict[str, Any] = Field(description="Additional source metadata")

class CollectedData(BaseModel):
    """Collected data from sources"""
    source: InformationSource = Field(description="Source of the data")
    content: str = Field(description="Raw content")
    publication_date: datetime.datetime = Field(description="Publication date")
    author: Optional[str] = Field(description="Content author if available")
    title: str = Field(description="Content title")
    url: str = Field(description="Content URL")
    metadata: Dict[str, Any] = Field(description="Additional content metadata")

class DataAnalysis(BaseModel):
    """Analysis of collected data"""
    key_findings: List[str] = Field(description="Key findings from the analysis")
    trends: List[Dict[str, Any]] = Field(description="Identified trends")
    sentiment: Dict[str, float] = Field(description="Sentiment analysis results")
    entities: List[Dict[str, Any]] = Field(description="Named entities identified")
    topic_clusters: List[Dict[str, Any]] = Field(description="Topic clusters")
    key_metrics: Dict[str, Any] = Field(description="Important metrics")

class ClassifiedInformation(BaseModel):
    """Classified and organized information"""
    categories: Dict[str, List[str]] = Field(description="Information by category")
    priority_items: List[Dict[str, Any]] = Field(description="High-priority information")
    impact_assessment: Dict[str, Any] = Field(description="Impact assessment by category")
    action_items: List[str] = Field(description="Required actions")
    knowledge_gaps: List[str] = Field(description="Identified knowledge gaps")

class MonitoringReport(BaseModel):
    """Final monitoring report"""
    theme: ThemeDefinition = Field(description="Theme definition")
    period: str = Field(description="Monitoring period")
    executive_summary: str = Field(description="Executive summary")
    key_insights: List[str] = Field(description="Key insights")
    trends_analysis: Dict[str, Any] = Field(description="Trends analysis")
    recommendations: List[str] = Field(description="Recommendations")
    next_steps: List[str] = Field(description="Next steps")
    sources_summary: List[Dict[str, Any]] = Field(description="Summary of sources")
    impact_evaluation: Dict[str, Any] = Field(description="Impact evaluation")

# Workflow Nodes
@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_theme_definition.j2"),
    output="theme_definition",
    response_model=ThemeDefinition,
    prompt_file=get_template_path("prompt_theme_definition.j2")
)
async def define_theme(topic: str, context: Dict[str, Any], model: str) -> ThemeDefinition:
    """Define the monitoring theme and scope."""
    logger.info(f"Defining monitoring theme for topic: {topic}")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_source_identification.j2"),
    output="information_sources",
    response_model=List[InformationSource],
    prompt_file=get_template_path("prompt_source_identification.j2")
)
async def identify_sources(theme: ThemeDefinition, model: str) -> List[InformationSource]:
    """Identify relevant information sources."""
    logger.info("Identifying information sources")
    pass

@Nodes.define
async def collect_data(
    sources: List[InformationSource],
    theme: ThemeDefinition
) -> List[CollectedData]:
    """Collect data from identified sources."""
    logger.info("Collecting data from sources")
    # Implementation would include:
    # - Web scraping
    # - RSS feed parsing
    # - API calls
    # - Data cleaning
    collected_data = []
    return collected_data

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_data_analysis.j2"),
    output="data_analysis",
    response_model=DataAnalysis,
    prompt_file=get_template_path("prompt_data_analysis.j2")
)
async def analyze_data(
    collected_data: List[CollectedData],
    theme: ThemeDefinition,
    model: str
) -> DataAnalysis:
    """Analyze collected data."""
    logger.info("Analyzing collected data")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_information_classification.j2"),
    output="classified_information",
    response_model=ClassifiedInformation,
    prompt_file=get_template_path("prompt_information_classification.j2")
)
async def classify_information(
    data_analysis: DataAnalysis,
    theme: ThemeDefinition,
    model: str
) -> ClassifiedInformation:
    """Classify and organize analyzed information."""
    logger.info("Classifying information")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_monitoring_report.j2"),
    output="monitoring_report",
    response_model=MonitoringReport,
    prompt_file=get_template_path("prompt_monitoring_report.j2")
)
async def prepare_monitoring_report(
    theme: ThemeDefinition,
    data_analysis: DataAnalysis,
    classified_info: ClassifiedInformation,
    model: str
) -> MonitoringReport:
    """Prepare the final monitoring report."""
    logger.info("Preparing monitoring report")
    pass

def create_topic_monitoring_workflow() -> Workflow:
    """Create the workflow for topic monitoring."""
    workflow = (
        Workflow("define_theme")
        .then("identify_sources")
        .then("collect_data")
        .then("analyze_data")
        .then("classify_information")
        .then("prepare_monitoring_report")
    )
    
    workflow.node_input_mappings = {
        "define_theme": {
            "model": "llm_model",
            "topic": "topic",
            "context": "context"
        },
        "identify_sources": {
            "model": "llm_model"
        },
        "collect_data": {},
        "analyze_data": {
            "model": "llm_model"
        },
        "classify_information": {
            "model": "llm_model"
        },
        "prepare_monitoring_report": {
            "model": "llm_model"
        }
    }
    
    return workflow

async def monitor_topic(
    topic: str,
    context: Dict[str, Any],
    llm_model: str = "gemini/gemini-2.0-flash",
    _handle_event: Optional[Callable[[str, dict], None]] = None
) -> MonitoringReport:
    """Run the complete topic monitoring workflow."""
    
    initial_context = {
        "topic": topic,
        "context": context,
        "llm_model": llm_model
    }
    
    workflow = create_topic_monitoring_workflow()
    engine = workflow.build()
    
    result = await engine.run(initial_context)
    
    logger.info(f"Topic monitoring completed for: {topic}")
    return result["monitoring_report"]

def cli_monitor_topic(
    topic: str,
    context_file: str,
    output_file: Optional[str] = None,
    model: str = "gemini/gemini-2.0-flash"
):
    """CLI wrapper for topic monitoring."""
    # Read context file
    with open(context_file, 'r', encoding='utf-8') as f:
        context = eval(f.read())  # Note: In production, use proper JSON parsing
    
    # Run monitoring
    result = asyncio.run(monitor_topic(
        topic=topic,
        context=context,
        llm_model=model
    ))
    
    # Output results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.model_dump_json(indent=2))
    else:
        print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    import typer
    typer.run(cli_monitor_topic)
