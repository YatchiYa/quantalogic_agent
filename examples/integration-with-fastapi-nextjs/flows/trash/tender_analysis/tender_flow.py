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
#     "rich>=13.0.0"
# ]
# ///

import asyncio
from collections.abc import Callable
import datetime
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from loguru import logger
from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def get_template_path(template_name: str) -> str:
    """Get the full path to a template file."""
    return os.path.join(TEMPLATES_DIR, template_name)

# Data Models
class RequirementAnalysis(BaseModel):
    """Analysis of tender requirements"""
    technical_requirements: List[str] = Field(description="Technical requirements identified")
    functional_requirements: List[str] = Field(description="Functional requirements identified")
    legal_requirements: List[str] = Field(description="Legal and administrative requirements")
    deadlines: Dict[str, datetime.datetime] = Field(description="Key deadlines and dates")
    budget_info: Optional[Dict[str, Any]] = Field(description="Budget related information")

class SWOTAnalysis(BaseModel):
    """SWOT Analysis results"""
    strengths: List[str] = Field(description="Company strengths for this tender")
    weaknesses: List[str] = Field(description="Company weaknesses for this tender")
    opportunities: List[str] = Field(description="Opportunities identified in the tender")
    threats: List[str] = Field(description="Potential threats and risks")

class PESTELAnalysis(BaseModel):
    """PESTEL Analysis results"""
    political: List[str] = Field(description="Political factors")
    economic: List[str] = Field(description="Economic factors")
    social: List[str] = Field(description="Social factors")
    technological: List[str] = Field(description="Technological factors")
    environmental: List[str] = Field(description="Environmental factors")
    legal: List[str] = Field(description="Legal factors")

class CompetitorAnalysis(BaseModel):
    """Competitor Analysis results"""
    identified_competitors: List[Dict[str, Any]] = Field(description="List of potential competitors")
    market_position: Dict[str, Any] = Field(description="Market position analysis")
    competitive_advantages: List[str] = Field(description="Our competitive advantages")
    risks: List[str] = Field(description="Competitive risks")

class ScoringMatrix(BaseModel):
    """Scoring Matrix for tender evaluation"""
    criteria: List[Dict[str, Any]] = Field(description="Evaluation criteria")
    scores: Dict[str, float] = Field(description="Scores for each criterion")
    weights: Dict[str, float] = Field(description="Weights for each criterion")
    total_score: float = Field(description="Total weighted score")
    recommendations: List[str] = Field(description="Recommendations based on scoring")

class TenderAnalysisReport(BaseModel):
    """Final tender analysis report"""
    tender_id: str = Field(description="Tender identifier")
    executive_summary: str = Field(description="Executive summary of the analysis")
    requirements: RequirementAnalysis = Field(description="Requirements analysis")
    swot: SWOTAnalysis = Field(description="SWOT analysis")
    pestel: PESTELAnalysis = Field(description="PESTEL analysis")
    competition: CompetitorAnalysis = Field(description="Competition analysis")
    scoring: ScoringMatrix = Field(description="Scoring matrix")
    final_recommendation: str = Field(description="Final recommendation (Go/No-Go)")
    risk_assessment: List[Dict[str, Any]] = Field(description="Key risks and mitigation strategies")
    next_steps: List[str] = Field(description="Recommended next steps")

# Workflow Nodes
@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_preliminary_analysis.j2"),
    output="initial_requirements",
    response_model=RequirementAnalysis,
    prompt_file=get_template_path("prompt_preliminary_analysis.j2")
)
async def preliminary_analysis(tender_content: str, model: str) -> RequirementAnalysis:
    """Perform initial reading and requirements analysis of the tender."""
    logger.info("Performing preliminary analysis of tender")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_swot.j2"),
    output="swot_analysis",
    response_model=SWOTAnalysis,
    prompt_file=get_template_path("prompt_swot.j2")
)
async def analyze_swot(
    requirements: RequirementAnalysis,
    company_profile: Dict[str, Any],
    model: str
) -> SWOTAnalysis:
    """Perform SWOT analysis based on requirements and company profile."""
    logger.info("Performing SWOT analysis")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_pestel.j2"),
    output="pestel_analysis",
    response_model=PESTELAnalysis,
    prompt_file=get_template_path("prompt_pestel.j2")
)
async def analyze_pestel(
    requirements: RequirementAnalysis,
    market_data: Dict[str, Any],
    model: str
) -> PESTELAnalysis:
    """Perform PESTEL analysis."""
    logger.info("Performing PESTEL analysis")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_competition.j2"),
    output="competition_analysis",
    response_model=CompetitorAnalysis,
    prompt_file=get_template_path("prompt_competition.j2")
)
async def analyze_competition(
    requirements: RequirementAnalysis,
    market_data: Dict[str, Any],
    swot: SWOTAnalysis,
    model: str
) -> CompetitorAnalysis:
    """Analyze competitive landscape."""
    logger.info("Analyzing competition")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_scoring.j2"),
    output="scoring_matrix",
    response_model=ScoringMatrix,
    prompt_file=get_template_path("prompt_scoring.j2")
)
async def create_scoring_matrix(
    requirements: RequirementAnalysis,
    swot: SWOTAnalysis,
    pestel: PESTELAnalysis,
    competition: CompetitorAnalysis,
    model: str
) -> ScoringMatrix:
    """Create and fill scoring matrix."""
    logger.info("Creating scoring matrix")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_final_report.j2"),
    output="final_report",
    response_model=TenderAnalysisReport,
    prompt_file=get_template_path("prompt_final_report.j2")
)
async def prepare_final_report(
    tender_id: str,
    requirements: RequirementAnalysis,
    swot: SWOTAnalysis,
    pestel: PESTELAnalysis,
    competition: CompetitorAnalysis,
    scoring: ScoringMatrix,
    model: str
) -> TenderAnalysisReport:
    """Prepare the final analysis report."""
    logger.info("Preparing final analysis report")
    pass

def create_tender_analysis_workflow() -> Workflow:
    """Create the workflow for tender analysis."""
    workflow = (
        Workflow("preliminary_analysis")
        .then("analyze_swot")
        .then("analyze_pestel")
        .then("analyze_competition")
        .then("create_scoring_matrix")
        .then("prepare_final_report")
    )
    
    workflow.node_input_mappings = {
        "preliminary_analysis": {
            "model": "llm_model",
            "tender_content": "tender_content"
        },
        "analyze_swot": {
            "model": "llm_model",
            "company_profile": "company_profile"
        },
        "analyze_pestel": {
            "model": "llm_model",
            "market_data": "market_data"
        },
        "analyze_competition": {
            "model": "llm_model",
            "market_data": "market_data"
        },
        "create_scoring_matrix": {
            "model": "llm_model"
        },
        "prepare_final_report": {
            "model": "llm_model",
            "tender_id": "tender_id"
        }
    }
    
    return workflow

async def analyze_tender(
    tender_content: str,
    tender_id: str,
    company_profile: Dict[str, Any],
    market_data: Dict[str, Any],
    llm_model: str = "gemini/gemini-2.0-flash",
    _handle_event: Optional[Callable[[str, dict], None]] = None
) -> TenderAnalysisReport:
    """Run the complete tender analysis workflow."""
    
    initial_context = {
        "tender_content": tender_content,
        "tender_id": tender_id,
        "company_profile": company_profile,
        "market_data": market_data,
        "llm_model": llm_model
    }
    
    workflow = create_tender_analysis_workflow()
    engine = workflow.build()
    
    result = await engine.run(initial_context)
    
    logger.info(f"Tender analysis completed for tender ID: {tender_id}")
    return result["final_report"]

def cli_analyze_tender(
    tender_file: str,
    tender_id: str,
    company_profile_file: str,
    market_data_file: str,
    output_file: Optional[str] = None,
    model: str = "gemini/gemini-2.0-flash"
):
    """CLI wrapper for tender analysis."""
    # Read input files
    with open(tender_file, 'r', encoding='utf-8') as f:
        tender_content = f.read()
    
    with open(company_profile_file, 'r', encoding='utf-8') as f:
        company_profile = eval(f.read())  # Note: In production, use proper JSON parsing
    
    with open(market_data_file, 'r', encoding='utf-8') as f:
        market_data = eval(f.read())  # Note: In production, use proper JSON parsing
    
    # Run analysis
    result = asyncio.run(analyze_tender(
        tender_content=tender_content,
        tender_id=tender_id,
        company_profile=company_profile,
        market_data=market_data,
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
    typer.run(cli_analyze_tender)
