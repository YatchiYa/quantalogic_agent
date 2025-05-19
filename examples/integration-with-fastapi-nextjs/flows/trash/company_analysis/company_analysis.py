import asyncio
from typing import Any, Dict, List, Optional, Callable
import os
from datetime import datetime

import anyio
from loguru import logger
from pydantic import BaseModel

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from quantalogic.tools.linkup_tool import LinkupTool 

# Configure logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Define structured output models
class CompanyInfo(BaseModel):
    name: str
    industry: str
    description: str
    revenue: str
    employee_count: str
    market_cap: str
    growth_rate: str
    recent_developments: List[str]
    
    class Config:
        frozen = True

class CompetitorAnalysis(BaseModel):
    company_name: str
    market_share: str
    strengths: List[str]
    weaknesses: List[str]
    
    class Config:
        frozen = True

class MarketingStrategy(BaseModel):
    strategy_name: str
    description: str
    target_audience: List[str]
    key_actions: List[str]
    timeline: str
    
    class Config:
        frozen = True

class CompanyAnalysis(BaseModel):
    company_info: CompanyInfo
    competitors: List[CompetitorAnalysis]
    marketing_strategies: List[MarketingStrategy]
    analysis_date: str
    
    class Config:
        frozen = True

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def get_template_path(template_name):
    return os.path.join(TEMPLATES_DIR, template_name)

# Custom Observer for Workflow Events
async def analysis_progress_observer(event: WorkflowEvent):
    if event.event_type == WorkflowEventType.WORKFLOW_STARTED:
        print(f"\n{'='*50}\n🔍 Starting Company Analysis 🔍\n{'='*50}")
    elif event.event_type == WorkflowEventType.NODE_STARTED:
        print(f"\n🔄 [{event.node_name}] Starting...")
    elif event.event_type == WorkflowEventType.NODE_COMPLETED:
        print(f"✅ [{event.node_name}] Completed")
    elif event.event_type == WorkflowEventType.WORKFLOW_COMPLETED:
        print(f"\n{'='*50}\n🎉 Company Analysis Completed 🎉\n{'='*50}")

# Workflow Nodes
@Nodes.define(output="company_basic_info")
async def gather_company_info(company_name: str) -> dict:
    linkup = LinkupTool()
    query = f"{company_name} company overview revenue employees industry"
    result = linkup.execute(query=query, output_type="sourcedAnswer")
    logger.info(f"Gathered basic info for {company_name}")
    return {"company_info_raw": result}

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_process_company_info.j2"),
    output="company_info",
    response_model=CompanyInfo,
    prompt_file=get_template_path("prompt_process_company_info.j2"),
    temperature=0.3,  # Lower temperature for more consistent output
    max_retries=2,  # Add retries for robustness
    validation_context={"company_name": "company_name"}  # Add validation context
)
async def process_company_info(model: str, company_info_raw: str, company_name: str) -> CompanyInfo:
    """Process raw company information into structured format.
    
    Args:
        model: The LLM model to use
        company_info_raw: Raw company information text
        company_name: Name of the company being analyzed
    
    Returns:
        CompanyInfo: Structured company information
    """
    logger.debug("Processing company information")
    pass

@Nodes.define(output="recent_news")
async def gather_company_news(company_name: str) -> dict:
    linkup = LinkupTool()
    query = f"{company_name} latest news last 3 months"
    result = linkup.execute(query=query, output_type="searchResults")
    logger.info(f"Gathered recent news for {company_name}")
    return {"recent_news_raw": result}

@Nodes.define(output="competitors_info")
async def identify_competitors(company_name: str, industry: str) -> dict:
    linkup = LinkupTool()
    query = f"top competitors of {company_name} in {industry} market share analysis"
    result = linkup.execute(query=query, output_type="sourcedAnswer")
    logger.info(f"Identified competitors for {company_name}")
    return {"competitors_raw": result}

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_analyze_competitors.j2"),
    output="competitors",
    response_model=List[CompetitorAnalysis],
    prompt_file=get_template_path("prompt_analyze_competitors.j2"),
)
async def analyze_competitors(model: str, competitors_raw: str) -> List[CompetitorAnalysis]:
    logger.debug("Analyzing competitors")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_generate_strategies.j2"),
    output="strategies",
    response_model=List[MarketingStrategy],
    prompt_file=get_template_path("prompt_generate_strategies.j2"),
)
async def generate_marketing_strategies(
    model: str, 
    company_info: CompanyInfo, 
    competitors: List[CompetitorAnalysis],
    recent_news_raw: str
) -> List[MarketingStrategy]:
    logger.debug("Generating marketing strategies")
    pass

@Nodes.define(output="final_analysis")
async def compile_analysis(
    company_info: CompanyInfo,
    competitors: List[CompetitorAnalysis],
    strategies: List[MarketingStrategy]
) -> CompanyAnalysis:
    analysis = CompanyAnalysis(
        company_info=company_info,
        competitors=competitors,
        marketing_strategies=strategies,
        analysis_date=datetime.now().strftime("%Y-%m-%d")
    )
    return analysis

# Define the Workflow
workflow = (
    Workflow("gather_company_info")
    .then("process_company_info")
    .then("gather_company_news")
    .then("identify_competitors")
    .then("analyze_competitors")
    .then("generate_marketing_strategies")
    .then("compile_analysis")
)

def analyze_company(
    company_name: str,
    model: str = "gemini/gemini-2.0-flash",
    task_id: str = "default",
    _handle_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
):
    """
    Run a comprehensive company analysis workflow.
    
    Args:
        company_name: Name of the company to analyze
        model: LLM model to use for analysis
        task_id: Unique identifier for the task
        _handle_event: Optional event handler for workflow events
    
    Returns:
        CompanyAnalysis: Comprehensive analysis of the company
    """
    initial_context = {
        "company_name": company_name,
        "model": model,
    }

    logger.info(f"Starting company analysis for {company_name}")
    engine = workflow.build() 
    
    result = anyio.run(engine.run, initial_context)
    logger.info("Company analysis completed successfully 🎉")
    return result

if __name__ == "__main__":
    # Test the company analysis flow with Microsoft
    try:
        analysis = analyze_company("Microsoft")
        
        # Print the analysis results in a structured format
        print("\n📊 Company Analysis Results 📊")
        print("="*50)
        
        # Company Info
        print("\n🏢 Company Information:")
        print(f"Name: {analysis.company_info.name}")
        print(f"Industry: {analysis.company_info.industry}")
        print(f"Description: {analysis.company_info.description}")
        print("\nKey Metrics:")
        print(f"Revenue: {analysis.company_info.revenue}")
        print(f"Employee Count: {analysis.company_info.employee_count}")
        print(f"Market Cap: {analysis.company_info.market_cap}")
        print(f"Growth Rate: {analysis.company_info.growth_rate}")
        
        # Competitors
        print("\n🏆 Top Competitors:")
        for comp in analysis.competitors:
            print(f"\n{comp.company_name} (Market Share: {comp.market_share}%)")
            print("Strengths:", ", ".join(comp.strengths[:3]))
            print("Weaknesses:", ", ".join(comp.weaknesses[:3]))
        
        # Marketing Strategies
        print("\n📈 Marketing Strategies:")
        for strategy in analysis.marketing_strategies:
            print(f"\n{strategy.strategy_name}")
            print(f"Description: {strategy.description}")
            print(f"Target Audience: {', '.join(strategy.target_audience)}")
            print(f"Timeline: {strategy.timeline}")
        
        print("\n" + "="*50)
        print(f"Analysis Date: {analysis.analysis_date}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise
