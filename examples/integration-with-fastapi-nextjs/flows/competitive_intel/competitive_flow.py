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
#     "pandas>=2.0.0",
#     "aiohttp>=3.9.0"
# ]
# ///

import asyncio
from collections.abc import Callable
import datetime
from typing import List, Optional, Dict, Any, Set
from pydantic import BaseModel, Field
import os
from loguru import logger
from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from quantalogic.tools.linkup_tool import LinkupTool
from jinja2 import Environment, FileSystemLoader

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def get_template_path(template_name: str) -> str:
    """Get the full path to a template file."""
    return os.path.join(TEMPLATES_DIR, template_name)

# Data Models
class CompanyProfile(BaseModel):
    """Company profile information"""
    name: str = Field(description="Company name")
    industry: str = Field(description="Industry sector")
    main_products: List[str] = Field(description="Main products or services")
    target_markets: List[str] = Field(description="Target markets")
    key_strengths: List[str] = Field(description="Key company strengths")
    known_strategies: List[str] = Field(description="Known business strategies")

class MarketSegment(BaseModel):
    """Market segment information"""
    name: str = Field(description="Name of the market segment")
    competitors: List[str] = Field(description="List of competitors in this segment")

class CompetitorScore(BaseModel):
    """Competitor relevance score"""
    name: str = Field(description="Competitor name")
    score: float = Field(description="Relevance score (0-1)", ge=0, le=1)

class CompetitorList(BaseModel):
    """List of identified competitors"""
    competitors: List[str] = Field(description="List of competitor names")
    segments: List[MarketSegment] = Field(description="Competitors grouped by market segment")
    scores: List[CompetitorScore] = Field(description="Relevance scores for competitors")

class BasicCompanyProfile(BaseModel):
    """Basic company profile"""
    company_type: str = Field(description="Type of company (e.g., Public, Private)")
    headquarters: str = Field(description="Company headquarters location")
    founded: str = Field(description="Year founded")
    employees: str = Field(description="Approximate number of employees")
    revenue: str = Field(description="Annual revenue if public")

class MarketPosition(BaseModel):
    """Company's market position"""
    global_rank: str = Field(description="Global market position")
    market_share: str = Field(description="Approximate market share")
    target_segments: List[str] = Field(description="Target market segments")
    geographic_presence: List[str] = Field(description="Key geographic markets")

class Activity(BaseModel):
    """Recent company activity"""
    activity_type: str = Field(description="Type of activity (e.g., Product Launch)")
    date: str = Field(description="Activity date")
    description: str = Field(description="Activity description")
    impact: str = Field(description="Potential market impact")

class Product(BaseModel):
    """Product or service information"""
    name: str = Field(description="Product/Service name")
    category: str = Field(description="Product category")
    description: str = Field(description="Brief description")
    market_position: str = Field(description="Market position for this product")

class SWOT(BaseModel):
    """SWOT analysis"""
    strengths: List[str] = Field(description="Key strengths")
    weaknesses: List[str] = Field(description="Key weaknesses")
    opportunities: List[str] = Field(description="Opportunities")
    threats: List[str] = Field(description="Threats")

class FinancialMetrics(BaseModel):
    """Key financial metrics"""
    revenue_growth: str = Field(description="YoY revenue growth")
    market_cap: str = Field(description="Market capitalization if public")
    r_and_d_investment: str = Field(description="R&D investment")

class CompetitorInfo(BaseModel):
    """Detailed competitor information"""
    name: str = Field(description="Competitor name")
    profile: BasicCompanyProfile = Field(description="Company profile")
    market_position: MarketPosition = Field(description="Market position analysis")
    recent_activities: List[Activity] = Field(description="Recent activities and news")
    products: List[Product] = Field(description="Products and services")
    swot: SWOT = Field(description="SWOT analysis")
    financials: Optional[FinancialMetrics] = Field(description="Financial metrics if available")

class CompetitorAnalysis(BaseModel):
    """Competitor analysis response"""
    competitors: List[CompetitorInfo] = Field(description="List of analyzed competitors")

class NewsImpactAnalysis(BaseModel):
    """Analysis of news impact"""
    market_impact: str = Field(description="Impact on the market")
    competitive_implications: str = Field(description="Implications for competition")
    industry_trends: List[str] = Field(description="Related industry trends")

class MarketNews(BaseModel):
    """Market and competitor news analysis"""
    timestamp: str = Field(description="News timestamp")
    company: str = Field(description="Company involved")
    title: str = Field(description="News title")
    summary: str = Field(description="News summary")
    source: str = Field(description="News source")
    impact_analysis: NewsImpactAnalysis = Field(description="Impact analysis")
    categories: List[str] = Field(description="News categories")

class MarketNewsResponse(BaseModel):
    """Market news response"""
    news_items: List[MarketNews] = Field(description="List of market news items")

class TrendImpact(BaseModel):
    """Impact assessment of a market trend"""
    market_impact: str = Field(description="Impact on the market")
    timeframe: str = Field(description="Expected timeframe")
    confidence_level: str = Field(description="Confidence level in the assessment")

class MarketTrend(BaseModel):
    """Individual market trend analysis"""
    name: str = Field(description="Name of the trend")
    category: str = Field(description="Category of the trend")
    description: str = Field(description="Detailed description")
    drivers: List[str] = Field(description="Key drivers of the trend")
    impact_assessment: TrendImpact = Field(description="Impact assessment")
    affected_companies: List[str] = Field(description="Companies affected by the trend")
    recommendations: List[str] = Field(description="Strategic recommendations")

class MetaAnalysis(BaseModel):
    """Meta-analysis of market trends"""
    key_insights: List[str] = Field(description="Key insights from trend analysis")
    market_outlook: str = Field(description="Overall market outlook")
    risk_factors: List[str] = Field(description="Key risk factors")
    opportunities: List[str] = Field(description="Key opportunities")

class MarketTrendsResponse(BaseModel):
    """Market trends analysis response"""
    trends: List[MarketTrend] = Field(description="List of identified market trends")
    meta_analysis: MetaAnalysis = Field(description="Meta-analysis of trends")

class StrategicRecommendation(BaseModel):
    """Strategic recommendation"""
    action: str = Field(description="Recommended action")
    priority: str = Field(description="Priority level (High/Medium/Low)")
    expected_impact: str = Field(description="Expected impact of the action")
    timeline: str = Field(description="Implementation timeline")
    resources_needed: List[str] = Field(description="Required resources")

class RiskAssessmentItem(BaseModel):
    """Risk assessment item"""
    risk_type: str = Field(description="Type of risk")
    description: str = Field(description="Risk description")
    severity: str = Field(description="Risk severity (High/Medium/Low)")
    likelihood: str = Field(description="Risk likelihood (High/Medium/Low)")
    mitigation_steps: List[str] = Field(description="Risk mitigation steps")

class Opportunity(BaseModel):
    """Market opportunity"""
    name: str = Field(description="Opportunity name")
    description: str = Field(description="Opportunity description")
    potential_impact: str = Field(description="Potential business impact")
    requirements: List[str] = Field(description="Requirements to capture opportunity")
    timeline: str = Field(description="Timeline to realize opportunity")

class CompetitiveReport(BaseModel):
    """Comprehensive competitive intelligence report"""
    company_name: str = Field(description="Company name")
    report_date: str = Field(description="Report generation date")
    executive_summary: List[str] = Field(description="Key findings and summary")
    competitors: List[CompetitorInfo] = Field(description="Detailed competitor analysis")
    market_news: List[MarketNews] = Field(description="Recent market news")
    market_trends: List[MarketTrend] = Field(description="Identified market trends")
    strategic_recommendations: List[StrategicRecommendation] = Field(description="Strategic recommendations")
    risk_assessment: List[RiskAssessmentItem] = Field(description="Risk assessment")
    opportunities: List[Opportunity] = Field(description="Identified opportunities")
    monitoring_priorities: List[str] = Field(description="Areas to monitor closely")

# Workflow Nodes
@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_competitor_identification.j2"),
    output="identified_competitors",
    response_model=CompetitorList,
    prompt_file=get_template_path("prompt_competitor_identification.j2")
)
async def identify_competitors(
    company_profile: CompanyProfile,
    linkup_tool: LinkupTool,
    model: str
) -> CompetitorList:
    """Identify main competitors for the company."""
    try:
        logger.info(f"Identifying competitors for {company_profile.name}")
        
        # Use Linkup to search for competitors
        query = f"main competitors of {company_profile.name} in {company_profile.industry}"
        results = linkup_tool.execute(query=query, output_type="sourcedAnswer")
        logger.debug(f"Linkup search results: {results}")
        
        # Process results with LLM
        pass

    except Exception as e:
        logger.error(f"Error in identify_competitors: {str(e)}")
        raise ValueError(f"Failed to identify competitors: {str(e)}")

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_competitor_analysis.j2"),
    prompt_file=get_template_path("prompt_competitor_analysis.j2"),
    response_model=CompetitorAnalysis,
    output_key="competitors"
)
async def analyze_competitors(
    competitors: CompetitorList,
    company_profile: CompanyProfile,
    linkup_tool: LinkupTool,
    model: str
) -> List[CompetitorInfo]:
    """Analyze each identified competitor."""
    try:
        logger.info("Analyzing competitors")
        competitor_details = []
        
        for competitor in competitors.competitors:
            try:
                # Get competitor's relevance score
                score = next((s.score for s in competitors.scores if s.name == competitor), 0.0)
                logger.info(f"Analyzing {competitor} (relevance score: {score})")
                
                # Use Linkup to gather detailed information
                queries = [
                    f"company profile and background of {competitor}",
                    f"market position and strategy of {competitor}",
                    f"products and services offered by {competitor}",
                    f"recent news and developments about {competitor} in {company_profile.industry}"
                ]
                
                results = []
                for query in queries:
                    try:
                        result = linkup_tool.execute(query=query, output_type="sourcedAnswer")
                        results.append(result)
                        logger.debug(f"Linkup search results for {competitor} - {query}: {result}")
                    except Exception as e:
                        logger.warning(f"Error executing query '{query}' for {competitor}: {str(e)}")
                        results.append(None)
                
                # Process results with LLM to create CompetitorInfo
                # The LLM will handle this based on the prompt template
                pass
                
            except Exception as e:
                logger.error(f"Error analyzing competitor {competitor}: {str(e)}")
                # Continue with next competitor
                continue
        
        return competitor_details

    except Exception as e:
        logger.error(f"Error in analyze_competitors: {str(e)}")
        raise ValueError(f"Failed to analyze competitors: {str(e)}")

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_market_news.j2"),
    prompt_file=get_template_path("prompt_market_news.j2"),
    response_model=MarketNewsResponse,
    output_key="news_items"
)
async def gather_market_news(
    company_profile: CompanyProfile,
    competitors: CompetitorList,
    linkup_tool: LinkupTool,
    model: str
) -> List[MarketNews]:
    """Gather and analyze recent market news."""
    try:
        logger.info("Gathering market news")
        
        # Use Linkup to gather news
        companies = [company_profile.name] + competitors.competitors
        news_items = []
        
        for company in companies:
            try:
                query = f"recent news and developments about {company} in {company_profile.industry}"
                results = linkup_tool.execute(query=query, output_type="searchResults")
                logger.debug(f"Linkup search results for {company} news: {results}")
                
                # Process results with LLM
                result = await Nodes.structured_llm_node.process(
                    system_prompt_file=get_template_path("system_market_news.j2"),
                    prompt_file=get_template_path("prompt_market_news.j2"),
                    response_model=MarketNewsResponse,
                    output_key="news_items",
                    company=company,
                    search_results=results,
                    model=model
                )
                
                if result and hasattr(result, 'news_items'):
                    news_items.extend(result.news_items)
                
            except Exception as e:
                logger.error(f"Error processing news for {company}: {str(e)}")
                continue
        
        # Sort news by timestamp (newest first)
        news_items.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Return the top N most recent news items
        max_news = 10
        return news_items[:max_news]
        
    except Exception as e:
        logger.error(f"Error in gather_market_news: {str(e)}")
        raise ValueError(f"Failed to gather market news: {str(e)}")

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_market_trends.j2"),
    prompt_file=get_template_path("prompt_market_trends.j2"),
    response_model=MarketTrendsResponse,
    output_key="trends"
)
async def analyze_market_trends(
    company_profile: CompanyProfile,
    competitors: CompetitorList,
    market_news: List[MarketNews],
    model: str
) -> List[MarketTrend]:
    """Analyze market trends based on news and competitor analysis."""
    try:
        logger.info("Analyzing market trends")
        
        # Process with LLM using the provided template
        result = await Nodes.structured_llm_node.process(
            system_prompt_file=get_template_path("system_market_trends.j2"),
            prompt_file=get_template_path("prompt_market_trends.j2"),
            response_model=MarketTrendsResponse,
            output_key="trends",
            company_profile=company_profile,
            competitors=competitors,
            market_news=market_news,
            model=model
        )
        
        if not result:
            logger.warning("No trends analysis result received")
            return []
            
        if not hasattr(result, 'trends'):
            logger.warning("Trends analysis result missing 'trends' attribute")
            return []
            
        logger.info(f"Successfully analyzed {len(result.trends)} market trends")
        return result.trends
        
    except Exception as e:
        logger.error(f"Error in analyze_market_trends: {str(e)}")
        raise ValueError(f"Failed to analyze market trends: {str(e)}")

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_competitive_report.j2"),
    prompt_file=get_template_path("prompt_competitive_report.j2"),
    response_model=CompetitiveReport,
    output="competitive_report",
    max_tokens=4096  # Increase max tokens
)
async def prepare_competitive_report(
    company_profile: CompanyProfile,
    competitor_details: CompetitorAnalysis,
    market_news: List[MarketNews],
    market_trends: List[MarketTrend],
    model: str
) -> CompetitiveReport:
    """Prepare the final competitive intelligence report."""
    try:
        logger.info("Preparing competitive intelligence report")
        
        # Process competitors in smaller batches
        processed_competitors = []
        batch_size = 3
        competitors = competitor_details.competitors
        
        for i in range(0, len(competitors), batch_size):
            batch = competitors[i:i + batch_size]
            try:
                # Process competitor batch
                batch_result = await Nodes.structured_llm_node.process(
                    system_prompt_file=get_template_path("system_competitor_analysis.j2"),
                    prompt_file=get_template_path("prompt_competitor_analysis.j2"),
                    response_model=CompetitorAnalysis,
                    output="competitors",
                    competitors=batch,
                    model=model
                )
                if batch_result and hasattr(batch_result, 'competitors'):
                    processed_competitors.extend(batch_result.competitors)
            except Exception as e:
                logger.warning(f"Error processing competitor batch {i}-{i+batch_size}: {str(e)}")
                continue
        
        # Process market news in batches
        processed_news = []
        for i in range(0, len(market_news), batch_size):
            news_batch = market_news[i:i + batch_size]
            try:
                batch_result = await Nodes.structured_llm_node.process(
                    system_prompt_file=get_template_path("system_market_news.j2"),
                    prompt_file=get_template_path("prompt_market_news.j2"),
                    response_model=MarketNewsResponse,
                    output="news_items",
                    news_items=news_batch,
                    model=model
                )
                if batch_result and hasattr(batch_result, 'news_items'):
                    processed_news.extend(batch_result.news_items)
            except Exception as e:
                logger.warning(f"Error processing news batch {i}-{i+batch_size}: {str(e)}")
                continue
        
        # Process market trends in batches
        processed_trends = []
        for i in range(0, len(market_trends), batch_size):
            trend_batch = market_trends[i:i + batch_size]
            try:
                batch_result = await Nodes.structured_llm_node.process(
                    system_prompt_file=get_template_path("system_market_trends.j2"),
                    prompt_file=get_template_path("prompt_market_trends.j2"),
                    response_model=MarketTrendsResponse,
                    output="trends",
                    trends=trend_batch,
                    model=model
                )
                if batch_result and hasattr(batch_result, 'trends'):
                    processed_trends.extend(batch_result.trends)
            except Exception as e:
                logger.warning(f"Error processing trends batch {i}-{i+batch_size}: {str(e)}")
                continue
        
        # Generate executive summary and recommendations
        summary_result = await Nodes.structured_llm_node.process(
            system_prompt_file=get_template_path("system_executive_summary.j2"),
            prompt_file=get_template_path("prompt_executive_summary.j2"),
            response_model=CompetitiveReport,
            output="executive_summary",
            company_profile=company_profile,
            competitors=processed_competitors[:5],  # Limit to top 5 competitors
            market_news=processed_news[:5],  # Limit to top 5 news items
            market_trends=processed_trends[:5],  # Limit to top 5 trends
            model=model
        )
        
        # Combine all results into final report
        report = CompetitiveReport(
            company_name=company_profile.name,
            report_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            executive_summary=summary_result.executive_summary if summary_result else [],
            competitors=processed_competitors,
            market_news=processed_news,
            market_trends=processed_trends,
            strategic_recommendations=summary_result.strategic_recommendations if summary_result else [],
            risk_assessment=summary_result.risk_assessment if summary_result else [],
            opportunities=summary_result.opportunities if summary_result else [],
            monitoring_priorities=summary_result.monitoring_priorities if summary_result else []
        )
        
        logger.info("Successfully generated competitive intelligence report")
        return report
        
    except Exception as e:
        logger.error(f"Error in prepare_competitive_report: {str(e)}")
        raise ValueError(f"Failed to prepare competitive report: {str(e)}")

def create_competitive_intel_workflow() -> Workflow:
    """Create the workflow for competitive intelligence."""
    workflow = (
        Workflow("identify_competitors")
        .then("analyze_competitors")
        .then("gather_market_news")
        .then("analyze_market_trends")
        .then("prepare_competitive_report")
    )
    
    workflow.node_input_mappings = {
        "identify_competitors": {
            "model": "llm_model",
            "company_profile": "company_profile",
            "linkup_tool": "linkup_tool"
        },
        "analyze_competitors": {
            "model": "llm_model",
            "company_profile": "company_profile",
            "linkup_tool": "linkup_tool"
        },
        "gather_market_news": {
            "model": "llm_model",
            "company_profile": "company_profile",
            "linkup_tool": "linkup_tool"
        },
        "analyze_market_trends": {
            "model": "llm_model",
            "company_profile": "company_profile"
        },
        "prepare_competitive_report": {
            "model": "llm_model",
            "company_profile": "company_profile"
        }
    }
    
    return workflow

def generate_markdown_report(report: CompetitiveReport) -> str:
    """Generate a markdown report using the template."""
    try:
        # Set up Jinja environment
        env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
        template = env.get_template("report_template.md")
        
        # Render the template
        return template.render(report=report)
    except Exception as e:
        logger.error(f"Error generating markdown report: {str(e)}")
        raise ValueError(f"Failed to generate markdown report: {str(e)}")

async def analyze_competition(
    company_name: str,
    industry: str,
    products: List[str],
    markets: List[str],
    strengths: List[str],
    strategies: List[str],
    llm_model: str = "gemini/gemini-2.0-flash",
    output_file: Optional[str] = None,
    _handle_event: Optional[Callable[[str, dict], None]] = None
) -> CompetitiveReport:
    """Run the complete competitive intelligence workflow."""
    
    # Initialize LinkupTool
    linkup_tool = LinkupTool()
    
    # Create company profile
    company_profile = CompanyProfile(
        name=company_name,
        industry=industry,
        main_products=products,
        target_markets=markets,
        key_strengths=strengths,
        known_strategies=strategies
    )
    
    initial_context = {
        "company_profile": company_profile,
        "llm_model": llm_model,
        "linkup_tool": linkup_tool
    }
    
    workflow = create_competitive_intel_workflow()
    engine = workflow.build()
    
    result = await engine.run(initial_context)
    
    # Generate markdown report
    markdown_report = generate_markdown_report(result["competitive_report"])
    
    # Save markdown report if output file is specified
    if output_file:
        output_md = output_file.replace('.json', '.md')
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        logger.info(f"Markdown report saved to: {output_md}")
    
    logger.info(f"Competitive analysis completed for: {company_name}")
    return result["competitive_report"]

def cli_analyze_competition(
    company_name: str,
    industry: str,
    products_file: str,
    markets_file: str,
    strengths_file: str,
    strategies_file: str,
    output_file: Optional[str] = None,
    model: str = "gemini/gemini-2.0-flash"
):
    """CLI wrapper for competitive analysis."""
    # Read input files
    def read_list_file(file_path: str) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    products = read_list_file(products_file)
    markets = read_list_file(markets_file)
    strengths = read_list_file(strengths_file)
    strategies = read_list_file(strategies_file)
    
    # Run analysis
    result = asyncio.run(analyze_competition(
        company_name=company_name,
        industry=industry,
        products=products,
        markets=markets,
        strengths=strengths,
        strategies=strategies,
        llm_model=model,
        output_file=output_file
    ))
    
    # Output results
    if output_file:
        # Save JSON report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.model_dump_json(indent=2))
            
        # Save markdown report
        output_md = output_file.replace('.json', '.md')
        markdown_report = generate_markdown_report(result)
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
            
        print(f"Reports saved to:\n- JSON: {output_file}\n- Markdown: {output_md}")
    else:
        print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    import typer
    typer.run(cli_analyze_competition)
