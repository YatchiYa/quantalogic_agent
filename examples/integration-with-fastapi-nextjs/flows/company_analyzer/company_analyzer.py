## company analyzer flow 
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
from quantalogic.tools.google_packages.linkup_enhanced_tool import LinkupEnhancedTool
from quantalogic.tools.google_packages.duckduckgo_search_llm_tool_enhanced import DuckDuckGoSearchLLMTool
from quantalogic.tools.website_search.web_scraper_llm_tool import WebScraperLLMTool
from ..service import event_observer

# Initialize Typer app and rich console
app = typer.Typer(help="Analyze a company and generate a comprehensive report")
console = Console()

# Default models for different analysis phases
DEFAULT_LLM_MODEL = "gemini/gemini-2.0-flash"

# Define Pydantic models for structured output
class CompanyBasicInfo(BaseModel):
    """Basic information about a company."""
    name: str
    industry: str
    founded: str
    headquarters: str
    website: str
    description: str

class CompanyProduct(BaseModel):
    """Information about a company product or service."""
    name: str
    description: str
    key_features: List[str]
    target_market: str

class CompanyCompetitor(BaseModel):
    """Information about a company competitor."""
    name: str
    website: Optional[str] = None
    strengths: List[str]
    weaknesses: List[str]

class CompanyNews(BaseModel):
    """Recent news about a company."""
    title: str
    date: str
    summary: str
    source: str
    url: Optional[str] = None

class CompanyAnalysis(BaseModel):
    """Comprehensive analysis of a company."""
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]
    market_position: str
    future_outlook: str

class CompanyReport(BaseModel):
    """Complete company report."""
    basic_info: CompanyBasicInfo
    products_services: List[CompanyProduct]
    competitors: List[CompanyCompetitor]
    recent_news: List[CompanyNews]
    analysis: CompanyAnalysis
    report_date: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))

# Node 1: Search for Company Overview
@Nodes.define(output="company_overview_results")
async def search_company_overview(company_name: str) -> Dict:
    """Search for general information about the company."""
    try:
        tool = LinkupEnhancedTool()
        query = f"{company_name} company overview"
        question = f"What is {company_name}? Provide a comprehensive overview including industry, founding date, headquarters, and main business areas."
        
        results = await tool.async_execute(
            query=query,
            question=question,
            depth="standard",
            analysis_depth="standard",
            scrape_sources="true",
            max_sources_to_scrape="5",
            output_format="technical"
        )
        
        logger.info(f"Completed company overview search for {company_name}")
        return results
    except Exception as e:
        logger.error(f"Error searching for company overview: {e}")
        raise

# Node 2: Search for Company Products and Services
@Nodes.define(output="products_services_results")
async def search_products_services(company_name: str) -> Dict:
    """Search for information about the company's products and services."""
    try:
        tool = DuckDuckGoSearchLLMTool()
        query = f"{company_name} products services offerings"
        question = f"What are the main products and services offered by {company_name}? Describe each product/service and its key features."
        
        results = await tool.async_execute(
            query=query,
            question=question,
            max_results="7",
            search_type="text",
            scrape_content="true",
            output_format="technical"
        )
        
        logger.info(f"Completed products and services search for {company_name}")
        return results
    except Exception as e:
        logger.error(f"Error searching for products and services: {e}")
        raise

# Node 3: Search for Company Competitors
@Nodes.define(output="competitors_results")
async def search_competitors(company_name: str, industry: str) -> Dict:
    """Search for information about the company's competitors."""
    try:
        tool = LinkupEnhancedTool()
        query = f"{company_name} competitors {industry} market"
        question = f"Who are the main competitors of {company_name} in the {industry} industry? Compare their strengths and weaknesses."
        
        results = await tool.async_execute(
            query=query,
            question=question,
            depth="standard",
            analysis_depth="standard",
            scrape_sources="true",
            max_sources_to_scrape="4",
            output_format="technical"
        )
        
        logger.info(f"Completed competitors search for {company_name}")
        return results
    except Exception as e:
        logger.error(f"Error searching for competitors: {e}")
        raise

# Node 4: Search for Recent Company News
@Nodes.define(output="news_results")
async def search_recent_news(company_name: str) -> Dict:
    """Search for recent news about the company."""
    try:
        tool = DuckDuckGoSearchLLMTool()
        query = f"{company_name} recent news last 6 months"
        question = f"What are the most significant recent news stories about {company_name} from the last 6 months? Summarize each story and its implications."
        
        results = await tool.async_execute(
            query=query,
            question=question,
            max_results="8",
            search_type="news",
            scrape_content="true",
            output_format="article"
        )
        
        logger.info(f"Completed recent news search for {company_name}")
        return results
    except Exception as e:
        logger.error(f"Error searching for recent news: {e}")
        raise

# Node 5: Scrape Company Website
@Nodes.define(output="website_content")
async def scrape_company_website(website_url: str) -> Dict:
    """Scrape the company's official website for information."""
    try:
        tool = WebScraperLLMTool()
        query = "What are the main products, services, and company information presented on this website?"
        
        results = await tool.async_execute(
            url=website_url,
            query=query,
            max_retries="3",
            timeout="90.0"
        )
        
        logger.info(f"Completed scraping of company website: {website_url}")
        return results
    except Exception as e:
        logger.error(f"Error scraping company website: {e}")
        raise

# Node 6: Extract Company Basic Info
@Nodes.structured_llm_node(
    system_prompt="You are an expert business analyst tasked with extracting structured information about a company.",
    output="basic_info",
    response_model=CompanyBasicInfo,
    prompt_template="""
Extract basic information about the company from the following search results and website content:

COMPANY NAME: {{company_name}}
WEBSITE: {{website_url}}

OVERVIEW SEARCH RESULTS:
{{company_overview_results.answer}}

WEBSITE CONTENT:
{{website_content.answer}}

Extract and structure the following information:
- Full company name
- Primary industry
- Year founded
- Headquarters location
- Official website URL
- Brief company description (2-3 sentences)
""",
    inputs_mapping={"model": "llm_model"}
)
async def extract_basic_info(
    company_name: str,
    website_url: str,
    company_overview_results: Dict,
    website_content: Dict
) -> CompanyBasicInfo:
    """Extract structured basic information about the company."""
    pass

# Node 7: Extract Products and Services
@Nodes.llm_node(
    system_prompt="You are an expert product analyst tasked with extracting structured information about a company's products and services.",
    output="products_services_json",
    prompt_template="""
Extract information about the company's products and services from the following search results and website content:

COMPANY NAME: {{company_name}}

PRODUCTS/SERVICES SEARCH RESULTS:
{{products_services_results.answer}}

WEBSITE CONTENT:
{{website_content.answer}}

Extract and structure information about the top 3-5 products or services offered by the company.
For each product/service, include:
- Product/service name
- Brief description
- 3-5 key features or capabilities
- Target market or user base

Return the information in the following JSON format:
```json
[
  {
    "name": "Product Name",
    "description": "Brief description",
    "key_features": ["Feature 1", "Feature 2", "Feature 3"],
    "target_market": "Target market description"
  },
  ...
]
""",
    inputs_mapping={"model": "llm_model"}
)
async def extract_products_services_to_json(
    company_name: str,
    products_services_results: Dict,
    website_content: Dict
) -> str:
    """Extract structured information about the company's products and services as JSON string."""
    pass

# Node: Convert Products JSON to Objects
@Nodes.define(output="products_services")
async def convert_products_json_to_objects(products_services_json: str) -> List[CompanyProduct]:
    """Convert JSON string to list of CompanyProduct objects."""
    try:
        import json
        
        # Clean the JSON string if it's wrapped in markdown code blocks
        cleaned_json = products_services_json
        if cleaned_json.startswith("```json"):
            # Extract content between ```json and ``` markers
            start_marker = "```json"
            end_marker = "```"
            start_idx = cleaned_json.find(start_marker) + len(start_marker)
            end_idx = cleaned_json.rfind(end_marker)
            if start_idx > -1 and end_idx > start_idx:
                cleaned_json = cleaned_json[start_idx:end_idx].strip()
        
        # Parse the cleaned JSON
        products_data = json.loads(cleaned_json)
        products = []
        for item in products_data:
            product = CompanyProduct(
                name=item.get("name", "Unknown Product"),
                description=item.get("description", "No description available"),
                key_features=item.get("key_features", []),
                target_market=item.get("target_market", "Unknown market")
            )
            products.append(product)
        
        logger.info(f"Successfully converted {len(products)} products/services to structured objects")
        return products
    except Exception as e:
        logger.error(f"Error converting products JSON to objects: {e}")
        raise

# Node 8: Extract Competitors
@Nodes.llm_node(
    system_prompt="You are an expert competitive analyst tasked with extracting structured information about a company's competitors.",
    output="competitors_json",
    prompt_template="""
Extract information about the company's competitors from the following search results:

COMPANY NAME: {{company_name}}
INDUSTRY: {{basic_info.industry}}

COMPETITORS SEARCH RESULTS:
{{competitors_results.answer}}

Extract and structure information about the top 3-5 competitors of the company.
For each competitor, include:
- Competitor name
- Website (if available)
- 2-3 key strengths relative to {{company_name}}
- 2-3 key weaknesses relative to {{company_name}}

Return the information in the following JSON format:
```json
[
  {
    "name": "Competitor Name",
    "website": "https://competitor.com", 
    "strengths": ["Strength 1", "Strength 2"],
    "weaknesses": ["Weakness 1", "Weakness 2"]
  },
  ...
]
""",
    inputs_mapping={"model": "llm_model"}
)
async def extract_competitors_to_json(
    company_name: str,
    basic_info: CompanyBasicInfo,
    competitors_results: Dict
) -> str:
    """Extract structured information about the company's competitors as JSON string."""
    pass

# Node: Convert Competitors JSON to Objects
@Nodes.define(output="competitors")
async def convert_competitors_json_to_objects(competitors_json: str) -> List[CompanyCompetitor]:
    """Convert JSON string to list of CompanyCompetitor objects."""
    try:
        import json
        
        # Clean the JSON string if it's wrapped in markdown code blocks
        cleaned_json = competitors_json
        if cleaned_json.startswith("```json"):
            # Extract content between ```json and ``` markers
            start_marker = "```json"
            end_marker = "```"
            start_idx = cleaned_json.find(start_marker) + len(start_marker)
            end_idx = cleaned_json.rfind(end_marker)
            if start_idx > -1 and end_idx > start_idx:
                cleaned_json = cleaned_json[start_idx:end_idx].strip()
        
        # Parse the cleaned JSON
        competitors_data = json.loads(cleaned_json)
        competitors = []
        for item in competitors_data:
            competitor = CompanyCompetitor(
                name=item.get("name", "Unknown Competitor"),
                website=item.get("website"),
                strengths=item.get("strengths", []),
                weaknesses=item.get("weaknesses", [])
            )
            competitors.append(competitor)
        
        logger.info(f"Successfully converted {len(competitors)} competitors to structured objects")
        return competitors
    except Exception as e:
        logger.error(f"Error converting competitors JSON to objects: {e}")
        raise

# Node 9: Extract Recent News
@Nodes.llm_node(
    system_prompt="You are an expert business news analyst tasked with extracting structured information about recent company news.",
    output="recent_news_json",
    prompt_template="""
Extract information about recent news related to the company from the following search results:

COMPANY NAME: {{company_name}}

NEWS SEARCH RESULTS:
{{news_results.answer}}

Extract and structure information about the top 3-5 most significant recent news stories about the company.
For each news story, include:
- News title/headline
- Approximate date (format: YYYY-MM-DD or Month Year)
- Brief summary of the news (1-2 sentences)
- Source of the news
- URL (if available)

Return the information in the following JSON format:
```json
[
  {
    "title": "News Title",
    "date": "2025-01-01",
    "summary": "Brief summary of the news",
    "source": "News Source",
    "url": "https://news-source.com/article"
  },
  ...
]
""",
    inputs_mapping={"model": "llm_model"}
)
async def extract_recent_news_to_json(
    company_name: str,
    news_results: Dict
) -> str:
    """Extract structured information about recent company news as JSON string."""
    pass

# Node: Convert News JSON to Objects
@Nodes.define(output="recent_news")
async def convert_news_json_to_objects(recent_news_json: str) -> List[CompanyNews]:
    """Convert JSON string to list of CompanyNews objects."""
    try:
        import json
        
        # Clean the JSON string if it's wrapped in markdown code blocks
        cleaned_json = recent_news_json
        if cleaned_json.startswith("```json"):
            # Extract content between ```json and ``` markers
            start_marker = "```json"
            end_marker = "```"
            start_idx = cleaned_json.find(start_marker) + len(start_marker)
            end_idx = cleaned_json.rfind(end_marker)
            if start_idx > -1 and end_idx > start_idx:
                cleaned_json = cleaned_json[start_idx:end_idx].strip()
        
        # Parse the cleaned JSON
        news_data = json.loads(cleaned_json)
        news_items = []
        for item in news_data:
            news = CompanyNews(
                title=item.get("title", "Unknown News"),
                date=item.get("date", "Unknown date"),
                summary=item.get("summary", "No summary available"),
                source=item.get("source", "Unknown source"),
                url=item.get("url")
            )
            news_items.append(news)
        
        logger.info(f"Successfully converted {len(news_items)} news items to structured objects")
        return news_items
    except Exception as e:
        logger.error(f"Error converting news JSON to objects: {e}")
        raise

# Node 10: Generate Company Analysis
@Nodes.structured_llm_node(
    system_prompt="You are an expert business analyst tasked with performing a SWOT analysis and market assessment for a company.",
    output="company_analysis",
    response_model=CompanyAnalysis,
    prompt_template="""
Analyze the company based on all the information collected:

COMPANY NAME: {{company_name}}
INDUSTRY: {{basic_info.industry}}

BASIC INFO:
{{basic_info}}

PRODUCTS/SERVICES:
{{products_services if products_services is not none else []}}

COMPETITORS:
{{competitors if competitors is not none else []}}

RECENT NEWS:
{{recent_news if recent_news is not none else []}}

Perform a comprehensive analysis of the company including:
1. SWOT Analysis:
   - 3-5 key strengths of the company
   - 3-5 key weaknesses or challenges
   - 3-5 key opportunities in the market
   - 3-5 key threats or risks

2. Market Position:
   - Assessment of the company's current position in its market

3. Future Outlook:
   - Prediction of the company's future prospects (next 1-3 years)
""",
    inputs_mapping={"model": "llm_model"}
)
async def generate_company_analysis(
    company_name: str,
    basic_info: CompanyBasicInfo,
    products_services: Optional[List[CompanyProduct]] = None,
    competitors: Optional[List[CompanyCompetitor]] = None,
    recent_news: Optional[List[CompanyNews]] = None
) -> CompanyAnalysis:
    """Generate a comprehensive analysis of the company."""
    pass

# Node 11: Compile Full Report
@Nodes.define(output="full_report")
async def compile_full_report(
    basic_info: CompanyBasicInfo,
    company_analysis: CompanyAnalysis,
    products_services: Optional[List[CompanyProduct]] = None,
    competitors: Optional[List[CompanyCompetitor]] = None,
    recent_news: Optional[List[CompanyNews]] = None
) -> CompanyReport:
    """Compile all components into a comprehensive company report."""
    try:
        # Use empty lists for any missing components
        products = products_services if products_services is not None else []
        comps = competitors if competitors is not None else []
        news = recent_news if recent_news is not None else []
        
        report = CompanyReport(
            basic_info=basic_info,
            products_services=products,
            competitors=comps,
            recent_news=news,
            analysis=company_analysis
        )
        
        logger.info("Successfully compiled full company report")
        return report
    except Exception as e:
        logger.error(f"Error compiling full report: {e}")
        raise

# Node 12: Generate Markdown Report
@Nodes.define(output="markdown_report")
async def generate_markdown_report(full_report: CompanyReport) -> str:
    """Convert the full report to a well-formatted Markdown document."""
    try:
        company_name = full_report.basic_info.name
        
        # Format the report as Markdown
        markdown = [
            f"# {company_name} - Company Analysis Report",
            f"*Report Date: {full_report.report_date}*",
            
            "## 1. Company Overview",
            f"**Name:** {full_report.basic_info.name}",
            f"**Industry:** {full_report.basic_info.industry}",
            f"**Founded:** {full_report.basic_info.founded}",
            f"**Headquarters:** {full_report.basic_info.headquarters}",
            f"**Website:** {full_report.basic_info.website}",
            "",
            f"**Description:**",
            f"{full_report.basic_info.description}",
            
            "## 2. Products and Services",
        ]
        
        # Add products and services
        for i, product in enumerate(full_report.products_services, 1):
            markdown.extend([
                f"### 2.{i}. {product.name}",
                f"**Description:** {product.description}",
                "",
                "**Key Features:**"
            ])
            for feature in product.key_features:
                markdown.append(f"- {feature}")
            markdown.append(f"**Target Market:** {product.target_market}")
            markdown.append("")
        
        # Add competitors
        markdown.append("## 3. Competitive Landscape")
        for i, competitor in enumerate(full_report.competitors, 1):
            markdown.extend([
                f"### 3.{i}. {competitor.name}",
                f"**Website:** {competitor.website or 'N/A'}",
                "",
                "**Strengths:**"
            ])
            for strength in competitor.strengths:
                markdown.append(f"- {strength}")
            markdown.append("**Weaknesses:**")
            for weakness in competitor.weaknesses:
                markdown.append(f"- {weakness}")
            markdown.append("")
        
        # Add recent news
        markdown.append("## 4. Recent News")
        for i, news in enumerate(full_report.recent_news, 1):
            markdown.extend([
                f"### 4.{i}. {news.title}",
                f"**Date:** {news.date}",
                f"**Source:** {news.source}",
                f"**URL:** {news.url or 'N/A'}",
                "",
                f"**Summary:** {news.summary}",
                ""
            ])
        
        # Add SWOT analysis
        markdown.extend([
            "## 5. SWOT Analysis",
            
            "### 5.1. Strengths",
        ])
        for strength in full_report.analysis.strengths:
            markdown.append(f"- {strength}")
        
        markdown.append("\n### 5.2. Weaknesses")
        for weakness in full_report.analysis.weaknesses:
            markdown.append(f"- {weakness}")
        
        markdown.append("\n### 5.3. Opportunities")
        for opportunity in full_report.analysis.opportunities:
            markdown.append(f"- {opportunity}")
        
        markdown.append("\n### 5.4. Threats")
        for threat in full_report.analysis.threats:
            markdown.append(f"- {threat}")
        
        # Add market position and future outlook
        markdown.extend([
            "\n## 6. Market Position and Future Outlook",
            "### 6.1. Current Market Position",
            full_report.analysis.market_position,
            "",
            "### 6.2. Future Outlook",
            full_report.analysis.future_outlook,
            "",
            "---",
            "*This report was generated automatically using the Quantalogic Company Analyzer.*",
            f"*Generated on: {full_report.report_date}*"
        ])
        
        markdown_content = "\n\n".join(markdown)
        logger.info(f"Successfully generated Markdown report for {company_name}")
        return markdown_content
    except Exception as e:
        logger.error(f"Error generating Markdown report: {e}")
        raise

# Node 13: Save Report to File
@Nodes.define(output="report_file_path")
async def save_report_to_file(markdown_report: str, company_name: str, output_dir: Optional[str] = None) -> str:
    """Save the Markdown report to a file."""
    try:
        # Create sanitized filename from company name
        safe_company_name = "".join(c if c.isalnum() else "_" for c in company_name)
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        
        # Determine output directory
        if output_dir:
            output_dir = os.path.expanduser(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.getcwd()
        
        # Create file path
        file_path = os.path.join(output_dir, f"{safe_company_name}_analysis_{date_str}.md")
        
        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        
        logger.info(f"Successfully saved report to: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving report to file: {e}")
        raise

# Define the Workflow
def create_company_analyzer_workflow() -> Workflow:
    """Create a workflow to analyze a company and generate a comprehensive report."""
    wf = Workflow("search_company_overview")
    
    # Add all nodes with input mappings
    wf.node("search_company_overview")
    wf.node("search_products_services")
    wf.node("search_competitors")
    wf.node("search_recent_news")
    wf.node("scrape_company_website")
    wf.node("extract_basic_info", inputs_mapping={"model": "llm_model"})
    wf.node("extract_products_services_to_json", inputs_mapping={"model": "llm_model"})
    wf.node("convert_products_json_to_objects")
    wf.node("extract_competitors_to_json", inputs_mapping={"model": "llm_model"})
    wf.node("convert_competitors_json_to_objects")
    wf.node("extract_recent_news_to_json", inputs_mapping={"model": "llm_model"})
    wf.node("convert_news_json_to_objects")
    wf.node("generate_company_analysis", inputs_mapping={"model": "llm_model"})
    wf.node("compile_full_report")
    wf.node("generate_markdown_report")
    wf.node("save_report_to_file")
    
    # Define the workflow structure with explicit transitions
    wf.current_node = "search_company_overview"
    
    # Define parallel searches
    wf.transitions["search_company_overview"] = [
        ("search_products_services", None),
        ("search_recent_news", None),
        ("scrape_company_website", None)
    ]
    
    # After product search, start competitor search (needs industry info)
    wf.transitions["search_products_services"] = [("extract_basic_info", None)]
    
    # Extract basic info after overview search and website scraping
    wf.transitions["scrape_company_website"] = [("extract_basic_info", None)]
    
    # After basic info extraction, search for competitors and extract products
    wf.transitions["extract_basic_info"] = [
        ("search_competitors", None),
        ("extract_products_services_to_json", None)
    ]
    
    # Extract products after search
    wf.transitions["search_competitors"] = [("extract_competitors_to_json", None)]
    
    # Extract news after search
    wf.transitions["search_recent_news"] = [("extract_recent_news_to_json", None)]
    
    # Extract products after basic info
    wf.transitions["extract_products_services_to_json"] = [("convert_products_json_to_objects", None)]
    
    # After all extractions, generate analysis
    wf.transitions["convert_products_json_to_objects"] = [("generate_company_analysis", None)]
    wf.transitions["extract_competitors_to_json"] = [("convert_competitors_json_to_objects", None)]
    wf.transitions["convert_competitors_json_to_objects"] = [("generate_company_analysis", None)]
    wf.transitions["extract_recent_news_to_json"] = [("convert_news_json_to_objects", None)]
    wf.transitions["convert_news_json_to_objects"] = [("generate_company_analysis", None)]
    
    # Compile full report after analysis
    wf.transitions["generate_company_analysis"] = [("compile_full_report", None)]
    
    # Generate markdown after full report
    wf.transitions["compile_full_report"] = [("generate_markdown_report", None)]
    
    # Save to file after markdown generation
    wf.transitions["generate_markdown_report"] = [("save_report_to_file", None)]
    
    return wf

# Function to Run the Workflow
async def analyze_company(
    company_name: str,
    website_url: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    output_dir: Optional[str] = None,
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Execute the workflow to analyze a company and generate a report."""
    if output_dir:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Initial context
    initial_context = {
        "company_name": company_name,
        "website_url": website_url,
        "llm_model": llm_model,
        "output_dir": output_dir
    }
    
    try:
        workflow = create_company_analyzer_workflow()
        engine = workflow.build()
        
        # Add the event observer if _handle_event is provided
        if _handle_event:
            # Create a lambda to bind task_id to the observer
            bound_observer = lambda event: asyncio.create_task(
                event_observer(event, task_id=task_id, _handle_event=_handle_event)
            )
            engine.add_observer(bound_observer)
        
        result = await engine.run(initial_context)
        
        if "full_report" not in result:
            logger.warning("No company report generated.")
            raise ValueError("Workflow completed but no report was generated.")
        
        logger.info("Company analysis workflow completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise

async def display_results(markdown_report: str, report_file_path: str):
    """Async helper function to display results with animation."""
    console.print("\n[bold green]Company Analysis Report Generated:[/]")
    console.print(Panel(Markdown(markdown_report[:2000] + "...\n\n[Full report in saved file]"), 
                        border_style="blue", 
                        title="Report Preview"))
    
    console.print(f"[green]✓ Full report saved to:[/] {report_file_path}")

@app.command()
def analyze(
    company_name: Annotated[str, typer.Argument(help="Name of the company to analyze")],
    website_url: Annotated[str, typer.Argument(help="URL of the company's official website")],
    llm_model: Annotated[str, typer.Option(help="LLM model to use for analysis")] = DEFAULT_LLM_MODEL,
    output_dir: Annotated[Optional[str], typer.Option(help="Directory to save output files (supports ~ expansion)")] = None,
):
    """Analyze a company and generate a comprehensive report."""
    try:
        with console.status(f"[bold blue]Analyzing {company_name}...[/]"):
            result = asyncio.run(analyze_company(
                company_name=company_name,
                website_url=website_url,
                llm_model=llm_model,
                output_dir=output_dir
            ))
        
        markdown_report = result["markdown_report"]
        report_file_path = result["report_file_path"]
        
        # Run the async display function
        asyncio.run(display_results(markdown_report, report_file_path))
        
    except Exception as e:
        logger.error(f"Failed to run workflow: {e}")
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    # Direct testing without Typer
    import sys
    
    # Default values
    company_name = "Microsoft"
    website_url = "https://www.microsoft.com"
    llm_model = DEFAULT_LLM_MODEL
    output_dir = None
    
    # Simple argument parsing
    if len(sys.argv) > 1:
        # Check if using the typer interface
        if sys.argv[1] == "analyze":
            app()
            sys.exit(0)
        
        # Otherwise use direct arguments
        company_name = sys.argv[1]
        
        if len(sys.argv) > 2:
            website_url = sys.argv[2]
        
        if len(sys.argv) > 3:
            llm_model = sys.argv[3]
        
        if len(sys.argv) > 4:
            output_dir = sys.argv[4]
    
    # Get user input if no command line arguments provided
    if len(sys.argv) <= 1:
        print("Company Analyzer - Generate comprehensive company reports")
        print("-" * 50)
        company_name = input("Enter company name (default: Microsoft): ").strip() or company_name
        website_url = input(f"Enter company website URL (default: {website_url}): ").strip() or website_url
        use_default_model = input(f"Use default LLM model ({DEFAULT_LLM_MODEL})? (y/n): ").strip().lower()
        if use_default_model != "y" and use_default_model != "":
            llm_model = input("Enter LLM model name: ").strip()
        output_dir_input = input("Enter output directory (optional, press Enter to use current directory): ").strip()
        if output_dir_input:
            output_dir = output_dir_input
    
    print(f"\nAnalyzing {company_name} ({website_url})...")
    print(f"Using LLM model: {llm_model}")
    print(f"Output directory: {output_dir or 'Current directory'}")
    print("-" * 50)
    
    try:
        # Run the analysis
        result = asyncio.run(analyze_company(
            company_name=company_name,
            website_url=website_url,
            llm_model=llm_model,
            output_dir=output_dir
        ))
        
        # Display results
        markdown_report = result["markdown_report"]
        report_file_path = result["report_file_path"]
        
        asyncio.run(display_results(markdown_report, report_file_path))
        
        print(f"\nAnalysis complete! Report saved to: {report_file_path}")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)