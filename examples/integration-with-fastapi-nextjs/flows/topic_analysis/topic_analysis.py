## Topic Analysis Flow
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
app = typer.Typer(help="Analyze a topic and generate a comprehensive report")
console = Console()

# Default models for different analysis phases
DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"

# Output directory for reports
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "reports")

# Define Pydantic models for structured output
class TopicBasicInfo(BaseModel):
    """Basic information about a topic."""
    name: str
    category: str
    description: str
    key_aspects: List[str]
    related_fields: List[str]
    summary_paragraph: str = Field(default="", description="A comprehensive summary paragraph about the topic")
    historical_context: str = Field(default="", description="Brief historical context or evolution of the topic")
    importance_statement: str = Field(default="", description="Statement about why this topic is important")

class TopicResource(BaseModel):
    """Information about a resource related to the topic."""
    title: str
    type: str  # article, book, video, course, etc.
    author: Optional[str] = None
    source: str
    url: Optional[str] = None
    description: str
    key_points: List[str]
    publication_date: Optional[str] = None
    relevance_score: Optional[float] = None
    content_summary: Optional[str] = None

class TopicExpert(BaseModel):
    """Information about an expert in the topic."""
    name: str
    affiliation: Optional[str] = None
    expertise: List[str]
    contributions: List[str]
    contact: Optional[str] = None  # website, social media, etc.
    bio: Optional[str] = None
    notable_works: Optional[List[str]] = None
    influence_areas: Optional[List[str]] = None

class TopicTrend(BaseModel):
    """Information about a trend related to the topic."""
    name: str
    description: str
    timeframe: str  # current, emerging, declining, etc.
    impact_areas: List[str]
    key_drivers: List[str]
    prediction: Optional[str] = None
    supporting_evidence: Optional[List[str]] = None
    related_technologies: Optional[List[str]] = None

class TopicAnalysis(BaseModel):
    """Comprehensive analysis of a topic."""
    importance: str
    current_state: str
    challenges: List[str]
    opportunities: List[str]
    future_directions: str
    practical_applications: List[str]
    industry_impact: Optional[str] = None
    societal_implications: Optional[str] = None
    ethical_considerations: Optional[str] = None
    research_gaps: Optional[List[str]] = None

class TopicReport(BaseModel):
    """Complete topic report."""
    basic_info: TopicBasicInfo
    resources: List[TopicResource]
    experts: List[TopicExpert]
    trends: List[TopicTrend]
    analysis: TopicAnalysis
    report_date: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))
    executive_summary: Optional[str] = None

# Node 1: Search for Topic Overview
@Nodes.define(output="topic_overview_results")
async def search_topic_overview(topic_name: str) -> Dict:
    """Search for general information about the topic."""
    try:
        tool = LinkupEnhancedTool()
        query = f"{topic_name} overview definition explanation"
        question = f"What is {topic_name}? Provide a comprehensive overview including definition, key aspects, and related fields."
        
        results = await tool.async_execute(
            query=query,
            question=question,
            depth="standard",
            analysis_depth="standard",
            scrape_sources="true",
            max_sources_to_scrape="5",
            output_format="technical"
        )
        
        logger.info(f"Completed topic overview search for {topic_name}")
        return results
    except Exception as e:
        logger.error(f"Error searching for topic overview: {e}")
        raise

# Node 2: Search for Topic Resources
@Nodes.define(output="resources_results")
async def search_topic_resources(topic_name: str) -> Dict:
    """Search for resources related to the topic."""
    try:
        tool = DuckDuckGoSearchLLMTool()
        query = f"{topic_name} best resources articles books courses"
        question = f"What are the most valuable resources (articles, books, courses, videos) for learning about {topic_name}? Describe each resource and its key points."
        
        results = await tool.async_execute(
            query=query,
            question=question,
            max_results="7",
            search_type="text",
            scrape_content="true",
            output_format="technical"
        )
        
        logger.info(f"Completed resources search for {topic_name}")
        return results
    except Exception as e:
        logger.error(f"Error searching for topic resources: {e}")
        raise

# Node 3: Search for Topic Experts
@Nodes.define(output="experts_results")
async def search_topic_experts(topic_name: str) -> Dict:
    """Search for experts in the topic."""
    try:
        tool = LinkupEnhancedTool()
        query = f"{topic_name} leading experts researchers authorities"
        question = f"Who are the leading experts, researchers, or authorities on {topic_name}? Describe their expertise and contributions."
        
        results = await tool.async_execute(
            query=query,
            question=question,
            depth="standard",
            analysis_depth="standard",
            scrape_sources="true",
            max_sources_to_scrape="4",
            output_format="technical"
        )
        
        logger.info(f"Completed experts search for {topic_name}")
        return results
    except Exception as e:
        logger.error(f"Error searching for topic experts: {e}")
        raise

# Node 4: Search for Topic Trends
@Nodes.define(output="trends_results")
async def search_topic_trends(topic_name: str) -> Dict:
    """Search for current trends related to the topic."""
    try:
        tool = DuckDuckGoSearchLLMTool()
        query = f"{topic_name} current trends developments future directions"
        question = f"What are the current trends, recent developments, and future directions in {topic_name}? Describe each trend and its potential impact."
        
        results = await tool.async_execute(
            query=query,
            question=question,
            max_results="8",
            search_type="news",
            scrape_content="true",
            output_format="article"
        )
        
        logger.info(f"Completed trends search for {topic_name}")
        return results
    except Exception as e:
        logger.error(f"Error searching for topic trends: {e}")
        raise

# Node 5: Extract Topic Basic Info
@Nodes.structured_llm_node(
    system_prompt="You are an expert researcher tasked with extracting structured information about a topic.",
    output="basic_info",
    response_model=TopicBasicInfo,
    prompt_template="""
Extract basic information about the topic from the following search results:

TOPIC NAME: {{topic_name}}

OVERVIEW SEARCH RESULTS:
{{topic_overview_results.answer}}

Extract and structure the following information:
- Full topic name
- Category or field the topic belongs to
- Comprehensive description (3-4 sentences)
- 4-6 key aspects or components of the topic
- 3-5 related fields or disciplines
- A summary paragraph about the topic
- Brief historical context or evolution of the topic
- Statement about why this topic is important
""",
)
async def extract_basic_info(
    topic_name: str,
    topic_overview_results: Dict,
    model: str = DEFAULT_LLM_MODEL
) -> TopicBasicInfo:
    """Extract structured basic information about the topic."""
    pass

# Node 6: Extract Topic Resources
@Nodes.llm_node(
    system_prompt="You are an expert researcher tasked with extracting structured information about resources related to a topic.",
    output="resources_json",
    prompt_template="""
Extract information about resources related to the topic from the following search results:

TOPIC NAME: {{topic_name}}

RESOURCES SEARCH RESULTS:
{{resources_results.answer}}

Extract and structure information about the top 5-7 most valuable resources for learning about this topic.
For each resource, include:
- Resource title
- Type (article, book, video, course, website, etc.)
- Author or creator (if available)
- Source (publisher, platform, institution)
- URL (if available)
- Brief description
- 3-5 key points or takeaways from the resource
- Publication date (if available)
- Relevance score (if available)
- Content summary (if available)

Return the information in the following JSON format:
```json
[
  {
    "title": "Resource Title",
    "type": "article/book/video/etc",
    "author": "Author Name",
    "source": "Source Name",
    "url": "https://resource-url.com",
    "description": "Brief description",
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "publication_date": "YYYY-MM-DD",
    "relevance_score": 0.8,
    "content_summary": "Summary of the content"
  },
  ...
]
```
""",
)
async def extract_resources_to_json(
    topic_name: str,
    resources_results: Dict,
    model: str = DEFAULT_LLM_MODEL
) -> str:
    """Extract structured information about topic resources as JSON string."""
    pass

# Node 7: Convert Resources JSON to Objects
@Nodes.define(output="resources")
async def convert_resources_json_to_objects(resources_json: str) -> List[TopicResource]:
    """Convert JSON string to list of TopicResource objects."""
    try:
        import json
        
        # Clean the JSON string if it's wrapped in markdown code blocks
        cleaned_json = resources_json
        if cleaned_json.startswith("```json"):
            # Extract content between ```json and ``` markers
            start_marker = "```json"
            end_marker = "```"
            start_idx = cleaned_json.find(start_marker) + len(start_marker)
            end_idx = cleaned_json.rfind(end_marker)
            if start_idx > -1 and end_idx > start_idx:
                cleaned_json = cleaned_json[start_idx:end_idx].strip()
        
        # Parse the cleaned JSON
        resources_data = json.loads(cleaned_json)
        resources = []
        for item in resources_data:
            resource = TopicResource(
                title=item.get("title", "Unknown Resource"),
                type=item.get("type", "Unknown type"),
                author=item.get("author"),
                source=item.get("source", "Unknown source"),
                url=item.get("url"),
                description=item.get("description", "No description available"),
                key_points=item.get("key_points", []),
                publication_date=item.get("publication_date"),
                relevance_score=item.get("relevance_score"),
                content_summary=item.get("content_summary")
            )
            resources.append(resource)
        
        logger.info(f"Successfully converted {len(resources)} resources to structured objects")
        return resources
    except Exception as e:
        logger.error(f"Error converting resources JSON to objects: {e}")
        raise

# Node 8: Extract Topic Experts
@Nodes.llm_node(
    system_prompt="You are an expert researcher tasked with extracting structured information about experts in a topic.",
    output="experts_json",
    prompt_template="""
Extract information about experts in the topic from the following search results:

TOPIC NAME: {{topic_name}}

EXPERTS SEARCH RESULTS:
{{experts_results.answer}}

Extract and structure information about the top 4-6 leading experts or authorities in this topic.
For each expert, include:
- Full name
- Affiliation (university, company, organization, if available)
- Areas of expertise within the topic (2-4 specific areas)
- Key contributions or works (2-4 significant contributions)
- Contact information or public profile (website, social media, if available)
- Brief bio (if available)
- Notable works (if available)
- Influence areas (if available)

Return the information in the following JSON format:
```json
[
  {
    "name": "Expert Name",
    "affiliation": "Expert Affiliation",
    "expertise": ["Area 1", "Area 2", "Area 3"],
    "contributions": ["Contribution 1", "Contribution 2", "Contribution 3"],
    "contact": "https://expert-website.com",
    "bio": "Brief bio",
    "notable_works": ["Work 1", "Work 2", "Work 3"],
    "influence_areas": ["Area 1", "Area 2", "Area 3"]
  },
  ...
]
```
""",
)
async def extract_experts_to_json(
    topic_name: str,
    experts_results: Dict,
    model: str = DEFAULT_LLM_MODEL
) -> str:
    """Extract structured information about topic experts as JSON string."""
    pass

# Node 9: Convert Experts JSON to Objects
@Nodes.define(output="experts")
async def convert_experts_json_to_objects(experts_json: str) -> List[TopicExpert]:
    """Convert JSON string to list of TopicExpert objects."""
    try:
        import json
        
        # Clean the JSON string if it's wrapped in markdown code blocks
        cleaned_json = experts_json
        if cleaned_json.startswith("```json"):
            # Extract content between ```json and ``` markers
            start_marker = "```json"
            end_marker = "```"
            start_idx = cleaned_json.find(start_marker) + len(start_marker)
            end_idx = cleaned_json.rfind(end_marker)
            if start_idx > -1 and end_idx > start_idx:
                cleaned_json = cleaned_json[start_idx:end_idx].strip()
        
        # Parse the cleaned JSON
        experts_data = json.loads(cleaned_json)
        experts = []
        for item in experts_data:
            expert = TopicExpert(
                name=item.get("name", "Unknown Expert"),
                affiliation=item.get("affiliation"),
                expertise=item.get("expertise", []),
                contributions=item.get("contributions", []),
                contact=item.get("contact"),
                bio=item.get("bio"),
                notable_works=item.get("notable_works"),
                influence_areas=item.get("influence_areas")
            )
            experts.append(expert)
        
        logger.info(f"Successfully converted {len(experts)} experts to structured objects")
        return experts
    except Exception as e:
        logger.error(f"Error converting experts JSON to objects: {e}")
        raise

# Node 10: Extract Topic Trends
@Nodes.llm_node(
    system_prompt="You are an expert researcher tasked with extracting structured information about trends related to a topic.",
    output="trends_json",
    prompt_template="""
Extract information about current trends related to the topic from the following search results:

TOPIC NAME: {{topic_name}}

TRENDS SEARCH RESULTS:
{{trends_results.answer}}

Extract and structure information about the top 4-6 most significant current trends or developments in this topic.
For each trend, include:
- Trend name or title
- Brief description (1-2 sentences)
- Timeframe (current, emerging, declining, etc.)
- Areas impacted by this trend (2-4 areas)
- Key drivers or causes of this trend (2-4 factors)
- Prediction about the trend (if available)
- Supporting evidence for the trend (if available)
- Related technologies or innovations (if available)

Return the information in the following JSON format:
```json
[
  {
    "name": "Trend Name",
    "description": "Brief description",
    "timeframe": "current/emerging/declining",
    "impact_areas": ["Area 1", "Area 2", "Area 3"],
    "key_drivers": ["Driver 1", "Driver 2", "Driver 3"],
    "prediction": "Prediction about the trend",
    "supporting_evidence": ["Evidence 1", "Evidence 2", "Evidence 3"],
    "related_technologies": ["Technology 1", "Technology 2", "Technology 3"]
  },
  ...
]
```
""",
)
async def extract_trends_to_json(
    topic_name: str,
    trends_results: Dict,
    model: str = DEFAULT_LLM_MODEL
) -> str:
    """Extract structured information about topic trends as JSON string."""
    pass

# Node 11: Convert Trends JSON to Objects
@Nodes.define(output="trends")
async def convert_trends_json_to_objects(trends_json: str) -> List[TopicTrend]:
    """Convert JSON string to list of TopicTrend objects."""
    try:
        import json
        
        # Clean the JSON string if it's wrapped in markdown code blocks
        cleaned_json = trends_json
        if cleaned_json.startswith("```json"):
            # Extract content between ```json and ``` markers
            start_marker = "```json"
            end_marker = "```"
            start_idx = cleaned_json.find(start_marker) + len(start_marker)
            end_idx = cleaned_json.rfind(end_marker)
            if start_idx > -1 and end_idx > start_idx:
                cleaned_json = cleaned_json[start_idx:end_idx].strip()
        
        # Parse the cleaned JSON
        trends_data = json.loads(cleaned_json)
        trends = []
        for item in trends_data:
            trend = TopicTrend(
                name=item.get("name", "Unknown Trend"),
                description=item.get("description", "No description available"),
                timeframe=item.get("timeframe", "Unknown timeframe"),
                impact_areas=item.get("impact_areas", []),
                key_drivers=item.get("key_drivers", []),
                prediction=item.get("prediction"),
                supporting_evidence=item.get("supporting_evidence"),
                related_technologies=item.get("related_technologies")
            )
            trends.append(trend)
        
        logger.info(f"Successfully converted {len(trends)} trends to structured objects")
        return trends
    except Exception as e:
        logger.error(f"Error converting trends JSON to objects: {e}")
        raise

# Node 12: Generate Topic Analysis
@Nodes.structured_llm_node(
    system_prompt="""You are an expert researcher and analyst specializing in creating comprehensive, professional, and insightful reports.
Your analysis should be thorough, well-structured, and provide deep insights that would be valuable to professionals, researchers, and decision-makers.
Use an authoritative, clear, and professional tone throughout your analysis.
Support your points with evidence from the provided information.
Identify connections, patterns, and implications that might not be immediately obvious.
Provide balanced perspectives, considering multiple viewpoints where appropriate.""",
    output="topic_analysis",
    response_model=TopicAnalysis,
    prompt_template="""
Analyze the topic based on all the information collected:

TOPIC NAME: {{topic_name}}
CATEGORY: {{basic_info.category}}

BASIC INFO:
{{basic_info}}

RESOURCES:
{{resources if resources is not none else []}}

EXPERTS:
{{experts if experts is not none else []}}

TRENDS:
{{trends if trends is not none else []}}

WEBSITE INSIGHTS:
{{websites_content_results.combined_insights if websites_content_results is not none else "No website insights available."}}

Perform a comprehensive, professional-grade analysis of the topic including:

1. Importance:
   - Provide a detailed assessment of why this topic is important or significant in its field
   - Explain its relevance to various stakeholders and industries
   - Discuss its place in the broader context of its domain

2. Current State:
   - Deliver a comprehensive overview of the current state of knowledge or development
   - Analyze the maturity level of this field/topic
   - Identify the leading approaches, methodologies, or technologies
   - Discuss recent breakthroughs or significant developments

3. Challenges:
   - Identify 5-7 key challenges or obstacles in this topic area
   - For each challenge, provide context on why it's significant and its implications
   - Consider technical, practical, and theoretical challenges

4. Opportunities:
   - Identify 5-7 key opportunities or promising areas for advancement
   - For each opportunity, explain its potential impact and feasibility
   - Consider short-term and long-term opportunities

5. Future Directions:
   - Provide a detailed prediction of how this topic might evolve in the next 3-5 years
   - Identify potential paradigm shifts or transformative developments
   - Discuss factors that might accelerate or hinder progress

6. Practical Applications:
   - Identify 5-7 practical applications or implementations of this topic
   - For each application, describe its potential impact and current status
   - Include both existing and potential future applications

7. Industry Impact:
   - Analyze how this topic is affecting or will affect relevant industries
   - Identify which sectors are most likely to be disrupted or transformed
   - Discuss economic implications where relevant

8. Societal Implications:
   - Examine how this topic might impact society at large
   - Consider effects on daily life, work, education, etc.
   - Discuss potential benefits and risks to society

9. Ethical Considerations:
   - Identify ethical questions or concerns related to this topic
   - Discuss potential approaches to addressing these concerns
   - Consider different stakeholder perspectives

10. Research Gaps:
    - Identify 3-5 significant gaps in current research or understanding
    - Suggest potential approaches to address these gaps
    - Discuss why filling these gaps is important

Your analysis should be detailed, nuanced, and provide genuine insights beyond what's obvious from the source materials.
""",
)
async def generate_topic_analysis(
    topic_name: str,
    basic_info: TopicBasicInfo,
    resources: Optional[List[TopicResource]] = None,
    experts: Optional[List[TopicExpert]] = None,
    trends: Optional[List[TopicTrend]] = None,
    websites_content_results: Optional[Dict] = None,
    model: str = DEFAULT_LLM_MODEL
) -> TopicAnalysis:
    """Generate a comprehensive analysis of the topic."""
    pass

# Node 13: Generate Executive Summary
@Nodes.llm_node(
    system_prompt="""You are an expert at creating concise, compelling, and informative executive summaries.
Your summaries distill complex information into clear, actionable insights while maintaining professional language.
Focus on the most important aspects that decision-makers and professionals need to understand.
Use an authoritative, clear, and professional tone throughout your summary.""",
    output="executive_summary",
    prompt_template="""
Create a professional executive summary for a comprehensive report on the following topic:

TOPIC NAME: {{topic_name}}

BASIC INFORMATION:
{{basic_info}}

ANALYSIS HIGHLIGHTS:
Importance: {{topic_analysis.importance}}
Current State: {{topic_analysis.current_state}}
Key Challenges: {{topic_analysis.challenges}}
Key Opportunities: {{topic_analysis.opportunities}}
Future Directions: {{topic_analysis.future_directions}}
Practical Applications: {{topic_analysis.practical_applications}}

Write a concise yet comprehensive executive summary (400-600 words) that:
1. Introduces the topic and its significance
2. Summarizes the current state and key developments
3. Highlights the most important challenges and opportunities
4. Outlines future directions and implications
5. Provides key takeaways for professionals in this field

The executive summary should be professional, insightful, and valuable for decision-makers who need to quickly understand the topic's importance and implications.
""",
)
async def generate_executive_summary(
    topic_name: str,
    basic_info: TopicBasicInfo,
    topic_analysis: TopicAnalysis,
    model: str = DEFAULT_LLM_MODEL
) -> str:
    """Generate an executive summary for the topic report."""
    pass

# Node 14: Compile Full Report
@Nodes.define(output="full_report")
async def compile_full_report(
    basic_info: TopicBasicInfo,
    topic_analysis: TopicAnalysis,
    executive_summary: str,
    resources: Optional[List[TopicResource]] = None,
    experts: Optional[List[TopicExpert]] = None,
    trends: Optional[List[TopicTrend]] = None
) -> TopicReport:
    """Compile all components into a comprehensive topic report."""
    try:
        # Use empty lists for any missing components
        res = resources if resources is not None else []
        exp = experts if experts is not None else []
        trd = trends if trends is not None else []
        
        report = TopicReport(
            basic_info=basic_info,
            resources=res,
            experts=exp,
            trends=trd,
            analysis=topic_analysis,
            executive_summary=executive_summary
        )
        
        logger.info("Successfully compiled full topic report")
        return report
    except Exception as e:
        logger.error(f"Error compiling full report: {e}")
        raise

# Node 15: Generate Markdown Report
@Nodes.define(output="markdown_report")
async def generate_markdown_report(full_report: TopicReport) -> str:
    """Convert the full report to a well-formatted Markdown document."""
    try:
        topic_name = full_report.basic_info.name
        
        # Format the report as Markdown
        markdown = [
            f"# {topic_name} - Topic Analysis Report",
            f"*Report Date: {full_report.report_date}*",
            
            "## Executive Summary",
            full_report.executive_summary if full_report.executive_summary else "*No executive summary available.*",
            
            "## 1. Topic Overview",
            f"**Name:** {full_report.basic_info.name}",
            f"**Category:** {full_report.basic_info.category}",
            "",
            f"**Description:**",
            f"{full_report.basic_info.description}",
            "",
            "**Key Aspects:**"
        ]
        
        # Add key aspects
        for aspect in full_report.basic_info.key_aspects:
            markdown.append(f"- {aspect}")
        
        markdown.append("\n**Related Fields:**")
        for field in full_report.basic_info.related_fields:
            markdown.append(f"- {field}")
        
        markdown.append("\n**Summary Paragraph:**")
        markdown.append(full_report.basic_info.summary_paragraph)
        
        markdown.append("\n**Historical Context:**")
        markdown.append(full_report.basic_info.historical_context)
        
        markdown.append("\n**Importance Statement:**")
        markdown.append(full_report.basic_info.importance_statement)
        
        # Add resources
        markdown.append("\n## 2. Key Resources")
        for i, resource in enumerate(full_report.resources, 1):
            markdown.extend([
                f"### 2.{i}. {resource.title}",
                f"**Type:** {resource.type}",
                f"**Author:** {resource.author or 'N/A'}",
                f"**Source:** {resource.source}",
                f"**URL:** {resource.url or 'N/A'}",
                "",
                f"**Description:** {resource.description}",
                "",
                "**Key Points:**"
            ])
            for point in resource.key_points:
                markdown.append(f"- {point}")
            markdown.append("")
        
        # Add experts
        markdown.append("## 3. Leading Experts")
        for i, expert in enumerate(full_report.experts, 1):
            markdown.extend([
                f"### 3.{i}. {expert.name}",
                f"**Affiliation:** {expert.affiliation or 'N/A'}",
                f"**Contact:** {expert.contact or 'N/A'}",
                "",
                "**Areas of Expertise:**"
            ])
            for area in expert.expertise:
                markdown.append(f"- {area}")
            markdown.append("\n**Key Contributions:**")
            for contribution in expert.contributions:
                markdown.append(f"- {contribution}")
            markdown.append("")
        
        # Add trends
        markdown.append("## 4. Current Trends")
        for i, trend in enumerate(full_report.trends, 1):
            markdown.extend([
                f"### 4.{i}. {trend.name}",
                f"**Timeframe:** {trend.timeframe}",
                "",
                f"**Description:** {trend.description}",
                "",
                "**Impact Areas:**"
            ])
            for area in trend.impact_areas:
                markdown.append(f"- {area}")
            markdown.append("\n**Key Drivers:**")
            for driver in trend.key_drivers:
                markdown.append(f"- {driver}")
            markdown.append("")
        
        # Add analysis
        markdown.extend([
            "## 5. Topic Analysis",
            
            "### 5.1. Importance",
            full_report.analysis.importance,
            "",
            "### 5.2. Current State",
            full_report.analysis.current_state,
            "",
            "### 5.3. Challenges"
        ])
        for challenge in full_report.analysis.challenges:
            markdown.append(f"- {challenge}")
        
        markdown.append("\n### 5.4. Opportunities")
        for opportunity in full_report.analysis.opportunities:
            markdown.append(f"- {opportunity}")
        
        markdown.append("\n### 5.5. Future Directions")
        markdown.append(full_report.analysis.future_directions)
        
        markdown.append("\n### 5.6. Practical Applications")
        for application in full_report.analysis.practical_applications:
            markdown.append(f"- {application}")
        
        markdown.append("\n### 5.7. Industry Impact")
        markdown.append(full_report.analysis.industry_impact)
        
        markdown.append("\n### 5.8. Societal Implications")
        markdown.append(full_report.analysis.societal_implications)
        
        markdown.append("\n### 5.9. Ethical Considerations")
        markdown.append(full_report.analysis.ethical_considerations)
        
        markdown.append("\n### 5.10. Research Gaps")
        for gap in full_report.analysis.research_gaps:
            markdown.append(f"- {gap}")
        
        # Add footer
        markdown.extend([
            "",
            "---",
            "*This report was generated automatically using the Quantalogic Topic Analyzer.*",
            f"*Generated on: {full_report.report_date}*"
        ])
        
        markdown_content = "\n\n".join(markdown)
        logger.info(f"Successfully generated Markdown report for {topic_name}")
        return markdown_content
    except Exception as e:
        logger.error(f"Error generating Markdown report: {e}")
        raise

# Node 16: Save Report to File
@Nodes.define(output="report_file_path")
async def save_report_to_file(markdown_report: str, topic_name: str, output_dir: Optional[str] = None) -> str:
    """Save the Markdown report to a file."""
    try:
        # Create sanitized filename from topic name
        safe_topic_name = "".join(c if c.isalnum() else "_" for c in topic_name)
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        
        # Determine output directory
        if output_dir:
            output_dir = os.path.expanduser(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = os.getcwd()
        
        # Create file path
        file_path = os.path.join(output_dir, f"{safe_topic_name}_analysis_{date_str}.md")
        
        # Write to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        
        logger.info(f"Successfully saved report to: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving report to file: {e}")
        raise

# Node 17: Search for Key Websites Content
@Nodes.define(output="websites_content_results")
async def search_websites_content(topic_name: str, resources_results: Dict) -> Dict:
    """Search for and analyze content from key websites related to the topic."""
    try:
        # Extract top URLs from resources results if available
        urls = []
        if "answer" in resources_results:
            # Try to extract URLs from the answer text
            import re
            url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
            found_urls = re.findall(url_pattern, resources_results["answer"])
            urls.extend(found_urls[:3])  # Take up to 3 URLs
        
        # If no URLs found or not enough, use a generic search
        if len(urls) < 3:
            tool = LinkupEnhancedTool()
            query = f"{topic_name} official website authoritative source"
            question = f"What are the most authoritative websites about {topic_name}? Provide URLs to official sources."
            
            search_results = await tool.async_execute(
                query=query,
                question=question,
                depth="standard",
                analysis_depth="standard",
                scrape_sources="true",
                max_sources_to_scrape="3",
                output_format="technical"
            )
            
            # Extract URLs from the search results
            if "answer" in search_results:
                url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
                found_urls = re.findall(url_pattern, search_results["answer"])
                urls.extend(found_urls[:3 - len(urls)])  # Fill up to 3 URLs
        
        # Analyze content from each URL
        scraper_tool = WebScraperLLMTool()
        combined_results = {
            "websites": [],
            "combined_insights": ""
        }
        
        for url in urls:
            try:
                logger.info(f"Analyzing content from: {url}")
                
                # Define a specific question about the topic for this website
                question = f"What are the key insights, facts, and information about {topic_name} on this website? Focus on unique and authoritative information."
                
                # Scrape and analyze the website
                result = await scraper_tool.async_execute(
                    url=url,
                    query=question,
                    timeout="90.0"  # Longer timeout for thorough analysis
                )
                
                # Add to combined results
                combined_results["websites"].append({
                    "url": url,
                    "title": result.get("content_summary", {}).get("title", "Unknown"),
                    "insights": result.get("answer", "No insights available")
                })
                
            except Exception as e:
                logger.error(f"Error analyzing website {url}: {e}")
                combined_results["websites"].append({
                    "url": url,
                    "title": "Error",
                    "insights": f"Failed to analyze: {str(e)}"
                })
        
        # Combine insights from all websites
        insights = [f"From {site['url']}: {site['insights']}" for site in combined_results["websites"]]
        combined_results["combined_insights"] = "\n\n".join(insights)
        
        logger.info(f"Completed website content analysis for {topic_name}")
        return combined_results
    except Exception as e:
        logger.error(f"Error in website content analysis: {e}")
        return {"websites": [], "combined_insights": f"Error in analysis: {str(e)}"}

# Define the Workflow
def create_topic_analyzer_workflow() -> Workflow:
    """Create a workflow to analyze a topic and generate a comprehensive report."""
    wf = Workflow("search_topic_overview")
    
    # Add all nodes with input mappings
    wf.node("search_topic_overview")
    wf.node("search_topic_resources")
    wf.node("search_topic_experts")
    wf.node("search_topic_trends")
    wf.node("search_websites_content")
    wf.node("extract_basic_info")
    wf.node("extract_resources_to_json")
    wf.node("convert_resources_json_to_objects")
    wf.node("extract_experts_to_json")
    wf.node("convert_experts_json_to_objects")
    wf.node("extract_trends_to_json")
    wf.node("convert_trends_json_to_objects")
    wf.node("generate_topic_analysis")
    wf.node("generate_executive_summary")
    wf.node("compile_full_report")
    wf.node("generate_markdown_report")
    wf.node("save_report_to_file")
    
    # Define the workflow structure with explicit transitions
    wf.current_node = "search_topic_overview"
    
    # Define parallel searches
    wf.transitions["search_topic_overview"] = [
        ("extract_basic_info", lambda ctx: True),
        ("search_topic_resources", lambda ctx: True),
        ("search_topic_experts", lambda ctx: True),
        ("search_topic_trends", lambda ctx: True)
    ]
    
    # Extract resources after search
    wf.transitions["search_topic_resources"] = [
        ("extract_resources_to_json", lambda ctx: True),
        ("search_websites_content", lambda ctx: True)
    ]
    wf.transitions["extract_resources_to_json"] = [("convert_resources_json_to_objects", lambda ctx: True)]
    
    # Extract experts after search
    wf.transitions["search_topic_experts"] = [("extract_experts_to_json", lambda ctx: True)]
    wf.transitions["extract_experts_to_json"] = [("convert_experts_json_to_objects", lambda ctx: True)]
    
    # Extract trends after search
    wf.transitions["search_topic_trends"] = [("extract_trends_to_json", lambda ctx: True)]
    wf.transitions["extract_trends_to_json"] = [("convert_trends_json_to_objects", lambda ctx: True)]
    
    # After all extractions, generate analysis
    wf.transitions["extract_basic_info"] = [("generate_topic_analysis", lambda ctx: True)]
    wf.transitions["convert_resources_json_to_objects"] = [("generate_topic_analysis", lambda ctx: True)]
    wf.transitions["convert_experts_json_to_objects"] = [("generate_topic_analysis", lambda ctx: True)]
    wf.transitions["convert_trends_json_to_objects"] = [("generate_topic_analysis", lambda ctx: True)]
    wf.transitions["search_websites_content"] = [("generate_topic_analysis", lambda ctx: True)]
    
    # Generate executive summary after analysis
    wf.transitions["generate_topic_analysis"] = [("generate_executive_summary", lambda ctx: True)]
    
    # Compile full report after executive summary
    wf.transitions["generate_executive_summary"] = [("compile_full_report", lambda ctx: True)]
    
    # Generate markdown after full report
    wf.transitions["compile_full_report"] = [("generate_markdown_report", lambda ctx: True)]
    
    # Save to file after markdown generation
    wf.transitions["generate_markdown_report"] = [("save_report_to_file", lambda ctx: True)]
    
    return wf

# Function to Run the Workflow
async def analyze_topic(
    topic_name: str,
    model: str = DEFAULT_LLM_MODEL,
    output_dir: Optional[str] = None,
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Execute the workflow to analyze a topic and generate a report."""
    if output_dir:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Initial context
    initial_context = {
        "topic_name": topic_name,
        "model": model,
        "output_dir": output_dir
    }
    
    try:
        workflow = create_topic_analyzer_workflow()
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
            logger.warning("No topic report generated.")
            raise ValueError("Workflow completed but no report was generated.")
        
        logger.info("Topic analysis workflow completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during workflow execution: {e}")
        raise

async def display_results(markdown_report: str, report_file_path: str):
    """Async helper function to display results with animation."""
    console.print("\n[bold green]Topic Analysis Report Generated:[/]")
    console.print(Panel(Markdown(markdown_report[:2000] + "...\n\n[Full report in saved file]"), 
                        border_style="blue", 
                        title="Report Preview"))
    
    console.print(f"[green]✓ Full report saved to:[/] {report_file_path}")

@app.command()
def analyze(
    topic_name: Annotated[str, typer.Argument(help="Name of the topic to analyze")],
    model: Annotated[str, typer.Option(help="LLM model to use for analysis")] = DEFAULT_LLM_MODEL,
    output_dir: Annotated[Optional[str], typer.Option(help="Directory to save output files (supports ~ expansion)")] = None,
):
    """Analyze a topic and generate a comprehensive report."""
    try:
        with console.status(f"[bold blue]Analyzing {topic_name}...[/]"):
            result = asyncio.run(analyze_topic(
                topic_name=topic_name,
                model=model,
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
    topic_name = "Artificial Intelligence"
    model = DEFAULT_LLM_MODEL
    output_dir = None
    
    # Simple argument parsing
    if len(sys.argv) > 1:
        # Check if using the typer interface
        if sys.argv[1] == "analyze":
            app()
            sys.exit(0)
        
        # Otherwise use direct arguments
        topic_name = sys.argv[1]
        
        if len(sys.argv) > 2:
            model = sys.argv[2]
        
        if len(sys.argv) > 3:
            output_dir = sys.argv[3]
    
    # Get user input if no command line arguments provided
    if len(sys.argv) <= 1:
        print("Topic Analyzer - Generate comprehensive topic reports")
        print("-" * 50)
        topic_name = input("Enter topic name (default: Artificial Intelligence): ").strip() or topic_name
        use_default_model = input(f"Use default LLM model ({DEFAULT_LLM_MODEL})? (y/n): ").strip().lower()
        if use_default_model != "y" and use_default_model != "":
            model = input("Enter LLM model name: ").strip()
        output_dir_input = input("Enter output directory (optional, press Enter to use current directory): ").strip()
        if output_dir_input:
            output_dir = output_dir_input
    
    print(f"\nAnalyzing {topic_name}...")
    print(f"Using LLM model: {model}")
    print(f"Output directory: {output_dir or 'Current directory'}")
    print("-" * 50)
    
    try:
        # Run the analysis
        result = asyncio.run(analyze_topic(
            topic_name=topic_name,
            model=model,
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