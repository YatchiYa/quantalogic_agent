#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru>=0.7.2",
#     "litellm>=1.0.0",
#     "pydantic>=2.0.0",
#     "asyncio",
#     "jinja2>=3.1.0",
#     "py-zerox",
#     "pdf2image",
#     "pillow",
#     "quantalogic",
#     "instructor>=0.5.2",
#     "typer>=0.9.0",
#     "rich>=13.0.0",
#     "pyperclip>=1.8.2",
#     "python-dateutil>=2.8.2"
# ]
# ///

import asyncio
from collections.abc import Callable
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Optional, Dict

import pyperclip
import typer
from dateutil.parser import parse as parse_date
from loguru import logger
from pydantic import BaseModel, Field
from pyzerox import zerox
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from quantalogic.flow.flow import Nodes, Workflow
from ..service import event_observer

# Initialize Typer app and rich console
console = Console()

# Default models for different phases
DEFAULT_TEXT_EXTRACTION_MODEL = "gemini/gemini-2.0-flash"
DEFAULT_ANALYSIS_MODEL = "gemini/gemini-2.0-flash"
DEFAULT_WRITING_MODEL = "gemini/gemini-2.0-flash"

# Pydantic models for structured data
class TenderDates(BaseModel):
    submission_deadline: datetime
    start_date: Optional[datetime]
    clarification_deadline: Optional[datetime]
    evaluation_period: Optional[str]

class TechnicalRequirement(BaseModel):
    category: str
    description: str
    mandatory: bool
    specifications: List[str]

class ComplianceRequirement(BaseModel):
    category: str
    requirements: List[str]
    documentation_needed: List[str]

class TenderInfo(BaseModel):
    title: str
    reference_number: Optional[str]
    procuring_entity: str
    estimated_value: Optional[str]
    location: Optional[str]
    sector: str
    tender_type: str
    dates: TenderDates
    technical_requirements: List[TechnicalRequirement]
    compliance_requirements: List[ComplianceRequirement]
    eligibility_criteria: List[str]
    evaluation_criteria: List[str]

class FinancialRequirement(BaseModel):
    budget_range: Optional[str]
    payment_terms: List[str]
    financial_criteria: List[str]
    required_guarantees: List[str]
    currency: Optional[str]

class RiskAssessment(BaseModel):
    risk_category: str
    probability: str  # Low, Medium, High
    impact: str      # Low, Medium, High
    mitigation_strategies: List[str]

class RiskAssessmentResponse(BaseModel):
    risks: List[RiskAssessment] = Field(description="List of identified risks and their assessments")
    total_risks: int = Field(description="Total number of identified risks")
    risk_categories: List[str] = Field(description="Categories of risks identified")
    overall_risk_level: str = Field(description="Overall risk level assessment (Low/Medium/High)")

class RiskAssessmentList(BaseModel):
    risks: List[RiskAssessment]

class CompetitiveAnalysis(BaseModel):
    market_position: str
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]
    competitor_insights: List[str]

class ResourceRequirement(BaseModel):
    category: str  # Human Resources, Equipment, Infrastructure, etc.
    description: str
    quantity: Optional[int]
    qualifications: List[str]
    availability_period: str

class ResourceRequirementList(BaseModel):
    resources: List[ResourceRequirement]

class DetailedTenderAnalysis(BaseModel):
    financial_requirements: FinancialRequirement
    risks: List[RiskAssessment]
    competitive_analysis: CompetitiveAnalysis
    resource_requirements: List[ResourceRequirement]
    success_probability: str
    recommended_actions: List[str]
    critical_decision_factors: List[str]

class BudgetBreakdown(BaseModel):
    category: str
    estimated_cost: str
    cost_type: str  # Fixed, Variable, One-time, Recurring
    notes: List[str]

class BudgetSummary(BaseModel):
    total_estimated_cost: str
    contingency_percentage: float
    assumptions: List[str]
    cost_risks: List[str]

class BudgetItemList(BaseModel):
    items: List[BudgetBreakdown]

class DetailedBudgetAnalysis(BaseModel):
    total_estimated_cost: str
    contingency_percentage: float
    items: List[BudgetBreakdown]
    assumptions: List[str]
    cost_risks: List[str]

class TimelinePhase(BaseModel):
    phase_name: str
    duration: str
    start_offset: str
    dependencies: List[str]
    key_deliverables: List[str]
    critical_path: bool

class StakeholderInfo(BaseModel):
    category: str  # Internal, External, Key Decision Maker, etc.
    interests: List[str]
    influence_level: str  # High, Medium, Low
    engagement_strategy: List[str]
    key_concerns: List[str]

class ProjectTimeline(BaseModel):
    total_duration: str
    phases: List[TimelinePhase]
    key_milestones: List[str]
    critical_path_duration: str
    buffer_periods: List[str]

class StakeholderAnalysis(BaseModel):
    stakeholders: List[StakeholderInfo]
    key_relationships: List[str]
    communication_strategy: List[str]
    approval_requirements: List[str]

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
    elif ext in [".txt", ".doc", ".docx"]:
        return "text"
    elif ext == ".md":
        return "markdown"
    else:
        logger.error(f"Unsupported file type: {ext}")
        raise ValueError(f"Unsupported file type: {ext}")

# Node: Convert PDF to Text
@Nodes.define(output="document_content")
async def convert_pdf_to_text(
    file_path: str,
    model: str,
    custom_system_prompt: Optional[str] = None
) -> str:
    """Convert a PDF to text using a vision model."""
    if custom_system_prompt is None:
        custom_system_prompt = (
            "Convert this tender document to clean, structured text. "
            "Preserve all dates, requirements, specifications, and formatting. "
            "Pay special attention to tables, lists, and emphasized text. "
            "Maintain the hierarchical structure of sections and subsections."
        )

    try:
        zerox_result = await zerox(
            file_path=file_path,
            model=model,
            system_prompt=custom_system_prompt
        )
        return str(zerox_result)
    except Exception as e:
        logger.error(f"Error converting PDF to text: {e}")
        raise

# Node: Read Text File
@Nodes.define(output="document_content")
async def read_text_file(file_path: str) -> str:
    """Read content from a text file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        logger.error(f"Error reading text file: {e}")
        raise

# Node: Extract Tender Information
@Nodes.structured_llm_node(
    system_prompt="You are an expert in analyzing tender documents. Extract key information accurately and thoroughly.",
    output="tender_info",
    response_model=TenderInfo,
    prompt_template="""
Analyze the following tender document and extract all key information in a structured format.
Pay special attention to dates, requirements, and specifications.

Document Content:
{{document_content}}

Extract and structure the information according to the TenderInfo model, ensuring all dates are in ISO format (YYYY-MM-DD).
Be thorough and precise in identifying technical and compliance requirements.
"""
)
async def extract_tender_info(document_content: str) -> TenderInfo:
    """Extract structured tender information from the document content."""
    pass

# Node: Analyze Financial Requirements
@Nodes.structured_llm_node(
    system_prompt="You are a financial analyst specializing in tender evaluation.",
    output="financial_analysis",
    response_model=FinancialRequirement,
    prompt_template="""
Analyze the financial aspects of this tender document:

{{document_content}}

Focus on:
1. Budget constraints and ranges
2. Payment schedules and terms
3. Financial stability requirements
4. Required guarantees and bonds
5. Currency considerations and exchange risks

Provide a detailed financial requirements analysis.
"""
)
async def analyze_financial_requirements(document_content: str) -> FinancialRequirement:
    """Analyze financial requirements and constraints."""
    pass

# Node: Analyze Risks
@Nodes.structured_llm_node(
    system_prompt="You are a risk management expert specializing in tender analysis.",
    output="risk_analysis",
    response_model=RiskAssessmentResponse,
    prompt_template="""
Conduct a comprehensive risk assessment for this tender:

Tender Information:
- Title: {{tender_info.title}}
- Reference: {{tender_info.reference_number}}
- Type: {{tender_info.tender_type}}
- Location: {{tender_info.location}}
- Sector: {{tender_info.sector}}

Technical Requirements:
{% for req in tender_info.technical_requirements %}
- {{req.category}}: {{req.description}}
  {% for spec in req.specifications %}
  * {{spec}}
  {% endfor %}
{% endfor %}

Compliance Requirements:
{% for req in tender_info.compliance_requirements %}
- {{req.category}}:
  {% for r in req.requirements %}
  * {{r}}
  {% endfor %}
{% endfor %}

Document Content:
{{document_content}}

Analyze and identify risks in these categories:
1. Technical risks
2. Financial risks
3. Operational risks
4. Compliance risks
5. Market risks
6. Resource risks

For each risk provide:
- Risk category
- Probability (Low/Medium/High)
- Impact (Low/Medium/High)
- Specific mitigation strategies

Format the response as a structured assessment including:
1. List of detailed risk assessments
2. Total number of risks identified
3. List of risk categories found
4. Overall risk level assessment (Low/Medium/High)
"""
)
async def analyze_risks(
    tender_info: TenderInfo,
    document_content: str
) -> RiskAssessmentResponse:
    """Perform detailed risk assessment."""
    pass

# Node: Analyze Competitive Position
@Nodes.structured_llm_node(
    system_prompt="You are a market analysis expert specializing in competitive tender analysis.",
    output="competitive_analysis",
    response_model=CompetitiveAnalysis,
    prompt_template="""
Analyze the competitive landscape for this tender:

Tender Information:
{{tender_info}}

Document Content:
{{document_content}}

Provide:
1. Market positioning analysis
2. SWOT analysis
3. Competitor landscape insights
4. Unique value propositions
5. Market differentiators
"""
)
async def analyze_competitive_position(
    tender_info: TenderInfo,
    document_content: str
) -> CompetitiveAnalysis:
    """Analyze competitive position and market landscape."""
    pass

# Node: Analyze Resource Requirements
@Nodes.structured_llm_node(
    system_prompt="You are a resource planning expert specializing in tender requirements.",
    output="resource_analysis",
    response_model=ResourceRequirementList,
    prompt_template="""
Analyze the resource requirements for this tender:

Technical Requirements:
{{tender_info.technical_requirements}}

Document Content:
{{document_content}}

Identify all required:
1. Human resources and expertise
2. Equipment and machinery
3. Infrastructure needs
4. Software and technology requirements
5. Certifications and qualifications

For each resource requirement, provide:
- Category (HR, Equipment, Infrastructure, etc.)
- Detailed description
- Quantity (if applicable)
- Required qualifications
- Required availability period

Format each resource requirement with clear categorization and specifications.
"""
)
async def analyze_resource_requirements(
    tender_info: TenderInfo,
    document_content: str
) -> ResourceRequirementList:
    """Analyze required resources and capabilities."""
    pass

# Node: Generate Detailed Analysis
@Nodes.structured_llm_node(
    system_prompt="You are a tender evaluation expert providing strategic recommendations.",
    output="detailed_analysis",
    response_model=DetailedTenderAnalysis,
    prompt_template="""
Synthesize all analyses into a comprehensive evaluation:

Financial Analysis:
{{financial_analysis}}

Risk Assessment:
Total Risks: {{risk_analysis.total_risks}}
Overall Risk Level: {{risk_analysis.overall_risk_level}}
Risk Categories: {{risk_analysis.risk_categories}}
Detailed Risks:
{% for risk in risk_analysis.risks %}
- {{risk.risk_category}}: {{risk.description}}
  Probability: {{risk.probability}}
  Impact: {{risk.impact}}
  Mitigation:
  {% for strategy in risk.mitigation_strategies %}
  * {{strategy}}
  {% endfor %}
{% endfor %}

Competitive Analysis:
{{competitive_analysis}}

Resource Requirements:
{{resource_analysis}}

Provide:
1. Overall success probability
2. Recommended actions
3. Critical decision factors
4. Strategic recommendations
"""
)
async def generate_detailed_analysis(
    financial_analysis: FinancialRequirement,
    risk_analysis: RiskAssessmentResponse,
    competitive_analysis: CompetitiveAnalysis,
    resource_analysis: List[ResourceRequirement]
) -> DetailedTenderAnalysis:
    """Generate comprehensive tender analysis and recommendations."""
    pass

# Node: Generate Strategic Summary
@Nodes.llm_node(
    system_prompt="You are a strategic advisor creating executive briefings.",
    output="strategic_summary",
    prompt_template="""
Create a strategic executive briefing based on the detailed analysis:

{{detailed_analysis}}

Focus on:
1. Key strategic considerations
2. Critical success factors
3. Major risks and opportunities
4. Resource implications
5. Strategic recommendations

Format as a clear, actionable executive summary with bullet points and clear sections.
"""
)
async def generate_strategic_summary(detailed_analysis: DetailedTenderAnalysis) -> str:
    """Generate strategic executive summary."""
    pass

# Node: Generate Executive Summary
@Nodes.llm_node(
    system_prompt="You are a tender analysis expert who creates clear, concise executive summaries.",
    output="executive_summary",
    prompt_template="""
Create a professional executive summary for the following tender:

Title: {{tender_info.title}}
Reference: {{tender_info.reference_number}}
Entity: {{tender_info.procuring_entity}}

Focus on key aspects:
1. Project scope and objectives
2. Key dates and deadlines
3. Critical requirements
4. Value proposition
5. Key success factors

Format the summary in clear, professional language suitable for management review.
"""
)
async def generate_executive_summary(tender_info: TenderInfo) -> str:
    """Generate an executive summary of the tender."""
    pass

# Node: Generate Technical Analysis
@Nodes.llm_node(
    system_prompt="You are a technical expert analyzing tender specifications and requirements.",
    output="technical_analysis",
    prompt_template="""
Provide a detailed technical analysis of the tender requirements:

Technical Requirements:
{{tender_info.technical_requirements}}

Compliance Requirements:
{{tender_info.compliance_requirements}}

Include:
1. Detailed analysis of each technical requirement
2. Potential challenges and considerations
3. Critical success factors
4. Required expertise and resources
5. Risk assessment
"""
)
async def generate_technical_analysis(tender_info: TenderInfo) -> str:
    """Generate detailed technical analysis of the tender requirements."""
    pass

# Node: Generate Compliance Checklist
@Nodes.define(output="compliance_checklist")
async def generate_compliance_checklist(tender_info: TenderInfo) -> str:
    """Generate a detailed compliance checklist."""
    checklist = "# Tender Compliance Checklist\n\n"
    
    # Add tender identification
    checklist += f"## Tender Information\n"
    checklist += f"- Title: {tender_info.title}\n"
    checklist += f"- Reference: {tender_info.reference_number}\n"
    checklist += f"- Entity: {tender_info.procuring_entity}\n\n"
    
    # Add key dates
    checklist += f"## Key Dates\n"
    checklist += f"- Submission Deadline: {tender_info.dates.submission_deadline}\n"
    if tender_info.dates.start_date:
        checklist += f"- Start Date: {tender_info.dates.start_date}\n"
    if tender_info.dates.clarification_deadline:
        checklist += f"- Clarification Deadline: {tender_info.dates.clarification_deadline}\n\n"
    
    # Add compliance requirements
    checklist += "## Compliance Requirements\n"
    for req in tender_info.compliance_requirements:
        checklist += f"\n### {req.category}\n"
        for item in req.requirements:
            checklist += f"- [ ] {item}\n"
        if req.documentation_needed:
            checklist += "\nRequired Documentation:\n"
            for doc in req.documentation_needed:
                checklist += f"- [ ] {doc}\n"
    
    return checklist

# Node: Generate Full Report
@Nodes.define(output="full_report")
async def generate_full_report(
    tender_info: TenderInfo,
    executive_summary: str,
    technical_analysis: str,
    compliance_checklist: str,
    strategic_summary: str,
    detailed_analysis: DetailedTenderAnalysis,
    budget_analysis: DetailedBudgetAnalysis,
    timeline_analysis: ProjectTimeline,
    stakeholder_analysis: StakeholderAnalysis,
    mermaid_diagrams: str
) -> str:
    """Compile the full tender analysis report."""
    report = f"""# Tender Analysis Report

## Executive Summary
{executive_summary}

## Tender Overview
- **Title:** {tender_info.title}
- **Reference:** {tender_info.reference_number}
- **Entity:** {tender_info.procuring_entity}
- **Sector:** {tender_info.sector}
- **Type:** {tender_info.tender_type}
- **Location:** {tender_info.location}
- **Estimated Value:** {tender_info.estimated_value}

## Key Dates
- **Submission Deadline:** {tender_info.dates.submission_deadline}
- **Start Date:** {tender_info.dates.start_date}
- **Clarification Deadline:** {tender_info.dates.clarification_deadline}
- **Evaluation Period:** {tender_info.dates.evaluation_period}

## Technical Analysis
{technical_analysis}

## Compliance Requirements
{compliance_checklist}

## Strategic Analysis
{strategic_summary}

## Detailed Analysis

### Financial Requirements
- **Budget Range:** {detailed_analysis.financial_requirements.budget_range}
- **Currency:** {detailed_analysis.financial_requirements.currency}

#### Payment Terms
"""
    for term in detailed_analysis.financial_requirements.payment_terms:
        report += f"- {term}\n"

    report += "\n#### Financial Criteria\n"
    for criterion in detailed_analysis.financial_requirements.financial_criteria:
        report += f"- {criterion}\n"

    report += "\n### Risk Assessment\n"
    for risk in detailed_analysis.risks:
        report += f"""
#### {risk.risk_category}
- **Probability:** {risk.probability}
- **Impact:** {risk.impact}
- **Mitigation Strategies:**
"""
        for strategy in risk.mitigation_strategies:
            report += f"  - {strategy}\n"

    report += "\n### Competitive Analysis\n"
    report += f"""
#### Market Position
{detailed_analysis.competitive_analysis.market_position}

#### SWOT Analysis
- **Strengths:**
"""
    for strength in detailed_analysis.competitive_analysis.strengths:
        report += f"  - {strength}\n"
    
    report += "\n- **Weaknesses:**\n"
    for weakness in detailed_analysis.competitive_analysis.weaknesses:
        report += f"  - {weakness}\n"
    
    report += "\n- **Opportunities:**\n"
    for opportunity in detailed_analysis.competitive_analysis.opportunities:
        report += f"  - {opportunity}\n"
    
    report += "\n- **Threats:**\n"
    for threat in detailed_analysis.competitive_analysis.threats:
        report += f"  - {threat}\n"

    report += "\n## Budget Analysis\n"
    report += f"""
- **Total Estimated Cost:** {budget_analysis.total_estimated_cost}
- **Contingency Percentage:** {budget_analysis.contingency_percentage}%

### Budget Breakdown
"""
    for item in budget_analysis.items:
        report += f"""
#### {item.category}
- **Estimated Cost:** {item.estimated_cost}
- **Cost Type:** {item.cost_type}
- **Notes:**
"""
        for note in item.notes:
            report += f"  - {note}\n"

    report += "\n## Project Timeline\n"
    report += f"""
- **Total Duration:** {timeline_analysis.total_duration}
- **Critical Path Duration:** {timeline_analysis.critical_path_duration}

### Project Phases
"""
    for phase in timeline_analysis.phases:
        report += f"""
#### {phase.phase_name}
- **Duration:** {phase.duration}
- **Start Offset:** {phase.start_offset}
- **Critical Path:** {'Yes' if phase.critical_path else 'No'}
- **Key Deliverables:**
"""
        for deliverable in phase.key_deliverables:
            report += f"  - {deliverable}\n"

    report += "\n## Stakeholder Analysis\n"
    for stakeholder in stakeholder_analysis.stakeholders:
        report += f"""
### {stakeholder.category}
- **Influence Level:** {stakeholder.influence_level}
- **Interests:**
"""
        for interest in stakeholder.interests:
            report += f"  - {interest}\n"
        
        report += "\n**Engagement Strategy:**\n"
        for strategy in stakeholder.engagement_strategy:
            report += f"  - {strategy}\n"

    report += "\n## Visualizations\n"
    report += "\n### Project Timeline\n"
    report += "```mermaid\n"
    report += mermaid_diagrams
    report += "\n```\n"

    report += "\n## Evaluation Criteria\n"
    for criterion in tender_info.evaluation_criteria:
        report += f"- {criterion}\n"
    
    return report

# Node: Save Report
@Nodes.define(output="report_path")
async def save_report(full_report: str, file_path: str) -> str:
    """Save the full report to a file."""
    try:
        output_path = Path(file_path).with_suffix('.report.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        raise

# Node: Analyze Budget Summary
@Nodes.structured_llm_node(
    system_prompt="You are a financial expert specializing in budget analysis for tenders.",
    output="budget_summary",
    response_model=BudgetSummary,
    prompt_template="""
Analyze the budget summary for this tender:

Financial Requirements:
{{financial_analysis}}

Technical Requirements:
{{tender_info.technical_requirements}}

Document Content:
{{document_content}}

Provide:
1. Total estimated cost
2. Recommended contingency percentage
3. Key assumptions
4. Cost risks
"""
)
async def analyze_budget_summary(
    financial_analysis: FinancialRequirement,
    tender_info: TenderInfo,
    document_content: str
) -> BudgetSummary:
    """Generate budget summary analysis."""
    pass

# Node: Analyze Budget Items
@Nodes.structured_llm_node(
    system_prompt="You are a financial expert specializing in detailed cost breakdown analysis.",
    output="budget_items",
    response_model=BudgetItemList,
    prompt_template="""
Create a detailed budget breakdown for this tender:

Financial Requirements:
{{financial_analysis}}

Technical Requirements:
{{tender_info.technical_requirements}}

Document Content:
{{document_content}}

Break down the costs into specific items, including:
1. Category name
2. Estimated cost
3. Cost type (Fixed, Variable, One-time, Recurring)
4. Notes and justifications
"""
)
async def analyze_budget_items(
    financial_analysis: FinancialRequirement,
    tender_info: TenderInfo,
    document_content: str
) -> BudgetItemList:
    """Generate detailed budget items breakdown."""
    pass

# Node: Combine Budget Analysis
@Nodes.define(output="budget_analysis")
async def combine_budget_analysis(
    budget_summary: BudgetSummary,
    budget_items: BudgetItemList
) -> DetailedBudgetAnalysis:
    """Combine budget summary and items into complete analysis."""
    return DetailedBudgetAnalysis(
        total_estimated_cost=budget_summary.total_estimated_cost,
        contingency_percentage=budget_summary.contingency_percentage,
        items=budget_items.items,
        assumptions=budget_summary.assumptions,
        cost_risks=budget_summary.cost_risks
    )

# Node: Analyze Project Timeline
@Nodes.structured_llm_node(
    system_prompt="You are a project planning expert specializing in tender timeline analysis.",
    output="timeline_analysis",
    response_model=ProjectTimeline,
    prompt_template="""
Create a detailed project timeline analysis for this tender:

Tender Dates:
{{tender_info.dates}}

Technical Requirements:
{{tender_info.technical_requirements}}

Resource Requirements:
{{resource_analysis}}

Analyze and provide:
1. Detailed phase breakdown
2. Critical path identification
3. Key milestones and dependencies
4. Buffer periods and contingencies
5. Resource allocation timing
"""
)
async def analyze_project_timeline(
    tender_info: TenderInfo,
    resource_analysis: ResourceRequirementList
) -> ProjectTimeline:
    """Generate detailed project timeline analysis."""
    pass

# Node: Analyze Stakeholders
@Nodes.structured_llm_node(
    system_prompt="You are a stakeholder management expert specializing in tender analysis.",
    output="stakeholder_analysis",
    response_model=StakeholderAnalysis,
    prompt_template="""
Perform a comprehensive stakeholder analysis for this tender:

Tender Information:
{{tender_info}}

Document Content:
{{document_content}}

Analyze:
1. Key stakeholder identification
2. Interest and influence mapping
3. Engagement strategies
4. Communication requirements
5. Approval processes and dependencies
"""
)
async def analyze_stakeholders(
    tender_info: TenderInfo,
    document_content: str
) -> StakeholderAnalysis:
    """Generate detailed stakeholder analysis."""
    pass

# Node: Generate Visualization Diagrams
@Nodes.llm_node(
    system_prompt="You are a visualization expert creating tender analysis diagrams.",
    output="mermaid_diagrams",
    prompt_template="""
Create Mermaid diagram code for visualizing the tender analysis:

Timeline:
{{timeline_analysis}}

Stakeholders:
{{stakeholder_analysis}}

Create three diagrams:
1. Gantt chart for project timeline
2. Mindmap for stakeholder relationships
3. Flowchart for approval process

Use proper Mermaid syntax and ensure diagrams are clear and informative.
"""
)
async def generate_visualization_diagrams(
    timeline_analysis: ProjectTimeline,
    stakeholder_analysis: StakeholderAnalysis
) -> str:
    """Generate Mermaid diagrams for visual representation."""
    pass

# Create Workflow
def create_tender_analysis_workflow() -> Workflow:
    """Create a workflow for analyzing tender documents."""
    wf = Workflow("check_file_type")
    
    # Add base nodes
    wf.node("check_file_type")
    wf.node("convert_pdf_to_text", inputs_mapping={"model": "text_extraction_model"})
    wf.node("read_text_file")
    wf.node("extract_tender_info", inputs_mapping={"model": "analysis_model"})
    
    # Add advanced analysis nodes
    wf.node("analyze_financial_requirements", inputs_mapping={"model": "analysis_model"})
    wf.node("analyze_risks", inputs_mapping={"model": "analysis_model"})
    wf.node("analyze_competitive_position", inputs_mapping={"model": "analysis_model"})
    wf.node("analyze_resource_requirements", inputs_mapping={"model": "analysis_model"})
    wf.node("generate_detailed_analysis", inputs_mapping={"model": "analysis_model"})
    wf.node("generate_strategic_summary", inputs_mapping={"model": "writing_model"})
    
    # Add specialized analysis nodes
    wf.node("analyze_budget_summary", inputs_mapping={"model": "analysis_model"})
    wf.node("analyze_budget_items", inputs_mapping={"model": "analysis_model"})
    wf.node("combine_budget_analysis")
    wf.node("analyze_project_timeline", inputs_mapping={"model": "analysis_model"})
    wf.node("analyze_stakeholders", inputs_mapping={"model": "analysis_model"})
    wf.node("generate_visualization_diagrams", inputs_mapping={"model": "analysis_model"})
    
    # Add report generation nodes
    wf.node("generate_executive_summary", inputs_mapping={"model": "writing_model"})
    wf.node("generate_technical_analysis", inputs_mapping={"model": "writing_model"})
    wf.node("generate_compliance_checklist")
    wf.node("generate_full_report", inputs_mapping={
        "tender_info": "tender_info",
        "executive_summary": "executive_summary",
        "technical_analysis": "technical_analysis",
        "compliance_checklist": "compliance_checklist",
        "strategic_summary": "strategic_summary",
        "detailed_analysis": "detailed_analysis",
        "budget_analysis": "budget_analysis",
        "timeline_analysis": "timeline_analysis",
        "stakeholder_analysis": "stakeholder_analysis",
        "mermaid_diagrams": "mermaid_diagrams"
    })
    wf.node("save_report")
    
    # Define workflow structure
    wf.current_node = "check_file_type"
    wf.branch([
        ("convert_pdf_to_text", lambda ctx: ctx["file_type"] == "pdf"),
        ("read_text_file", lambda ctx: ctx["file_type"] in ["text", "markdown"])
    ])
    
    # Define transitions for content extraction
    wf.transitions["convert_pdf_to_text"] = [("extract_tender_info", None)]
    wf.transitions["read_text_file"] = [("extract_tender_info", None)]
    
    # Define transitions for advanced analysis
    wf.transitions["extract_tender_info"] = [("analyze_financial_requirements", None)]
    wf.transitions["analyze_financial_requirements"] = [("analyze_risks", None)]
    wf.transitions["analyze_risks"] = [("analyze_competitive_position", None)]
    wf.transitions["analyze_competitive_position"] = [("analyze_resource_requirements", None)]
    wf.transitions["analyze_resource_requirements"] = [("generate_detailed_analysis", None)]
    wf.transitions["generate_detailed_analysis"] = [("analyze_budget_summary", None)]
    wf.transitions["analyze_budget_summary"] = [("analyze_budget_items", None)]
    wf.transitions["analyze_budget_items"] = [("combine_budget_analysis", None)]
    wf.transitions["combine_budget_analysis"] = [("analyze_project_timeline", None)]
    wf.transitions["analyze_project_timeline"] = [("analyze_stakeholders", None)]
    wf.transitions["analyze_stakeholders"] = [("generate_visualization_diagrams", None)]
    wf.transitions["generate_visualization_diagrams"] = [("generate_strategic_summary", None)]
    
    # Define transitions for report generation
    wf.transitions["generate_strategic_summary"] = [("generate_executive_summary", None)]
    wf.transitions["generate_executive_summary"] = [("generate_technical_analysis", None)]
    wf.transitions["generate_technical_analysis"] = [("generate_compliance_checklist", None)]
    wf.transitions["generate_compliance_checklist"] = [("generate_full_report", None)]
    wf.transitions["generate_full_report"] = [("save_report", None)]

    return wf

# Main async function
async def analyze_tender(
    file_path: str,
    text_extraction_model: str = DEFAULT_TEXT_EXTRACTION_MODEL,
    analysis_model: str = DEFAULT_ANALYSIS_MODEL,
    writing_model: str = DEFAULT_WRITING_MODEL,
    output_dir: Optional[str] = None,
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Analyze a tender document and generate a detailed report."""
    try:
        # Prepare initial context
        initial_context = {
            "file_path": file_path,
            "text_extraction_model": text_extraction_model,
            "analysis_model": analysis_model,
            "writing_model": writing_model,
            "output_dir": output_dir or str(Path(file_path).parent)
        }

        # Create and run workflow
        workflow = create_tender_analysis_workflow()
        engine = workflow.build()

        # Add the event observer if _handle_event is provided
        if _handle_event:
            # Create a lambda to bind task_id to the observer
            bound_observer = lambda event: asyncio.create_task(
                event_observer(event, task_id=task_id, _handle_event=_handle_event)
            )
            engine.add_observer(bound_observer)
        
        console.print(f"\n[bold blue]Starting analysis of:[/] {file_path}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Analyzing tender document...", total=None)
            result = await engine.run(initial_context)
            progress.update(task, completed=True)
        
        # Display results
        console.print("\n[bold green]Tender Analysis Complete![/]")
        console.print(f"[green]✓ Full report saved to:[/] {result['report_path']}")
        
        # Preview executive summary
        if "executive_summary" in result:
            console.print("\n[bold blue]Executive Summary Preview:[/]")
            console.print(Panel(Markdown(result["executive_summary"]), border_style="blue"))
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to analyze tender document: {e}")
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise

async def main():
    """Main entry point for the tender analyzer."""
    # Example usage
    file_path = "/home/yarab/Bureau/trash_agents_tests/f1/docs/test/test.pdf"  # Replace with actual path
    try:
        result = await analyze_tender(
            file_path=file_path,
            output_dir="output"  # Optional output directory
        )
        logger.info("Analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(main())
