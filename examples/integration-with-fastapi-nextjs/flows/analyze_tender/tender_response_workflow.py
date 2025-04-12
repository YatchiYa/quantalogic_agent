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
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from quantalogic.flow.flow import Nodes, Workflow
from ..service import event_observer

# Initialize Rich console
console = Console()

# Default models for different phases
DEFAULT_ANALYSIS_MODEL = "gemini/gemini-2.0-flash"
DEFAULT_WRITING_MODEL = "gemini/gemini-2.0-flash"

# Additional models for structured data
class PersonnelDetails(BaseModel):
    role: str = Field(description="Role/position in the company")
    qualifications: str = Field(description="Professional qualifications and experience")
    expertise: str = Field(description="Areas of expertise")

class TechnicalSpecification(BaseModel):
    parameter: str = Field(description="Technical parameter name")
    value: str = Field(description="Technical parameter value")

class PaymentMilestone(BaseModel):
    milestone: str = Field(description="Payment milestone description")
    amount: str = Field(description="Payment amount")
    timeline: str = Field(description="Expected timeline")

class CostItem(BaseModel):
    item: str = Field(description="Cost item name")
    amount: str = Field(description="Cost amount")
    description: str = Field(description="Item description")

class StakeholderCommunication(BaseModel):
    stakeholder: str = Field(description="Stakeholder type/role")
    communication_methods: List[str] = Field(description="Communication methods for this stakeholder")

# Models for company profile
class CompanyExperience(BaseModel):
    project_name: str = Field(description="Name of the past project")
    client: str = Field(description="Name of the client")
    value: str = Field(description="Project value")
    completion_date: datetime = Field(description="Project completion date")
    description: str = Field(description="Brief project description")
    key_achievements: List[str] = Field(description="List of key achievements in the project")

class CompanyProfile(BaseModel):
    name: str = Field(description="Company name")
    established: int = Field(description="Year company was established")
    core_expertise: List[str] = Field(description="List of core expertise areas")
    key_personnel: List[PersonnelDetails] = Field(description="List of key personnel with their roles and qualifications")
    certifications: List[str] = Field(description="List of relevant certifications")
    past_experience: List[CompanyExperience] = Field(description="List of relevant past experience")
    annual_turnover: str = Field(description="Annual turnover of the company")
    unique_selling_points: List[str] = Field(description="List of unique selling points")
    employee_count: int = Field(description="Number of employees")
    key_differentiators: List[str] = Field(description="List of key competitive advantages")

# Models for technical response
class TechnicalSolution(BaseModel):
    component: str = Field(description="Name of the technical component")
    description: str = Field(description="Detailed description of the solution")
    innovation_points: List[str] = Field(description="List of innovative features or approaches")
    technical_specs: List[TechnicalSpecification] = Field(description="Technical specifications with details")
    compliance_details: List[str] = Field(description="List of compliance details")

class TechnicalSolutionResponse(BaseModel):
    solutions: List[TechnicalSolution] = Field(description="List of technical solutions")

class MethodologyPhase(BaseModel):
    name: str = Field(description="Name of the project phase")
    duration: str = Field(description="Duration of the phase")
    key_activities: List[str] = Field(description="List of key activities in this phase")
    deliverables: List[str] = Field(description="List of deliverables for this phase")
    resources: List[str] = Field(description="Required resources for this phase")
    quality_measures: List[str] = Field(description="Quality assurance measures")

class ProjectMethodology(BaseModel):
    approach: str = Field(description="Overall project approach description")
    phases: List[MethodologyPhase] = Field(description="List of project phases")
    quality_assurance: List[str] = Field(description="Quality assurance measures")
    risk_mitigation: List[str] = Field(description="Risk mitigation strategies")
    communication_plan: List[StakeholderCommunication] = Field(description="Communication plan by stakeholder type")

# Models for commercial response
class CostBreakdown(BaseModel):
    category: str = Field(description="Cost category name")
    items: List[CostItem] = Field(description="List of cost items with details")
    subtotal: str = Field(description="Subtotal for this category")
    notes: List[str] = Field(description="Additional notes and justifications")

class CommercialOffer(BaseModel):
    total_cost: str = Field(description="Total cost of the proposal")
    payment_schedule: List[PaymentMilestone] = Field(description="Payment milestones and amounts")
    cost_breakdown: List[CostBreakdown] = Field(description="Detailed cost breakdown")
    value_additions: List[str] = Field(description="Additional value propositions")
    commercial_terms: List[str] = Field(description="Key commercial terms and conditions")

# Models for presentation
class PresentationSection(BaseModel):
    title: str = Field(description="Section title")
    content: str = Field(description="Section content")
    key_points: List[str] = Field(description="Key points to highlight", default_factory=list)
    visuals: Optional[List[str]] = Field(description="Visual aids or diagrams", default_factory=list)

class ExecutivePresentation(BaseModel):
    sections: List[PresentationSection] = Field(description="Presentation sections", default_factory=list)
    key_messages: List[str] = Field(description="Key messages to convey", default_factory=list)
    visual_aids: List[str] = Field(description="List of visual aids to include", default_factory=list)

# Node: Load Company Profile
@Nodes.structured_llm_node(
    system_prompt="You are a company profile expert creating compelling company presentations.",
    output="company_profile",
    response_model=CompanyProfile,
    prompt_template="""
Create a detailed company profile highlighting our strengths for this tender:

Company Information:
- Name: QuantaLogic Solutions
- Established: 2020
- Core Focus: AI and Advanced Analytics Solutions

Include:
1. Core expertise areas
2. Key personnel with their qualifications
3. Relevant certifications
4. Past similar projects
5. Company size and turnover
6. Key differentiators

Format the response to emphasize our capabilities relevant to the tender requirements.
""",
    inputs_mapping={"model": "model"}
)
async def load_company_profile(model: str) -> CompanyProfile:
    """Generate company profile with relevant experience."""
    pass

# Node: Design Technical Solution
@Nodes.structured_llm_node(
    system_prompt="You are a technical solution architect specializing in tender responses.",
    output="technical_solution",
    response_model=TechnicalSolutionResponse,
    prompt_template="""
Design a comprehensive technical solution addressing the tender requirements:

Tender Requirements:
{tender_requirements}

Company Profile:
{company_profile}

Guidelines:
1. Break down the solution into clear components
2. For each component, provide:
   - Detailed description
   - Innovative features
   - Technical specifications
   - Compliance details
3. Ensure alignment with our company's expertise
4. Address all tender requirements
5. Highlight innovative approaches

Format the response as a list of technical components, each with complete details.
""",
    inputs_mapping={"tender_requirements": "tender_requirements", "company_profile": "company_profile", "model": "model"}
)
async def design_technical_solution(
    tender_requirements: str,
    company_profile: CompanyProfile,
    model: str
) -> TechnicalSolutionResponse:
    """Design technical solution meeting tender requirements."""
    pass

# Node: Develop Project Methodology
@Nodes.structured_llm_node(
    system_prompt="You are a project methodology expert specializing in tender responses.",
    output="project_methodology",
    response_model=ProjectMethodology,
    prompt_template="""
Develop a comprehensive project methodology:

Technical Solution:
{{technical_solution}}

Requirements:
{{tender_requirements}}

Create a methodology that includes:
1. Project approach and framework
2. Detailed phase breakdown
3. Quality assurance measures
4. Risk mitigation strategies
5. Communication and reporting plan
""",
    inputs_mapping={"model": "model"}
)
async def develop_project_methodology(
    technical_solution: TechnicalSolutionResponse,
    tender_requirements: str,
    model: str
) -> ProjectMethodology:
    """Develop project execution methodology."""
    pass

# Node: Prepare Commercial Offer
@Nodes.structured_llm_node(
    system_prompt="You are a commercial proposal expert creating competitive offers.",
    output="commercial_offer",
    response_model=CommercialOffer,
    prompt_template="""
Prepare a competitive commercial offer:

Technical Solution:
{{technical_solution}}

Project Methodology:
{{project_methodology}}

Create a detailed commercial proposal including:
1. Detailed cost breakdown
2. Payment schedule
3. Value additions
4. Commercial terms
5. Competitive advantages
""",
    inputs_mapping={"model": "model"}
)
async def prepare_commercial_offer(
    technical_solution: TechnicalSolutionResponse,
    project_methodology: ProjectMethodology,
    model: str
) -> CommercialOffer:
    """Prepare competitive commercial offer."""
    pass

# Node: Generate Win Strategy
@Nodes.llm_node(
    system_prompt="You are a tender strategy expert creating winning approaches.",
    output="win_strategy",
    prompt_template="""
Create a comprehensive win strategy:

Company Profile:
{{company_profile}}

Technical Solution:
{{technical_solution}}

Commercial Offer:
{{commercial_offer}}

Develop a strategy that:
1. Highlights our unique value proposition
2. Addresses potential competitor advantages
3. Emphasizes our differentiators
4. Outlines key win themes
5. Provides negotiation strategies
""",
    inputs_mapping={"model": "writing_model"}
)
async def generate_win_strategy(
    company_profile: CompanyProfile,
    technical_solution: TechnicalSolutionResponse,
    commercial_offer: CommercialOffer,
    model: str
) -> str:
    """Generate winning strategy."""
    pass

# Node: Create Executive Presentation
@Nodes.structured_llm_node(
    system_prompt="""You are a presentation expert creating compelling tender presentations.
Focus on creating clear, impactful sections that effectively communicate our value proposition.""",
    output="executive_presentation",
    response_model=ExecutivePresentation,
    prompt_template="""
Create an executive presentation for the tender:

Win Strategy:
{{win_strategy}}

Company Profile:
{% for expertise in company_profile.core_expertise %}
- {{expertise}}
{% endfor %}

Technical Solution Highlights:
{% for solution in technical_solution.solutions %}
- {{solution.component}}: {{solution.description}}
{% endfor %}

Commercial Highlights:
- Total Cost: {{commercial_offer.total_cost}}
- Value Additions:
{% for value in commercial_offer.value_additions %}
  - {{value}}
{% endfor %}

Create a presentation that:
1. Tells a compelling story
2. Highlights key differentiators
3. Demonstrates clear value
4. Addresses client needs
5. Includes impactful visuals

Structure each section with:
- Clear title
- Focused content
- Key bullet points
- Relevant visual aids
""",
    inputs_mapping={"model": "writing_model"}
)
async def create_executive_presentation(
    win_strategy: str,
    company_profile: CompanyProfile,
    technical_solution: TechnicalSolutionResponse,
    commercial_offer: CommercialOffer,
    model: str
) -> ExecutivePresentation:
    """Create executive presentation."""
    pass

# Node: Generate Process Diagrams
@Nodes.llm_node(
    system_prompt="You are a diagram expert creating clear process visualizations.",
    output="process_diagrams",
    prompt_template="""
Create Mermaid diagrams for the tender response:

Methodology:
{{project_methodology}}

Create diagrams for:
1. Project execution flow
2. Organization structure
3. Communication flow
4. Quality assurance process
5. Risk management approach

Use proper Mermaid syntax for each diagram type.
""",
    inputs_mapping={"model": "writing_model"}
)
async def generate_process_diagrams(
    project_methodology: ProjectMethodology,
    model: str
) -> str:
    """Generate process flow diagrams."""
    pass

# Node: Compile Final Response
@Nodes.define(output="final_response")
async def compile_final_response(
    company_profile: CompanyProfile,
    technical_solution: TechnicalSolutionResponse,
    project_methodology: ProjectMethodology,
    commercial_offer: CommercialOffer,
    win_strategy: str,
    executive_presentation: ExecutivePresentation,
    process_diagrams: str
) -> str:
    """Compile complete tender response."""
    response = """# Tender Response Document

## Executive Summary
"""
    # Add executive summary from win strategy
    response += f"{win_strategy}\n\n"

    # Add company profile
    response += "## Company Profile\n"
    response += f"### About {company_profile.name}\n"
    response += f"- Established: {company_profile.established}\n"
    response += "\n### Core Expertise\n"
    for expertise in company_profile.core_expertise:
        response += f"- {expertise}\n"

    # Add technical solution
    response += "\n## Technical Solution\n"
    for solution in technical_solution.solutions:
        response += f"\n### {solution.component}\n"
        response += f"{solution.description}\n\n"
        response += "#### Innovation Points\n"
        for point in solution.innovation_points:
            response += f"- {point}\n"

    # Add methodology
    response += "\n## Project Methodology\n"
    response += f"{project_methodology.approach}\n\n"
    response += "### Project Phases\n"
    for phase in project_methodology.phases:
        response += f"\n#### {phase.name}\n"
        response += f"Duration: {phase.duration}\n\n"
        response += "Key Activities:\n"
        for activity in phase.key_activities:
            response += f"- {activity}\n"

    # Add commercial section
    response += "\n## Commercial Proposal\n"
    response += f"Total Cost: {commercial_offer.total_cost}\n\n"
    response += "### Cost Breakdown\n"
    for breakdown in commercial_offer.cost_breakdown:
        response += f"\n#### {breakdown.category}\n"
        response += f"Subtotal: {breakdown.subtotal}\n"

    # Add process diagrams
    response += "\n## Process Diagrams\n"
    response += "```mermaid\n"
    response += process_diagrams
    response += "\n```\n"

    # Add presentation
    response += "\n## Presentation\n"
    for section in executive_presentation.sections:
        response += f"\n### {section.title}\n"
        response += f"{section.content}\n\n"
        response += "Key Points:\n"
        for point in section.key_points:
            response += f"- {point}\n"

    return response

# Node: Save Response
@Nodes.define(output="response_path")
async def save_response(
    final_response: str,
    output_dir: str
) -> str:
    """Save the tender response to a file."""
    try:
        output_path = Path(output_dir) / "tender_response.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_response)
        return str(output_path)
    except Exception as e:
        logger.error(f"Error saving response: {e}")
        raise

def create_tender_response_workflow() -> Workflow:
    """Create workflow for generating tender response."""
    wf = Workflow("load_company_profile")
    
    # Add nodes with model mapping
    wf.node("load_company_profile", inputs_mapping={"model": "model"})
    wf.node("design_technical_solution", inputs_mapping={"tender_requirements": "tender_requirements", "company_profile": "company_profile", "model": "model"})
    wf.node("develop_project_methodology", inputs_mapping={"model": "model"})
    wf.node("prepare_commercial_offer", inputs_mapping={"model": "model"})
    wf.node("generate_win_strategy", inputs_mapping={"model": "writing_model"})
    wf.node("create_executive_presentation", inputs_mapping={"model": "writing_model"})
    wf.node("generate_process_diagrams", inputs_mapping={"model": "writing_model"})
    wf.node("compile_final_response")
    wf.node("save_response")
    
    # Define workflow structure
    wf.transitions["load_company_profile"] = [("design_technical_solution", None)]
    wf.transitions["design_technical_solution"] = [("develop_project_methodology", None)]
    wf.transitions["develop_project_methodology"] = [("prepare_commercial_offer", None)]
    wf.transitions["prepare_commercial_offer"] = [("generate_win_strategy", None)]
    wf.transitions["generate_win_strategy"] = [("create_executive_presentation", None)]
    wf.transitions["create_executive_presentation"] = [("generate_process_diagrams", None)]
    wf.transitions["generate_process_diagrams"] = [("compile_final_response", None)]
    wf.transitions["compile_final_response"] = [("save_response", None)]
    
    return wf

async def generate_tender_response(
    tender_requirements_path: str,
    output_dir: Optional[str] = None,
    analysis_model: str = DEFAULT_ANALYSIS_MODEL,
    writing_model: str = DEFAULT_WRITING_MODEL,
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Generate a comprehensive tender response."""
    try:
        # Prepare initial context
        initial_context = {
            "tender_requirements_path": tender_requirements_path,
            "output_dir": output_dir or str(Path(tender_requirements_path).parent / "response"),
            "model": analysis_model,  # Default model for analysis nodes
            "writing_model": writing_model  # Model for content generation nodes
        }

        # Read tender requirements
        with open(tender_requirements_path, 'r', encoding='utf-8') as f:
            tender_requirements = f.read()
        initial_context["tender_requirements"] = tender_requirements

        # Create and run workflow
        workflow = create_tender_response_workflow()
        engine = workflow.build()
        # Add the event observer if _handle_event is provided
        if _handle_event:
            # Create a lambda to bind task_id to the observer
            bound_observer = lambda event: asyncio.create_task(
                event_observer(event, task_id=task_id, _handle_event=_handle_event)
            )
            engine.add_observer(bound_observer)
        
        console.print(f"\n[bold blue]Starting tender response generation[/]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Generating tender response...", total=None)
            result = await engine.run(initial_context)
            progress.update(task, completed=True)
        
        # Display results
        console.print("\n[bold green]Tender Response Generated![/]")
        console.print(f"[green]✓ Full response saved to:[/] {result['response_path']}")
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to generate tender response: {e}")
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise

async def main():
    """Main entry point for tender response generator."""
    # Example usage
    tender_requirements_path = "/home/yarab/Bureau/trash_agents_tests/f1/docs/test/test.report.md"  # Replace with actual path
    try:
        result = await generate_tender_response(
            tender_requirements_path=tender_requirements_path,
            output_dir="output/response"  # Optional output directory
        )
        logger.info("Response generation completed successfully")
        return result
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(main())
