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
#     "python-docx>=0.8.11"
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
class ExecutiveSummary(BaseModel):
    """Executive summary section of the tender response"""
    understanding: str = Field(description="Understanding of requirements")
    approach: str = Field(description="High-level approach")
    value_proposition: str = Field(description="Key value propositions")
    differentiators: List[str] = Field(description="Key differentiators")
    benefits: List[str] = Field(description="Client benefits")

class MethodologySection(BaseModel):
    """Detailed methodology and approach"""
    overall_approach: str = Field(description="Overall methodology")
    phases: List[Dict[str, Any]] = Field(description="Project phases")
    tools_technologies: List[Dict[str, Any]] = Field(description="Tools and technologies")
    quality_assurance: Dict[str, Any] = Field(description="Quality assurance approach")
    risk_management: List[Dict[str, Any]] = Field(description="Risk management strategy")

class TeamStructure(BaseModel):
    """Team structure and resources"""
    key_personnel: List[Dict[str, Any]] = Field(description="Key team members")
    roles_responsibilities: List[Dict[str, Any]] = Field(description="Roles and responsibilities")
    expertise: List[Dict[str, Any]] = Field(description="Team expertise")
    availability: Dict[str, Any] = Field(description="Resource availability")
    backup_plan: Dict[str, Any] = Field(description="Backup and contingency plans")

class Deliverables(BaseModel):
    """Project deliverables and timeline"""
    milestones: List[Dict[str, Any]] = Field(description="Project milestones")
    outputs: List[Dict[str, Any]] = Field(description="Deliverable outputs")
    timeline: Dict[str, Any] = Field(description="Delivery timeline")
    acceptance_criteria: List[str] = Field(description="Acceptance criteria")

class PricingDetails(BaseModel):
    """Pricing and commercial details"""
    cost_breakdown: Dict[str, Any] = Field(description="Detailed cost breakdown")
    payment_schedule: List[Dict[str, Any]] = Field(description="Payment milestones")
    assumptions: List[str] = Field(description="Pricing assumptions")
    additional_costs: Optional[Dict[str, Any]] = Field(description="Any additional costs")

class ComplianceMatrix(BaseModel):
    """Compliance with tender requirements"""
    technical_compliance: List[Dict[str, Any]] = Field(description="Technical requirements compliance")
    functional_compliance: List[Dict[str, Any]] = Field(description="Functional requirements compliance")
    legal_compliance: List[Dict[str, Any]] = Field(description="Legal requirements compliance")
    deviations: Optional[List[Dict[str, Any]]] = Field(description="Any deviations or non-compliance")

class TenderResponse(BaseModel):
    """Complete tender response document"""
    tender_id: str = Field(description="Tender identifier")
    executive_summary: ExecutiveSummary = Field(description="Executive summary")
    methodology: MethodologySection = Field(description="Methodology section")
    team: TeamStructure = Field(description="Team structure")
    deliverables: Deliverables = Field(description="Deliverables section")
    pricing: PricingDetails = Field(description="Pricing details")
    compliance: ComplianceMatrix = Field(description="Compliance matrix")
    appendices: List[Dict[str, Any]] = Field(description="Additional supporting documents")

# Workflow Nodes
@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_executive_summary.j2"),
    output="executive_summary",
    response_model=ExecutiveSummary,
    prompt_file=get_template_path("prompt_executive_summary.j2")
)
async def create_executive_summary(
    tender_analysis: Dict[str, Any],
    company_profile: Dict[str, Any],
    model: str
) -> ExecutiveSummary:
    """Create the executive summary section."""
    logger.info("Creating executive summary")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_methodology.j2"),
    output="methodology",
    response_model=MethodologySection,
    prompt_file=get_template_path("prompt_methodology.j2")
)
async def develop_methodology(
    tender_analysis: Dict[str, Any],
    company_capabilities: Dict[str, Any],
    model: str
) -> MethodologySection:
    """Develop the methodology section."""
    logger.info("Developing methodology section")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_team.j2"),
    output="team_structure",
    response_model=TeamStructure,
    prompt_file=get_template_path("prompt_team.j2")
)
async def define_team_structure(
    methodology: MethodologySection,
    available_resources: Dict[str, Any],
    model: str
) -> TeamStructure:
    """Define the team structure and resources."""
    logger.info("Defining team structure")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_deliverables.j2"),
    output="deliverables",
    response_model=Deliverables,
    prompt_file=get_template_path("prompt_deliverables.j2")
)
async def specify_deliverables(
    methodology: MethodologySection,
    tender_analysis: Dict[str, Any],
    model: str
) -> Deliverables:
    """Specify project deliverables and timeline."""
    logger.info("Specifying deliverables")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_pricing.j2"),
    output="pricing",
    response_model=PricingDetails,
    prompt_file=get_template_path("prompt_pricing.j2")
)
async def develop_pricing(
    deliverables: Deliverables,
    team_structure: TeamStructure,
    cost_data: Dict[str, Any],
    model: str
) -> PricingDetails:
    """Develop pricing and commercial details."""
    logger.info("Developing pricing details")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_compliance.j2"),
    output="compliance",
    response_model=ComplianceMatrix,
    prompt_file=get_template_path("prompt_compliance.j2")
)
async def create_compliance_matrix(
    tender_analysis: Dict[str, Any],
    methodology: MethodologySection,
    deliverables: Deliverables,
    model: str
) -> ComplianceMatrix:
    """Create the compliance matrix."""
    logger.info("Creating compliance matrix")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_final_response.j2"),
    output="final_response",
    response_model=TenderResponse,
    prompt_file=get_template_path("prompt_final_response.j2")
)
async def compile_final_response(
    tender_id: str,
    executive_summary: ExecutiveSummary,
    methodology: MethodologySection,
    team_structure: TeamStructure,
    deliverables: Deliverables,
    pricing: PricingDetails,
    compliance: ComplianceMatrix,
    supporting_docs: List[Dict[str, Any]],
    model: str
) -> TenderResponse:
    """Compile the final tender response."""
    logger.info("Compiling final response")
    pass

def create_response_workflow() -> Workflow:
    """Create the workflow for tender response generation."""
    workflow = (
        Workflow("create_executive_summary")
        .then("develop_methodology")
        .then("define_team_structure")
        .then("specify_deliverables")
        .then("develop_pricing")
        .then("create_compliance_matrix")
        .then("compile_final_response")
    )
    
    workflow.node_input_mappings = {
        "create_executive_summary": {
            "model": "llm_model",
            "tender_analysis": "tender_analysis",
            "company_profile": "company_profile"
        },
        "develop_methodology": {
            "model": "llm_model",
            "tender_analysis": "tender_analysis",
            "company_capabilities": "company_capabilities"
        },
        "define_team_structure": {
            "model": "llm_model",
            "available_resources": "available_resources"
        },
        "specify_deliverables": {
            "model": "llm_model",
            "tender_analysis": "tender_analysis"
        },
        "develop_pricing": {
            "model": "llm_model",
            "cost_data": "cost_data"
        },
        "create_compliance_matrix": {
            "model": "llm_model",
            "tender_analysis": "tender_analysis"
        },
        "compile_final_response": {
            "model": "llm_model",
            "tender_id": "tender_id",
            "supporting_docs": "supporting_docs"
        }
    }
    
    return workflow

async def generate_tender_response(
    tender_id: str,
    tender_analysis: Dict[str, Any],
    company_profile: Dict[str, Any],
    company_capabilities: Dict[str, Any],
    available_resources: Dict[str, Any],
    cost_data: Dict[str, Any],
    supporting_docs: List[Dict[str, Any]],
    llm_model: str = "gemini/gemini-2.0-flash",
    output_format: str = "docx",
    _handle_event: Optional[Callable[[str, dict], None]] = None
) -> TenderResponse:
    """Generate a complete tender response."""
    
    initial_context = {
        "tender_id": tender_id,
        "tender_analysis": tender_analysis,
        "company_profile": company_profile,
        "company_capabilities": company_capabilities,
        "available_resources": available_resources,
        "cost_data": cost_data,
        "supporting_docs": supporting_docs,
        "llm_model": llm_model
    }
    
    workflow = create_response_workflow()
    engine = workflow.build()
    
    result = await engine.run(initial_context)
    
    logger.info(f"Tender response generated for tender ID: {tender_id}")
    return result["final_response"]

def cli_generate_response(
    tender_analysis_file: str,
    company_profile_file: str,
    capabilities_file: str,
    resources_file: str,
    cost_data_file: str,
    supporting_docs_file: str,
    tender_id: str,
    output_file: Optional[str] = None,
    output_format: str = "docx",
    model: str = "gemini/gemini-2.0-flash"
):
    """CLI wrapper for tender response generation."""
    # Read input files
    with open(tender_analysis_file, 'r', encoding='utf-8') as f:
        tender_analysis = eval(f.read())
    
    with open(company_profile_file, 'r', encoding='utf-8') as f:
        company_profile = eval(f.read())
    
    with open(capabilities_file, 'r', encoding='utf-8') as f:
        capabilities = eval(f.read())
    
    with open(resources_file, 'r', encoding='utf-8') as f:
        resources = eval(f.read())
    
    with open(cost_data_file, 'r', encoding='utf-8') as f:
        cost_data = eval(f.read())
    
    with open(supporting_docs_file, 'r', encoding='utf-8') as f:
        supporting_docs = eval(f.read())
    
    # Generate response
    result = asyncio.run(generate_tender_response(
        tender_id=tender_id,
        tender_analysis=tender_analysis,
        company_profile=company_profile,
        company_capabilities=capabilities,
        available_resources=resources,
        cost_data=cost_data,
        supporting_docs=supporting_docs,
        llm_model=model,
        output_format=output_format
    ))
    
    # Output results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.model_dump_json(indent=2))
    else:
        print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    import typer
    typer.run(cli_generate_response)
