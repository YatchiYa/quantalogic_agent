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
#     "instructor"
# ]
# ///

import os
from typing import Dict, List, Optional

import anyio
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
class VisualStyle(BaseModel):
    preferred_style: str
    color_preferences: Optional[str]
    imagery_type: str

class RequirementsAnalysis(BaseModel):
    target_audience: str
    industry: str
    key_sections: List[str]
    visual_style: VisualStyle
    interactive_elements: List[str]
    content_focus: str
    technical_requirements: List[str]
    branding_elements: List[str]

class ColorScheme(BaseModel):
    primary: str
    secondary: str
    accent: str
    text: str
    background: str

class NavigationItem(BaseModel):
    text: str
    link: str
    is_button: bool

class NavigationGroup(BaseModel):
    title: str
    links: List[NavigationItem]

class HeaderSection(BaseModel):
    logo: str
    navigation: List[NavigationItem]
    is_sticky: bool
    has_mobile_menu: bool

class HeroSection(BaseModel):
    title: str
    subtitle: str
    cta_primary: NavigationItem
    cta_secondary: Optional[NavigationItem]
    background_type: str
    background_content: str

class FeatureCard(BaseModel):
    icon: str
    title: str
    description: str

class FeaturesSection(BaseModel):
    title: str
    subtitle: str
    features: List[FeatureCard]
    layout: str

class TestimonialCard(BaseModel):
    quote: str
    author: str
    role: str
    company: str
    image: Optional[str]

class TestimonialsSection(BaseModel):
    title: str
    subtitle: str
    testimonials: List[TestimonialCard]

class CTASection(BaseModel):
    title: str
    subtitle: str
    button: NavigationItem
    background_style: str

class FooterSection(BaseModel):
    logo: str
    description: str
    social_links: List[NavigationItem]
    nav_columns: List[NavigationGroup]
    bottom_text: str

class LandingPageStructure(BaseModel):
    title: str
    description: str
    color_scheme: ColorScheme
    header: HeaderSection
    hero: HeroSection
    features: FeaturesSection
    testimonials: TestimonialsSection
    cta: CTASection
    footer: FooterSection

class SectionContent(BaseModel):
    html: str
    css: str
    js: str

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def get_template_path(template_name):
    return os.path.join(TEMPLATES_DIR, template_name)

# Custom Observer for Workflow Events
async def landing_page_progress_observer(event: WorkflowEvent):
    if event.event_type == WorkflowEventType.WORKFLOW_STARTED:
        print(f"\n{'='*50}\n🚀 Starting Landing Page Generation 🚀\n{'='*50}")
    elif event.event_type == WorkflowEventType.NODE_STARTED:
        print(f"\n🔄 [{event.node_name}] Starting...")
    elif event.event_type == WorkflowEventType.NODE_COMPLETED:
        if event.node_name == "generate_structure":
            logger.debug(f"Generated structure: {event.result}")
        elif event.node_name.startswith("generate_section_"):
            section = event.node_name.replace("generate_section_", "")
            print(f"✅ [{section}] Section generated")
        print(f"✅ [{event.node_name}] Completed")
    elif event.event_type == WorkflowEventType.WORKFLOW_COMPLETED:
        print(f"\n{'='*50}\n🎉 Landing Page Generation Finished 🎉\n{'='*50}")
    elif event.event_type == WorkflowEventType.TRANSITION_EVALUATED:
        logger.debug(f"Transition evaluated: {event.transition_from} -> {event.transition_to}")

# Workflow Nodes
@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_analyze_requirements.j2"),
    output="requirements",
    response_model=RequirementsAnalysis,
    prompt_file=get_template_path("prompt_analyze_requirements.j2"),
    temperature=0.7,
)
async def analyze_requirements(model: str, client_request: str) -> RequirementsAnalysis:
    logger.debug("Analyzing client requirements")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_generate_structure.j2"),
    output="structure",
    response_model=LandingPageStructure,
    prompt_file=get_template_path("prompt_generate_structure.j2"),
    temperature=0.5,
)
async def generate_structure(model: str, requirements: RequirementsAnalysis) -> LandingPageStructure:
    logger.debug("Generating landing page structure")
    pass

@Nodes.define(output=None)
async def initialize_sections() -> dict:
    logger.debug("Initializing section tracking")
    return {
        "completed_sections": [],
        "current_section": "header",
        "sections_order": ["header", "hero", "features", "testimonials", "cta", "footer"]
    }

@Nodes.llm_node(
    system_prompt_file=get_template_path("system_generate_html.j2"),
    output="section_html",
    prompt_file=get_template_path("prompt_generate_html.j2"),
    temperature=0.3,
)
async def generate_section_html(model: str, structure: LandingPageStructure, current_section: str) -> str:
    logger.debug(f"Generating HTML for {current_section} section")
    pass

@Nodes.llm_node(
    system_prompt_file=get_template_path("system_generate_css.j2"),
    output="section_css",
    prompt_file=get_template_path("prompt_generate_css.j2"),
    temperature=0.3,
)
async def generate_section_css(model: str, structure: LandingPageStructure, current_section: str, section_html: str) -> str:
    logger.debug(f"Generating CSS for {current_section} section")
    pass

@Nodes.llm_node(
    system_prompt_file=get_template_path("system_generate_js.j2"),
    output="section_js",
    prompt_file=get_template_path("prompt_generate_js.j2"),
    temperature=0.3,
)
async def generate_section_js(model: str, structure: LandingPageStructure, current_section: str, section_html: str) -> str:
    logger.debug(f"Generating JavaScript for {current_section} section")
    pass

@Nodes.define(output="section_content")
async def validate_section(section_html: str, section_css: str, section_js: str, current_section: str) -> SectionContent:
    logger.info(f"Validating {current_section} section")
    return SectionContent(html=section_html, css=section_css, js=section_js)

@Nodes.define(output=None)
async def update_sections(
    completed_sections: List[dict],
    section_content: SectionContent,
    current_section: str,
    sections_order: List[str]
) -> dict:
    completed_sections.append({
        "name": current_section,
        "content": section_content.model_dump()
    })
    current_idx = sections_order.index(current_section)
    next_section = sections_order[current_idx + 1] if current_idx + 1 < len(sections_order) else None
    return {
        "completed_sections": completed_sections,
        "current_section": next_section
    }

@Nodes.define(output=None)
async def compile_landing_page(completed_sections: List[dict], structure: LandingPageStructure) -> dict:
    # Combine all sections into final files
    html_parts = []
    css_parts = []
    js_parts = []
    
    # Add HTML header
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{structure.title}</title>
    <meta name="description" content="{structure.description}">
    <link rel="stylesheet" href="styles.css">
</head>
<body>""")
    
    # Add each section's content
    for section in completed_sections:
        html_parts.append(f"<!-- {section['name'].upper()} SECTION -->")
        html_parts.append(section['content']['html'])
        css_parts.append(f"/* {section['name'].upper()} SECTION */")
        css_parts.append(section['content']['css'])
        if section['content']['js']:
            js_parts.append(f"// {section['name'].upper()} SECTION")
            js_parts.append(section['content']['js'])
    
    # Add HTML footer
    html_parts.append("""
    <script src="main.js"></script>
</body>
</html>""")
    
    return {
        "html_content": "\n\n".join(html_parts),
        "css_content": "\n\n".join(css_parts),
        "js_content": "\n\n".join(js_parts)
    }

@Nodes.define(output=None)
async def save_files(html_content: str, css_content: str, js_content: str, output_dir: str) -> None:
    """Save the generated files to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean content function
    def clean_content(content: str) -> str:
        return "\n".join(
            line for line in content.splitlines()
            if not line.strip().startswith("[Snippet identifier=") and 
               not line.strip().startswith("```") and
               not line.strip() == "[/Snippet]"
        ).strip()
    
    # Save files
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(clean_content(html_content))
    
    with open(os.path.join(output_dir, "styles.css"), "w", encoding="utf-8") as f:
        f.write(clean_content(css_content))
    
    with open(os.path.join(output_dir, "main.js"), "w", encoding="utf-8") as f:
        f.write(clean_content(js_content))
    
    logger.info(f"Saved files to {output_dir}")

# Define the Workflow
workflow = (
    Workflow("analyze_requirements")
    .add_observer(landing_page_progress_observer)
    .then("generate_structure")
    .then("initialize_sections")
    .then("generate_section_html")
    .then("generate_section_css")
    .then("generate_section_js")
    .then("validate_section")
    .then("update_sections")
    .branch([
        ("generate_section_html", lambda ctx: ctx["current_section"] is not None),
        ("compile_landing_page", lambda ctx: ctx["current_section"] is None)
    ])
    .then("save_files")
)

async def create_landing_page(
    client_request: str,
    output_dir: str = "./output",
    model: str = "gemini/gemini-2.0-flash",
):
    """
    Create a landing page based on client requirements.
    
    Args:
        client_request: Client's requirements for the landing page
        output_dir: Directory to save the generated files
        model: LLM model to use
    """
    initial_context = {
        "client_request": client_request,
        "output_dir": output_dir,
        "model": model,
    }
    
    logger.info("Starting landing page generation")
    engine = workflow.build()
    result = await engine.run(initial_context)
    logger.info("Landing page generation completed successfully 🎉")
    return result

async def main():
    # Example client request
    client_request = """Create a modern tech startup landing page with:
    - Sticky header with logo and navigation
    - Hero section with video background
    - Features grid with icons
    - Testimonials carousel
    - Call-to-action section
    - Footer with social links
    """
    await create_landing_page(client_request)

if __name__ == "__main__":
    anyio.run(main)
