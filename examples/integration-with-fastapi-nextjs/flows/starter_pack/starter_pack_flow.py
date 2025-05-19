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
from typing import List, Optional
from pydantic import BaseModel, Field

from loguru import logger
from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType 
from quantalogic.tools.llm_vision_tool import LLMVisionTool, DEFAULT_MODEL_NAME
from quantalogic.tools.image_generation.stable_diffusion import StableDiffusionTool, STABLE_DIFFUSION_CONFIG
from quantalogic.tools.image_generation.dalle_e import LLMImageGenerationTool, DALLE_CONFIG

from ..service import event_observer
# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Helper function to get template paths
def get_template_path(template_name):
    return os.path.join(TEMPLATES_DIR, template_name)

# Define Pydantic models for structured output
class PersonAnalysis(BaseModel):
    """Analysis of a person in the image"""
    facial_features: str = Field(description="Detailed description of facial features")
    body_type: str = Field(description="Body type and build description")
    clothing: str = Field(description="Detailed description of clothing")
    pose: str = Field(description="Description of pose and posture")
    expression: str = Field(description="Facial expression and emotion")
    distinguishing_features: str = Field(description="Any distinguishing features or characteristics")

class ImageAnalysisResult(BaseModel):
    """Detailed analysis of the reference image"""
    people: List[PersonAnalysis] = Field(description="Analysis of each person in the image")
    overall_style: str = Field(description="Overall style and aesthetic of the image")
    background: str = Field(description="Description of the background/setting")
    lighting: str = Field(description="Lighting conditions and effects")
    color_scheme: List[str] = Field(description="Dominant colors in the image")
    composition: str = Field(description="Analysis of image composition")
    mood: str = Field(description="Overall mood and atmosphere")

class StarterPackPrompt(BaseModel):
    """Generated prompt for starter pack creation"""
    main_prompt: str = Field(description="Main prompt for the collectible figure")
    negative_prompt: str = Field(description="Elements to avoid in generation")
    style_preset: str = Field(description="Style preset to use")
    composition_guide: str = Field(description="Guide for composition")
    packaging_details: str = Field(description="Details for blister packaging")
    background_elements: str = Field(description="Background and environmental elements")
    lighting_setup: str = Field(description="Lighting configuration")
    additional_effects: str = Field(description="Additional special effects or details")

class GenerationResult(BaseModel):
    """Final result of the starter pack generation"""
    image_path: str = Field(description="Path to the generated image")
    prompt_used: StarterPackPrompt = Field(description="The prompt that was used")
    reference_analysis: ImageAnalysisResult = Field(description="Analysis of the reference image")

# Define workflow nodes
@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_analyze_reference.j2"),
    output="reference_analysis",
    response_model=ImageAnalysisResult,
    prompt_file=get_template_path("prompt_analyze_reference.j2")
)
async def analyze_reference_image(image_url: str, model: str) -> ImageAnalysisResult:
    """Analyze the reference image in detail."""
    logger.debug(f"Analyzing reference image with model: {model}")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_generate_starter_prompt.j2"),
    output="starter_pack_prompt",
    response_model=StarterPackPrompt,
    prompt_file=get_template_path("prompt_generate_starter.j2")
)
async def generate_starter_pack_prompt(
    reference_analysis: ImageAnalysisResult,
    packaging_style: str,
    model: str
) -> StarterPackPrompt:
    """Generate the prompt for starter pack creation."""
    logger.debug(f"Generating starter pack prompt with model: {model}")
    pass

@Nodes.define(output="generation_result")
async def generate_starter_pack(starter_pack_prompt: StarterPackPrompt, generator: str = "stable_diffusion") -> str:
    """Generate the starter pack image using either Stable Diffusion or DALL-E.
    
    Args:
        starter_pack_prompt: Generated prompt for the image
        generator: Image generator to use ('stable_diffusion' or 'dalle')
    """
    try:
        # Initialize the appropriate tool
        if generator == "stable_diffusion":
            tool = StableDiffusionTool()
            size = "1024x1024"  # Default size for collectible figure
            cfg_scale = 7.5     # Default cfg_scale for balanced generation
            steps = 50          # Default steps for quality
            
            # Generate the image
            result = await tool.async_execute(
                prompt=starter_pack_prompt.main_prompt,
                negative_prompt=starter_pack_prompt.negative_prompt,
                style=starter_pack_prompt.style_preset,
                size=size,
                cfg_scale=cfg_scale,
                steps=steps
            )
        elif generator == "dalle":
            tool = LLMImageGenerationTool()
            # Generate the image using DALL-E
            result = await tool.async_execute(
                prompt=starter_pack_prompt.main_prompt,
                size="1024x1024",  # Square format for collectible figure
                quality="hd",      # High quality for best results
                style="vivid"      # Vivid style for collectible look
            )
        else:
            raise ValueError(f"Unsupported generator: {generator}")
        
        logger.info(f"Starter pack generated successfully using {generator}: {result}")
        return result
    except Exception as e:
        logger.error(f"Error generating starter pack with {generator}: {e}")
        raise

# Create the workflow
def create_starter_pack_workflow() -> Workflow:
    """Create the workflow for starter pack generation."""
    workflow = (
        Workflow("analyze_reference_image")
        .then("generate_starter_pack_prompt")
        .then("generate_starter_pack")
    )
    
    # Add input mappings
    workflow.node_input_mappings = {
        "analyze_reference_image": {
            "model": "vision_model"
        },
        "generate_starter_pack_prompt": {
            "model": "prompt_model",
            "packaging_style": "packaging_style"
        },
        "generate_starter_pack": {
            "generator": "generator"
        }
    }
    
    return workflow

# Main function
async def generate_starter_pack_workflow(
    reference_image_url: str,
    packaging_style: str = "collector_blister",
    vision_model: str = DEFAULT_MODEL_NAME,
    prompt_model: str = "gemini/gemini-2.0-flash",
    generator: str = "stable_diffusion",
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None
) -> GenerationResult:
    """Run the complete starter pack generation workflow."""
    
    # Create initial context
    initial_context = {
        "image_url": reference_image_url,
        "packaging_style": packaging_style,
        "vision_model": vision_model,
        "prompt_model": prompt_model,
        "generator": generator
    }
    
    # Create and run workflow
    workflow = create_starter_pack_workflow()
    engine = workflow.build()
    # Add the event observer if _handle_event is provided
    if _handle_event:
        # Create a lambda to bind task_id to the observer
        bound_observer = lambda event: asyncio.create_task(
            event_observer(event, task_id=task_id, _handle_event=_handle_event)
        )
        engine.add_observer(bound_observer)
    
    result = await engine.run(initial_context)
    
    # Create the final result
    final_result = GenerationResult(
        image_path=result["generation_result"],
        prompt_used=result["starter_pack_prompt"],
        reference_analysis=result["reference_analysis"]
    )
    
    logger.info("Starter pack generation completed successfully")
    return final_result

# CLI wrapper
def cli_generate_starter_pack(
    reference_image_url: str,
    packaging_style: str = "collector_blister",
    vision_model: str = DEFAULT_MODEL_NAME,
    prompt_model: str = "gemini/gemini-2.0-flash",
    generator: str = "stable_diffusion"
):
    """CLI wrapper for the starter pack generation function."""
    result = asyncio.run(generate_starter_pack_workflow(
        reference_image_url=reference_image_url,
        packaging_style=packaging_style,
        vision_model=vision_model,
        prompt_model=prompt_model,
        generator=generator
    ))
    print(f"\nGenerated starter pack: {result.image_path}")
    print("\nPrompt used:")
    print(result.prompt_used.model_dump_json(indent=2))

if __name__ == "__main__":
    import typer
    cli_generate_starter_pack(
        reference_image_url="/home/yarab/Téléchargements/IMG_20250413_123943.jpg",
        packaging_style="collector_blister",
        vision_model=DEFAULT_MODEL_NAME,
        prompt_model="openai/gpt-4o-mini",
        generator="dalle")
