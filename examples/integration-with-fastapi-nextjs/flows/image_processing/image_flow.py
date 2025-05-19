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
from quantalogic.tools.llm_vision_tool import LLMVisionTool, DEFAULT_MODEL_NAME
from quantalogic.tools.image_generation.stable_diffusion import StableDiffusionTool
from quantalogic.tools.image_generation.dalle_e import LLMImageGenerationTool

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def get_template_path(template_name: str) -> str:
    """Get the full path to a template file."""
    return os.path.join(TEMPLATES_DIR, template_name)

class ImageAnalysisConfig(BaseModel):
    """Configuration for image analysis"""
    aspects_to_analyze: List[str] = Field(
        default=["objects", "colors", "composition", "style", "mood"],
        description="Aspects of the image to analyze"
    )
    custom_instructions: Optional[str] = Field(
        default=None,
        description="Custom instructions for analysis"
    )
    analysis_depth: str = Field(
        default="detailed",
        description="Depth of analysis (basic, detailed, or comprehensive)"
    )

class ImageGenerationConfig(BaseModel):
    """Configuration for image generation"""
    style_preset: Optional[str] = Field(
        default=None,
        description="Style preset to use for generation"
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        description="Elements to avoid in generation"
    )
    custom_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional generation parameters"
    )

class ImageAnalysisResult(BaseModel):
    """Detailed analysis of an image"""
    objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Objects detected in the image"
    )
    colors: List[str] = Field(
        description="Dominant colors in the image"
    )
    composition: str = Field(
        description="Analysis of image composition"
    )
    style: str = Field(
        description="Overall style and aesthetic"
    )
    mood: str = Field(
        description="Overall mood and atmosphere"
    )
    custom_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results of custom analysis aspects"
    )

class GenerationPrompt(BaseModel):
    """Generated prompt for image creation"""
    main_prompt: str = Field(
        description="Main generation prompt"
    )
    style_settings: Dict[str, Any] = Field(
        description="Style and composition settings"
    )
    technical_settings: Dict[str, Any] = Field(
        description="Technical generation settings"
    )

class GenerationResult(BaseModel):
    """Final result of the image generation"""
    image_path: str = Field(
        description="Path to the generated image"
    )
    prompt_used: GenerationPrompt = Field(
        description="The prompt that was used"
    )
    reference_analysis: Optional[ImageAnalysisResult] = Field(
        default=None,
        description="Analysis of the reference image if used"
    )

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_analyze_image.j2"),
    output="image_analysis",
    response_model=ImageAnalysisResult,
    prompt_file=get_template_path("prompt_analyze_image.j2")
)
async def analyze_image(
    image_url: str,
    analysis_config: ImageAnalysisConfig,
    model: str
) -> ImageAnalysisResult:
    """Analyze the image based on provided configuration."""
    logger.debug(f"Analyzing image with model: {model} and config: {analysis_config}")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_generate_prompt.j2"),
    output="generation_prompt",
    response_model=GenerationPrompt,
    prompt_file=get_template_path("prompt_generate_image.j2")
)
async def generate_image_prompt(
    image_analysis: Optional[ImageAnalysisResult],
    generation_config: ImageGenerationConfig,
    user_prompt: str,
    model: str
) -> GenerationPrompt:
    """Generate the prompt for image creation."""
    logger.debug(f"Generating image prompt with model: {model}")
    pass

@Nodes.define(output="generation_result")
async def generate_image(
    generation_prompt: GenerationPrompt,
    generator: str = "stable_diffusion"
) -> str:
    """Generate the image using specified generator."""
    try:
        if generator == "stable_diffusion":
            tool = StableDiffusionTool()
            result = await tool.async_execute(
                prompt=generation_prompt.main_prompt,
                negative_prompt=generation_prompt.style_settings.get("negative_prompt", ""),
                style=generation_prompt.style_settings.get("style_preset", ""),
                **generation_prompt.technical_settings
            )
        elif generator == "dalle":
            tool = LLMImageGenerationTool()
            result = await tool.async_execute(
                prompt=generation_prompt.main_prompt,
                **generation_prompt.technical_settings
            )
        else:
            raise ValueError(f"Unsupported generator: {generator}")
        
        logger.info(f"Image generated successfully using {generator}: {result}")
        return result
    except Exception as e:
        logger.error(f"Error generating image with {generator}: {e}")
        raise

def create_image_workflow(use_reference: bool = False) -> Workflow:
    """Create the workflow for image analysis and generation."""
    if use_reference:
        workflow = (
            Workflow("analyze_image")
            .then("generate_image_prompt")
            .then("generate_image")
        )
    else:
        workflow = (
            Workflow("generate_image_prompt")
            .then("generate_image")
        )
    
    workflow.node_input_mappings = {
        "analyze_image": {
            "model": "vision_model",
            "analysis_config": "analysis_config"
        },
        "generate_image_prompt": {
            "model": "prompt_model",
            "generation_config": "generation_config",
            "user_prompt": "user_prompt"
        },
        "generate_image": {
            "generator": "generator"
        }
    }
    
    return workflow

async def process_image_workflow(
    user_prompt: str,
    reference_image_url: Optional[str] = None,
    analysis_config: Optional[ImageAnalysisConfig] = None,
    generation_config: Optional[ImageGenerationConfig] = None,
    vision_model: str = DEFAULT_MODEL_NAME,
    prompt_model: str = "gemini/gemini-2.0-flash",
    generator: str = "stable_diffusion",
    _handle_event: Optional[Callable[[str, dict], None]] = None,
    task_id: Optional[str] = None
) -> GenerationResult:
    """Run the complete image processing workflow."""
    
    # Set default configs if not provided
    analysis_config = analysis_config or ImageAnalysisConfig()
    generation_config = generation_config or ImageGenerationConfig()

    # Create initial context
    initial_context = {
        "user_prompt": user_prompt,
        "vision_model": vision_model,
        "prompt_model": prompt_model,
        "generator": generator,
        "analysis_config": analysis_config,
        "generation_config": generation_config
    }

    if reference_image_url:
        initial_context["image_url"] = reference_image_url

    # Create and run workflow
    workflow = create_image_workflow(use_reference=bool(reference_image_url))
    engine = workflow.build()
    
    result = await engine.run(initial_context)
    
    # Create the final result
    final_result = GenerationResult(
        image_path=result["generation_result"],
        prompt_used=result["generation_prompt"],
        reference_analysis=result.get("image_analysis")
    )
    
    logger.info("Image processing completed successfully")
    return final_result

def cli_process_image(
    user_prompt: str,
    reference_image_url: Optional[str] = None,
    analysis_depth: str = "detailed",
    style_preset: Optional[str] = None,
    generator: str = "stable_diffusion",
    vision_model: str = DEFAULT_MODEL_NAME,
    prompt_model: str = "gemini/gemini-2.0-flash"
):
    """CLI wrapper for the image processing function."""
    analysis_config = ImageAnalysisConfig(analysis_depth=analysis_depth)
    generation_config = ImageGenerationConfig(style_preset=style_preset)
    
    result = asyncio.run(process_image_workflow(
        user_prompt=user_prompt,
        reference_image_url=reference_image_url,
        analysis_config=analysis_config,
        generation_config=generation_config,
        vision_model=vision_model,
        prompt_model=prompt_model,
        generator=generator
    ))
    
    print(f"\nGenerated image: {result.image_path}")
    if result.reference_analysis:
        print("\nReference Image Analysis:")
        print(result.reference_analysis.model_dump_json(indent=2))
    print("\nGeneration Prompt Used:")
    print(result.prompt_used.model_dump_json(indent=2))

if __name__ == "__main__":
    import typer
    typer.run(cli_process_image)
