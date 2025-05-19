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
#     "pyperclip>=1.8.2",
# ]
# ///

import asyncio
from collections.abc import Callable
import datetime
import base64
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType 
from quantalogic.tools.image_generation.stable_diffusion import StableDiffusionTool, STABLE_DIFFUSION_CONFIG
from quantalogic.tools.image_generation.dalle_e import LLMImageGenerationTool, DALLE_CONFIG
from ..service import event_observer

console = Console()

# Constants for image sizes
FACEBOOK_SIZES = {
    "square": "1080x1080",
    "portrait": "1080x1350",
    "landscape": "1080x608"
}

STABLE_DIFFUSION_SIZES = [
    "1024x1024",  # square
    "1152x896",   # landscape
    "896x1152",   # portrait
    "1280x1024",  # wide
    "1536x1024"   # ultra-wide
]

DALLE_SIZES = [
    "1024x1024",    # square
    "1024x1792",    # portrait
    "1792x1024"     # landscape
]

# Constants for token limits
MAX_TOKENS = {
    "analysis": 1000,  # Increased for more detailed analysis
    "post_structure": 1500,  # Increased for richer content
    "image_spec": 800  # Increased for better image prompts
}

def convert_facebook_to_model_size(facebook_size: str, model_type: str) -> str:
    """Convert Facebook size to appropriate model size."""
    width, height = map(int, facebook_size.split("x"))
    aspect_ratio = width / height

    if model_type == "stable_diffusion":
        if aspect_ratio == 1:  # Square
            return "1024x1024"
        elif aspect_ratio > 1:  # Landscape
            return "1152x896"
        else:  # Portrait
            return "896x1152"
    elif model_type == "dalle":
        if aspect_ratio == 1:  # Square
            return "1024x1024"
        elif aspect_ratio > 1:  # Landscape
            return "1792x1024"
        else:  # Portrait
            return "1024x1792"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Define Pydantic models for structured output
class ContentTheme(BaseModel):
    """Analysis of the content theme for consistent image and text generation."""
    main_topic: str = Field(description="Main topic or subject of the content")
    key_themes: List[str] = Field(description="Key themes to emphasize")
    visual_style: str = Field(description="Overall visual style for images")
    tone: str = Field(description="Tone of the content (e.g., professional, casual, inspirational)")
    target_audience: List[str] = Field(description="Target audience segments")
    color_scheme: List[str] = Field(description="Suggested color scheme")
    business_goals: List[str] = Field(description="Business objectives to achieve")

class FacebookPostStructure(BaseModel):
    """Structure for the Facebook post."""
    caption: str = Field(description="Main caption text")
    hashtags: List[str] = Field(description="Relevant hashtags")
    image_prompts: List[str] = Field(description="Image generation prompts")
    carousel_text: Optional[List[str]] = Field(description="Text for carousel slides if multiple images")

class ImageGenerationSpec(BaseModel):
    """Specification for image generation."""
    main_prompt: str = Field(description="Enhanced main prompt")
    negative_prompt: str = Field(description="Elements to avoid")
    style_preset: str = Field(description="Style preset to use")
    size: str = Field(description="Image dimensions")
    cfg_scale: float = Field(description="Configuration scale")
    steps: int = Field(description="Number of generation steps")
    model_type: str = Field(description="Model to use (stable_diffusion or dalle)")

class GeneratedImage(BaseModel):
    """Result of image generation."""
    image_path: str = Field(description="Path to the generated image")
    base64_data: str = Field(description="Base64 encoded image data")

class FacebookPost(BaseModel):
    """Final Facebook post content."""
    caption: str = Field(description="Formatted caption")
    hashtags: str = Field(description="Formatted hashtags")
    images: List[GeneratedImage] = Field(description="Generated images")
    carousel_text: Optional[List[str]] = Field(description="Text for carousel slides")

# Node: Analyze Content Theme
@Nodes.structured_llm_node(
    system_prompt="""You are an expert Facebook content strategist and business communication specialist.
    Your task is to analyze content themes and create professional, engagement-driving content strategies.
    Consider Facebook's algorithm, business engagement patterns, and professional audience behavior.
    Focus on creating content that builds brand authority and drives meaningful business engagement.
    MOST IMPORTANTLY: You must stay completely faithful to the user's original content and intent.
    Never deviate from or generalize the specific topic provided by the user.""",
    output="content_theme",
    response_model=ContentTheme,
    max_tokens=MAX_TOKENS["analysis"],
    prompt_template="""
Analyze the following content for professional Facebook optimization:

Content: {{content}}

IMPORTANT: Your analysis MUST be strictly focused on the exact content provided above. Do not generalize or pivot to adjacent topics. Stay anchored to the specific subject matter and details provided.

Create a comprehensive theme analysis focused on business impact:

1. Professional Direction:
- Define a professional visual aesthetic that aligns with business goals
- Identify color schemes that enhance brand recognition
- Suggest visual elements that convey authority and expertise

2. Content Strategy:
- Identify key business messaging elements
- Map professional engagement triggers for target audience
- Define content pillars that align with business objectives

3. Engagement Optimization:
- Determine optimal posting times for professional audience
- Identify relevant business hashtags
- Plan professional CTAs and discussion points

Ensure the analysis covers:
- B2B and B2C best practices
- Algorithm-friendly content structures
- Professional engagement patterns
- Business growth elements

REMINDER: Your analysis must be 100% anchored to the user's original content. Do not drift from the specific topic provided.
"""
)
async def analyze_content_theme(content: str, model: str) -> ContentTheme:
    """Analyze the content theme for consistent generation."""
    logger.debug(f"analyze_content_theme called with model: {model}")
    pass

# Node: Generate Post Structure
@Nodes.structured_llm_node(
    system_prompt="""You are an elite Facebook content creator specializing in professional business communication.
    Your expertise lies in crafting compelling business narratives, strategic CTAs, and engagement-driving content.
    Focus on creating content that builds authority, drives meaningful engagement, and achieves business objectives.
    CRITICAL: You must create content that is 100% faithful to the user's original content and intent.
    Never deviate from the specific topic provided by the user or introduce unrelated themes.""",
    output="post_structure",
    response_model=FacebookPostStructure,
    max_tokens=MAX_TOKENS["post_structure"],
    prompt_template="""
Create a professional Facebook post based on:

Theme Analysis:
{{content_theme.visual_style}}
{{content_theme.tone}}
{{content_theme.key_themes}}
{{content_theme.business_goals}}

Number of Images: {{num_images}}

CRITICAL INSTRUCTION: Your post MUST be strictly focused on the exact content themes derived from the user's original input. Do not generalize or pivot to adjacent topics. Every element of your post must directly relate to the specific subject matter provided.

Craft a post that maximizes:
1. Professional Engagement
2. Brand Authority
3. Business Value
4. Community Building

Structure Requirements:

📝 Main Text (4-6 paragraphs):
- Professional Hook: Compelling opening that establishes authority and directly references the main topic
- Value Proposition: Clear business or professional value specific to the content provided
- Supporting Points: Well-structured evidence or insights that directly support the specific topic
- Strategic CTA: Professional call-to-action that encourages engagement with the specific topic

#️⃣ Hashtags (5-10 total):
- Industry-specific hashtags directly related to the content
- Professional community tags relevant to the specific topic
- Brand-relevant tags mentioned in or relevant to the content
- Campaign-specific tags if mentioned in the content
- Local business tags (if applicable and relevant to the specific content)

🖼️ Image Prompts:
- Professional, high-quality visual descriptions that EXACTLY match the topic
- Brand-aligned composition details specific to the content
- Business-appropriate aesthetics that represent the specific subject matter

Optimization Requirements:
- Professional tone throughout
- Strategic emoji usage (business-appropriate)
- Include discussion prompts related to the specific topic
- Proper formatting for readability
- Actionable business insights derived directly from the content
- Shareable professional content that remains true to the original content

FINAL CHECK: Verify that your entire post remains 100% focused on the specific topic from the user's original content. Do not drift to adjacent or general topics.
"""
)
async def generate_post_structure(
    content_theme: ContentTheme,
    num_images: int,
    model: str
) -> FacebookPostStructure:
    """Generate the structure for a Facebook post."""
    logger.debug(f"generate_post_structure called with model: {model}")
    pass

# Node: Generate Single Image Spec
@Nodes.structured_llm_node(
    system_prompt="""You are a professional AI image prompt engineer specializing in business-focused visuals.
    Your expertise lies in creating prompts that generate professional, brand-aligned images.
    Focus on current business visual trends, professional appeal, and brand consistency.
    CRITICAL: Your image prompts must be 100% faithful to the user's original content and intent.
    Never deviate from the specific topic provided by the user.""",
    output="image_spec",
    response_model=ImageGenerationSpec,
    max_tokens=MAX_TOKENS["image_spec"],
    prompt_template="""
Create a Facebook-optimized professional image specification:

Content Theme:
{{content_theme.visual_style}}
{{content_theme.color_scheme}}
{{content_theme.business_goals}}

Base Prompt:
{{current_prompt}}

Technical Parameters:
- Style Presets: {{available_styles|join(", ")}}
- Available Sizes: {{available_sizes|join(", ")}}
- Generator: {{image_generator}}

CRITICAL INSTRUCTION: Your image prompt MUST be strictly focused on visualizing the exact content from the user's original input. Do not generalize or pivot to adjacent topics. The image must directly represent the specific subject matter provided.

Generate a specification that creates:
1. Main Prompt:
   - Include professional composition details that directly visualize the topic
   - Define business-appropriate lighting and atmosphere for the specific subject
   - Specify brand-aligned color palette from theme
   - Add high-quality technical markers
   - Include Facebook feed optimization elements

2. Negative Prompt:
   - Exclude unprofessional elements
   - Remove business-inappropriate content
   - Prevent brand-misaligned elements
   - Avoid controversial content
   - Exclude any elements that would shift focus away from the main topic

3. Technical Settings:
   - Professional style preset selection
   - Facebook-optimized dimensions
   - High-quality parameters
   - Performance-tuned settings

Focus on creating images that:
- Convey professionalism
- Build brand authority
- Maintain business standards
- Follow platform best practices
- Drive meaningful engagement
- MOST IMPORTANTLY: Accurately represent the specific topic from the user's content

FINAL CHECK: Verify that your image prompt will generate visuals that are 100% on-topic and faithful to the user's original content.
"""
)
async def generate_single_image_spec(
    content_theme: ContentTheme,
    current_prompt: str,
    available_styles: List[str],
    available_sizes: List[str],
    image_generator: str,
    model: str
) -> ImageGenerationSpec:
    """Generate enhanced specification for a single image."""
    logger.debug(f"generate_single_image_spec called with model: {model}")
    pass

# Node: Generate Image Specs
@Nodes.define(output="image_specs")
async def generate_image_specs(
    content_theme: ContentTheme,
    post_structure: FacebookPostStructure,
    available_styles: List[str],
    available_sizes: List[str],
    image_generator: str,
    model: str
) -> List[ImageGenerationSpec]:
    """Generate enhanced specifications for each image."""
    logger.debug(f"generate_image_specs called with model: {model}")
    
    specs = []
    for prompt in post_structure.image_prompts:
        spec = await generate_single_image_spec(
            content_theme=content_theme,
            current_prompt=prompt,
            available_styles=available_styles,
            available_sizes=available_sizes,
            image_generator=image_generator,
            model=model
        )
        specs.append(spec)
    
    return specs

# Node: Generate Images
@Nodes.define(output="generated_images")
async def generate_images(
    image_specs: List[ImageGenerationSpec],
    image_generator: str
) -> List[GeneratedImage]:
    """Generate images based on the specifications."""
    logger.debug(f"Generating images with {image_generator}")
    
    # If no image specs, return empty list
    if not image_specs:
        return []
    
    generated_images = []
    output_dir = Path("generated_images")
    output_dir.mkdir(exist_ok=True)
    
    try:
        if image_generator == "stable_diffusion":
            tool = StableDiffusionTool()
            
            for spec in image_specs:
                # Convert Facebook size to Stable Diffusion size
                model_size = convert_facebook_to_model_size(spec.size, "stable_diffusion")
                
                local_path = await tool.async_execute(
                    prompt=spec.main_prompt,
                    negative_prompt=spec.negative_prompt,
                    style=spec.style_preset,
                    size=model_size,
                    cfg_scale=spec.cfg_scale,
                    steps=spec.steps
                )
                
                # Verify the file exists
                if not os.path.exists(local_path):
                    raise FileNotFoundError(f"Generated image file not found at: {local_path}")
                
                # Convert to base64
                with open(local_path, "rb") as image_file:
                    base64_data = base64.b64encode(image_file.read()).decode()
                
                # Add to results
                generated_images.append(GeneratedImage(
                    image_path=local_path,
                    base64_data=base64_data
                ))
                
                logger.info(f"Generated image using {image_generator}: {local_path}")
                
        elif image_generator == "dalle":
            tool = LLMImageGenerationTool()
            
            for spec in image_specs:
                # Convert Facebook size to DALL-E size
                model_size = convert_facebook_to_model_size(spec.size, "dalle")
                
                # Generate image and get local file path
                local_path = await tool.async_execute(
                    prompt=spec.main_prompt,
                    size=model_size,
                    quality="standard",
                    style="vivid"
                )
                
                # Wait a moment for file to be written
                await asyncio.sleep(0.5)
                
                # Check if the file exists in the generated_images directory
                if not os.path.exists(local_path):
                    logger.error(f"Generated image not found at expected path: {local_path}")
                    # Try to find the most recent file in the directory
                    recent_files = sorted(output_dir.glob("dalle_*.png"), key=os.path.getmtime, reverse=True)
                    if recent_files:
                        local_path = str(recent_files[0])
                        logger.info(f"Found most recent DALL-E image at: {local_path}")
                    else:
                        raise FileNotFoundError(f"No DALL-E images found in {output_dir}")
                
                # Convert to base64
                with open(local_path, "rb") as image_file:
                    base64_data = base64.b64encode(image_file.read()).decode()
                
                # Add to results
                generated_images.append(GeneratedImage(
                    image_path=local_path,
                    base64_data=base64_data
                ))
                
                # logger.info(f"Generated image using {image_generator}: {local_path}")
                
    except Exception as e:
        logger.error(f"Error generating image with {image_generator}: {str(e)}")
        raise
    
    return generated_images

# Node: Compile Facebook Post
@Nodes.define(output="facebook_post")
async def compile_facebook_post(
    post_structure: FacebookPostStructure,
    generated_images: Optional[List[GeneratedImage]] = None
) -> FacebookPost:
    """Compile the final Facebook post with images and formatted text."""
    try:
        # Format hashtags with proper spacing and line breaks
        formatted_hashtags = "\n\n" + " ".join(f"#{tag}" for tag in post_structure.hashtags)
        
        # Create the final post object
        post = FacebookPost(
            caption=post_structure.caption + formatted_hashtags,
            hashtags=formatted_hashtags,
            images=generated_images or [],  # Use empty list if generated_images is None
            carousel_text=post_structure.carousel_text
        )
        
        logger.info("Compiled Facebook post successfully")
        return post
    
    except Exception as e:
        logger.error(f"Error compiling Facebook post: {e}")
        raise

# Create the workflow
def create_facebook_content_workflow(generate_images: bool = True) -> Workflow:
    """Create a workflow for Facebook content generation."""
    if generate_images:
        workflow = (
            Workflow("analyze_content_theme")
            .then("generate_post_structure")
            .then("generate_image_specs")
            .then("generate_images")
            .then("compile_facebook_post")
        )
    else:
        workflow = (
            Workflow("analyze_content_theme")
            .then("generate_post_structure")
            .then("compile_facebook_post")
        )
    
    # Add input mappings
    workflow.node_input_mappings = {
        "analyze_content_theme": {
            "model": "analysis_model"
        },
        "generate_post_structure": {
            "model": "content_model",
            "num_images": "num_images" if generate_images else lambda _: 0
        }
    }
    
    if generate_images:
        workflow.node_input_mappings.update({
            "generate_image_specs": {
                "model": "image_model",
                "available_styles": lambda _: list(STABLE_DIFFUSION_CONFIG["styles"]),
                "available_sizes": lambda _: list(FACEBOOK_SIZES.values()),
                "image_generator": "image_generator"
            }
        })
    
    return workflow

async def generate_facebook_content(
    content_context: str,
    num_images: int = 1,
    generate_images: bool = True,
    analysis_model: str = "gemini/gemini-2.0-flash",
    content_model: str = "gemini/gemini-2.0-flash",
    image_model: str = "gemini/gemini-2.0-flash",
    image_generator: str = "stable_diffusion",
    task_id: str = "default",
    _handle_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
) -> FacebookPost:
    """Generate Facebook content with optional images from a context."""
    
    if generate_images:
        if num_images < 0 or num_images > 3:
            raise ValueError("Number of images must be between 0 and 3")
        
        if image_generator not in ["stable_diffusion", "dalle"]:
            raise ValueError("image_generator must be either 'stable_diffusion' or 'dalle'")
    else:
        num_images = 0
    
    # Ensure content is not empty or just whitespace
    if not content_context or content_context.strip() == "":
        raise ValueError("Content context cannot be empty")
    
    # Clean up the content to remove extra whitespace
    content_context = content_context.strip()
    
    logger.info(f"Processing content: {content_context[:100]}...")
    
    initial_context = {
        "content": content_context,  # Added to match the prompt template parameter
        "content_context": content_context,
        "num_images": num_images,
        "analysis_model": analysis_model,
        "content_model": content_model,
        "image_model": image_model,
        "image_generator": image_generator
    }
    
    logger.info(f"Starting Facebook content generation {'with' if generate_images else 'without'} images")
    
    try:
        workflow = create_facebook_content_workflow(generate_images)
        engine = workflow.build() 
        
        # Add the event observer if _handle_event is provided
        if _handle_event:
            # Create a lambda to bind task_id to the observer
            bound_observer = lambda event: asyncio.create_task(
                event_observer(event, task_id=task_id, _handle_event=_handle_event)
            )
            engine.add_observer(bound_observer)

        result = await engine.run(initial_context)
        
        if not isinstance(result.get("facebook_post"), FacebookPost):
            raise ValueError("Workflow did not produce a valid Facebook post")
        
        logger.info("Facebook content generation completed successfully")
        return result["facebook_post"]
        
    except Exception as e:
        logger.error(f"Error generating Facebook content: {e}")
        raise

# Add main function for testing
async def main():
    """Test function for the Facebook content generation workflow."""
    # Test content with specific details to ensure adherence
    test_content = """
    The impact of AI on retail business: How artificial intelligence is transforming 
    inventory management, customer experience, and sales forecasting. Retailers are 
    increasingly adopting AI tools to optimize supply chains, personalize shopping 
    experiences, and make data-driven decisions that boost revenue.
    """
    
    try:
        # Test without images
        console.print("\n[bold blue]Testing without images...[/]")
        post_no_images = await generate_facebook_content(
            content_context=test_content,
            generate_images=False
        )
        console.print("[bold green]✓ Text-only post generated[/]")
        console.print(Panel(Markdown(post_no_images.caption), title="Generated Facebook Post (No Images)"))
        
        # Verify content adherence for first test
        console.print("\n[bold yellow]Content Adherence Check (Text Only):[/]")
        key_terms = ["AI", "retail", "inventory management", "customer experience", "sales forecasting", "supply chains"]
        found_terms = [term for term in key_terms if term.lower() in post_no_images.caption.lower()]
        console.print(f"[bold]Original content key terms found in post:[/] {len(found_terms)}/{len(key_terms)}")
        console.print(f"Terms found: {', '.join(found_terms)}")
        
        # Optional image test - only run if explicitly requested
        # Commented out to avoid unnecessary API calls during testing
        """
        # Test with DALL-E
        console.print("\n[bold blue]Testing with DALL-E...[/]")
        post_dalle = await generate_facebook_content(
            content_context=test_content, 
            num_images=1,
            generate_images=True,      
            image_generator="dalle"
        )
        console.print("[bold green]✓ DALL-E test completed[/]")
        console.print(Panel(Markdown(post_dalle.caption), title="Generated Facebook Post with DALL-E Image"))
        """
        
    except Exception as e:
        console.print(f"[bold red]Error during testing:[/] {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
