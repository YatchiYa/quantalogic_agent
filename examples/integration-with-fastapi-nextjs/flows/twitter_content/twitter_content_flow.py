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

console = Console()

# Constants for image sizes
TWITTER_SIZES = {
    "standard": "1200x675",    # 16:9 aspect ratio
    "square": "1200x1200",     # 1:1 aspect ratio
    "portrait": "1080x1350",   # 4:5 aspect ratio (max height)
    "card": "800x418"          # Twitter card size
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
    "analysis": 1000,      # For detailed content analysis
    "tweet_structure": 800, # For tweet thread structure
    "image_spec": 800      # For image generation
}

def convert_twitter_to_model_size(twitter_size: str, model_type: str) -> str:
    """Convert Twitter size to appropriate model size."""
    width, height = map(int, twitter_size.split("x"))
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
    """Analysis of the content theme for consistent tweet and image generation."""
    main_topic: str = Field(description="Main topic or subject of the content")
    key_themes: List[str] = Field(description="Key themes to emphasize")
    visual_style: str = Field(description="Overall visual style for images")
    tone: str = Field(description="Tone of the content (e.g., professional, authoritative, engaging)")
    target_audience: List[str] = Field(description="Target audience segments")
    color_scheme: List[str] = Field(description="Suggested color scheme")
    keywords: List[str] = Field(description="SEO and engagement-optimized keywords")

class TwitterThreadStructure(BaseModel):
    """Structure for the Twitter thread."""
    main_tweet: str = Field(description="Main tweet text (max 1024 chars)")
    thread_tweets: List[str] = Field(description="Follow-up tweets in thread")
    image_prompts: List[str] = Field(description="Image generation prompts")
    hashtags: List[str] = Field(description="Strategic hashtags")
    mentions: List[str] = Field(description="Relevant accounts to mention")
    call_to_action: str = Field(description="Engagement-driving CTA")

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

class TwitterPost(BaseModel):
    """Final Twitter post content."""
    main_tweet: str = Field(description="Formatted main tweet")
    thread_tweets: List[str] = Field(description="Formatted thread tweets")
    images: List[GeneratedImage] = Field(description="Generated images")
    hashtags: str = Field(description="Formatted hashtags")
    mentions: str = Field(description="Formatted mentions")
    call_to_action: str = Field(description="Engagement-driving CTA")

# Node: Analyze Content Theme
@Nodes.structured_llm_node(
    system_prompt="""You are an expert Twitter content strategist and thought leader.
    Your expertise lies in crafting viral threads, professional insights, and engaging content.
    Focus on Twitter-specific best practices:
    - Professional and authoritative tone
    - Clear and concise messaging
    - Strategic thread structure
    - Engagement-driving elements
    - Data-backed insights
    - Industry expertise signals""",
    output="content_theme",
    response_model=ContentTheme,
    max_tokens=MAX_TOKENS["analysis"],
    prompt_template="""
Analyze the following content for Twitter optimization:

Content: {{content}}

Create a comprehensive theme analysis optimized for Twitter's professional audience:

1. Content Strategy:
- Identify key professional insights and thought leadership angles
- Map content to current industry trends and discussions
- Define content pillars that showcase expertise
- Structure information for maximum value delivery

2. Engagement Optimization:
- Professional hashtag categories (#Tech, #AI, #Innovation, etc.)
- Key opinion leader (KOL) engagement opportunities
- Thread structure recommendations
- Viral-worthy elements for professional content

3. Visual Direction:
- Professional visual aesthetic that builds credibility
- Data visualization opportunities
- Brand-consistent visual elements
- High-impact image compositions

Ensure the analysis focuses on:
- Professional authority building
- Industry thought leadership
- Credibility signals
- Engagement from decision makers
- Network growth opportunities
- Knowledge sharing value
"""
)
async def analyze_content_theme(content: str, model: str) -> ContentTheme:
    """Analyze the content theme for consistent generation."""
    logger.debug(f"analyze_content_theme called with model: {model}")
    pass

# Node: Generate Post Structure
@Nodes.structured_llm_node(
    system_prompt="""You are an elite Twitter content strategist specializing in professional thought leadership.
    Your expertise lies in crafting detailed, data-rich threads that establish deep expertise.
    Focus on creating comprehensive content that:
    - Maximizes the 1024-character limit for each tweet
    - Delivers rich, actionable insights
    - Supports claims with specific data points
    - Includes relevant case studies
    - Provides step-by-step strategies
    - Uses professional formatting for readability""",
    output="post_structure",
    response_model=TwitterThreadStructure,
    max_tokens=MAX_TOKENS["tweet_structure"],
    prompt_template="""
Create a detailed, professionally optimized Twitter thread based on:

Theme Analysis:
{{content_theme.visual_style}}
{{content_theme.tone}}
{{content_theme.key_themes}}

Number of Images: {{num_images}}

Craft a comprehensive thread that maximizes:
1. Information Density
2. Professional Authority
3. Actionable Insights
4. Data-Backed Claims
5. Engagement Through Value

Structure Requirements:

🎯 Main Tweet (USE FULL 1024 CHARS):
- Start with a powerful statistic or insight
- Include specific numbers/percentages
- Highlight clear business value
- Add intrigue for thread continuation
- Use professional formatting

🧵 Thread Structure (5-10 tweets, EACH 1024 CHARS):
1. Context Setting Tweet:
   - Industry background
   - Current challenges
   - Market dynamics

2. Data & Analysis Tweets:
   - Specific statistics
   - Research findings
   - Market trends
   - Expert opinions

3. Strategy Tweets:
   - Step-by-step approaches
   - Implementation tips
   - Best practices
   - Common pitfalls

4. Case Study Tweets:
   - Real-world examples
   - Success metrics
   - Key learnings
   - ROI data

5. Action Plan Tweets:
   - Immediate next steps
   - Resource recommendations
   - Implementation timeline
   - Success metrics

#️⃣ Strategic Hashtags (2-3 per tweet):
- Industry-specific (#AI, #Tech, #Innovation)
- Role-based (#CTO, #TechLeaders)
- Topic-specific (#DataScience, #MachineLearning)

🖼️ Image Requirements:
- Data visualizations
- Process diagrams
- Comparison charts
- Strategy frameworks
- Result snapshots

Optimization Requirements:
- Use full character limit (1024) for each tweet
- Include specific numbers and percentages
- Add line breaks for readability
- Use professional bullet points
- Include source citations
- Tag relevant experts
- Add data visualizations
"""
)
async def generate_post_structure(
    content_theme: ContentTheme,
    num_images: int,
    model: str
) -> TwitterThreadStructure:
    """Generate the structure for a Twitter thread."""
    logger.debug(f"generate_post_structure called with model: {model}")
    pass

# Node: Generate Single Image Spec
@Nodes.structured_llm_node(
    system_prompt="""You are a professional AI image prompt engineer specializing in Twitter-optimized visuals.
    Your expertise lies in creating prompts that generate credible, professional, and engaging images.
    Focus on visual elements that build authority and drive engagement on Twitter.""",
    output="image_spec",
    response_model=ImageGenerationSpec,
    max_tokens=MAX_TOKENS["image_spec"],
    prompt_template="""
Create a Twitter-optimized image generation specification:

Content Theme:
{{content_theme.visual_style}}
{{content_theme.color_scheme}}

Base Prompt:
{{current_prompt}}

Technical Parameters:
- Style Presets: {{available_styles|join(", ")}}
- Available Sizes: {{available_sizes|join(", ")}}
- Generator: {{image_generator}}

Generate a specification that creates:
1. Main Prompt:
   - Professional composition
   - Authority-building elements
   - Clean, modern aesthetics
   - Brand-aligned visuals
   - Twitter-optimized layout

2. Negative Prompt:
   - Exclude unprofessional elements
   - Remove distracting features
   - Prevent credibility-damaging artifacts
   - Avoid controversial elements

3. Technical Settings:
   - Twitter-optimized dimensions
   - Professional quality settings
   - Clarity-focused parameters
   - Engagement-optimized layout

Focus on creating images that:
- Build professional authority
- Drive engagement
- Support credibility
- Enhance message impact
- Encourage sharing
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
    post_structure: TwitterThreadStructure,
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
                # Convert Twitter size to Stable Diffusion size
                model_size = convert_twitter_to_model_size(spec.size, "stable_diffusion")
                
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
                # Convert Twitter size to DALL-E size
                model_size = convert_twitter_to_model_size(spec.size, "dalle")
                
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
                
                # logger.info(f"Generated image using {image_generator}: {local_path}")
                
    except Exception as e:
        logger.error(f"Error generating image with {image_generator}: {str(e)}")
        raise
    
    return generated_images

# Node: Compile Twitter Post
@Nodes.define(output="twitter_post")
async def compile_twitter_post(
    post_structure: TwitterThreadStructure,
    generated_images: Optional[List[GeneratedImage]] = None
) -> TwitterPost:
    """Compile the final Twitter post with images and formatted text."""
    try:
        # Format hashtags with proper spacing and line breaks
        formatted_hashtags = "\n\n" + " ".join(f"#{tag}" for tag in post_structure.hashtags)
        
        # Create the final post object
        post = TwitterPost(
            main_tweet=post_structure.main_tweet,
            thread_tweets=post_structure.thread_tweets,
            images=generated_images or [],  # Use empty list if generated_images is None
            hashtags=formatted_hashtags,
            mentions=" ".join(f"@{mention}" for mention in post_structure.mentions),
            call_to_action=post_structure.call_to_action
        )
        
        logger.info("Compiled Twitter post successfully")
        return post
    
    except Exception as e:
        logger.error(f"Error compiling Twitter post: {e}")
        raise

# Create the workflow
def create_twitter_content_workflow(generate_images: bool = True) -> Workflow:
    """Create a workflow for Twitter content generation."""
    if generate_images:
        workflow = (
            Workflow("analyze_content_theme")
            .then("generate_post_structure")
            .then("generate_image_specs")
            .then("generate_images")
            .then("compile_twitter_post")
        )
    else:
        workflow = (
            Workflow("analyze_content_theme")
            .then("generate_post_structure")
            .then("compile_twitter_post")
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
                "available_sizes": lambda _: list(TWITTER_SIZES.values()),
                "image_generator": "image_generator"
            }
        })
    
    return workflow

async def generate_twitter_content(
    content_context: str,
    num_images: int = 1,
    generate_images: bool = True,
    analysis_model: str = "gemini/gemini-2.0-flash",
    content_model: str = "gemini/gemini-2.0-flash",
    image_model: str = "gemini/gemini-2.0-flash",
    image_generator: str = "stable_diffusion",
    task_id: str = "default",
    _handle_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
) -> TwitterPost:
    """Generate Twitter content with optional images from a context."""
    
    if generate_images:
        if num_images < 0 or num_images > 3:
            raise ValueError("Number of images must be between 0 and 3")
        
        if image_generator not in ["stable_diffusion", "dalle"]:
            raise ValueError("image_generator must be either 'stable_diffusion' or 'dalle'")
    else:
        num_images = 0
    
    initial_context = {
        "content_context": content_context,
        "num_images": num_images,
        "analysis_model": analysis_model,
        "content_model": content_model,
        "image_model": image_model,
        "image_generator": image_generator
    }
    
    logger.info(f"Starting Twitter content generation {'with' if generate_images else 'without'} images")
    
    try:
        workflow = create_twitter_content_workflow(generate_images)
        engine = workflow.build() 
        
        result = await engine.run(initial_context)
        
        if not isinstance(result.get("twitter_post"), TwitterPost):
            raise ValueError("Workflow did not produce a valid Twitter post")
        
        logger.info("Twitter content generation completed successfully")
        return result["twitter_post"]
        
    except Exception as e:
        logger.error(f"Error generating Twitter content: {e}")
        raise

# Add main function for testing
async def main():
    """Test function for the Twitter content generation workflow."""
    # Test content
    test_content = """
    Exploring the future of AI and machine learning! 🤖
    
    Today we're diving deep into how artificial intelligence is transforming 
    the way we work and live. From smart assistants to autonomous systems,
    the possibilities are endless.
    
    Key areas we're exploring:
    - Machine Learning
    - Neural Networks
    - Natural Language Processing
    - Computer Vision
    """
    
    try:
        # Test without images
        console.print("\n[bold blue]Testing without images...[/]")
        post_no_images = await generate_twitter_content(
            content_context=test_content,
            generate_images=False
        )
        console.print("[bold green]✓ Text-only post generated[/]")
        console.print(Panel(Markdown(post_no_images.main_tweet), title="Generated Main Tweet (No Images)"))
        
        # Test with DALL-E
        console.print("\n[bold blue]Testing with DALL-E...[/]")
        post_dalle = await generate_twitter_content(
            content_context=test_content,
            num_images=0,
            generate_images=False,
            image_generator="dalle"
        )
        console.print("[bold green]✓ DALL-E test completed[/]")
        console.print(Panel(Markdown(post_dalle.main_tweet), title="Generated Main Tweet with Image"))
        
    except Exception as e:
        console.print(f"[bold red]Error during testing:[/] {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
