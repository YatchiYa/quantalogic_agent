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

MEDIUM_SIZES = [
    "1024x1024",    # square
    "1024x1792",    # portrait
    "1792x1024"     # landscape
]

# Constants for token limits
MAX_TOKENS = {
    "analysis": 1000,  # Increased for more detailed analysis
    "post_structure": 1500,  # Increased for richer content
    "image_spec": 800,  # Increased for better image prompts
    "article_structure": 2000  # Increased for more detailed article structure
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

class ArticleTheme(BaseModel):
    """Analysis of the article theme for consistent generation."""
    writing_style: str = Field(description="Writing style for the article")
    expertise_level: str = Field(description="Level of expertise for the article")
    key_themes: List[str] = Field(description="Key themes to emphasize")
    key_takeaways: List[str] = Field(description="Key takeaways for the reader")

class ArticleSection(BaseModel):
    """Structure for an article section."""
    heading: str = Field(description="Section heading")
    content: str = Field(description="Section content")

class ArticleStructure(BaseModel):
    """Structure for the Medium article."""
    title: str = Field(description="Title of the article")
    subtitle: str = Field(description="Subtitle of the article")
    introduction: str = Field(description="Introduction of the article")
    sections: List[ArticleSection] = Field(description="Main sections of the article")
    conclusion: str = Field(description="Conclusion of the article")
    tags: List[str] = Field(description="Tags for the article (max 5)")
    image_prompts: List[str] = Field(description="Image generation prompts")
    image_placements: List[str] = Field(description="Where to place each image")

class MediumArticle(BaseModel):
    """Final Medium article content."""
    title: str = Field(description="Title of the article")
    subtitle: str = Field(description="Subtitle of the article")
    content: str = Field(description="Content of the article in Markdown format")
    tags: List[str] = Field(description="Tags for the article")
    images: List[GeneratedImage] = Field(description="Generated images with placements")

# Node: Analyze Content Theme
@Nodes.structured_llm_node(
    system_prompt="""You are an expert Medium writer and content strategist.
    Your task is to analyze topics and create engaging, authoritative long-form content.
    Consider Medium's best practices, reader engagement patterns, and content quality standards.
    Focus on creating content that demonstrates expertise and provides deep, actionable insights.""",
    output="article_theme",
    response_model=ArticleTheme,
    max_tokens=MAX_TOKENS["analysis"],
    prompt_template="""
Analyze the following content for a high-quality Medium article:

Content: {{content}}

Create a comprehensive theme analysis focused on knowledge sharing:

1. Content Direction:
- Define the core thesis and key arguments
- Identify supporting evidence needed
- Map the intellectual journey for readers
- Plan knowledge scaffolding approach

2. Writing Strategy:
- Determine appropriate technical depth
- Plan narrative structure and flow
- Identify areas needing examples/code
- Define key concepts to explain

3. Reader Engagement:
- Map prerequisite knowledge
- Plan progressive disclosure of complex topics
- Identify potential questions/concerns
- Structure learning objectives

Ensure the analysis covers:
- Technical accuracy and depth
- Logical flow and progression
- Knowledge gaps to address
- Reader learning outcomes
"""
)
async def analyze_content_theme(content: str, model: str) -> ArticleTheme:
    """Analyze the content theme for consistent generation."""
    logger.debug(f"analyze_content_theme called with model: {model}")
    pass

# Node: Generate Post Structure
@Nodes.structured_llm_node(
    system_prompt="""You are an elite Facebook content creator specializing in professional business communication.
    Your expertise lies in crafting compelling business narratives, strategic CTAs, and engagement-driving content.
    Focus on creating content that builds authority, drives meaningful engagement, and achieves business objectives.""",
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

Craft a post that maximizes:
1. Professional Engagement
2. Brand Authority
3. Business Value
4. Community Building

Structure Requirements:

📝 Main Text (4-6 paragraphs):
- Professional Hook: Compelling opening that establishes authority
- Value Proposition: Clear business or professional value
- Supporting Points: Well-structured evidence or insights
- Strategic CTA: Professional call-to-action

#️⃣ Hashtags (5-10 total):
- Industry-specific hashtags
- Professional community tags
- Brand-relevant tags
- Campaign-specific tags
- Local business tags (if applicable)

🖼️ Image Prompts:
- Professional, high-quality visual descriptions
- Brand-aligned composition details
- Business-appropriate aesthetics

Optimization Requirements:
- Professional tone throughout
- Strategic emoji usage (business-appropriate)
- Include discussion prompts
- Proper formatting for readability
- Actionable business insights
- Shareable professional content
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
    Focus on current business visual trends, professional appeal, and brand consistency.""",
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

Generate a specification that creates:
1. Main Prompt:
   - Include professional composition details
   - Define business-appropriate lighting
   - Specify brand-aligned color palette
   - Add high-quality technical markers
   - Include Facebook feed optimization elements

2. Negative Prompt:
   - Exclude unprofessional elements
   - Remove business-inappropriate content
   - Prevent brand-misaligned elements
   - Avoid controversial content

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

# Node: Generate Article Structure
@Nodes.structured_llm_node(
    system_prompt="""You are an elite Medium writer specializing in technical and professional content.
    Your expertise lies in crafting engaging, authoritative articles that educate and inspire.
    Focus on creating content that demonstrates expertise while remaining accessible and actionable.""",
    output="article_structure",
    response_model=ArticleStructure,
    max_tokens=MAX_TOKENS["article_structure"],
    prompt_template="""
Create a comprehensive Medium article based on:

Theme Analysis:
{{article_theme.writing_style}}
{{article_theme.expertise_level}}
{{article_theme.key_themes}}
{{article_theme.key_takeaways}}

Number of Images: {{num_images}}

Craft an article that delivers:
1. Deep Understanding
2. Actionable Insights
3. Clear Examples
4. Expert Analysis

Structure Requirements:

📝 Title & Subtitle:
- Compelling, specific title (50-60 chars)
- Clear, descriptive subtitle (120-140 chars)

📋 Introduction:
- Hook with real-world relevance
- Clear problem statement
- Article roadmap
- Value proposition

📚 Main Sections:
- Clear section headings
- Progressive complexity
- Code examples if relevant
- Visual aids placement
- Transition paragraphs

✍️ Writing Style:
- Clear topic sentences
- Evidence-based arguments
- Technical accuracy
- Engaging examples
- Professional tone

🎯 Conclusion:
- Key takeaways
- Next steps/applications
- Call for discussion
- Resource links

🏷️ Tags (max 5):
- Topic-specific
- Technology-related
- Industry-relevant
- Skill-level appropriate

🖼️ Image Prompts:
- Professional illustrations
- Technical diagrams
- Concept visualizations
- Strategic placement

Optimization Requirements:
- Proper Markdown formatting
- Code block formatting
- Pull quote opportunities
- Strategic subheadings
- Mobile-friendly paragraphs
"""
)
async def generate_article_structure(
    article_theme: ArticleTheme,
    num_images: int,
    model: str
) -> ArticleStructure:
    """Generate the structure for a Medium article."""
    logger.debug(f"generate_article_structure called with model: {model}")
    pass

# Node: Compile Medium Article
@Nodes.define(output="medium_article")
async def compile_medium_article(
    article_structure: ArticleStructure,
    generated_images: Optional[List[GeneratedImage]] = None
) -> MediumArticle:
    """Compile the final Medium article with images and formatted text."""
    try:
        # Build the article content in Markdown format
        content_parts = []
        
        # Add title and subtitle
        content_parts.append(f"# {article_structure.title}\n")
        content_parts.append(f"### {article_structure.subtitle}\n")
        
        # Add introduction
        content_parts.append(f"\n{article_structure.introduction}\n")
        
        # Add main sections with images
        image_index = 0
        for section in article_structure.sections:
            content_parts.append(f"\n## {section.heading}\n")
            
            # Insert image if available and specified for this section
            if generated_images and image_index < len(generated_images):
                if generated_images[image_index].placement == section.heading:
                    content_parts.append(f"\n![{section.heading}]({generated_images[image_index].image_path})\n")
                    image_index += 1
            
            content_parts.append(f"\n{section.content}\n")
        
        # Add conclusion
        content_parts.append(f"\n## Conclusion\n")
        content_parts.append(f"\n{article_structure.conclusion}\n")
        
        # Create the final article object
        article = MediumArticle(
            title=article_structure.title,
            subtitle=article_structure.subtitle,
            content="\n".join(content_parts),
            tags=article_structure.tags,
            images=generated_images or []
        )
        
        logger.info("Compiled Medium article successfully")
        return article
    
    except Exception as e:
        logger.error(f"Error compiling Medium article: {e}")
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

def create_medium_content_workflow(generate_images: bool = True) -> Workflow:
    """Create a workflow for Medium article generation."""
    if generate_images:
        workflow = (
            Workflow("analyze_content_theme")
            .then("generate_article_structure")
            .then("generate_image_specs")
            .then("generate_images")
            .then("compile_medium_article")
        )
    else:
        workflow = (
            Workflow("analyze_content_theme")
            .then("generate_article_structure")
            .then("compile_medium_article")
        )
    
    # Add input mappings
    workflow.node_input_mappings = {
        "analyze_content_theme": {
            "model": "analysis_model"
        },
        "generate_article_structure": {
            "model": "content_model",
            "num_images": "num_images" if generate_images else lambda _: 0
        }
    }
    
    if generate_images:
        workflow.node_input_mappings.update({
            "generate_image_specs": {
                "model": "image_model",
                "available_styles": lambda _: list(STABLE_DIFFUSION_CONFIG["styles"]),
                "available_sizes": lambda _: list(MEDIUM_SIZES.values()),
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
    
    initial_context = {
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
        
        result = await engine.run(initial_context)
        
        if not isinstance(result.get("facebook_post"), FacebookPost):
            raise ValueError("Workflow did not produce a valid Facebook post")
        
        logger.info("Facebook content generation completed successfully")
        return result["facebook_post"]
        
    except Exception as e:
        logger.error(f"Error generating Facebook content: {e}")
        raise

async def generate_medium_content(
    content_context: str,
    num_images: int = 1,
    generate_images: bool = True,
    analysis_model: str = "gemini/gemini-2.0-flash",
    content_model: str = "gemini/gemini-2.0-flash",
    image_model: str = "gemini/gemini-2.0-flash",
    image_generator: str = "stable_diffusion",
    task_id: str = "default",
    _handle_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
) -> MediumArticle:
    """Generate Medium article with optional images from a context."""
    
    if generate_images:
        if num_images < 0 or num_images > 5:  # Medium articles typically support more images
            raise ValueError("Number of images must be between 0 and 5")
        
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
    
    logger.info(f"Starting Medium article generation {'with' if generate_images else 'without'} images")
    
    try:
        workflow = create_medium_content_workflow(generate_images)
        engine = workflow.build() 
        
        result = await engine.run(initial_context)
        
        if not isinstance(result.get("medium_article"), MediumArticle):
            raise ValueError("Workflow did not produce a valid Medium article")
        
        logger.info("Medium article generation completed successfully")
        return result["medium_article"]
        
    except Exception as e:
        logger.error(f"Error generating Medium article: {e}")
        raise

# Add main function for testing
async def main():
    """Test function for the Facebook content generation workflow."""
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
        # console.print("\n[bold blue]Testing without images...[/]")
        # post_no_images = await generate_facebook_content(
        #     content_context=test_content,
        #     generate_images=False
        # )
        # console.print("[bold green]✓ Text-only post generated[/]")
        # console.print(Panel(Markdown(post_no_images.caption), title="Generated Caption (No Images)"))
        
        # # Test with DALL-E
        # console.print("\n[bold blue]Testing with DALL-E...[/]")
        # post_dalle = await generate_facebook_content(
        #     content_context=test_content, 
        #     generate_images=False, 
        # )
        # console.print("[bold green]✓ Post with images completed[/]")
        # console.print(Panel(Markdown(post_dalle.caption), title="Generated Caption with Images"))
        
        # Test Medium article without images
        # console.print("\n[bold blue]Testing Medium article without images...[/]")
        # article_no_images = await generate_medium_content(
        #     content_context=test_content,
        #     generate_images=False
        # )
        # console.print("[bold green]✓ Text-only article generated[/]")
        # console.print(Panel(Markdown(article_no_images.content[:500] + "..."), title="Generated Article Preview"))
        
        # Test Medium article with DALL-E
        console.print("\n[bold blue]Testing Medium article with DALL-E...[/]")
        article_dalle = await generate_medium_content(
            content_context=test_content, 
            generate_images=False, 
        )
        console.print("[bold green]✓ Article with images completed[/]")
        console.print(Panel(Markdown(article_dalle.content + "..."), title="Generated Article Preview with Images"))
        
    except Exception as e:
        console.print(f"[bold red]Error during testing:[/] {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
