"""LLM Vision Tool for analyzing images using a language model."""

import asyncio
import base64
from typing import Callable, Optional
import os.path
from loguru import logger
from pydantic import ConfigDict, Field

from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.tools.tool import Tool, ToolArgument

# DEFAULT_MODEL_NAME = "ollama/llama3.2-vision"
DEFAULT_MODEL_NAME = "openai/gpt-4o-mini"


class LLMVisionTool(Tool):
    """Tool to analyze images using a specified language model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="llm_vision_tool")
    description: str = Field(
        default=(
            "Analyzes images and generates responses using a specified language model. "
            "Supports multimodal input combining text and images."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="system_prompt",
                arg_type="string",
                description="The system prompt to guide the model's behavior",
                required=True,
                example="You are an expert in image analysis and visual understanding.",
            ),
            ToolArgument(
                name="prompt",
                arg_type="string",
                description="The question or instruction about the image",
                required=True,
                example="What is shown in this image?",
            ),
            ToolArgument(
                name="image_url",
                arg_type="string",
                description="URL of the image to analyze, or local file path prefixed with file://",
                required=True,
                example="https://example.com/image.jpg",
            ),
            ToolArgument(
                name="temperature",
                arg_type="string",
                description='Sampling temperature between "0.0" and "1.0"',
                required=True,
                default="0.7",
                example="0.7",
            ),
        ]
    )

    model_name: str = Field(default=DEFAULT_MODEL_NAME, description="The name of the language model to use")
    on_token: Callable | None = Field(default=None, exclude=True)
    generative_model: Optional[GenerativeModel] = Field(default=None)

    @classmethod
    def from_tool_config(cls, config: dict) -> 'LLMVisionTool':
        """Create a LLMVisionTool instance from a tool configuration dictionary.
        
        Args:
            config: Dictionary containing tool configuration parameters
            
        Returns:
            LLMVisionTool instance
        """
        logger.info(f"Creating LLMVisionTool from config: {config}")
        
        # Extract parameters from config
        parameters = config.get('parameters', {})
        
        # Get model name from parameters, with fallback to vision_model_name
        model_name = parameters.get('model_name') or parameters.get('vision_model_name')
        logger.info(f"Extracted model_name from config: {model_name}")
        
        # Create tool instance
        return cls(model_name=model_name)

    def __init__(
        self,
        model_name: str | None = None,
        on_token: Callable | None = None,
        name: str = "llm_vision_tool",
        generative_model: GenerativeModel | None = None,
    ):
        """Initialize the LLMVisionTool with model configuration and optional callback."""
        logger.info(f"Initializing LLMVisionTool with model_name={model_name}, name={name}")
        
        # Use default model if none provided
        if model_name is None:
            model_name = DEFAULT_MODEL_NAME
            logger.info(f"No model name provided, using default: {DEFAULT_MODEL_NAME}")
        
        # Use dict to pass validated data to parent constructor
        super().__init__(
            **{
                "model_name": model_name,
                "on_token": on_token,
                "name": name,
                "generative_model": generative_model,
            }
        )

        # Initialize the generative model
        self.model_post_init(None)
        logger.info(f"LLMVisionTool initialization complete for model {model_name}")

    def model_post_init(self, __context):
        """Initialize the generative model after model initialization."""
        logger.info("Starting model_post_init")
        if self.generative_model is None:
            logger.info(f"Creating new GenerativeModel instance with model: {self.model_name}")
            self.generative_model = GenerativeModel(model=self.model_name)
            logger.info(f"Initialized LLMVisionTool with model: {self.model_name}")

        # Only set up event listener if on_token is provided
        if self.on_token is not None:
            logger.info("Setting up event listener for token streaming")
            self.generative_model.event_emitter.on("stream_chunk", self.on_token)
            logger.info("Event listener setup complete")
        logger.info("Completed model_post_init")

    def execute(self, system_prompt: str, prompt: str, image_url: str, temperature: str = "0.7") -> str:
        """Execute the tool to analyze an image and generate a response."""
        logger.info(f"Starting synchronous execution with prompt: {prompt[:50]}...")
        return asyncio.run(
            self.async_execute(system_prompt=system_prompt, prompt=prompt, image_url=image_url, temperature=temperature)
        )

    async def async_execute(self, system_prompt: str, prompt: str, image_url: str, temperature: str = "0.7") -> str:
        """Execute the tool to analyze an image and generate a response asynchronously."""
        logger.info(f"Starting async execution with temperature={temperature}")
        logger.info(f"System prompt: {system_prompt[:50]}...")
        logger.info(f"User prompt: {prompt[:50]}...")
        logger.info(f"Image URL: {image_url[:50]}...")

        try:
            logger.info("Validating temperature")
            temp = float(temperature)
            if not (0.0 <= temp <= 1.0):
                logger.error(f"Temperature {temp} is outside valid range [0.0, 1.0]")
                raise ValueError("Temperature must be between 0 and 1.")
        except ValueError as ve:
            logger.error(f"Invalid temperature value: {temperature}")
            raise ValueError(f"Invalid temperature value: {temperature}") from ve

        # Handle local files
        logger.info("Processing image URL")
        if image_url.startswith("file://"):
            file_path = image_url[7:]  # Remove 'file://' prefix
            logger.info(f"Processing local file: {file_path}")
            
            if not os.path.exists(file_path):
                logger.error(f"Local file not found: {file_path}")
                raise ValueError(f"Local file not found: {file_path}")
            
            # Read and encode the image
            try:
                logger.info("Reading and encoding local image file")
                with open(file_path, "rb") as image_file:
                    image_data = image_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    image_url = f"data:image/png;base64,{base64_image}"
                logger.info("Successfully encoded local image to base64")
            except Exception as e:
                logger.error(f"Error reading local file: {e}")
                raise ValueError(f"Error reading local file: {e}") from e
        elif not image_url.startswith(("http://", "https://", "data:")):
            logger.error(f"Invalid image URL format: {image_url[:50]}...")
            raise ValueError("Image URL must start with http://, https://, file://, or be a base64 data URL")

        # Prepare the messages history
        logger.info("Preparing message history")
        messages_history = [
            Message(role="system", content=system_prompt),
        ]

        if self.generative_model is None:
            logger.info(f"Creating new GenerativeModel instance with model: {self.model_name}")
            self.generative_model = GenerativeModel(model=self.model_name)

        self.generative_model.temperature = temp
        logger.info(f"Set model temperature to {temp}")

        try:
            is_streaming = self.on_token is not None
            logger.info(f"Starting generation with streaming={is_streaming}")
            response_stats = await self.generative_model.async_generate_with_history(
                messages_history=messages_history, prompt=prompt, image_url=image_url, streaming=is_streaming
            )

            if is_streaming:
                logger.info("Processing streaming response")
                response = ""
                async for chunk in response_stats:
                    response += chunk
                    logger.info(f"Received chunk of length: {len(chunk)}")
            else:
                logger.info("Processing non-streaming response")
                response = response_stats.response.strip()

            logger.info(f"Generated response of length: {len(response)}")
            logger.info(f"Response preview: {response[:100]}...")
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise Exception(f"Error generating response: {e}") from e


if __name__ == "__main__":
    # Set up logging configuration
    logger.remove()  # Remove default handler
    logger.add(
        "llm_vision_tool.log",
        rotation="1 MB",
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(lambda msg: print(msg), level="INFO")  # Console output for INFO and above
    
    logger.info("Starting LLMVisionTool example usage")
    # Example usage
    tool = LLMVisionTool(model_name=DEFAULT_MODEL_NAME)
    system_prompt = "You are a vision expert."
    question = "What is shown in this image? Describe it with details."
    temperature = "0.7"
    
    # Test with random online image
    # image_url = "https://fastly.picsum.photos/id/767/200/300.jpg?hmac=j5YA1cRw-jS6fK3Mx2ooPwl2_TS3RSyLmFmiM9TqLC4"
    # print("\nTesting with online image:")
    # answer = tool.execute(system_prompt=system_prompt, prompt=question, image_url=image_url, temperature=temperature)
    # print(answer)
    
    # Test with local image
    local_image_path = "file:///home/yarab/Bureau/trash_agents_tests/f1/generated_images/dalle_20250419_190711.png"
    print("\nTesting with local image:")
    answer = tool.execute(system_prompt=system_prompt, prompt=question, image_url=local_image_path, temperature=temperature)
    print(answer)
