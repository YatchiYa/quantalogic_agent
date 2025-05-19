"""Oriented LLM Tool for generating answers to questions using a language model with customizable role."""

import asyncio
from typing import Callable, Optional

from loguru import logger
from pydantic import ConfigDict, Field

from quantalogic.console_print_token import console_print_token
from quantalogic.event_emitter import EventEmitter
from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.tools.tool import Tool, ToolArgument


class OrientedLLMTool(Tool):
    """Tool to generate answers using a specified language model with a customizable role.
    
    This tool extends the basic LLM functionality by allowing explicit definition of:
    - A specific role for the model (e.g., "scientist", "poet", "programmer")
    - Custom name and description for better integration in agent frameworks
    - Streamlined initialization with all parameters in a single constructor
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(...)
    description: str = Field(...)
    arguments: list = Field(
        default=[
            ToolArgument(
                name="prompt",
                arg_type="string",
                description="The question or instruction for the language model. Use interpolation if needed (e.g., $var1$).",
                required=True,
                example="Explain the concept of $var1$ in simple terms.",
            ),
            ToolArgument(
                name="temperature",
                arg_type="string",
                description='Sampling temperature between "0.0" and "1.0": "0.0" for deterministic responses, "1.0" for maximum creativity. (float)',
                required=False,
                default="0.5",
                example="0.7",
            ),
        ]
    )

    model_name: str = Field(..., description="The name of the language model to use")
    role: str = Field(..., description="The role or persona the language model should adopt")
    on_token: Optional[Callable] = Field(default=None, exclude=True)
    generative_model: Optional[GenerativeModel] = Field(default=None, exclude=True)
    event_emitter: Optional[EventEmitter] = Field(default=None, exclude=True)

    def __init__(
        self,
        model_name: str,
        role: str,
        name: str,
        description: str,
        on_token: Optional[Callable] = None,
        generative_model: Optional[GenerativeModel] = None,
        event_emitter: Optional[EventEmitter] = None,
    ):
        """Initialize the OrientedLLMTool with model configuration, role, and metadata.

        Args:
            model_name (str): The name of the language model to use.
            role (str): The role or persona the language model should adopt.
            name (str): Custom name for the tool instance.
            description (str): Custom description of what the tool does.
            on_token (Callable, optional): Callback function for streaming tokens.
            generative_model (GenerativeModel, optional): Pre-initialized generative model.
            event_emitter (EventEmitter, optional): Event emitter for handling events.
        """
        # Use dict to pass validated data to parent constructor
        super().__init__(
            **{
                "model_name": model_name,
                "role": role,
                "name": name,
                "description": description,
                "on_token": on_token,
                "generative_model": generative_model,
                "event_emitter": event_emitter,
            }
        )
        
        # Initialize the generative model
        self.model_post_init(None)

    def model_post_init(self, __context):
        """Initialize the generative model after model initialization."""
        if self.generative_model is None:
            self.generative_model = GenerativeModel(
                model=self.model_name,
                event_emitter=self.event_emitter
            )
            logger.debug(f"Initialized OrientedLLMTool with model: {self.model_name}")

        # Only set up event listener if on_token is provided
        if self.on_token is not None:
            logger.debug(f"Setting up event listener for OrientedLLMTool with model: {self.model_name}")
            self.generative_model.event_emitter.on("stream_chunk", self.on_token)

    def _construct_system_prompt(self) -> str:
        """Construct the system prompt using the defined role.
        
        Returns:
            str: The formatted system prompt.
        """
        return f"You are {self.role}. Respond to all queries in this role."

    async def async_execute(self, prompt: str, temperature: Optional[str] = "0.5", step_number: Optional[int] = None, total_steps: Optional[int] = None) -> str:
        """Execute the tool to generate an answer asynchronously.

        Args:
            prompt (str): The question or instruction to be answered.
            temperature (str, optional): Sampling temperature. Defaults to "0.5".
            step_number (int, optional): Current step number in a multi-step process.
            total_steps (int, optional): Total number of steps in a multi-step process.

        Returns:
            str: The generated answer with progress tracking metadata.

        Raises:
            ValueError: If temperature is not a valid float between 0 and 1.
            Exception: If there's an error during response generation.
        """
        try:
            temp = float(temperature)
            if not (0.0 <= temp <= 1.0):
                raise ValueError("Temperature must be between 0 and 1.")
        except ValueError as ve:
            logger.error(f"Invalid temperature value: {temperature}")
            raise ValueError(f"Invalid temperature value: {temperature}") from ve

        system_prompt = self._construct_system_prompt()

        # Prepare the messages history
        messages_history = [
            Message(role="system", content=system_prompt),
        ]

        is_streaming = self.on_token is not None

        # Set the model's temperature
        if self.generative_model:
            self.generative_model.temperature = temp

            # Generate the response asynchronously using the generative model
            try:
                result = await self.generative_model.async_generate_with_history(
                    messages_history=messages_history, prompt=prompt, streaming=is_streaming
                )

                if is_streaming:
                    response = ""
                    async for chunk in result:
                        response += chunk
                        # Note: on_token is handled via the event emitter set in model_post_init
                else:
                    response = result.response

                logger.debug(f"Generated async response: {response}")
                
                # Create a structured response with progress tracking metadata
                progress_info = ""
                if step_number is not None and total_steps is not None:
                    progress_percentage = round((step_number / total_steps) * 100)
                    progress_info = f"\n\n[PROGRESS_METADATA]\nStep: {step_number}/{total_steps}\nProgress: {progress_percentage}%\nTool: {self.name}\nStatus: SUCCESS\n[/PROGRESS_METADATA]"
                
                # Format the response with tool name and progress tracking
                formatted_response = f"{self.name} Tool executed successfully, \nResult: \n{response}{progress_info}"
                
                return formatted_response
            except Exception as e:
                logger.error(f"Error generating async response: {e}")
                error_msg = f"Error generating async response: {e}"
                
                # Include error in progress metadata
                if step_number is not None and total_steps is not None:
                    error_msg += f"\n\n[PROGRESS_METADATA]\nStep: {step_number}/{total_steps}\nTool: {self.name}\nStatus: ERROR\nError: {str(e)}\n[/PROGRESS_METADATA]"
                
                raise Exception(error_msg) from e
        else:
            raise ValueError("Generative model not initialized")

    def execute(self, prompt: str, temperature: Optional[str] = "0.5", step_number: Optional[int] = None, total_steps: Optional[int] = None) -> str:
        """Execute the tool to generate an answer synchronously.

        Args:
            prompt (str): The question or instruction to be answered.
            temperature (str, optional): Sampling temperature. Defaults to "0.5".
            step_number (int, optional): Current step number in a multi-step process.
            total_steps (int, optional): Total number of steps in a multi-step process.

        Returns:
            str: The generated answer with progress tracking metadata.
        """
        return asyncio.run(self.async_execute(prompt=prompt, temperature=temperature, step_number=step_number, total_steps=total_steps))


if __name__ == "__main__":
    # Example usage of OrientedLLMTool
    scientist_tool = OrientedLLMTool(
        model_name="openrouter/openai/gpt-4o-mini",
        role="an expert scientist specialized in quantum physics",
        name="quantum_expert",
        description="A tool that provides expert explanations on quantum physics concepts",
        on_token=console_print_token
    )
    
    question = "What is quantum entanglement?"
    
    # Synchronous execution
    answer = scientist_tool.execute(prompt=question)
    print("\nSynchronous Answer:")
    print(answer)
    
    # Asynchronous execution with streaming
    print("\nAsynchronous Answer with streaming:")
    async_answer = asyncio.run(scientist_tool.async_execute(prompt=question))
    
    # Display tool configuration in Markdown
    print("\nTool Configuration:")
    print(scientist_tool.to_markdown())
