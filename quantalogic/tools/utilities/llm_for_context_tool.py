"""LLM Tool with Context for generating answers to questions using a language model with provided context."""

import asyncio
from typing import Callable, Optional

from loguru import logger
from pydantic import ConfigDict, Field

from quantalogic.event_emitter import EventEmitter
from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.tools.tool import Tool, ToolArgument


class DocumentLLMTool(Tool):
    """Tool to generate answers using a specified language model with additional context.
    
    This tool extends the basic LLM functionality by allowing context to be provided during initialization.
    The context can be used in prompts to provide additional information to the language model.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="llm_for_context_tool")
    description: str = Field(
        default=(
            "Answers questions about documents or content that has been uploaded to the chat. "
            "This tool receives parsed document content as context and uses it to provide accurate, "
            "contextually relevant responses about the documents. The context string contains "
            "the content of one or more files that have been parsed and prepared for analysis. "
            "Use this tool when the user asks questions about documents they've uploaded."
        )
    )
    context: str = Field(
        default="", 
        description="Context string that can be referenced in prompts"
    )
    model_name: str = Field(..., description="The name of the language model to use")
    system_prompt: Optional[str] = Field(default=None)
    on_token: Optional[Callable] = Field(default=None, exclude=True)
    generative_model: Optional[GenerativeModel] = Field(default=None, exclude=True)
    event_emitter: Optional[EventEmitter] = Field(default=None, exclude=True)
    
    arguments: list = Field(
        default=[
            ToolArgument(
                name="system_prompt",
                arg_type="string",
                description=("The persona or system prompt to guide the language model's behavior. "
                             "Can reference context using $context$ syntax."),
                required=True,
                example=("You are an expert in natural language processing. Here is some context: $context$"),
            ),
            ToolArgument(
                name="prompt",
                arg_type="string",
                description=("The question to ask the language model. "
                             "Can reference context using $context$ syntax."),
                required=True,
                example="Based on the provided context, what is the significance of neural networks?",
            ),
            ToolArgument(
                name="temperature",
                arg_type="string",
                description='Sampling temperature between "0.0" and "1.0": "0.0" no creativity, "1.0" full creativity. (float)',
                required=False,
                default="0.5",
                example="0.5",
            ),
        ]
    )

    def __init__(
        self,
        model_name: str,
        context: str = "",
        system_prompt: str | None = None,
        on_token: Callable | None = None,
        name: str = "llm_context_tool",
        generative_model: GenerativeModel | None = None,
        event_emitter: EventEmitter | None = None,
    ):
        """Initialize the LLMForContextTool with model configuration, context, and optional callback.

        Args:
            model_name (str): The name of the language model to use.
            context (str, optional): Context string that can be referenced in prompts.
            system_prompt (str, optional): Default system prompt for the model.
            on_token (Callable, optional): Callback function for streaming tokens.
            name (str): Name of the tool instance. Defaults to "llm_context_tool".
            generative_model (GenerativeModel, optional): Pre-initialized generative model.
            event_emitter (EventEmitter, optional): Event emitter for handling events.
        """
        # Use dict to pass validated data to parent constructor
        super().__init__(
            **{
                "model_name": model_name,
                "context": context or "",
                "system_prompt": system_prompt,
                "on_token": on_token,
                "name": name,
                "generative_model": generative_model,
                "event_emitter": event_emitter,
            }
        )
        
        # Initialize the generative model
        self.model_post_init(None)
        
        logger.debug(f"Initialized LLMForContextTool with model: {self.model_name} and context length: {len(self.context)}")

    def _interpolate_context(self, text: str) -> str:
        """Replace context references in text with the context string.
        
        Args:
            text (str): Text containing context references in $context$ format.
            
        Returns:
            str: Text with context references replaced with the context string.
        """
        if not text or not self.context:
            return text
            
        # Replace all instances of $context$ with the context string
        return text.replace("$context$", self.context)

    def model_post_init(self, __context):
        """Initialize the generative model after model initialization."""
        if self.generative_model is None:
            self.generative_model = GenerativeModel(
                model=self.model_name,
                event_emitter=self.event_emitter
            )
            logger.debug(f"Initialized LLMForContextTool with model: {self.model_name}")

        # Only set up event listener if on_token is provided
        if self.on_token is not None:
            logger.debug(f"Setting up event listener for LLMForContextTool with model: {self.model_name}")
            self.generative_model.event_emitter.on("stream_chunk", self.on_token)
            
    def execute(self, system_prompt: str | None = None, prompt: str | None = None, temperature: str | None = None) -> str:
        """Execute the tool to generate an answer.

        Args:
            system_prompt (str, optional): The system prompt to guide the model.
            prompt (str, optional): The question to be answered.
            temperature (str, optional): Sampling temperature. Defaults to "0.5".

        Returns:
            str: The generated answer.

        Raises:
            ValueError: If temperature is not a valid float between 0 and 1.
            Exception: If there's an error during response generation.
        """
        return asyncio.run(self.async_execute(system_prompt, prompt, temperature))
    
    async def async_execute(
        self, system_prompt: str | None = None, prompt: str | None = None, temperature: str | None = "0.5"
    ) -> str:
        """Execute the tool to generate an answer asynchronously with context interpolation.

        Args:
            system_prompt (str, optional): The system prompt to guide the model.
            prompt (str, optional): The question to be answered.
            temperature (str, optional): Sampling temperature. Defaults to "0.5".

        Returns:
            str: The generated answer.

        Raises:
            ValueError: If temperature is not a valid float between 0 and 1.
            Exception: If there's an error during response generation.
        """
        try:
            temp = float(temperature) if temperature is not None else 0.5
            if not (0.0 <= temp <= 1.0):
                raise ValueError("Temperature must be between 0 and 1.")
        except ValueError as ve:
            logger.error(f"Invalid temperature value: {temperature}")
            raise ValueError(f"Invalid temperature value: {temperature}") from ve

        # Use provided system prompt or default
        used_system_prompt = system_prompt or self.system_prompt or "You are a helpful assistant."
        
        # Interpolate context in system prompt and prompt
        interpolated_system_prompt = self._interpolate_context(used_system_prompt)
        interpolated_prompt = self._interpolate_context(prompt)

        # Prepare the messages history
        messages_history = [
            Message(role="system", content=interpolated_system_prompt),
        ]

        is_streaming = self.on_token is not None

        # Set the model's temperature
        if self.generative_model:
            self.generative_model.temperature = temp

            # Generate the response asynchronously using the generative model
            try:
                result = await self.generative_model.async_generate_with_history(
                    messages_history=messages_history, prompt=interpolated_prompt, streaming=is_streaming
                )

                if is_streaming:
                    response = ""
                    async for chunk in result:
                        response += chunk
                        # Note: on_token is handled via the event emitter set in model_post_init
                else:
                    response = result.response

                logger.debug(f"Generated async response: {response}")
                return response
            except Exception as e:
                logger.error(f"Error generating async response: {e}")
                raise Exception(f"Error generating async response: {e}") from e
        else:
            raise ValueError("Generative model not initialized")


if __name__ == "__main__":
    # Example usage of LLMForContextTool
    import asyncio
    from quantalogic.console_print_token import console_print_token
    
    # Create context
    context = "Neural networks are a fundamental component of modern machine learning, consisting of interconnected nodes organized in layers. They process data through weighted connections, learning patterns through backpropagation. Key types include CNNs for image processing, RNNs for sequential data, and transformers for NLP tasks."
    
    # Create tool with context
    tool = LLMForContextTool(
        model_name="openrouter/openai/gpt-4o-mini",
        context=context
    )
    
    # System prompt and question with context references
    system_prompt = 'You are an expert in machine learning. Use this context to inform your answers: $context$'
    question = "Based on the provided context, explain the importance of neural networks in modern AI applications."
    temperature = "0.7"

    # Synchronous execution
    answer = tool.execute(system_prompt=system_prompt, prompt=question, temperature=temperature)
    print("Synchronous Answer:")
    print(answer)

    # Asynchronous execution with streaming
    streaming_tool = LLMForContextTool(
        model_name="openrouter/openai/gpt-4o-mini", 
        context=context,
        on_token=console_print_token
    )
    
    async_answer = asyncio.run(
        streaming_tool.async_execute(system_prompt=system_prompt, prompt=question, temperature=temperature)
    )
    
    print("\nAsynchronous Answer:")
    print(f"Answer: {async_answer}")
