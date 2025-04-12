"""Tool for performing deep searches using the Perplexity API via litellm."""

import os
from typing import Any, Dict, List, Optional, Union, Generator
from loguru import logger
from quantalogic.tools.tool import Tool, ToolArgument
from litellm import completion

class PerplexityRequestsTool(Tool):
    """Tool for performing web searches using Perplexity API with support for various models and parameters."""

    name: str = "perplexity_requests"
    description: str = "Perform web searches using Perplexity API with support for various models, parameters and search options"
    need_validation: bool = False
    api_key: str | None = os.getenv("PERPLEXITYAI_API_KEY")  # Updated env var name to match litellm

    arguments: list = [
        ToolArgument(
            name="query",
            arg_type="string",
            description="The search query to perform",
            required=True,
            example="What are the latest developments in quantum computing?",
        ),
        ToolArgument(
            name="model",
            arg_type="string",
            description="Model to use (sonar-deep-research, sonar-reasoning-pro, sonar-reasoning, sonar-pro, sonar, r1-1776)",
            required=False,
            default="sonar-pro",
            example="sonar-pro",
        ),
        ToolArgument(
            name="max_tokens",
            arg_type="int",
            description="Maximum number of tokens to generate",
            required=False,
            default="1024",
            example="1024",
        ),
        ToolArgument(
            name="temperature",
            arg_type="float",
            description="Sampling temperature (0.0 to 2.0)",
            required=False,
            default="0.7",
            example="0.7",
        ),
        ToolArgument(
            name="top_p",
            arg_type="float",
            description="Nucleus sampling parameter",
            required=False,
            default="0.9",
            example="0.9",
        ),
        ToolArgument(
            name="stream",
            arg_type="boolean",
            description="Whether to stream the response",
            required=False,
            default="false",
            example="false",
        ),
        ToolArgument(
            name="presence_penalty",
            arg_type="float",
            description="Penalty for token presence",
            required=False,
            default="0.0",
            example="0.0",
        ),
        ToolArgument(
            name="frequency_penalty",
            arg_type="float",
            description="Penalty for token frequency",
            required=False,
            default="0.0",
            example="0.0",
        ),
    ]

    def _validate_api_key(self) -> None:
        """Validate that the API key is set.

        Raises:
            ValueError: If the API key is not set
        """
        if not self.api_key:
            raise ValueError(
                "Perplexity API key not found. Please set the PERPLEXITYAI_API_KEY environment variable."
            )

    def _prepare_messages(self, query: str) -> List[Dict[str, str]]:
        """Prepare the messages for the API request."""
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant that provides accurate, "
                    "detailed, and well-researched answers. Please include sources "
                    "for your information when available."
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

    def _format_response(self, response: Any) -> Dict[str, Any]:
        """Format the litellm response."""
        try:
            if hasattr(response, 'choices') and response.choices:
                result = {
                    "content": response.choices[0].message.content,
                    "model": response.model,
                    "usage": response.usage._asdict() if hasattr(response.usage, '_asdict') else dict(response.usage),
                }
                return result
            return {"content": "", "error": True}

        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return {"content": f"Error formatting response: {str(e)}", "error": True}

    def execute(
        self,
        query: str,
        model: str = "sonar-pro",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """Execute a search using the Perplexity API via litellm."""
        try:
            self._validate_api_key()

            # Validate and format model name
            valid_models = ["sonar-deep-research", "sonar-reasoning-pro", "sonar-reasoning", "sonar-pro", "sonar", "r1-1776"]
            if model not in valid_models:
                model = "sonar-pro"
                logger.warning(f"Invalid model '{model}', defaulting to 'sonar-pro'")

            model_name = f"perplexity/{model}"
            messages = self._prepare_messages(query)

            # Prepare parameters
            params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            }

            # Make the API request using litellm
            if stream:
                def generate_chunks():
                    for chunk in completion(**params):
                        yield self._format_response(chunk)
                return generate_chunks()
            else:
                response = completion(**params)
                return self._format_response(response)

        except Exception as e:
            error_msg = f"Error performing Perplexity search: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)


if __name__ == "__main__":
    tool = PerplexityRequestsTool()
    print(tool.to_markdown())
