"""Tool for performing deep searches using the Perplexity API via OpenAI client."""

import os
from typing import Literal, Any, Dict, List

from loguru import logger
from openai import OpenAI
from quantalogic.tools.tool import Tool, ToolArgument


class PerplexityDeepSearchTool(Tool):
    """Tool for performing deep web searches using the Perplexity API with enhanced search capabilities."""

    name: str = "perplexity_deep_search"
    description: str = "Perform deep web searches using Perplexity API with support for different models, streaming, and source retrieval."
    need_validation: bool = False
    api_key: str | None = os.getenv("PERPLEXITY_API_KEY")
    base_url: str = "https://api.perplexity.ai"

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
            description="Model to use (sonar-small-chat, sonar-medium-chat, sonar-pro)",
            required=False,
            default="sonar-pro",
            example="sonar-pro",
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
            name="include_sources",
            arg_type="boolean",
            description="Whether to include sources in the response",
            required=False,
            default="true",
            example="true",
        ),
    ]

    def _validate_api_key(self) -> None:
        """Validate that the API key is set.

        Raises:
            ValueError: If the API key is not set
        """
        if not self.api_key:
            raise ValueError(
                "Perplexity API key not found. Please set the PERPLEXITY_API_KEY environment variable."
            )

    def _format_response(self, response: Any, stream: bool = False, include_sources: bool = True) -> Dict[str, Any]:
        """Format the Perplexity API response.

        Args:
            response: The raw API response
            stream: Whether the response is streamed
            include_sources: Whether to include sources in the response

        Returns:
            Dict[str, Any]: Formatted response with content and sources
        """
        try:
            result = {}
            
            if stream:
                # For streaming responses, return the content
                content = response.choices[0].delta.content if hasattr(response.choices[0].delta, 'content') else ""
                result["content"] = content
            else:
                # For non-streaming responses, return the full message
                content = response.choices[0].message.content
                result["content"] = content

                # Extract sources if available and requested
                if include_sources and hasattr(response, 'citations'):
                    result["sources"] = response.citations
                elif include_sources:
                    # Try to extract sources from the tool_calls if available
                    tool_calls = getattr(response.choices[0].message, 'tool_calls', None)
                    if tool_calls:
                        sources = []
                        for call in tool_calls:
                            if call.function.name == "search":
                                sources.extend(call.function.arguments.get("sources", []))
                        result["sources"] = sources

            return result

        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return {"content": f"Error formatting response: {str(e)}", "sources": []}

    def execute(
        self,
        query: str,
        model: str = "sonar-pro",
        stream: bool = False,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """Perform a search using the Perplexity API.

        Args:
            query: The search query to perform
            model: Model to use (sonar-small-chat, sonar-medium-chat, sonar-pro)
            stream: Whether to stream the response
            include_sources: Whether to include sources in the response

        Returns:
            Dict[str, Any]: Dictionary containing response content and sources

        Raises:
            ValueError: If the API key is not set or if there's an error with the request
        """
        try:
            self._validate_api_key()

            # Validate model parameter
            valid_models = ["sonar-small-chat", "sonar-medium-chat", "sonar-pro"]
            if model not in valid_models:
                model = "sonar-pro"
                logger.warning(f"Invalid model '{model}', defaulting to 'sonar-pro'")

            # Initialize OpenAI client with Perplexity base URL
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            # Prepare messages
            messages = [
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

            # Make the API request
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                tools=[{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {"sources": {"type": "array", "items": {"type": "string"}}}}}}] if include_sources else None,
            )

            # Handle streaming and non-streaming responses
            if stream:
                full_response = {"content": "", "sources": []}
                for chunk in response:
                    logger.debug(f"Chunk: {chunk.choices[0].delta.content if hasattr(chunk.choices[0].delta, 'content') else ''}")
                    chunk_result = self._format_response(chunk, stream=True, include_sources=include_sources)
                    if chunk_result["content"]:
                        full_response["content"] += chunk_result["content"]
                return full_response
            else:
                return self._format_response(response, stream=False, include_sources=include_sources)

        except Exception as e:
            error_msg = f"Error performing Perplexity search: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)


if __name__ == "__main__":
    tool = PerplexityDeepSearchTool()
    print(tool.to_markdown())
