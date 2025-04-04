"""Tool for performing deep searches using the Perplexity API via requests."""

import os
from typing import Any, Dict, List, Optional, Union, Generator
import requests
import json
from loguru import logger
from quantalogic.tools.tool import Tool, ToolArgument

class PerplexityRequestsTool(Tool):
    """Tool for performing web searches using Perplexity API with enhanced search capabilities and direct requests."""

    name: str = "perplexity_requests"
    description: str = "Perform web searches using Perplexity API with support for various models, parameters and search options"
    need_validation: bool = False
    api_key: str | None = os.getenv("PERPLEXITY_API_KEY")
    base_url: str = "https://api.perplexity.ai/chat/completions"

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
            name="search_domain_filter",
            arg_type="string",
            description="List of domains to filter search results (comma-separated)",
            required=False,
            default="<any>",
            example="arxiv.org,science.org",
        ),
        ToolArgument(
            name="return_images",
            arg_type="boolean",
            description="Whether to return images in the response",
            required=False,
            default="false",
            example="false",
        ),
        ToolArgument(
            name="return_related_questions",
            arg_type="boolean",
            description="Whether to return related questions",
            required=False,
            default="false",
            example="false",
        ),
        ToolArgument(
            name="search_recency_filter",
            arg_type="string",
            description="Filter for search result recency",
            required=False,
            default="",
            example="last_day",
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
                "Perplexity API key not found. Please set the PERPLEXITY_API_KEY environment variable."
            )

    def _prepare_payload(
        self,
        query: str,
        model: str = "sonar-pro",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        search_domain_filter: str = "<any>",
        return_images: bool = False,
        return_related_questions: bool = False,
        search_recency_filter: Optional[str] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> Dict[str, Any]:
        """Prepare the payload for the API request."""
        # Convert comma-separated domain filter to list
        domain_filter = [d.strip() for d in search_domain_filter.split(",")] if search_domain_filter else ["<any>"]
        
        payload = {
            "model": model,
            "messages": [
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
            ],
        }

        # Add optional parameters only if they have non-default values
        if max_tokens != 1024:
            payload["max_tokens"] = max_tokens
        if temperature != 0.7:
            payload["temperature"] = temperature
        if top_p != 0.9:
            payload["top_p"] = top_p
        if stream:
            payload["stream"] = stream
        if domain_filter != ["<any>"]:
            payload["search_domain_filter"] = domain_filter
        if return_images:
            payload["return_images"] = return_images
        if return_related_questions:
            payload["return_related_questions"] = return_related_questions
        if search_recency_filter:
            payload["search_recency_filter"] = search_recency_filter
        if presence_penalty != 0.0:
            payload["presence_penalty"] = presence_penalty
        if frequency_penalty != 0.0:
            payload["frequency_penalty"] = frequency_penalty
        
        # Add web search options only if we're using domain or recency filters
        if domain_filter != ["<any>"] or search_recency_filter:
            payload["web_search_options"] = {"search_context_size": "high"}

        return payload

    def _format_response(self, response: Union[requests.Response, bytes, str]) -> Dict[str, Any]:
        """Format the Perplexity API response."""
        try:
            # Handle streaming response
            if isinstance(response, (bytes, str)):
                try:
                    # Try to decode if it's bytes
                    if isinstance(response, bytes):
                        response = response.decode('utf-8')
                    # Parse the SSE data
                    if response.startswith('data: '):
                        response = response[6:]  # Remove 'data: ' prefix
                    response_data = json.loads(response)
                    return {"content": response_data.get("choices", [{}])[0].get("delta", {}).get("content", "")}
                except Exception as e:
                    logger.error(f"Error parsing streaming response: {str(e)}")
                    return {"content": "", "error": True}

            # Handle regular response
            response_data = response.json()
            result = {
                "content": response_data["choices"][0]["message"]["content"],
                "model": response_data.get("model", ""),
                "usage": response_data.get("usage", {}),
            }

            # Add additional data if available
            if "related_questions" in response_data:
                result["related_questions"] = response_data["related_questions"]
            if "images" in response_data:
                result["images"] = response_data["images"]
            if "sources" in response_data:
                result["sources"] = response_data["sources"]

            return result

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
        search_domain_filter: str = "<any>",
        return_images: bool = False,
        return_related_questions: bool = False,
        search_recency_filter: Optional[str] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """Execute a search using the Perplexity API."""
        try:
            self._validate_api_key()

            # Validate model parameter
            valid_models = ["sonar-small-chat", "sonar-medium-chat", "sonar-pro"]
            if model not in valid_models:
                model = "sonar-pro"
                logger.warning(f"Invalid model '{model}', defaulting to 'sonar-pro'")

            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Prepare payload
            payload = self._prepare_payload(
                query=query,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
                search_domain_filter=search_domain_filter,
                return_images=return_images,
                return_related_questions=return_related_questions,
                search_recency_filter=search_recency_filter,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )

            # Make the API request
            response = requests.post(self.base_url, json=payload, headers=headers, stream=stream)
            response.raise_for_status()

            # Handle streaming responses
            if stream:
                def generate_chunks():
                    for line in response.iter_lines():
                        if line:
                            yield self._format_response(line)
                return generate_chunks()
            else:
                return self._format_response(response)

        except requests.exceptions.RequestException as e:
            error_msg = f"Error making request to Perplexity API: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error performing Perplexity search: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)


if __name__ == "__main__":
    tool = PerplexityRequestsTool()
    print(tool.to_markdown())
