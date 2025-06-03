"""Tool for performing web searches using the Linkup API."""

import os
from typing import Literal, Any

from loguru import logger
from linkup import LinkupClient
from quantalogic.tools.tool import Tool, ToolArgument


class LinkupTool(Tool):
    """Tool for performing web searches using the Linkup API with different output types."""

    name: str = "linkup_tool"
    description: str = "Perform web searches using Linkup API with support for both search results and sourced answers."
    need_validation: bool = False
    api_key: str | None = os.getenv("LINKUP_API_KEY")

    arguments: list = [
        ToolArgument(
            name="query",
            arg_type="string",
            description="The search query to perform",
            required=True,
            example="What is Microsoft's 2024 revenue?",
        ),
        ToolArgument(
            name="depth",
            arg_type="string",
            description="Search depth (standard or deep)",
            required=False,
            default="standard",
            example="standard",
        ),
        ToolArgument(
            name="output_type",
            arg_type="string",
            description="Type of output (searchResults or sourcedAnswer)",
            required=False,
            default="sourcedAnswer",
            example="sourcedAnswer",
        ),
    ]

    def _validate_api_key(self) -> None:
        """Validate that the API key is set.

        Raises:
            ValueError: If the API key is not set
        """
        if not self.api_key:
            raise ValueError(
                "Linkup API key not found. Please set the YOUR_LINKUP_API_KEY environment variable."
            )

    def _format_response(
        self, 
        response: Any,
        output_type: Literal["searchResults", "sourcedAnswer"]
    ) -> dict:
        """Format the Linkup API response based on output type.

        Args:
            response: The raw API response
            output_type: Type of output to format

        Returns:
            dict: Formatted search results or sourced answer as a structured dictionary
        """
        try:
            if output_type == "sourcedAnswer":
                # For sourced answers, return a structured dictionary similar to LinkupEnhancedTool
                # Extract answer if available
                answer = response.answer if hasattr(response, "answer") else ""
                
                # Process sources if available
                sources = []
                if hasattr(response, "sources") and response.sources:
                    for source in response.sources:
                        source_dict = {
                            "title": source.title if hasattr(source, "title") else "No title",
                            "url": source.url if hasattr(source, "url") else "No URL",
                            "content": source.content if hasattr(source, "content") else "No content"
                        }
                        sources.append(source_dict)
                
                # Return structured response
                return {
                    "answer": answer,
                    "sources": sources,
                    "sources_count": len(sources),
                    "output_type": output_type
                }
                
            elif output_type == "searchResults":
                # For search results, return a structured dictionary
                results = []
                if hasattr(response, "results"):
                    for result in response.results:
                        result_dict = {
                            "content": result.content if hasattr(result, "content") else "No content",
                            "url": result.url if hasattr(result, "url") else "No URL"
                        }
                        results.append(result_dict)
                
                return {
                    "results": results,
                    "results_count": len(results),
                    "output_type": output_type
                }
            else:
                return {"error": "Invalid output type specified."}
        except Exception as e:
            error_msg = f"Error formatting response: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def execute(
        self,
        query: str,
        depth: str = "standard",
        output_type: str = "sourcedAnswer",
    ) -> dict:
        """Perform a web search using the Linkup API.

        Args:
            query: The search query to perform
            depth: Search depth (standard or deep)
            output_type: Type of output (searchResults or sourcedAnswer)

        Returns:
            dict: Structured dictionary with search results or sourced answer

        Raises:
            ValueError: If the API key is not set or if there's an error with the request
        """
        try:
            self._validate_api_key()

            # Validate depth parameter
            if depth not in ["standard", "deep"]:
                depth = "deep"
                logger.warning(f"Invalid depth '{depth}', defaulting to 'deep'")

            # Validate output_type parameter
            if output_type not in ["searchResults", "sourcedAnswer"]:
                output_type = "sourcedAnswer"
                logger.warning(f"Invalid output_type '{output_type}', defaulting to 'sourcedAnswer'")

            # Initialize Linkup client
            client = LinkupClient(api_key=self.api_key)

            # Make the API request
            response = client.search(
                query=query,
                depth=depth,
                output_type=output_type
            )

            # Format the response as a structured dictionary
            formatted_response = self._format_response(response, output_type)
            
            # Add query and depth to the response
            formatted_response["query"] = query
            formatted_response["depth"] = depth
            
            return formatted_response

        except Exception as e:
            error_msg = f"Error performing Linkup search: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "query": query}


if __name__ == "__main__":
    tool = LinkupTool()
    print(tool.to_markdown())
