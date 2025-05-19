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
            default="deep",
            example="deep",
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
    ) -> str:
        """Format the Linkup API response based on output type.

        Args:
            response: The raw API response
            output_type: Type of output to format

        Returns:
            str: Formatted search results or sourced answer
        """
        try:
            if output_type == "sourcedAnswer":
                # For sourced answers, return the complete response which includes
                # both the answer and sources
                return response
            elif output_type == "searchResults":
                # For search results, format the results list
                results = []
                for i, result in enumerate(response.results, start=1):
                    results.append(
                        f"{i}. Content: {result.content}\n   Source: {result.url}"
                    )
                return "Search Results:\n" + "\n\n".join(results)
            else:
                return "Invalid output type specified."
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return f"Error formatting response: {str(e)}"

    def execute(
        self,
        query: str,
        depth: str = "standard",
        output_type: str = "sourcedAnswer",
    ) -> str:
        """Perform a web search using the Linkup API.

        Args:
            query: The search query to perform
            depth: Search depth (standard or deep)
            output_type: Type of output (searchResults or sourcedAnswer)

        Returns:
            str: Formatted search results or sourced answer

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

            # Format and return the response
            return self._format_response(response, output_type)

        except Exception as e:
            error_msg = f"Error performing Linkup search: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)


if __name__ == "__main__":
    tool = LinkupTool()
    print(tool.to_markdown())
