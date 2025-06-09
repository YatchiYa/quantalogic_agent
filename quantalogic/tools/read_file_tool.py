"""Tool for reading a file or HTTP content and returning its content."""

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from typing import Any, Callable, Dict, Union

from loguru import logger
from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.utils.read_file import read_file
from quantalogic.utils.read_http_text_content import read_http_text_content

MAX_LINES = 3000


class ReadFileTool(Tool):
    """Tool for reading a file from agent-specific directory under /tmp or HTTP content and returning its content."""

    name: str = "read_file_tool"
    description: str = (
        f"Reads a local file content from agent's directory under /tmp and returns its content. "
        f"Cut to {MAX_LINES} first lines.\n"
        "Don't use on HTML files and large files. "
        "Prefer to use read file block tool to don't fill the memory. "
        "THE FILE PATH MUST BE WITHIN THE AGENT'S DIRECTORY UNDER /tmp."
    )
    agent_id: Optional[str] = None
    arguments: list = [
        ToolArgument(
            name="file_path",
            arg_type="string",
            description="The path to the file to read (must be within /tmp directory).",
            required=True,
            example="/path/to/file.txt",
        ),
    ]

    def _is_url(self, path: str) -> bool:
        """Check if the given path is a valid URL."""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _truncate_content(self, content: str) -> str:
        """Truncate the content to the first MAX_LINES lines."""
        lines = content.splitlines()
        truncated_lines = lines[:MAX_LINES]
        truncated_content = "\n".join(truncated_lines)
        if len(lines) > MAX_LINES:
            truncated_content += f"\n\n[The content is too long. Truncated at {MAX_LINES} lines.]"
        return truncated_content

    def execute(self, file_path: str, agent_id: str = None) -> Union[str, Dict[str, Any]]:
        """Reads a file or HTTP content and returns its content.

        Args:
            file_path (str): The path to the file or URL to read.

        Returns:
            str: The content of the file or HTTP content.
        """
        if self._is_url(file_path):
            # Handle HTTP content
            content, error = read_http_text_content(file_path)
            if error:
                return f"Error reading URL {file_path}: {error}"
            truncated_content = self._truncate_content(content)
            result = f"{truncated_content}"
            return result
        else:
            # Handle local file
            try:
                # Determine the base directory based on agent_id
                if self.agent_id and str(self.agent_id).strip():
                    base_dir = f"/tmp/{self.agent_id}"
                    logger.debug(f"Using agent-specific directory: {base_dir}")
                else:
                    base_dir = "/tmp"
                    logger.debug("Using default /tmp directory (no agent_id provided)")
                
                # Handle agent-specific paths
                if self.agent_id and str(self.agent_id).strip():
                    # If path is in /tmp but not in agent directory, adjust it
                    if file_path.startswith("/tmp/") and not file_path.startswith(f"/tmp/{self.agent_id}/"):
                        # Extract the part after /tmp/
                        relative_path = file_path[5:]
                        file_path = f"/tmp/{self.agent_id}/{relative_path}"
                        logger.debug(f"Adjusted path to agent directory: {file_path}")
                    # If path doesn't start with /tmp, prepend agent directory
                    elif not file_path.startswith("/tmp/"):
                        clean_path = file_path.lstrip("/")
                        file_path = f"/tmp/{self.agent_id}/{clean_path}"
                        logger.debug(f"Prepended agent directory: {file_path}")
                
                # Validate file path is within the correct directory
                abs_path = os.path.abspath(file_path)
                if not abs_path.startswith(base_dir):
                    return f"Security restriction: This tool can only read files within {base_dir}. '{abs_path}' is not allowed."
                
                logger.debug(f"Final file path: {abs_path}")
                
                content = read_file(file_path)
                truncated_content = self._truncate_content(content)
                result = f"{truncated_content}"
                return {
                        "status": "success",
                        "answer": result
                    }
            except Exception as e:
                return {
                        "status": "error",
                        "answer": f"Error reading file {file_path}: {str(e)}"
                    }


if __name__ == "__main__":
    tool = ReadFileTool()
    # Example usage with a file in /tmp
    # print(tool.execute("/tmp/example.txt"))
    print(tool.to_markdown())
