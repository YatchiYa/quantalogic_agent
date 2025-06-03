"""Tool for editing the whole content of a file."""

import os
from pathlib import Path
from typing import Optional

from loguru import logger
from quantalogic.tools.tool import Tool, ToolArgument


class EditWholeContentTool(Tool):
    """Tool for replace the whole content of a file in agent-specific directory."""

    name: str = "edit_whole_content_tool"
    description: str = "Edits the whole content of an existing file in agent's directory under /tmp."
    need_validation: bool = False
    agent_id: Optional[str] = None
    arguments: list = [
        ToolArgument(
            name="file_path",
            arg_type="string",
            description="The path to the file to edit. Using an absolute path is recommended.",
            required=True,
            example="/path/to/file.txt",
        ),
        ToolArgument(
            name="content",
            arg_type="string",
            description="""
            The content to write to the file. Use CDATA to escape special characters.
            Don't add newlines at the beginning or end of the content.
            """,
            required=True,
            example="Hello, world!",
        ),
    ]

    def execute(self, file_path: str, content: str, agent_id: str = None) -> str:
        """Writes a file with the given content.

        Args:
            file_path (str): The path to the file to write.
            content (str): The content to write to the file.

        Returns:
            str: The content of the file.

        Raises:
            FileExistsError: If the file already exists and append_mode is False and overwrite is False.
        """
        ## Handle tilde expansion
        if file_path.startswith("~"):
            # Expand the tilde to the user's home directory
            file_path = os.path.expanduser(file_path)
            
        # Handle agent-specific paths
        if self.agent_id and str(self.agent_id).strip():
            base_dir = f"/tmp/{self.agent_id}"
            logger.info(f"Using agent-specific directory: {base_dir}")
            
            # If path is in /tmp but not in agent directory, adjust it
            if file_path.startswith("/tmp/") and not file_path.startswith(f"/tmp/{self.agent_id}/"):
                # Extract the part after /tmp/
                relative_path = file_path[5:]
                file_path = f"/tmp/{self.agent_id}/{relative_path}"
                logger.info(f"Adjusted path to agent directory: {file_path}")
            # If path doesn't start with /tmp, prepend agent directory
            elif not file_path.startswith("/tmp/"):
                clean_path = file_path.lstrip("/")
                if not os.path.isabs(clean_path):
                    file_path = f"/tmp/{self.agent_id}/{clean_path}"
                    logger.info(f"Prepended agent directory: {file_path}")
        else:
            logger.info("Using default directory (no agent_id provided)")

        # Convert relative paths to absolute paths using current working directory
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(os.path.join(os.getcwd(), file_path))
            
        logger.info(f"Final file path: {file_path}")

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if not os.path.exists(file_path):
            raise FileExistsError(f"File {file_path} does not exist.")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        file_size = os.path.getsize(file_path)
        result = f"File {file_path} written successfully. Size: {file_size} bytes."
        logger.info(f"EditWholeContentTool result: {result}")
        return result


if __name__ == "__main__":
    tool = EditWholeContentTool()
    print(tool.to_markdown())
