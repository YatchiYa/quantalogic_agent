"""Tool for writing a file and returning its content."""

import os
from pathlib import Path
from typing import Optional

from loguru import logger
from quantalogic.tools.tool import Tool, ToolArgument


class WriteFileTool(Tool):
    """Tool for writing a text file in agent-specific directory under /tmp."""

    name: str = "write_file_tool"
    description: str = (
        "Writes a file with the given content in the agent's directory under /tmp/agent_id. "
        "If the file already exists, use overwrite=True to replace it or append_mode=True to add to it."
        "Otherwise, the operation will fail."
        "The file path must be within the agent's id directory."
        "Supports variable interpolation with $variable$ syntax if need_variables is True."
    )
    need_validation: bool = False
    need_variables: bool = True
    agent_id: Optional[str] = None
    arguments: list = [
        ToolArgument(
            name="file_path",
            arg_type="string",
            description="The name of the file to write in /tmp/agent_id directory. Can include subdirectories within /tmp/agent_id.",
            required=True,
            example="/tmp/agent_id/myfile.txt",
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
        ToolArgument(
            name="append_mode",
            arg_type="string",
            description="""Append mode. If true, the content will be appended to the end of the file.
            """,
            required=False,
            example="False",
        ),
        ToolArgument(
            name="overwrite",
            arg_type="string",
            description="Overwrite mode. If true, existing files can be overwritten. Defaults to False.",
            required=False,
            example="False",
            default="False",
        ),
        ToolArgument(
            name="variables",
            arg_type="string",
            description="Variables to interpolate in the content. Will be automatically provided when need_variables=True.",
            required=False,
            example="{}",
        ),
    ]

    def _ensure_tmp_path(self, file_path: str) -> str:
        """Ensures the file path is within the agent's directory under /tmp.

        Args:
            file_path (str): The original file path

        Returns:
            str: Normalized path within the agent's directory

        Raises:
            ValueError: If the path attempts to escape the agent's directory
        """
        # Log the current agent_id for debugging
        logger.debug(f"WriteFileTool._ensure_tmp_path called with agent_id: {self.agent_id}")
        logger.debug(f"Original file_path: {file_path}")
        
        # Determine the base directory based on agent_id
        if self.agent_id and str(self.agent_id).strip():
            base_dir = Path(f"/tmp/{self.agent_id}")
            base_prefix = f"/tmp/{self.agent_id}/"
            logger.debug(f"Using agent-specific directory: {base_dir}")
        else:
            base_dir = Path("/tmp")
            base_prefix = "/tmp/"
            logger.debug("Using default /tmp directory (no agent_id provided)")
            
        # Ensure base directory exists and is writable
        if not base_dir.exists():
            logger.debug(f"Creating directory: {base_dir}")
            os.makedirs(base_dir, exist_ok=True)
            
        if not os.access(base_dir, os.W_OK):
            raise ValueError(f"Error: {base_dir} directory is not accessible")

        # Handle different path formats
        # If the path already starts with the correct base prefix, keep it as is
        if file_path.startswith(base_prefix):
            logger.debug(f"Path already starts with correct prefix: {base_prefix}")
            normalized_path = file_path
        else:
            # Clean the path of any leading slashes and /tmp/ prefix if present
            clean_path = file_path.lstrip("/")
            if clean_path.startswith("tmp/"):
                clean_path = clean_path[4:]
                
            # If path starts with agent_id, remove it to avoid duplication
            if self.agent_id and clean_path.startswith(f"{self.agent_id}/"):
                clean_path = clean_path[len(f"{self.agent_id}/"):]
                
            normalized_path = os.path.join(str(base_dir), clean_path)
            logger.debug(f"Normalized path: {normalized_path}")

        # Resolve the absolute path and check if it's really in the agent's directory
        real_path = os.path.realpath(normalized_path)
        logger.debug(f"Real path after normalization: {real_path}")
        
        if not real_path.startswith(str(base_dir)):
            raise ValueError(f"Error: Cannot write files outside of {base_dir} directory")

        return real_path

    def _process_escape_sequences(self, content: str) -> str:
        """Process escape sequences in the content string.

        Args:
            content (str): The content string that may contain escape sequences

        Returns:
            str: Content with escape sequences properly processed
        """
        # Replace common escape sequences with their actual characters
        # This handles cases where the content contains literal \n that should be newlines
        replacements = {
            '\\n': '\n',  # newline
            '\\t': '\t',  # tab
            '\\r': '\r',  # carriage return
            '\\\\': '\\'  # backslash
        }
        
        # Process the content
        processed_content = content
        for escape_seq, char in replacements.items():
            processed_content = processed_content.replace(escape_seq, char)
            
        return processed_content
        
    def _interpolate_variables(self, content: str, variables: dict = None) -> str:
        """Interpolate variables in the content using $var$ syntax.
        
        Args:
            content (str): The content that may contain variable references
            variables (dict, optional): Dictionary of variables to interpolate
            
        Returns:
            str: Content with variables interpolated
        """
        if not isinstance(content, str) or not variables:
            return content
            
        try:
            import re
            
            # Interpolate each variable in the content
            for var_name, var_value in variables.items():
                if not var_name.startswith('$'):
                    # Create pattern for $var_name$
                    pattern = f"\${re.escape(var_name)}\$"
                    # Replace with the variable value
                    content = re.sub(pattern, str(var_value), content)
                    
            return content
        except Exception as e:
            logger.error(f"Error in _interpolate_variables: {str(e)}")
            return content

    def execute(self, file_path: str, content: str, append_mode: str = "False", overwrite: str = "False", agent_id: str = None, variables: dict = None) -> str:
        """Writes a file with the given content in the agent's directory under /tmp.

        Args:
            file_path (str): The path to the file to write (will be forced to agent's directory).
            content (str): The content to write to the file.
            append_mode (str, optional): If true, append content to existing file. Defaults to "False".
            overwrite (str, optional): If true, overwrite existing file. Defaults to "False".
            agent_id (str, optional): The agent ID to use for the directory. Defaults to None.

        Returns:
            str: Status message with file path and size.

        Raises:
            ValueError: If attempting to write outside agent's directory or if directory is not accessible.
            Exception: For other unexpected errors with detailed error message.
        """
        logger.info(f"WriteFileTool.execute called with agent_id={agent_id}, current self.agent_id={self.agent_id}")
        logger.info(f"File path: {file_path}")
        try:
            # Convert mode strings to booleans
            append_mode_bool = append_mode.lower() in ["true", "1", "yes"]
            overwrite_bool = overwrite.lower() in ["true", "1", "yes"]
            
            # Update agent_id if provided in this call
            if agent_id is not None:
                logger.info(f"Setting agent_id from parameter: {agent_id}")
                self.agent_id = agent_id
            else:
                logger.info(f"No agent_id provided in execute call, using existing: {self.agent_id}")

            # Ensure path is in agent's directory and normalize it
            file_path = self._ensure_tmp_path(file_path)

            # Ensure parent directory exists (only within agent's directory)
            parent_dir = os.path.dirname(file_path)
            base_prefix = f"/tmp/{self.agent_id}/" if self.agent_id else "/tmp/"
            if parent_dir.startswith(base_prefix):
                os.makedirs(parent_dir, exist_ok=True)

            # Check if file exists first
            file_exists = os.path.exists(file_path)
            
            # Handle based on file existence and specified modes
            if file_exists:
                if overwrite_bool:
                    # Overwrite existing file
                    processed_content = self._process_escape_sequences(content)
                    # Interpolate variables if provided
                    if variables:
                        processed_content = self._interpolate_variables(processed_content, variables)
                    with open(file_path, 'w', encoding="utf-8") as f:
                        f.write(processed_content)
                    file_size = os.path.getsize(file_path)
                    result = f"File {file_path} overwritten successfully. Size: {file_size} bytes."
                    logger.info(f"WriteFileTool result: {result}")
                    return result
                elif append_mode_bool:
                    # Append to existing file
                    processed_content = self._process_escape_sequences(content)
                    # Interpolate variables if provided
                    if variables:
                        processed_content = self._interpolate_variables(processed_content, variables)
                    with open(file_path, 'a', encoding="utf-8") as f:
                        f.write(processed_content)
                    file_size = os.path.getsize(file_path)
                    result = f"File {file_path} appended to successfully. Size: {file_size} bytes."
                    logger.info(f"WriteFileTool result: {result}")
                    return result
                else:
                    # File exists but neither overwrite nor append mode specified
                    result = f"Error: File {file_path} already exists. Use overwrite=True to replace it or append_mode=True to add to it."
                    logger.info(f"WriteFileTool result: {result}")
                    return result
            else:
                # File doesn't exist, create it
                processed_content = self._process_escape_sequences(content)
                # Interpolate variables if provided
                if variables:
                    processed_content = self._interpolate_variables(processed_content, variables)
                with open(file_path, 'w', encoding="utf-8") as f:
                    f.write(processed_content)
                file_size = os.path.getsize(file_path)
                result = f"File {file_path} created successfully. Size: {file_size} bytes."
                logger.info(f"WriteFileTool result: {result}")
                return result

        except ValueError as e:
            error_msg = f"Write file error: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error writing file: {str(e)}"
            logger.error(error_msg)
            raise Exception(f"Failed to write file: {str(e)}")


if __name__ == "__main__":
    tool = WriteFileTool()
    print(tool.to_markdown())
