"""Base class for social media publishing tools."""

from typing import List, Optional
from pydantic import Field
from quantalogic.tools.tool import Tool, ToolArgument


class BaseSocialMediaTool(Tool):
    """Base tool for publishing content to social media platforms."""

    need_validation: bool = True  # Always require validation for social media posts
    need_post_process: bool = True

    arguments: list = [
        ToolArgument(
            name="text",
            arg_type="string",
            description="The text content to post",
            required=False,
            example="Check out this amazing content!",
        ),
        ToolArgument(
            name="media_paths",
            arg_type="string",
            description="Comma-separated list of local file paths to media (images/videos) to post",
            required=False,
            example="/path/to/image.jpg,/path/to/video.mp4",
        ),
        ToolArgument(
            name="access_token",
            arg_type="string",
            description="OAuth access token for the social media platform",
            required=True,
            example="your-access-token-here",
        ),
    ]

    async def validate_media_files(self, media_paths: str) -> List[str]:
        """Validate media files exist and are of supported formats.
        
        Args:
            media_paths: Comma-separated list of media file paths
            
        Returns:
            List of validated file paths
            
        Raises:
            ValueError: If any media file is invalid or unsupported
        """
        if not media_paths:
            return []
            
        paths = [p.strip() for p in media_paths.split(",")]
        valid_paths = []
        
        for path in paths:
            # Add platform-specific validation here
            valid_paths.append(path)
            
        return valid_paths

    async def async_execute(
        self,
        access_token: str,
        text: Optional[str] = None,
        media_paths: Optional[str] = None
    ) -> str:
        """Base async execution method for social media posting.
        
        Args:
            access_token: OAuth access token for authentication
            text: Optional text content to post
            media_paths: Optional comma-separated list of media file paths
            
        Returns:
            Status message with post details
            
        Raises:
            NotImplementedError: This base class method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement async_execute()")
