"""Tool for publishing content to LinkedIn."""

import os
from typing import List, Optional

from loguru import logger
from pydantic import Field

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.social_media.base_social_media_tool import BaseSocialMediaTool


class LinkedInTool(BaseSocialMediaTool):
    """Tool for publishing posts to LinkedIn."""

    name: str = "linkedin_tool"
    description: str = "Publishes content (text, images, videos) to LinkedIn using the LinkedIn REST API"

    # LinkedIn-specific media constraints
    SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".gif"]
    SUPPORTED_VIDEO_TYPES = [".mp4", ".mov", ".avi"]
    MAX_MEDIA_COUNT = 9  # Maximum media items per post
    MAX_TEXT_LENGTH = 3000  # Maximum characters in post text

    async def validate_media_files(self, media_paths: str) -> List[str]:
        """Validate media files for LinkedIn compatibility.
        
        Args:
            media_paths: Comma-separated list of media file paths
            
        Returns:
            List of validated file paths
            
        Raises:
            ValueError: If media files are invalid or exceed LinkedIn limits
        """
        if not media_paths:
            return []
            
        paths = [p.strip() for p in media_paths.split(",")]
        
        if len(paths) > self.MAX_MEDIA_COUNT:
            raise ValueError(f"LinkedIn allows maximum {self.MAX_MEDIA_COUNT} media items per post")
            
        valid_paths = []
        for path in paths:
            if not os.path.exists(path):
                raise ValueError(f"Media file not found: {path}")
                
            ext = os.path.splitext(path)[1].lower()
            if ext not in self.SUPPORTED_IMAGE_TYPES + self.SUPPORTED_VIDEO_TYPES:
                raise ValueError(
                    f"Unsupported media type {ext}. Supported types: "
                    f"{', '.join(self.SUPPORTED_IMAGE_TYPES + self.SUPPORTED_VIDEO_TYPES)}"
                )
                
            valid_paths.append(path)
            
        return valid_paths

    async def async_execute(
        self,
        access_token: str,
        text: Optional[str] = None,
        media_paths: Optional[str] = None
    ) -> str:
        """Publish content to LinkedIn.
        
        Args:
            access_token: LinkedIn OAuth 2.0 access token
            text: Optional text content for the post
            media_paths: Optional comma-separated list of media file paths
            
        Returns:
            Status message with post details
            
        Raises:
            ValueError: If the post parameters are invalid
            Exception: If posting to LinkedIn fails
        """
        try:
            if not text and not media_paths:
                raise ValueError("Must provide either text or media content")

            if text and len(text) > self.MAX_TEXT_LENGTH:
                raise ValueError(f"Text exceeds LinkedIn's {self.MAX_TEXT_LENGTH} character limit")

            # Validate media files if provided
            media_files = await self.validate_media_files(media_paths) if media_paths else []
            
            # Here you would:
            # 1. Use LinkedIn REST API to upload media files
            # 2. Create a post with the media and text
            # 3. Handle rate limits and errors
            # 4. Return post URL or ID
            
            # Example implementation (replace with actual API calls):
            media_count = len(media_files)
            media_type = "media items" if media_count != 1 else "media item"
            
            return (
                f"Successfully posted to LinkedIn:\n"
                f"- Text: {text if text else 'No text'}\n"
                f"- Media: {media_count} {media_type}"
            )
            
        except Exception as e:
            logger.error(f"Failed to post to LinkedIn: {str(e)}")
            raise
