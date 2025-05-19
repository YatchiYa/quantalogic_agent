"""Tool for publishing content to Instagram."""

import os
from typing import List, Optional

from loguru import logger
from pydantic import Field

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.social_media.base_social_media_tool import BaseSocialMediaTool


class InstagramTool(BaseSocialMediaTool):
    """Tool for publishing posts to Instagram."""

    name: str = "instagram_tool"
    description: str = "Publishes content (text, images, videos) to Instagram using the Instagram Graph API"

    # Instagram-specific media constraints
    SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png"]
    SUPPORTED_VIDEO_TYPES = [".mp4", ".mov"]
    MAX_MEDIA_COUNT = 10  # Maximum media items per post

    async def validate_media_files(self, media_paths: str) -> List[str]:
        """Validate media files for Instagram compatibility.
        
        Args:
            media_paths: Comma-separated list of media file paths
            
        Returns:
            List of validated file paths
            
        Raises:
            ValueError: If media files are invalid or exceed Instagram limits
        """
        if not media_paths:
            return []
            
        paths = [p.strip() for p in media_paths.split(",")]
        
        if len(paths) > self.MAX_MEDIA_COUNT:
            raise ValueError(f"Instagram allows maximum {self.MAX_MEDIA_COUNT} media items per post")
            
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
        """Publish content to Instagram.
        
        Args:
            access_token: Instagram Graph API access token
            text: Optional caption for the post
            media_paths: Optional comma-separated list of media file paths
            
        Returns:
            Status message with post details
            
        Raises:
            ValueError: If the post parameters are invalid
            Exception: If posting to Instagram fails
        """
        try:
            if not text and not media_paths:
                raise ValueError("Must provide either text or media content")

            # Validate media files if provided
            media_files = await self.validate_media_files(media_paths) if media_paths else []
            
            # Here you would:
            # 1. Use Instagram Graph API to upload media files
            # 2. Create a post with the media and caption
            # 3. Handle rate limits and errors
            # 4. Return post URL or ID
            
            # Example implementation (replace with actual API calls):
            media_count = len(media_files)
            media_type = "media items" if media_count != 1 else "media item"
            
            return (
                f"Successfully posted to Instagram:\n"
                f"- Caption: {text if text else 'No caption'}\n"
                f"- Media: {media_count} {media_type}"
            )
            
        except Exception as e:
            logger.error(f"Failed to post to Instagram: {str(e)}")
            raise
