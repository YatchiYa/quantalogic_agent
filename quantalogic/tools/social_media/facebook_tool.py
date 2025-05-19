"""Tool for publishing content to Facebook."""

import os
from typing import List, Optional

from loguru import logger
from pydantic import Field

from quantalogic.tools.social_media.base_social_media_tool import BaseSocialMediaTool

from quantalogic.tools.tool import Tool, ToolArgument

class FacebookTool(BaseSocialMediaTool):
    """Tool for publishing posts to Facebook."""

    name: str = "facebook_tool"
    description: str = "Publishes content (text, images, videos) to Facebook using the Facebook Graph API"

    # Facebook-specific media constraints
    SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".gif"]
    SUPPORTED_VIDEO_TYPES = [".mp4", ".mov"]
    MAX_MEDIA_COUNT = 10  # Maximum media items per post
    MAX_TEXT_LENGTH = 63206  # Maximum characters in post text

    arguments = BaseSocialMediaTool.arguments + [
        ToolArgument(
            name="page_id",
            arg_type="string",
            description="ID of the Facebook page to post to (required for page posts)",
            required=False,
            example="123456789",
        ),
        ToolArgument(
            name="privacy",
            arg_type="string",
            description="Post privacy setting (EVERYONE, FRIENDS, ONLY_ME)",
            required=False,
            default="EVERYONE",
            example="FRIENDS",
        ),
    ]

    async def validate_media_files(self, media_paths: str) -> List[str]:
        """Validate media files for Facebook compatibility.
        
        Args:
            media_paths: Comma-separated list of media file paths
            
        Returns:
            List of validated file paths
            
        Raises:
            ValueError: If media files are invalid or exceed Facebook limits
        """
        if not media_paths:
            return []
            
        paths = [p.strip() for p in media_paths.split(",")]
        
        if len(paths) > self.MAX_MEDIA_COUNT:
            raise ValueError(f"Facebook allows maximum {self.MAX_MEDIA_COUNT} media items per post")
            
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
        media_paths: Optional[str] = None,
        page_id: Optional[str] = None,
        privacy: str = "EVERYONE"
    ) -> str:
        """Publish content to Facebook.
        
        Args:
            access_token: Facebook Graph API access token
            text: Optional text content for the post
            media_paths: Optional comma-separated list of media file paths
            page_id: Optional Facebook page ID to post to
            privacy: Post privacy setting (EVERYONE, FRIENDS, ONLY_ME)
            
        Returns:
            Status message with post details
            
        Raises:
            ValueError: If the post parameters are invalid
            Exception: If posting to Facebook fails
        """
        try:
            if not text and not media_paths:
                raise ValueError("Must provide either text or media content")

            if text and len(text) > self.MAX_TEXT_LENGTH:
                raise ValueError(f"Text exceeds Facebook's {self.MAX_TEXT_LENGTH} character limit")

            if privacy not in ["EVERYONE", "FRIENDS", "ONLY_ME"]:
                raise ValueError("Invalid privacy setting. Must be one of: EVERYONE, FRIENDS, ONLY_ME")

            # Validate media files if provided
            media_files = await self.validate_media_files(media_paths) if media_paths else []
            
            # Here you would:
            # 1. Use Facebook Graph API to upload media files
            # 2. Create a post with the media and text
            # 3. Set privacy settings
            # 4. Handle rate limits and errors
            # 5. Return post URL or ID
            
            # Example implementation (replace with actual API calls):
            media_count = len(media_files)
            media_type = "media items" if media_count != 1 else "media item"
            target = f"page {page_id}" if page_id else "personal timeline"
            
            return (
                f"Successfully posted to Facebook {target}:\n"
                f"- Text: {text if text else 'No text'}\n"
                f"- Media: {media_count} {media_type}\n"
                f"- Privacy: {privacy}"
            )
            
        except Exception as e:
            logger.error(f"Failed to post to Facebook: {str(e)}")
            raise
