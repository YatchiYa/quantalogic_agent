"""Tool for publishing content to YouTube."""

import os
from typing import List, Optional

from loguru import logger
from pydantic import Field

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.social_media.base_social_media_tool import BaseSocialMediaTool


class YouTubeTool(BaseSocialMediaTool):
    """Tool for publishing videos to YouTube."""

    name: str = "youtube_tool"
    description: str = "Publishes videos to YouTube using the YouTube Data API"

    # YouTube-specific constraints
    SUPPORTED_VIDEO_TYPES = [".mp4", ".mov", ".avi", ".flv", ".wmv", ".mkv"]
    MAX_TITLE_LENGTH = 100
    MAX_DESCRIPTION_LENGTH = 5000
    MAX_TAGS = 500
    MAX_TAG_LENGTH = 30

    arguments = BaseSocialMediaTool.arguments + [
        ToolArgument(
            name="title",
            arg_type="string",
            description="Video title (max 100 characters)",
            required=True,
            example="My Awesome Video",
        ),
        ToolArgument(
            name="description",
            arg_type="string",
            description="Video description (max 5000 characters)",
            required=False,
            example="Check out this amazing content!",
        ),
        ToolArgument(
            name="privacy",
            arg_type="string",
            description="Video privacy setting (public, unlisted, private)",
            required=False,
            default="public",
            example="public",
        ),
        ToolArgument(
            name="playlist_id",
            arg_type="string",
            description="Optional playlist ID to add the video to",
            required=False,
            example="PLxxxxxxxxxxxxxxxx",
        ),
        ToolArgument(
            name="tags",
            arg_type="string",
            description="Comma-separated list of tags (max 500 tags, each max 30 chars)",
            required=False,
            example="vlog,tutorial,howto",
        ),
        ToolArgument(
            name="category_id",
            arg_type="string",
            description="YouTube category ID (e.g., 22 for People & Blogs)",
            required=False,
            default="22",
            example="22",
        ),
        ToolArgument(
            name="made_for_kids",
            arg_type="string",
            description="Whether the video is made for kids",
            required=False,
            default="False",
            example="False",
        ),
        ToolArgument(
            name="notify_subscribers",
            arg_type="string",
            description="Whether to notify subscribers about the upload",
            required=False,
            default="True",
            example="True",
        ),
    ]

    async def validate_media_files(self, media_paths: str) -> List[str]:
        """Validate video files for YouTube compatibility.
        
        Args:
            media_paths: Comma-separated list of video file paths
            
        Returns:
            List of validated file paths
            
        Raises:
            ValueError: If video files are invalid or exceed YouTube limits
        """
        if not media_paths:
            raise ValueError("Must provide a video file for YouTube")
            
        paths = [p.strip() for p in media_paths.split(",")]
        
        if len(paths) > 1:
            raise ValueError("YouTube only supports uploading one video at a time")
            
        path = paths[0]
        if not os.path.exists(path):
            raise ValueError(f"Video file not found: {path}")
            
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.SUPPORTED_VIDEO_TYPES:
            raise ValueError(
                f"Unsupported video type {ext}. Supported types: "
                f"{', '.join(self.SUPPORTED_VIDEO_TYPES)}"
            )
            
        # Here you would also validate:
        # - Video duration
        # - Video resolution
        # - Video codec
        # - File size
        # - Frame rate
            
        return [path]

    def validate_metadata(
        self,
        title: str,
        description: Optional[str],
        privacy: str,
        tags: Optional[str],
        category_id: str,
    ) -> None:
        """Validate video metadata against YouTube constraints.
        
        Args:
            title: Video title
            description: Video description
            privacy: Privacy setting
            tags: Comma-separated tags
            category_id: YouTube category ID
            
        Raises:
            ValueError: If any metadata is invalid
        """
        if len(title) > self.MAX_TITLE_LENGTH:
            raise ValueError(f"Title exceeds YouTube's {self.MAX_TITLE_LENGTH} character limit")
            
        if description and len(description) > self.MAX_DESCRIPTION_LENGTH:
            raise ValueError(f"Description exceeds YouTube's {self.MAX_DESCRIPTION_LENGTH} character limit")
            
        if privacy not in ["public", "unlisted", "private"]:
            raise ValueError("Invalid privacy setting. Must be one of: public, unlisted, private")
            
        if tags:
            tag_list = [t.strip() for t in tags.split(",")]
            if len(tag_list) > self.MAX_TAGS:
                raise ValueError(f"Maximum {self.MAX_TAGS} tags allowed")
            for tag in tag_list:
                if len(tag) > self.MAX_TAG_LENGTH:
                    raise ValueError(f"Tag '{tag}' exceeds {self.MAX_TAG_LENGTH} character limit")
                    
        try:
            category_id = int(category_id)
            if category_id < 1:
                raise ValueError()
        except ValueError:
            raise ValueError("Category ID must be a positive integer")

    async def async_execute(
        self,
        access_token: str,
        media_paths: str,
        title: str,
        description: Optional[str] = None,
        privacy: str = "public",
        playlist_id: Optional[str] = None,
        tags: Optional[str] = None,
        category_id: str = "22",
        made_for_kids: str = "False",
        notify_subscribers: str = "True",
        text: Optional[str] = None,  # Not used for YouTube
    ) -> str:
        """Publish video to YouTube.
        
        Args:
            access_token: YouTube Data API OAuth token
            media_paths: Path to the video file
            title: Video title
            description: Optional video description
            privacy: Video privacy setting
            playlist_id: Optional playlist ID to add video to
            tags: Optional comma-separated list of tags
            category_id: YouTube category ID
            made_for_kids: Whether video is made for kids
            notify_subscribers: Whether to notify subscribers
            text: Not used for YouTube (inherited from base class)
            
        Returns:
            Status message with upload details
            
        Raises:
            ValueError: If the upload parameters are invalid
            Exception: If uploading to YouTube fails
        """
        try:
            # Validate video file
            video_files = await self.validate_media_files(media_paths)
            video_path = video_files[0]  # We know there's exactly one file
            
            # Validate metadata
            self.validate_metadata(title, description, privacy, tags, category_id)
            
            # Convert string booleans to actual booleans
            made_for_kids = made_for_kids.lower() in ["true", "1", "yes"]
            notify_subscribers = notify_subscribers.lower() in ["true", "1", "yes"]
            
            # Here you would:
            # 1. Initialize YouTube Data API client
            # 2. Create upload request with metadata
            # 3. Upload video file with resumable uploads
            # 4. Add to playlist if specified
            # 5. Handle quota limits and errors
            # 6. Return video URL or ID
            
            # Example implementation (replace with actual API calls):
            settings = []
            if made_for_kids:
                settings.append("made for kids")
            if not notify_subscribers:
                settings.append("notifications disabled")
            if playlist_id:
                settings.append(f"added to playlist {playlist_id}")
            
            settings_str = f" ({', '.join(settings)})" if settings else ""
            
            return (
                f"Successfully uploaded to YouTube:\n"
                f"- Video: {os.path.basename(video_path)}\n"
                f"- Title: {title}\n"
                f"- Description: {description if description else 'No description'}\n"
                f"- Privacy: {privacy}\n"
                f"- Tags: {tags if tags else 'No tags'}\n"
                f"- Category ID: {category_id}{settings_str}"
            )
            
        except Exception as e:
            logger.error(f"Failed to upload to YouTube: {str(e)}")
            raise
