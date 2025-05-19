"""Tool for publishing content to TikTok."""

import os
import tempfile
from typing import List, Optional

from loguru import logger
from pydantic import Field

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.social_media.base_social_media_tool import BaseSocialMediaTool


class TikTokTool(BaseSocialMediaTool):
    """Tool for publishing videos to TikTok."""

    name: str = "tiktok_tool"
    description: str = "Publishes videos to TikTok using the TikTok API. Can convert images to videos."

    # TikTok-specific constraints
    SUPPORTED_VIDEO_TYPES = [".mp4", ".mov"]
    SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png"]
    MAX_VIDEO_DURATION = 600  # 10 minutes in seconds
    MIN_VIDEO_DURATION = 1  # 1 second
    MAX_CAPTION_LENGTH = 2200  # Maximum characters in caption
    DEFAULT_IMAGE_DURATION = 5  # Duration in seconds for image-based videos

    arguments = BaseSocialMediaTool.arguments + [
        ToolArgument(
            name="privacy_level",
            arg_type="string",
            description="Video privacy setting (PUBLIC, FRIENDS, PRIVATE)",
            required=False,
            default="PUBLIC",
            example="PUBLIC",
        ),
        ToolArgument(
            name="disable_comments",
            arg_type="string",
            description="Whether to disable comments on the video",
            required=False,
            default="False",
            example="False",
        ),
        ToolArgument(
            name="disable_duet",
            arg_type="string",
            description="Whether to disable duets on the video",
            required=False,
            default="False",
            example="False",
        ),
        ToolArgument(
            name="disable_stitch",
            arg_type="string",
            description="Whether to disable stitching on the video",
            required=False,
            default="False",
            example="False",
        ),
    ]

    async def convert_image_to_video(self, image_path: str) -> str:
        """Convert an image to a video suitable for TikTok.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to the generated video file
            
        Raises:
            ValueError: If image conversion fails
        """
        try:
            # Create temporary file for the video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video_path = temp_video.name

            # Use ffmpeg to create video from image
            # -t: duration, -vf: video filter for zoom effect
            cmd = (
                f'ffmpeg -y -loop 1 -i "{image_path}" -t {self.DEFAULT_IMAGE_DURATION} '
                f'-vf "scale=1080:1920:force_original_aspect_ratio=increase,'
                f'crop=1080:1920,zoompan=z=\'if(lte(zoom,1.0),1.1,max(1.0,zoom-0.0015))\':'
                f'x=\'iw/2-(iw/zoom/2)\':y=\'ih/2-(ih/zoom/2)\':d={self.DEFAULT_IMAGE_DURATION*25}" '
                f'-c:v libx264 -pix_fmt yuv420p "{temp_video_path}"'
            )
            
            result = os.system(cmd)
            if result != 0:
                raise ValueError(f"Failed to convert image to video. ffmpeg error code: {result}")
                
            return temp_video_path
            
        except Exception as e:
            logger.error(f"Error converting image to video: {str(e)}")
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            raise ValueError(f"Failed to convert image to video: {str(e)}")

    async def validate_media_files(self, media_paths: str) -> List[str]:
        """Validate video/image files for TikTok compatibility.
        
        Args:
            media_paths: Comma-separated list of media file paths
            
        Returns:
            List of validated file paths
            
        Raises:
            ValueError: If media files are invalid or exceed TikTok limits
        """
        if not media_paths:
            raise ValueError("Must provide a video or image file for TikTok")
            
        paths = [p.strip() for p in media_paths.split(",")]
        
        if len(paths) > 1:
            raise ValueError("TikTok only supports uploading one media file at a time")
            
        path = paths[0]
        if not os.path.exists(path):
            raise ValueError(f"Media file not found: {path}")
            
        ext = os.path.splitext(path)[1].lower()
        
        # Check if it's an image
        if ext in self.SUPPORTED_IMAGE_TYPES:
            logger.info(f"Converting image {path} to video...")
            video_path = await self.convert_image_to_video(path)
            return [video_path]
            
        # Check if it's a video
        if ext not in self.SUPPORTED_VIDEO_TYPES:
            raise ValueError(
                f"Unsupported media type {ext}. Supported types: "
                f"Videos: {', '.join(self.SUPPORTED_VIDEO_TYPES)}, "
                f"Images: {', '.join(self.SUPPORTED_IMAGE_TYPES)}"
            )
            
        return [path]

    async def async_execute(
        self,
        access_token: str,
        text: Optional[str] = None,
        media_paths: Optional[str] = None,
        privacy_level: str = "PUBLIC",
        disable_comments: str = "False",
        disable_duet: str = "False",
        disable_stitch: str = "False"
    ) -> str:
        """Publish video to TikTok.
        
        Args:
            access_token: TikTok API access token
            text: Optional caption for the video
            media_paths: Path to the video file to upload
            privacy_level: Video privacy setting (PUBLIC, FRIENDS, PRIVATE)
            disable_comments: Whether to disable comments
            disable_duet: Whether to disable duets
            disable_stitch: Whether to disable stitching
            
        Returns:
            Status message with post details
            
        Raises:
            ValueError: If the post parameters are invalid
            Exception: If posting to TikTok fails
        """
        try:
            if not media_paths:
                raise ValueError("Must provide a video or image file for TikTok")

            if text and len(text) > self.MAX_CAPTION_LENGTH:
                raise ValueError(f"Caption exceeds TikTok's {self.MAX_CAPTION_LENGTH} character limit")

            if privacy_level not in ["PUBLIC", "FRIENDS", "PRIVATE"]:
                raise ValueError("Invalid privacy level. Must be one of: PUBLIC, FRIENDS, PRIVATE")

            # Convert string booleans to actual booleans
            disable_comments = disable_comments.lower() in ["true", "1", "yes"]
            disable_duet = disable_duet.lower() in ["true", "1", "yes"]
            disable_stitch = disable_stitch.lower() in ["true", "1", "yes"]

            # Validate video file
            video_files = await self.validate_media_files(media_paths)
            video_path = video_files[0]  # We know there's exactly one file
            
            # Here you would:
            # 1. Use TikTok API to upload the video
            # 2. Set caption and privacy settings
            # 3. Configure duet/stitch/comments settings
            # 4. Handle rate limits and errors
            # 5. Return video URL or ID
            
            # Example implementation (replace with actual API calls):
            settings = []
            if disable_comments:
                settings.append("comments disabled")
            if disable_duet:
                settings.append("duets disabled")
            if disable_stitch:
                settings.append("stitching disabled")
            
            settings_str = f" ({', '.join(settings)})" if settings else ""
            
            return (
                f"Successfully posted to TikTok:\n"
                f"- Video: {os.path.basename(video_path)}\n"
                f"- Caption: {text if text else 'No caption'}\n"
                f"- Privacy: {privacy_level}{settings_str}"
            )
            
        except Exception as e:
            logger.error(f"Failed to post to TikTok: {str(e)}")
            raise
