"""Test script for the TikTok publishing tool."""

import asyncio
import os
from PIL import Image
import numpy as np
from loguru import logger

from quantalogic.tools.social_media import TikTokTool


def create_test_image(path: str, size=(1080, 1920)):
    """Create a test image with text."""
    # Create a gradient background
    gradient = np.linspace(0, 255, size[1], dtype=np.uint8)
    gradient = np.tile(gradient, (size[0], 1))
    image = Image.fromarray(gradient).convert('RGB')
    
    # Save the image
    image.save(path)
    return path


async def test_tiktok_upload():
    """Test TikTok media upload with various settings."""
    
    # Initialize the TikTok tool
    tiktok = TikTokTool()
    
    # Create test image
    image_path = "/tmp/test_image.jpg"
    if not os.path.exists(image_path):
        logger.info("Creating a test image...")
        create_test_image(image_path)

    # Test cases with different settings
    test_cases = [
        {
            "name": "Basic Image Upload",
            "params": {
                "access_token": "your-test-token",
                "text": "Test image upload #test",
                "media_paths": image_path,
            }
        },
        {
            "name": "Private Image Upload",
            "params": {
                "access_token": "your-test-token",
                "text": "Private test image",
                "media_paths": image_path,
                "privacy_level": "PRIVATE",
            }
        },
        {
            "name": "Image Upload with All Settings",
            "params": {
                "access_token": "your-test-token",
                "text": "Full settings test image #test #demo",
                "media_paths": image_path,
                "privacy_level": "FRIENDS",
                "disable_comments": "True",
                "disable_duet": "True",
                "disable_stitch": "True",
            }
        }
    ]

    # Run test cases
    for test in test_cases:
        try:
            logger.info(f"\nRunning test: {test['name']}")
            result = await tiktok.async_execute(**test["params"])
            logger.success(f"Test '{test['name']}' succeeded:\n{result}")
        except Exception as e:
            logger.error(f"Test '{test['name']}' failed: {str(e)}")


async def test_error_cases():
    """Test various error cases."""
    
    tiktok = TikTokTool()
    
    # Create an invalid image file
    invalid_image = "/tmp/invalid.jpg"
    with open(invalid_image, "w") as f:
        f.write("This is not a valid image file")
    
    error_cases = [
        {
            "name": "Missing Media",
            "params": {
                "access_token": "your-test-token",
                "text": "Test post",
            }
        },
        {
            "name": "Invalid Privacy Level",
            "params": {
                "access_token": "your-test-token",
                "text": "Test post",
                "media_paths": "/tmp/test_image.jpg",
                "privacy_level": "INVALID",
            }
        },
        {
            "name": "Non-existent Image",
            "params": {
                "access_token": "your-test-token",
                "text": "Test post",
                "media_paths": "/tmp/nonexistent.jpg",
            }
        },
        {
            "name": "Invalid Image File",
            "params": {
                "access_token": "your-test-token",
                "text": "Test post",
                "media_paths": invalid_image,
            }
        }
    ]

    for test in error_cases:
        try:
            logger.info(f"\nTesting error case: {test['name']}")
            result = await tiktok.async_execute(**test["params"])
            logger.warning(f"Expected error but got success: {result}")
        except Exception as e:
            logger.success(f"Error case '{test['name']}' correctly failed: {str(e)}")
    
    # Clean up invalid image file
    if os.path.exists(invalid_image):
        os.unlink(invalid_image)


async def main():
    """Run all tests."""
    logger.info("Starting TikTok tool tests...")
    
    # Test successful uploads
    await test_tiktok_upload()
    
    # Test error cases
    await test_error_cases()
    
    logger.info("All tests completed!")


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        lambda msg: print(msg),
        format="<level>{level}</level> | {message}",
        colorize=True,
    )
    
    # Run tests
    asyncio.run(main())
