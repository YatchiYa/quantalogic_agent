#!/usr/bin/env python
"""Test script for logger configuration."""

import os
from loguru import logger

# Configure logger
logger.remove()  # Remove default handlers

# Set up file logger at the root of the project
project_root = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(project_root, "agent.logs")

logger.add(
    log_file_path,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
    rotation="10 MB",
    retention="1 week",
)

# Add console logger for testing
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Test logging
logger.info("Test log message saved to agent.logs at the root of the project")
logger.debug("This is a debug message")
logger.warning("This is a warning message")
logger.error("This is an error message")

print(f"\nLog file created at: {log_file_path}")
