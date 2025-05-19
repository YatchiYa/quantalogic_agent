"""
Git Tools Module

This module provides tools and utilities related to Git operations.
"""

from loguru import logger

# Explicit imports of all tools in the module

from quantalogic.tools.ecommerce.recommend_popular_products_tool import RecommendPopularProductsTool
from quantalogic.tools.ecommerce.product_identifier_tool import ProductIdentifierTool
from quantalogic.tools.ecommerce.product_memory_tool import ProductMemoryTool
from quantalogic.tools.ecommerce.command_validator_tool import ProductValidatorTool

# Define __all__ to control what is imported with `from ... import *`
__all__ = [ 
    'RecommendPopularProductsTool',
    'ProductIdentifierTool',
    'ProductMemoryTool',
    'ProductValidatorTool',
]

# Optional: Add logging for import confirmation
logger.info("Ecommerce tools module initialized successfully.")
