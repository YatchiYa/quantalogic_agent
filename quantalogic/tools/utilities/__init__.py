"""
Utilities Tools Module

This module provides general utility tools and helper functions.
"""

from loguru import logger

# Explicit imports of all tools in the module
from .csv_processor_tool import CSVProcessorTool
from .download_file_tool import PrepareDownloadTool
from .mermaid_validator_tool import MermaidValidatorTool
from .vscode_tool import VSCodeServerTool
from .llm_tool import LegalLLMTool 
from .oriented_llm_tool import OrientedLLMTool



from .legal_classifier_tool import LegalClassifierTool
from .legal_letter_analyzer_tool import LegalLetterAnalyzerTool
from .legal_case_triage_tool import LegalCaseTriageTool 
from .contract_comparison_tool import ContractComparisonTool
from .contract_extractor_tool import ContractExtractorTool
from .contextual_llm_tool import ContextualLLMTool
from .defender_llm_tool import DefenderLLMTool 
from .judicial_analytics_tool import JudicialAnalyticsTool
from .prosecutor_llm_tool import ProsecutorLLMTool  
from .llm_for_context_tool import DocumentLLMTool

# Define __all__ to control what is imported with `from ... import *`
__all__ = [
    'CSVProcessorTool',
    'PrepareDownloadTool',
    'MermaidValidatorTool',
    'VSCodeServerTool',
    'LegalLLMTool', 
    'OrientedLLMTool',
    'LegalClassifierTool',
    'LegalLetterAnalyzerTool',
    'LegalCaseTriageTool', 
    'ContractComparisonTool',
    'ContractExtractorTool',
    'ContextualLLMTool',
    'DefenderLLMTool',
    'JudicialAnalyticsTool',
    'ProsecutorLLMTool',
    'DocumentLLMTool',
]

# Optional: Add logging for import confirmation
logger.info("Utilities tools module initialized successfully.")
