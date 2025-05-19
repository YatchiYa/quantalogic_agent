import importlib
import os
from typing import Any, Optional, Type

from loguru import logger

from quantalogic.agent import Agent, AgentMemory
from quantalogic.console_print_events import console_print_events
from quantalogic.console_print_token import console_print_token
from quantalogic.event_emitter import EventEmitter
from quantalogic.tools.tool import Tool

import sys
from pathlib import Path
# Configure loguru logger
""" logger.remove()  # Remove default handler
# Add file handler
project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels to reach project root
log_file = project_root / "agent_log.log"
logger.add(
    log_file,
    rotation="1 day",  # Create a new file daily
    retention="7 days",  # Keep logs for 7 days
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    enqueue=True  # Thread-safe logging
)
# Configure loguru to output only INFO and above
logger.remove()  # Remove default handler
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
 """
# Helper function to import tool classes
def _import_tool(module_path: str, class_name: str) -> Type[Tool]:
    """
    Import a tool class from a module path using standard Python imports.
    
    Args:
        module_path: The module path to import from
        class_name: The name of the class to import
        
    Returns:
        The imported tool class, or None if import fails
    """
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        logger.warning(f"Failed to import {class_name} from {module_path}: {str(e)}")
        return None
    except AttributeError as e:
        logger.warning(f"Failed to find {class_name} in {module_path}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error importing {class_name} from {module_path}: {str(e)}")
        return None

# Helper function to create tool instances
def create_tool_instance(tool_class, **kwargs):
    """
    Creates an instance of a tool class with appropriate parameters.
    
    Args:
        tool_class: The tool class to instantiate
        **kwargs: Parameters to pass to the tool constructor
        
    Returns:
        An instance of the tool class, or None if instantiation fails
    """
    if tool_class is None:
        return None
        
    try:
        # Only set name if the class doesn't already define it
        if 'name' not in kwargs and not hasattr(tool_class, 'name'):
            class_name = tool_class.__name__
            kwargs['name'] = class_name.lower().replace('tool', '')
            
        # Create and return the tool instance
        instance = tool_class(**kwargs)
        logger.debug(f"Successfully created tool instance: {tool_class.__name__}")
        return instance
    except Exception as e:
        logger.error(f"Failed to instantiate {tool_class.__name__}: {str(e)}")
        return None

# Lazy loading tool imports using a dictionary of functions that return tool classes
TOOL_IMPORTS = {
    # LLM Tools
    "llm": lambda: _import_tool("quantalogic.tools.llm_tool", "LLMTool"),
    "co_worker_agent": lambda: _import_tool("quantalogic.tools.llm_tool", "LLMTool"),
    "llm_vision": lambda: _import_tool("quantalogic.tools.llm_vision_tool", "LLMVisionTool"),
    "llm_image_generation": lambda: _import_tool("quantalogic.tools.image_generation.dalle_e", "LLMImageGenerationTool"),
    "stable_diffusion": lambda: _import_tool("quantalogic.tools.image_generation.stable_diffusion", "StableDiffusionTool"),
    
    # File Tools
    "download_http_file": lambda: _import_tool("quantalogic.tools.utilities", "PrepareDownloadTool"),
    "write_file": lambda: _import_tool("quantalogic.tools.write_file_tool", "WriteFileTool"),
    "file_tracker": lambda: _import_tool("quantalogic.tools.file_tracker_tool", "FileTrackerTool"),
    "edit_whole_content": lambda: _import_tool("quantalogic.tools", "EditWholeContentTool"),
    "read_file_block": lambda: _import_tool("quantalogic.tools", "ReadFileBlockTool"),
    "read_file": lambda: _import_tool("quantalogic.tools", "ReadFileTool"),
    "replace_in_file": lambda: _import_tool("quantalogic.tools", "ReplaceInFileTool"),
    "list_directory": lambda: _import_tool("quantalogic.tools", "ListDirectoryTool"),
    
    # Search Tools
    "duck_duck_go_search": lambda: _import_tool("quantalogic.tools", "DuckDuckGoSearchTool"),
    "wikipedia_search": lambda: _import_tool("quantalogic.tools", "WikipediaSearchTool"),
    "google_news": lambda: _import_tool("quantalogic.tools.google_packages", "GoogleNewsTool"),
    "search_definition_names": lambda: _import_tool("quantalogic.tools", "SearchDefinitionNames"),
    "ripgrep": lambda: _import_tool("quantalogic.tools", "RipgrepTool"),
    
    # Execution Tools
    "execute_bash_command": lambda: _import_tool("quantalogic.tools", "ExecuteBashCommandTool"),
    "nodejs": lambda: _import_tool("quantalogic.tools", "NodeJsTool"),
    "python": lambda: _import_tool("quantalogic.tools", "PythonTool"),
    "safe_python_interpreter": lambda: _import_tool("quantalogic.tools", "SafePythonInterpreterTool"),
    
    # Database Tools
    "sql_query": lambda: _import_tool("quantalogic.tools", "SQLQueryTool"),
    "sql_query_advanced": lambda: _import_tool("quantalogic.tools.database", "SQLQueryToolAdvanced"),
    
    # Document Tools
    "markdown_to_pdf": lambda: _import_tool("quantalogic.tools.document_tools", "MarkdownToPdfTool"),
    "markdown_to_pptx": lambda: _import_tool("quantalogic.tools.document_tools", "MarkdownToPptxTool"),
    "markdown_to_html": lambda: _import_tool("quantalogic.tools.document_tools", "MarkdownToHtmlTool"),
    "markdown_to_epub": lambda: _import_tool("quantalogic.tools.document_tools", "MarkdownToEpubTool"),
    "markdown_to_ipynb": lambda: _import_tool("quantalogic.tools.document_tools", "MarkdownToIpynbTool"),
    "markdown_to_latex": lambda: _import_tool("quantalogic.tools.document_tools", "MarkdownToLatexTool"),
    "markdown_to_docx": lambda: _import_tool("quantalogic.tools.document_tools", "MarkdownToDocxTool"),
    
    # Git Tools
    "clone_repo_tool": lambda: _import_tool("quantalogic.tools.git", "CloneRepoTool"),
    "bitbucket_clone_repo_tool": lambda: _import_tool("quantalogic.tools.git", "BitbucketCloneTool"),
    "bitbucket_operations_tool": lambda: _import_tool("quantalogic.tools.git", "BitbucketOperationsTool"),
    "git_operations_tool": lambda: _import_tool("quantalogic.tools.git", "GitOperationsTool"),
    
    # NASA Tools
    "nasa_neows_tool": lambda: _import_tool("quantalogic.tools.nasa_packages", "NasaNeoWsTool"),
    "nasa_apod_tool": lambda: _import_tool("quantalogic.tools.nasa_packages", "NasaApodTool"),
    
    # Composio Tools
    "email_tool": lambda: _import_tool("quantalogic.tools.composio", "ComposioTool"),
    "callendar_tool": lambda: _import_tool("quantalogic.tools.composio", "ComposioTool"),
    "weather_tool": lambda: _import_tool("quantalogic.tools.composio", "ComposioTool"),
    
    # Product Hunt Tools
    "product_hunt_tool": lambda: _import_tool("quantalogic.tools.product_hunt", "ProductHuntTool"),
    
    # RAG Tools 
    "rag_tool_hf": lambda: _import_tool("quantalogic.tools.rag_tool", "RagToolHf"),
    "openai_legal_rag": lambda: _import_tool("quantalogic.tools.rag_tool", "OpenAILegalRAG"),
    "legal_embedding_rag": lambda: _import_tool("quantalogic.tools.rag_tool", "LegalEmbeddingRAG"),
    "general_rag": lambda: _import_tool("quantalogic.tools.rag_tool", "GeneralRAG"),
    "simple_rag": lambda: _import_tool("quantalogic.tools.rag_tool", "SimpleRagTool"),
    
    # Utility Tools
    "task_complete": lambda: _import_tool("quantalogic.tools.task_complete_tool", "TaskCompleteTool"),
    "input_question": lambda: _import_tool("quantalogic.tools.utilities", "InputQuestionTool"),
    "markitdown": lambda: _import_tool("quantalogic.tools.utilities", "MarkitdownTool"),
    "read_html": lambda: _import_tool("quantalogic.tools.read_html_tool", "ReadHTMLTool"),
    "oriented_llm_tool": lambda: _import_tool("quantalogic.tools.utilities", "OrientedLLMTool"),
    "document_llm_tool": lambda: _import_tool("quantalogic.tools.utilities", "DocumentLLMTool"),
    "legal_llm_tool": lambda: _import_tool("quantalogic.tools.utilities", "LegalLLMTool"),
    "presentation_llm": lambda: _import_tool("quantalogic.tools.presentation_tools", "PresentationLLMTool"),
    "sequence": lambda: _import_tool("quantalogic.tools.utilities", "SequenceTool"),
    "csv_processor": lambda: _import_tool("quantalogic.tools.utilities", "CSVProcessorTool"),
    "mermaid_validator_tool": lambda: _import_tool("quantalogic.tools.utilities", "MermaidValidatorTool"),
    "download_file_tool": lambda: _import_tool("quantalogic.tools.utilities", "PrepareDownloadTool"),
    "vscode_server_tool": lambda: _import_tool("quantalogic.tools.utilities.vscode_tool", "VSCodeServerTool"),
    "linkup_tool": lambda: _import_tool("quantalogic.tools", "LinkupTool"),
    
    "recommend_popular_products_tool": lambda: _import_tool("quantalogic.tools.ecommerce", "RecommendPopularProductsTool"),
    "product_identifier_tool": lambda: _import_tool("quantalogic.tools.ecommerce", "ProductIdentifierTool"),
    "product_memory_tool": lambda: _import_tool("quantalogic.tools.ecommerce", "ProductMemoryTool"),
    "product_validator_tool": lambda: _import_tool("quantalogic.tools.ecommerce", "ProductValidatorTool"),

    "legal_classifier_tool": lambda: _import_tool("quantalogic.tools.utilities", "LegalClassifierTool"),
    "legal_letter_analyzer_tool": lambda: _import_tool("quantalogic.tools.utilities", "LegalLetterAnalyzerTool"),
    "legal_case_triage_tool": lambda: _import_tool("quantalogic.tools.utilities", "LegalCaseTriageTool"),
    "contract_comparison_tool": lambda: _import_tool("quantalogic.tools.utilities", "ContractComparisonTool"),
    "contract_extractor_tool": lambda: _import_tool("quantalogic.tools.utilities", "ContractExtractorTool"),
    "contextual_llm_tool": lambda: _import_tool("quantalogic.tools.utilities", "ContextualLLMTool"),
    "defender_llm_tool": lambda: _import_tool("quantalogic.tools.utilities", "DefenderLLMTool"),
    "judicial_analytics_tool": lambda: _import_tool("quantalogic.tools.utilities", "JudicialAnalyticsTool"),
    "prosecutor_llm_tool": lambda: _import_tool("quantalogic.tools.utilities", "ProsecutorLLMTool"),

}

def create_custom_agent(
    model_name: str,
    vision_model_name: Optional[str] = None,
    no_stream: bool = False,
    compact_every_n_iteration: Optional[int] = None,
    max_tokens_working_memory: Optional[int] = None,
    specific_expertise: str = "",
    tools: Optional[list[dict[str, Any]]] = None,
    memory: Optional[AgentMemory] = None,
    agent_mode: str = "react",
    max_iterations: Optional[int] = 15,
) -> Agent:
    """Create an agent with lazy-loaded tools and graceful error handling.

    Args:
        model_name: Name of the model to use
        vision_model_name: Name of the vision model to use
        no_stream: If True, disable streaming
        compact_every_n_iteration: Frequency of memory compaction
        max_tokens_working_memory: Maximum tokens for working memory
        specific_expertise: Specific expertise of the agent
        tools: List of tool configurations with type and parameters
        memory: Memory object to use for the agent

    Returns:
        Agent: Configured agent instance
    """
    logger.info("Creating custom agent with model: {}".format(model_name))
    logger.info("tools: {}".format(tools))
    # Create storage directory for RAG
    storage_dir = os.path.join(os.path.dirname(__file__), "storage", "rag")
    os.makedirs(storage_dir, exist_ok=True)

    # Create event emitter
    event_emitter = EventEmitter()

    def get_llm_params(params: dict) -> dict:
        """Get common parameters for LLM-based tools."""
        return {
            "model_name": params.get("model_name", model_name),
            "on_token": console_print_token if not no_stream else None,
            "event_emitter": event_emitter
        }

    # Define tool configurations with default parameters
    tool_configs = {
        # LLM Tools with shared parameters
        "llm": lambda params: create_tool_instance(TOOL_IMPORTS["llm"](), **get_llm_params(params)),
        "co_worker_agent": lambda params: create_tool_instance(TOOL_IMPORTS["co_worker_agent"](), **get_llm_params(params)),
        "oriented_llm_tool": lambda params: create_tool_instance(TOOL_IMPORTS["oriented_llm_tool"](), 
            **get_llm_params(params),
            role=params.get("role", ""),
            name=params.get("name", "oriented_llm_tool"),
            description=params.get("description", "A tool that provides expert explanations on quantum physics concepts"),
        ),
        "document_llm_tool": lambda params: create_tool_instance(TOOL_IMPORTS["document_llm_tool"](), 
            **get_llm_params(params),
            context=params.get("context", ""), 
        ),
        "legal_llm_tool": lambda params: create_tool_instance(TOOL_IMPORTS["legal_llm_tool"](), **get_llm_params(params)),
        "llm_vision": lambda params: create_tool_instance(TOOL_IMPORTS["llm_vision"](),
            model_name=params.get("vision_model_name") or "gpt-4o-mini",
            on_token=console_print_token if not no_stream else None
        ),
        "llm_image_generation": lambda params: create_tool_instance(TOOL_IMPORTS["llm_image_generation"](),
            provider="dall-e",
            model_name="openai/dall-e-3",
            on_token=console_print_token if not no_stream else None
        ),
        "stable_diffusion": lambda params: create_tool_instance(TOOL_IMPORTS["stable_diffusion"](),
            on_token=console_print_token if not no_stream else None
        ),
        
        # Simple tools without parameters
        "download_http_file": lambda _: create_tool_instance(TOOL_IMPORTS["download_http_file"]()),
        "duck_duck_go_search": lambda _: create_tool_instance(TOOL_IMPORTS["duck_duck_go_search"]()),
        "write_file": lambda _: create_tool_instance(TOOL_IMPORTS["write_file"]()),
        "file_tracker": lambda _: create_tool_instance(TOOL_IMPORTS["file_tracker"]()),
        "task_complete": lambda _: create_tool_instance(TOOL_IMPORTS["task_complete"]()),
        "edit_whole_content": lambda _: create_tool_instance(TOOL_IMPORTS["edit_whole_content"]()),
        "execute_bash_command": lambda _: create_tool_instance(TOOL_IMPORTS["execute_bash_command"]()),
        "input_question": lambda _: create_tool_instance(TOOL_IMPORTS["input_question"]()),
        "list_directory": lambda _: create_tool_instance(TOOL_IMPORTS["list_directory"]()),
        "markitdown": lambda _: create_tool_instance(TOOL_IMPORTS["markitdown"]()),
        "nodejs": lambda _: create_tool_instance(TOOL_IMPORTS["nodejs"]()),
        "python": lambda _: create_tool_instance(TOOL_IMPORTS["python"]()),
        "read_file_block": lambda _: create_tool_instance(TOOL_IMPORTS["read_file_block"]()),
        "read_file": lambda _: create_tool_instance(TOOL_IMPORTS["read_file"]()),
        "read_html": lambda _: create_tool_instance(TOOL_IMPORTS["read_html"]()),
        "replace_in_file": lambda _: create_tool_instance(TOOL_IMPORTS["replace_in_file"]()),
        "ripgrep": lambda _: create_tool_instance(TOOL_IMPORTS["ripgrep"]()),
        "search_definition_names": lambda _: create_tool_instance(TOOL_IMPORTS["search_definition_names"]()),
        "wikipedia_search": lambda _: create_tool_instance(TOOL_IMPORTS["wikipedia_search"]()),
        "google_news": lambda _: create_tool_instance(TOOL_IMPORTS["google_news"]()),
        
        # Tools with specific configurations
        "safe_python_interpreter": lambda _: create_tool_instance(TOOL_IMPORTS["safe_python_interpreter"](),
            allowed_modules=["math", "numpy", "pandas", "datetime", "random", "statistics", "decimal"]
        ),
        
        # LLM-based tools with additional parameters
        "presentation_llm": lambda params: create_tool_instance(TOOL_IMPORTS["presentation_llm"](),
            **get_llm_params(params),
            additional_info=params.get("additional_info", "")
        ),
        "sequence": lambda params: create_tool_instance(TOOL_IMPORTS["sequence"](),
            **get_llm_params(params)
        ),
        
        # Database tools
        "sql_query": lambda params: create_tool_instance(TOOL_IMPORTS["sql_query"](),
            **get_llm_params(params),
            connection_string=params.get("connection_string", "")
        ),
        "sql_query_advanced": lambda params: create_tool_instance(TOOL_IMPORTS["sql_query_advanced"](),
            **get_llm_params(params),
            connection_string=params.get("connection_string", "")
        ),
        
        # Git tools
        "clone_repo_tool": lambda params: create_tool_instance(TOOL_IMPORTS["clone_repo_tool"](),
            auth_token=params.get("auth_token", "")
        ),
        "bitbucket_clone_repo_tool": lambda params: create_tool_instance(TOOL_IMPORTS["bitbucket_clone_repo_tool"](),
            access_token=params.get("access_token", "")
        ),
        "bitbucket_operations_tool": lambda params: create_tool_instance(TOOL_IMPORTS["bitbucket_operations_tool"](),
            access_token=params.get("access_token", "")
        ),
        "git_operations_tool": lambda params: create_tool_instance(TOOL_IMPORTS["git_operations_tool"](),
            auth_token=params.get("auth_token", "")
        ),
        
        # Document conversion tools
        "markdown_to_pdf": lambda params: create_tool_instance(TOOL_IMPORTS["markdown_to_pdf"](),**get_llm_params(params)),
        "markdown_to_pptx": lambda params: create_tool_instance(TOOL_IMPORTS["markdown_to_pptx"](),**get_llm_params(params)),
        "markdown_to_html": lambda params: create_tool_instance(TOOL_IMPORTS["markdown_to_html"](),**get_llm_params(params)),
        "markdown_to_epub": lambda params: create_tool_instance(TOOL_IMPORTS["markdown_to_epub"](),**get_llm_params(params)),
        "markdown_to_ipynb": lambda params: create_tool_instance(TOOL_IMPORTS["markdown_to_ipynb"](),**get_llm_params(params)),
        "markdown_to_latex": lambda params: create_tool_instance(TOOL_IMPORTS["markdown_to_latex"](),**get_llm_params(params)),
        "markdown_to_docx": lambda _: create_tool_instance(TOOL_IMPORTS["markdown_to_docx"]()),
        
        # Utility tools
        "csv_processor": lambda _: create_tool_instance(TOOL_IMPORTS["csv_processor"]()),
        "mermaid_validator_tool": lambda _: create_tool_instance(TOOL_IMPORTS["mermaid_validator_tool"]()),
        "download_file_tool": lambda _: create_tool_instance(TOOL_IMPORTS["download_file_tool"]()),
        
        # Composio tools
        "email_tool": lambda _: create_tool_instance(TOOL_IMPORTS["email_tool"](),
            action="GMAIL_SEND_EMAIL",
            name="email_tool",
            description="Send emails via Gmail",
            need_validation=False
        ),
        "callendar_tool": lambda _: create_tool_instance(TOOL_IMPORTS["callendar_tool"](),
            action="GOOGLECALENDAR_CREATE_EVENT",
            name="callendar_tool",
            description="Create events in Google Calendar",
            need_validation=False
        ),
        "weather_tool": lambda _: create_tool_instance(TOOL_IMPORTS["weather_tool"](),
            action="WEATHERMAP_WEATHER",
            name="weather_tool",
            description="Get weather information for a location"
        ),
        
        # NASA tools
        "nasa_neows_tool": lambda _: create_tool_instance(TOOL_IMPORTS["nasa_neows_tool"]()),
        "nasa_apod_tool": lambda _: create_tool_instance(TOOL_IMPORTS["nasa_apod_tool"]()),
        "product_hunt_tool": lambda _: create_tool_instance(TOOL_IMPORTS["product_hunt_tool"]()),
        
        # Multilingual RAG tool
        "rag_tool_hf": lambda params: create_tool_instance(TOOL_IMPORTS["rag_tool_hf"](),
            persist_dir=params.get("persist_dir", "./storage/multilingual_rag"),
            use_ocr_for_pdfs=params.get("use_ocr_for_pdfs", False),
            ocr_model=params.get("ocr_model", "openai/gpt-4o-mini"),
            embed_model=params.get("embed_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            document_paths=params.get("document_paths", ["ddd"])
        ),
        
        # Legal RAG tool
        "openai_legal_rag": lambda params: create_tool_instance(TOOL_IMPORTS["openai_legal_rag"](),
            persist_dir=params.get("persist_dir", "./storage/openai_legal_rag"),
            document_paths=params.get("document_paths", []),
            openai_api_key=params.get("openai_api_key", os.getenv("OPENAI_API_KEY")),
            embedding_model=params.get("embedding_model", "text-embedding-3-large"),
            chunk_size=params.get("chunk_size", 512),
            chunk_overlap=params.get("chunk_overlap", 128),
            bm25_weight=params.get("bm25_weight", 0.3),
            embedding_weight=params.get("embedding_weight", 0.7),
            force_reindex=params.get("force_reindex", True),
            cache_embeddings=params.get("cache_embeddings", True)
        ),
        
        # Simple RAG tool
        "simple_rag": lambda params: create_tool_instance(TOOL_IMPORTS["simple_rag"](),
            persist_dir=params.get("persist_dir", "./storage/simple_rag"),
            document_paths=params.get("document_paths", []),
            chunk_size=params.get("chunk_size", 1024),
            chunk_overlap=params.get("chunk_overlap", 256),
            force_reindex=params.get("force_reindex", True),
            model_name=params.get("embedding_model", "text-embedding-3-large"),
            use_temp_dir=params.get("use_temp_dir", True),
            llm_model=params.get("model_name", "gpt-4o-mini"),
            on_token=console_print_token if not no_stream else None,
            event_emitter=event_emitter
        ),
        
        # Legal Embedding RAG tool
        "legal_embedding_rag": lambda params: create_tool_instance(TOOL_IMPORTS["legal_embedding_rag"](),
            persist_dir=params.get("persist_dir", "./storage/legal_embedding_rag"),
            document_paths=params.get("document_paths", []),
            embed_model=params.get("embed_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
            chunk_size=params.get("chunk_size", 512),
            chunk_overlap=params.get("chunk_overlap", 128),
            force_reindex=params.get("force_reindex", True)
        ),
        
        # General RAG tool
        "general_rag": lambda params: create_tool_instance(TOOL_IMPORTS["general_rag"](),
            persist_dir=params.get("persist_dir", "./storage/general_rag"),
            document_paths=params.get("document_paths", []),
            model_name=params.get("embedding_model", "text-embedding-3-large"),
            chunk_size=params.get("chunk_size", 512),
            chunk_overlap=params.get("chunk_overlap", 128), 
        ),
        
        "vscode_server_tool": lambda _: create_tool_instance(TOOL_IMPORTS["vscode_server_tool"]()),
        "linkup_tool": lambda _: create_tool_instance(TOOL_IMPORTS["linkup_tool"]()),
        
        "recommend_popular_products_tool": lambda params: create_tool_instance(TOOL_IMPORTS["recommend_popular_products_tool"](),**get_llm_params(params)),
        "product_identifier_tool": lambda params: create_tool_instance(TOOL_IMPORTS["product_identifier_tool"](),**get_llm_params(params)),
        "product_memory_tool": lambda _: create_tool_instance(TOOL_IMPORTS["product_memory_tool"]()),
        "product_validator_tool": lambda params: create_tool_instance(TOOL_IMPORTS["product_validator_tool"](),**get_llm_params(params)),

        "legal_classifier_tool": lambda params: create_tool_instance(TOOL_IMPORTS["legal_classifier_tool"](), **get_llm_params(params)),
        "legal_letter_analyzer_tool": lambda params: create_tool_instance(TOOL_IMPORTS["legal_letter_analyzer_tool"](), **get_llm_params(params)),
        "legal_case_triage_tool": lambda params: create_tool_instance(TOOL_IMPORTS["legal_case_triage_tool"](), **get_llm_params(params)),
        "contract_comparison_tool": lambda params: create_tool_instance(TOOL_IMPORTS["contract_comparison_tool"](), **get_llm_params(params)),
        "contract_extractor_tool": lambda params: create_tool_instance(TOOL_IMPORTS["contract_extractor_tool"](), **get_llm_params(params)),
        "contextual_llm_tool": lambda params: create_tool_instance(TOOL_IMPORTS["contextual_llm_tool"](), **get_llm_params(params)),
        "defender_llm_tool": lambda params: create_tool_instance(TOOL_IMPORTS["defender_llm_tool"](), **get_llm_params(params)),
        "judicial_analytics_tool": lambda params: create_tool_instance(TOOL_IMPORTS["judicial_analytics_tool"](), **get_llm_params(params)),
        "prosecutor_llm_tool": lambda params: create_tool_instance(TOOL_IMPORTS["prosecutor_llm_tool"](), **get_llm_params(params)),

    }

    # Log available tool types before processing
    logger.debug("Available tool types:")
    for tool_type in tool_configs.keys():
        logger.debug(f"- {tool_type}")

    # Define write tools that trigger automatic download tool addition
    write_tools = {"write_file", "edit_whole_content", "replace_in_file"}
    agent_tools = []
    has_write_tool = False

    # Process requested tools
    if tools:
        logger.debug(f"Total tools to process: {len(tools)}")
        for tool_config in tools:
            tool_type = tool_config.get("type")
            logger.debug(f"Processing tool type: {tool_type}")
            
            if tool_type in tool_configs:
                try:
                    # Get the tool creation function
                    tool_create_func = tool_configs.get(tool_type)
                    if not tool_create_func:
                        logger.warning(f"No creation function found for tool type: {tool_type}")
                        continue
                    
                    # Create tool instance with parameters
                    tool_params = tool_config.get("parameters", {})
                    
                    # Create tool instance with parameters
                    tool_instance = tool_create_func(tool_params)
                    
                    if tool_instance:  # Check if tool creation was successful
                        agent_tools.append(tool_instance)
                        logger.info(f"Successfully added tool: {tool_type}")
                        if tool_type in write_tools:
                            has_write_tool = True
                    else:
                        logger.warning(f"Tool {tool_type} was not created (returned None)")
                except ImportError as e:
                    logger.warning(f"Failed to load tool {tool_type}: Required library missing - {str(e)}")
                except ValueError as e:
                    if "API_KEY" in str(e):
                        logger.warning(f"Failed to create tool {tool_type}: Missing API key - {str(e)}")
                    else:
                        logger.error(f"Failed to create tool {tool_type}: {str(e)}", exc_info=True)
                except Exception as e:
                    logger.error(f"Failed to create tool {tool_type}: {str(e)}", exc_info=True)
            else:
                logger.warning(f"Unknown tool type: {tool_type} - Skipping")


    # Create and return the agent
    try:
        return Agent(
            model_name=model_name,
            tools=agent_tools,
            event_emitter=event_emitter,
            compact_every_n_iterations=compact_every_n_iteration,
            max_tokens_working_memory=max_tokens_working_memory,
            specific_expertise=specific_expertise,
            memory=memory if memory else AgentMemory(),
            agent_mode=agent_mode,
            max_iterations=max_iterations,
        )
    except Exception as e:
        logger.error(f"Failed to create agent: {str(e)}")
        raise
