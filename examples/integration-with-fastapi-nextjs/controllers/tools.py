import asyncio
import time
from fastapi import APIRouter, HTTPException, Depends
from loguru import logger
from typing import Dict, Any, Optional
from pydantic import BaseModel 
from quantalogic.tools.linkup_tool import LinkupTool
from quantalogic.tools.google_packages.google_news_tool import GoogleNewsTool
from quantalogic.tools.duckduckgo_search_tool import DuckDuckGoSearchTool
from quantalogic.tools.google_packages.linkup_enhanced_tool import LinkupEnhancedTool
from ..middlewares.authenticate import require_auth

router = APIRouter(prefix="/api/agent/tools", tags=["tools"])

class SearchRequest(BaseModel):
    query: str
    mode: Optional[str] = "standard"  # standard, deep, search, news, or enhanced
    tool_type: Optional[str] = "linkup"  # linkup, news, search, or enhanced
    output_type: Optional[str] = "sourcedAnswer"  # sourcedAnswer or searchResults
    language: Optional[str] = "en"
    period: Optional[str] = "1d"
    max_results: Optional[int] = 20
    country: Optional[str] = "US"
    sort_by: Optional[str] = "relevance"
    analyze: Optional[bool] = True
    # Enhanced search parameters
    question: Optional[str] = None  # Question to ask about search results
    analysis_depth: Optional[str] = "standard"  # quick, standard, or deep
    scrape_sources: Optional[bool] = True  # Whether to scrape source content
    max_sources_to_scrape: Optional[int] = 5  # Maximum number of sources to scrape
    output_format: Optional[str] = "standard"  # standard, article, or technical

@router.post("/search_qllm")
async def search_qllm(
    request: SearchRequest, 
    user: Dict[str, Any] = require_auth,
):
    """Search using LinkupTool, GoogleNewsTool, DuckDuckGoSearchTool, or LinkupEnhancedTool."""
    logger.info(f"Executing search_qllm with mode: {request.mode}")
    try:
        # Handle mode-based tool selection first
        if request.mode in ["standard", "deep"]:
            tool = LinkupTool()
            response = tool.execute(
                query=request.query,
                depth=request.mode,
                output_type=request.output_type
            )
        elif request.mode == "news":
            tool = GoogleNewsTool()
            response = tool.execute(
                query=request.query,
                language=request.language,
                period=request.period,
                max_results=request.max_results,
                country=request.country,
                sort_by=request.sort_by,
                analyze=request.analyze
            )
        elif request.mode == "search":
            tool = DuckDuckGoSearchTool()
            response = tool.execute(
                query=request.query,
                language=request.language,
                period=request.period,
                max_results=request.max_results,
                country=request.country,
                sort_by=request.sort_by,
                analyze=request.analyze
            )
        elif request.mode == "enhanced":
            # Validate required parameters for enhanced mode
            if not request.question:
                raise HTTPException(status_code=400, detail="'question' parameter is required for enhanced search mode")
            
            tool = LinkupEnhancedTool()
            response = await tool.async_execute(
                query=request.query,
                question=request.question,
                depth="deep" if request.tool_type == "linkup" else "standard",
                analysis_depth=request.analysis_depth,
                scrape_sources=str(request.scrape_sources).lower() if request.scrape_sources else "true",
                max_sources_to_scrape=str(request.max_sources_to_scrape) if request.max_sources_to_scrape else "5",
                output_format=request.output_format
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}. Must be one of: standard, deep, news, search, or enhanced")

        return {
            "status": "success",
            "data": response,
            "tool_type": request.tool_type,
            "mode": request.mode
        }
        
    except Exception as e:
        logger.error(f"Error in search_qllm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))