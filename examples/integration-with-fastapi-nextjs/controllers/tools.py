import asyncio
import time
from fastapi import APIRouter, HTTPException, Depends
from loguru import logger
from typing import Dict, Any, Optional
from pydantic import BaseModel 
from quantalogic.tools.linkup_tool import LinkupTool
from quantalogic.tools.google_packages.google_news_tool import GoogleNewsTool
from quantalogic.tools.duckduckgo_search_tool import DuckDuckGoSearchTool
from ..middlewares.authenticate import require_auth

router = APIRouter(prefix="/api/agent/tools", tags=["tools"])

class SearchRequest(BaseModel):
    query: str
    mode: str = "standard"  # standard, deep, search, or news
    tool_type: str = "linkup"  # linkup, news, or search
    output_type: str = "sourcedAnswer"  # sourcedAnswer or searchResults
    language: str = "en"
    period: str = "1d"
    max_results: int = 20
    country: str = "US"
    sort_by: str = "relevance"
    analyze: bool = True

@router.post("/search_qllm")
async def search_qllm(
    request: SearchRequest, 
    user: Dict[str, Any] = require_auth,
):
    """Search using either LinkupTool, GoogleNewsTool, or DuckDuckGoSearchTool."""
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
        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}. Must be one of: standard, deep, news, or search")

        return {
            "status": "success",
            "data": response,
            "tool_type": request.tool_type,
            "mode": request.mode
        }
        
    except Exception as e:
        logger.error(f"Error in search_qllm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))