"""Website Search Tool using RAG for semantic search within websites."""

import asyncio
from typing import Optional, List, Dict, Any
import hashlib
import json
import os
from datetime import datetime, timedelta

import aiohttp
from bs4 import BeautifulSoup
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, field_validator
import openai
from tqdm import tqdm

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.web_scraper_tool import WebScraperTool, USER_AGENTS

class WebsiteCache(BaseModel):
    """Cache for website content and embeddings."""
    url: str
    content: Dict[str, Any]
    embeddings: Dict[str, List[float]]
    last_updated: datetime

    class Config:
        arbitrary_types_allowed = True

class WebsiteSearchTool(Tool):
    """Tool for semantic search within website content using RAG."""

    name: str = Field(default="website_search")
    description: str = Field(
        default=(
            "Performs semantic search within website content using RAG (Retrieval-Augmented Generation). "
            "Can search across any website or be restricted to a specific domain. "
            "Uses OpenAI embeddings for semantic similarity and content retrieval."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="website",
                arg_type="string",
                description="The website URL to search within. Required if not initialized with a specific website.",
                required=True,
                example="https://example.com",
            ),
            ToolArgument(
                name="query",
                arg_type="string",
                description="The search query to find relevant content.",
                required=True,
                example="What are the main features?",
            ),
            ToolArgument(
                name="max_results",
                arg_type="int",
                description="Maximum number of results to return",
                required=False,
                default="5",
            ),
            ToolArgument(
                name="refresh_cache",
                arg_type="boolean",
                description="Force refresh the website cache",
                required=False,
                default="false",
            ),
        ]
    )

    # Tool configuration
    fixed_website: Optional[str] = Field(default=None, description="Fixed website URL if tool is restricted to one site")
    cache_dir: str = Field(default=".website_cache", description="Directory to store website cache")
    cache_duration: timedelta = Field(default=timedelta(days=1), description="How long to keep cached content")
    chunk_size: int = Field(default=1000, description="Size of text chunks for embedding")
    
    def __init__(
        self,
        website: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        cache_dir: str = ".website_cache",
        **kwargs
    ):
        """Initialize the WebsiteSearchTool.
        
        Args:
            website: Optional fixed website URL to restrict searches to
            openai_api_key: OpenAI API key for embeddings
            cache_dir: Directory to store website cache
            **kwargs: Additional arguments passed to Tool
        """
        super().__init__(**kwargs)
        self.fixed_website = website
        self.cache_dir = cache_dir
        self.scraper = WebScraperTool()
        
        # Set up OpenAI client
        if openai_api_key:
            openai.api_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, url: str) -> str:
        """Get cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.json")

    async def _get_website_content(self, url: str, refresh: bool = False) -> WebsiteCache:
        """Get website content from cache or fetch and cache it."""
        cache_path = self._get_cache_path(url)
        
        # Check cache if refresh not forced
        if not refresh and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                cache = WebsiteCache(**cache_data)
                
                # Return cache if still valid
                if datetime.now() - cache.last_updated < self.cache_duration:
                    return cache
            except Exception as e:
                logger.warning(f"Error reading cache for {url}: {e}")
        
        # Fetch and process website content
        logger.info(f"Fetching and processing content for {url}")
        
        # Scrape website content
        content = await self.scraper.async_execute(website=url)
        
        # Split content into chunks
        chunks = self._split_content(content)
        
        # Get embeddings for chunks
        embeddings = {}
        for chunk_id, chunk in tqdm(chunks.items(), desc="Generating embeddings"):
            embedding = await self._get_embedding(chunk)
            embeddings[chunk_id] = embedding
        
        # Create and save cache
        cache = WebsiteCache(
            url=url,
            content=chunks,
            embeddings=embeddings,
            last_updated=datetime.now()
        )
        
        with open(cache_path, 'w') as f:
            json.dump(cache.dict(), f)
        
        return cache

    def _split_content(self, content: Dict[str, Any]) -> Dict[str, str]:
        """Split content into chunks for embedding."""
        chunks = {}
        chunk_id = 0
        
        # Process title
        if content.get("title"):
            chunks[f"chunk_{chunk_id}"] = f"Title: {content['title']}"
            chunk_id += 1
        
        # Process main text
        text = content.get("text", "")
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > self.chunk_size:
                chunks[f"chunk_{chunk_id}"] = " ".join(current_chunk)
                chunk_id += 1
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks[f"chunk_{chunk_id}"] = " ".join(current_chunk)
        
        return chunks

    async def _get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text."""
        try:
            response = await openai.Embedding.acreate(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise ValueError(f"Error getting embedding: {e}")

    def _calculate_similarities(self, query_embedding: List[float], cache: WebsiteCache) -> List[tuple]:
        """Calculate similarities between query and cached content."""
        similarities = []
        query_embedding = np.array(query_embedding)
        
        for chunk_id, chunk_embedding in cache.embeddings.items():
            similarity = np.dot(query_embedding, np.array(chunk_embedding))
            similarities.append((chunk_id, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)

    async def async_execute(
        self,
        website: Optional[str] = None,
        query: str = "",
        max_results: str = "5",
        refresh_cache: str = "false"
    ) -> Dict[str, Any]:
        """Execute semantic search on website content.
        
        Args:
            website: Website URL to search (required if no fixed_website)
            query: Search query
            max_results: Maximum number of results to return
            refresh_cache: Whether to force refresh the cache
            
        Returns:
            Dict containing search results and metadata
        """
        # Validate inputs
        target_url = self.fixed_website or website
        if not target_url:
            raise ValueError("Website URL must be provided either during initialization or execution")
        
        max_results = int(max_results)
        refresh_cache = refresh_cache.lower() == "true"
        
        # Get website content and embeddings
        cache = await self._get_website_content(target_url, refresh_cache)
        
        # Get query embedding
        query_embedding = await self._get_embedding(query)
        
        # Calculate similarities and get top results
        similarities = self._calculate_similarities(query_embedding, cache)
        top_results = similarities[:max_results]
        
        # Format results
        results = []
        for chunk_id, similarity in top_results:
            results.append({
                "content": cache.content[chunk_id],
                "relevance_score": float(similarity)
            })
        
        return {
            "website": target_url,
            "query": query,
            "results": results,
            "total_chunks": len(cache.content),
            "cache_age": (datetime.now() - cache.last_updated).total_seconds()
        }

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for async_execute."""
        return asyncio.run(self.async_execute(*args, **kwargs))


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize tool with OpenAI API key
        tool = WebsiteSearchTool(website="https://www.viveris.fr/")
        
        # Perform search
        results = await tool.async_execute(
            query="What are the main features?",
            max_results="3"
        )
        
        print("\nSearch Results:")
        for i, result in enumerate(results["results"], 1):
            print(f"\n{i}. Relevance Score: {result['relevance_score']:.3f}")
            print(f"Content: {result['content'][:200]}...")
        
        print(f"\nTotal chunks: {results['total_chunks']}")
        print(f"Cache age: {results['cache_age']:.1f} seconds")
    
    asyncio.run(main())
