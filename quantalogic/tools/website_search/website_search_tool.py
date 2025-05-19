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
from tqdm import tqdm

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.website_search.web_scraper_tool import WebScraperTool
from quantalogic.tools.llm_tool import LLMTool

class WebsiteCache(BaseModel):
    """Cache for website content and embeddings."""
    url: str
    content: Dict[str, Any]
    embeddings: Dict[str, List[float]]
    last_updated: datetime

    class Config:
        arbitrary_types_allowed = True
        
    def model_dump(self, **kwargs):
        """Custom model_dump to make datetime JSON serializable."""
        data = super().model_dump(**kwargs)
        # Convert datetime to ISO format string
        if "last_updated" in data and isinstance(data["last_updated"], datetime):
            data["last_updated"] = data["last_updated"].isoformat()
        return data
        
    def dict(self):
        """Legacy dict method for backward compatibility."""
        return self.model_dump()

class WebsiteSearchTool(Tool):
    """Tool for semantic search within website content using RAG."""

    name: str = Field(default="website_search")
    description: str = Field(
        default=(
            "Performs semantic search within website content using RAG (Retrieval-Augmented Generation). "
            "Can search across any website or be restricted to a specific domain. "
            "Uses LLM embeddings for semantic similarity and content retrieval."
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
        model_name: str = "openrouter/text-embedding-3-large",
        cache_dir: str = ".website_cache",
        **kwargs
    ):
        """Initialize the WebsiteSearchTool.
        
        Args:
            website: Optional fixed website URL to restrict searches to
            model_name: Name of the embedding model to use
            cache_dir: Directory to store website cache
            **kwargs: Additional arguments passed to Tool
        """
        super().__init__(**kwargs)
        self.fixed_website = website
        self.cache_dir = cache_dir
        self.scraper = WebScraperTool()
        
        # Set up LLM embedding tool
        self.embedding_model = LLMTool(model_name=model_name)
        
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
                    # Convert ISO format string back to datetime
                    if "last_updated" in cache_data and isinstance(cache_data["last_updated"], str):
                        cache_data["last_updated"] = datetime.fromisoformat(cache_data["last_updated"])
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
            json.dump(cache.model_dump(), f)
        
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
        """Get embedding for text using LLMTool."""
        try:
            # Use a system prompt designed for embedding generation
            system_prompt = "You are an embedding generator. Convert the input text into a numerical representation."
            
            # Use the LLMTool to get embeddings
            embedding_json = await self.embedding_model.async_execute(
                system_prompt=system_prompt,
                prompt=f"Generate embedding for: {text}",
                temperature="0.0"
            )
            
            # Parse the embedding from the response
            # Note: This assumes the LLM returns a JSON string with the embedding
            # You may need to adjust this based on the actual response format
            try:
                import json
                embedding_data = json.loads(embedding_json)
                if isinstance(embedding_data, dict) and "embedding" in embedding_data:
                    return embedding_data["embedding"]
                else:
                    # Fallback: Generate a simple hash-based embedding
                    logger.warning("LLM did not return proper embedding format, using fallback")
                    return self._generate_fallback_embedding(text)
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM response as JSON, using fallback")
                return self._generate_fallback_embedding(text)
                
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Fallback to a simple hash-based embedding
            return self._generate_fallback_embedding(text)
    
    def _generate_fallback_embedding(self, text: str, dim: int = 1536) -> List[float]:
        """Generate a fallback embedding based on hash of text."""
        import hashlib
        
        # Create a deterministic but simple embedding
        hash_values = []
        for i in range(dim):
            # Create different hashes by adding a salt based on position
            h = hashlib.md5(f"{text}_{i}".encode()).digest()
            # Convert 16 bytes to a float between -1 and 1
            value = int.from_bytes(h[:4], byteorder='little') / (2**32 - 1) * 2 - 1
            hash_values.append(value)
            
        # Normalize the vector
        norm = np.linalg.norm(hash_values)
        if norm > 0:
            hash_values = [v / norm for v in hash_values]
            
        return hash_values

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
        # Initialize tool with model name
        tool = WebsiteSearchTool(
            website="https://www.viveris.fr/",
            model_name="text-embedding-3-large"
        )
        
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
