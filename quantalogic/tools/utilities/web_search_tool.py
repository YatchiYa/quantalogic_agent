"""Web search and content fetching tools using DuckDuckGo.

This module provides tools for searching the web using DuckDuckGo and fetching content
from web pages, with rate limiting to avoid being blocked.
"""

import asyncio
import re
import sys
import traceback
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import httpx
from bs4 import BeautifulSoup
from loguru import logger

from quantalogic.tools.tool import Tool, ToolArgument


@dataclass
class SearchResult:
    """Represents a single search result from DuckDuckGo."""
    title: str
    link: str
    snippet: str
    position: int


class RateLimiter:
    """Rate limiter to prevent too many requests in a short period."""
    
    def __init__(self, requests_per_minute: int = 30):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum number of requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        """Acquire permission to make a request, waiting if necessary."""
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [
            req for req in self.requests if now - req < timedelta(minutes=1)
        ]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(now)


class DuckDuckGoSearcher:
    """Class for searching DuckDuckGo and parsing results."""
    
    BASE_URL = "https://html.duckduckgo.com/html"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    def __init__(self):
        """Initialize the DuckDuckGo searcher with rate limiting."""
        self.rate_limiter = RateLimiter()

    def format_results_for_output(self, results: List[SearchResult]) -> str:
        """Format search results in a readable format.
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            Formatted string with search results
        """
        if not results:
            return "No results were found for your search query. This could be due to DuckDuckGo's bot detection or the query returned no matches. Please try rephrasing your search or try again in a few minutes."

        output = []
        output.append(f"Found {len(results)} search results:\n")

        for result in results:
            output.append(f"{result.position}. {result.title}")
            output.append(f"   URL: {result.link}")
            output.append(f"   Summary: {result.snippet}")
            output.append("")  # Empty line between results

        return "\n".join(output)

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search DuckDuckGo for the given query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()

            # Create form data for POST request
            data = {
                "q": query,
                "b": "",
                "kl": "",
            }

            logger.info(f"Searching DuckDuckGo for: {query}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL, data=data, headers=self.HEADERS, timeout=30.0
                )
                response.raise_for_status()

            # Parse HTML response
            soup = BeautifulSoup(response.text, "html.parser")
            if not soup:
                logger.error("Failed to parse HTML response")
                return []

            results = []
            for result in soup.select(".result"):
                title_elem = result.select_one(".result__title")
                if not title_elem:
                    continue

                link_elem = title_elem.find("a")
                if not link_elem:
                    continue

                title = link_elem.get_text(strip=True)
                link = link_elem.get("href", "")

                # Skip ad results
                if "y.js" in link:
                    continue

                # Clean up DuckDuckGo redirect URLs
                if link.startswith("//duckduckgo.com/l/?uddg="):
                    link = urllib.parse.unquote(link.split("uddg=")[1].split("&")[0])

                snippet_elem = result.select_one(".result__snippet")
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                results.append(
                    SearchResult(
                        title=title,
                        link=link,
                        snippet=snippet,
                        position=len(results) + 1,
                    )
                )

                if len(results) >= max_results:
                    break

            logger.info(f"Successfully found {len(results)} results")
            return results

        except httpx.TimeoutException:
            logger.error("Search request timed out")
            return []
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during search: {str(e)}")
            traceback.print_exc(file=sys.stderr)
            return []


class WebContentFetcher:
    """Class for fetching and parsing web content."""
    
    # List of user agents to rotate through to reduce blocking
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
    ]
    
    def __init__(self):
        """Initialize the web content fetcher with rate limiting."""
        self.rate_limiter = RateLimiter(requests_per_minute=20)
        import random
        self.random = random

    async def fetch_and_parse(self, url: str) -> str:
        """Fetch and parse content from a webpage.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Parsed text content from the webpage
        """
        try:
            await self.rate_limiter.acquire()

            logger.info(f"Fetching content from: {url}")
            
            # Select a random user agent
            user_agent = self.random.choice(self.USER_AGENTS)
            
            # Add common headers to appear more like a real browser
            headers = {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.google.com/",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=headers,
                    follow_redirects=True,
                    timeout=30.0,
                )
                response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()

            # Get the text content
            text = soup.get_text()

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text).strip()

            logger.info(f"Successfully fetched and parsed content ({len(text)} characters)")
            return text

        except httpx.TimeoutException:
            logger.error(f"Request timed out for URL: {url}")
            return "Error: The request timed out while trying to fetch the webpage."
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.error(f"Access forbidden (403) for URL: {url} - Website likely blocks web scrapers")
                return "Error: This website blocks automated access. The content cannot be retrieved due to website restrictions."
            else:
                logger.error(f"HTTP error occurred while fetching {url}: {str(e)}")
                return f"Error: Could not access the webpage (HTTP {e.response.status_code})"
        except httpx.HTTPError as e:
            logger.error(f"HTTP error occurred while fetching {url}: {str(e)}")
            return f"Error: Could not access the webpage ({str(e)})"
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return f"Error: An unexpected error occurred while fetching the webpage ({str(e)})"


# Initialize searcher and fetcher instances
_searcher = DuckDuckGoSearcher()
_fetcher = WebContentFetcher()



class WebSearchTool(Tool):
    """Tool for searching the web using DuckDuckGo."""
    
    name: str = "web_search"
    description: str = "Search the web using DuckDuckGo and get formatted results"
    need_validation: bool = False
    arguments: list = [
        ToolArgument(
            name="query",
            arg_type="string",
            description="The search query string",
            required=True,
            example="python programming language",
        ),
        ToolArgument(
            name="max_results",
            arg_type="int",
            description="Maximum number of results to return (default: 10)",
            required=False,
            default="10",
            example="5",
        ),
    ]
    
    async def async_execute(self, query: str, max_results: str = "10") -> Union[str, Dict[str, Any]]:
        """Search the web using DuckDuckGo and return formatted results.
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return (default: 10)
            
        Returns:
            Formatted search results as a string
            
        Raises:
            Exception: If an error occurs during the search process
        """
        try:
            logger.info(f"Searching DuckDuckGo for: {query}")
            max_results_int = int(max_results)
            
            searcher = DuckDuckGoSearcher()
            results = await searcher.search(query, max_results_int)
            formatted_results = searcher.format_results_for_output(results)
            
            return {
                "status": "success",
                "message": f"Successfully searched for '{query}'",
                "results_count": len(results),
                "content": formatted_results
            }
        except Exception as e:
            error_msg = f"An error occurred while searching: {str(e)}"
            logger.error(error_msg)
            traceback.print_exc(file=sys.stderr)
            return {
                "status": "error",
                "message": error_msg,
                "content": "Failed to perform search. Please try again with a different query."
            }


class WebContentTool(Tool):
    """Tool for fetching content from web pages."""
    
    name: str = "web_content"
    description: str = "Fetch and parse content from a webpage URL"
    need_validation: bool = False
    arguments: list = [
        ToolArgument(
            name="url",
            arg_type="string",
            description="The webpage URL to fetch content from",
            required=True,
            example="https://example.com",
        ),
    ]
    
    async def async_execute(self, url: str) -> Union[str, Dict[str, Any]]:
        """Fetch and parse content from a webpage URL.
        
        Args:
            url: The webpage URL to fetch content from
            
        Returns:
            Dictionary containing status, message, and parsed content
            
        Raises:
            Exception: If an error occurs during the content fetching process
        """
        try:
            logger.info(f"Fetching content from: {url}")
            
            fetcher = WebContentFetcher()
            content = await fetcher.fetch_and_parse(url)
            
            return {
                "status": "success",
                "message": f"Successfully fetched content from {url}",
                "content_length": len(content),
                "content": content
            }
        except Exception as e:
            error_msg = f"An error occurred while fetching content: {str(e)}"
            logger.error(error_msg)
            traceback.print_exc(file=sys.stderr)
            return {
                "status": "error",
                "message": error_msg,
                "content": f"Failed to fetch content from {url}. Please check the URL and try again."
            }


class WebResearchTool(Tool):
    """Tool for searching the web and fetching content from search results."""
    
    name: str = "web_research"
    description: str = "Search the web using DuckDuckGo and fetch content from results"
    need_validation: bool = False
    arguments: list = [
        ToolArgument(
            name="query",
            arg_type="string",
            description="The search query string",
            required=True,
            example="python programming language",
        ),
        ToolArgument(
            name="max_results",
            arg_type="int",
            description="Maximum number of search results to return (default: 10)",
            required=False,
            default="20",
            example="10",
        ),
    ]
    
    async def async_execute(self, query: str, max_results: str = "10") -> Union[str, Dict[str, Any]]:
        """Search the web and optionally fetch content from top results.
        
        Args:
            query: The search query string
            max_results: Maximum number of search results to return (default: 5)
            fetch_content: Whether to fetch content from search results (default: True)
            max_content_results: Maximum number of search results to fetch content from (default: 3)
            
        Returns:
            Dictionary containing search results
            
        Raises:
            Exception: If an error occurs during the search or content fetching process
        """
        try:
            # Convert string parameters to appropriate types
            max_results_int = int(max_results)
            
            # Step 1: Search DuckDuckGo
            logger.info(f"Searching DuckDuckGo for: {query}")
            searcher = DuckDuckGoSearcher()
            search_results = await searcher.search(query, max_results_int)
            
            if not search_results:
                return {
                    "status": "warning",
                    "message": f"No search results found for '{query}'",
                    "search_results": [],
                    "content_results": {}
                }
            
            # Format search results
            formatted_results = searcher.format_results_for_output(search_results)
            
            # Step 2: Fetch content if requested
            content_results = {}
            if search_results:
                logger.info(f"Fetching content from top {max_results_int} results")
                fetcher = WebContentFetcher()
                
                # Only fetch from the specified number of top results
                for i, result in enumerate(search_results):
                    try:
                        logger.info(f"Fetching content from result #{i+1}: {result.title}")
                        content = await fetcher.fetch_and_parse(result.link)
                        
                        # Check if the content indicates an error
                        if content.startswith("Error:"):
                            logger.warning(f"Content fetch warning for {result.link}: {content}")
                            content_results[result.link] = {
                                "title": result.title,
                                "content": content,
                                "status": "error"
                            }
                        else:
                            content_results[result.link] = {
                                "title": result.title,
                                "content": content,
                                "status": "success"
                            }
                    except Exception as e:
                        logger.error(f"Failed to fetch content from {result.link}: {str(e)}")
                        content_results[result.link] = {
                            "title": result.title,
                            "content": f"Error fetching content: {str(e)}",
                            "status": "error"
                        }
            
            return {
                "status": "success",
                "message": f"Successfully researched '{query}'",
                "search_results_count": len(search_results),
                "content_results_count": len(content_results),
                "search_results": formatted_results,
                "content_results": content_results
            }
            
        except Exception as e:
            error_msg = f"An error occurred during web research: {str(e)}"
            logger.error(error_msg)
            traceback.print_exc(file=sys.stderr)
            return {
                "status": "error",
                "message": error_msg,
                "search_results": "",
                "content_results": {}
            }


# Export the tools
__all__ = ["WebSearchTool", "WebContentTool", "WebResearchTool"]


# Main section for testing
if __name__ == "__main__":
    import asyncio
    import json
    
    async def test_web_search():
        print("\n=== Testing WebSearchTool ===\n")
        tool = WebSearchTool()
        print(tool.to_markdown())
        result = await tool.async_execute("Python programming language")
        print(f"\nSearch Results:\n{json.dumps(result, indent=2)}")
    
    async def test_web_content():
        print("\n=== Testing WebContentTool ===\n")
        tool = WebContentTool()
        print(tool.to_markdown())
        result = await tool.async_execute("https://www.python.org")
        # Truncate content for display
        if "content" in result and isinstance(result["content"], str) and len(result["content"]) > 200:
            result["content"] = result["content"][:200] + "... [truncated]"
        print(f"\nContent Results:\n{json.dumps(result, indent=2)}")
    
    async def test_web_research():
        print("\n=== Testing WebResearchTool ===\n")
        tool = WebResearchTool()
        print(tool.to_markdown())
        result = await tool.async_execute("Python programming language", max_results="3")
        # Truncate content for display
        if "content_results" in result and isinstance(result["content_results"], dict):
            for url, data in result["content_results"].items():
                if "content" in data and isinstance(data["content"], str) and len(data["content"]) > 200:
                    data["content"] = data["content"][:200] + "... [truncated]"
        print(f"\nResearch Results:\n{json.dumps(result, indent=2)}")
    
    # Run all tests
    async def run_tests():
        await test_web_search()
        await test_web_content()
        await test_web_research()
    
    # Run the tests
    asyncio.run(run_tests())
