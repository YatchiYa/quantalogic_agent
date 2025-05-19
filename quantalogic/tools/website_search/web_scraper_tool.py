"""Web Scraper Tool for extracting data from websites with advanced features."""

import asyncio
import random
import time
from typing import Optional, List

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from quantalogic.tools.tool import Tool, ToolArgument

# Comprehensive User-Agent list for request rotation
USER_AGENTS = [
    # Chrome
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    
    # Firefox
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    
    # Safari
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    
    # Edge
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    
    # Opera
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
    
    # Mobile Browsers
    "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/120.0.0.0 Mobile/15E148 Safari/604.1",
]

# Additional headers to mimic real browser behavior
ADDITIONAL_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "DNT": "1",
}

class WebScraperTool(Tool):
    """Advanced web scraping tool with configurable options."""

    class Config(BaseModel):
        website: str = Field(..., description="Target website URL to scrape")
        selectors: Optional[List[str]] = Field(
            default=None,
            description="CSS selectors to extract specific elements (e.g., ['h1', '.content', '#main'])"
        )
        delay: float = Field(
            default=1.0,
            description="Delay between requests in seconds",
            ge=0.5,
            le=5.0
        )
        timeout: float = Field(
            default=30.0,
            description="Request timeout in seconds",
            ge=5.0,
            le=60.0
        )
        max_retries: int = Field(
            default=3,
            description="Maximum number of retry attempts",
            ge=1,
            le=5
        )

        @field_validator("website")
        def validate_website(cls, v: str) -> str:
            if not v.startswith(("http://", "https://")):
                raise ValueError("Website URL must start with http:// or https://")
            return v

    name: str = "web_scraper"
    description: str = (
        "Advanced web scraping tool that extracts data from websites with configurable options. "
        "Features include CSS selector filtering, rate limiting, retry logic, and error handling. "
        "Returns extracted content in a structured format."
    )
    arguments: list = [
        ToolArgument(
            name="website",
            arg_type="string",
            description="Target website URL to scrape",
            required=True,
            example="https://example.com"
        ),
        ToolArgument(
            name="selectors",
            arg_type="string",
            description="Comma-separated CSS selectors (e.g., 'h1,.content,#main')",
            required=False,
            default=None
        ),
        ToolArgument(
            name="delay",
            arg_type="float",
            description="Delay between requests in seconds (0.5-5.0)",
            required=False,
            default="1.0"
        ),
        ToolArgument(
            name="timeout",
            arg_type="float",
            description="Request timeout in seconds (5-60)",
            required=False,
            default="30.0"
        ),
        ToolArgument(
            name="max_retries",
            arg_type="int",
            description="Maximum number of retry attempts (1-5)",
            required=False,
            default="3"
        )
    ]

    async def _make_request(self, session: aiohttp.ClientSession, url: str, timeout: float, retries: int = 0) -> str:
        """Make an HTTP request with retry logic."""
        try:
            headers = ADDITIONAL_HEADERS.copy()
            headers["User-Agent"] = random.choice(USER_AGENTS)
            
            async with session.get(url, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                return await response.text()
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if retries < self.config.max_retries:
                wait_time = (2 ** retries) * self.config.delay
                logger.warning(f"Request failed, retrying in {wait_time}s... ({e})")
                await asyncio.sleep(wait_time)
                return await self._make_request(session, url, timeout, retries + 1)
            raise ValueError(f"Failed to fetch {url} after {retries} retries: {e}")

    def _parse_content(self, html: str, selectors: Optional[List[str]] = None) -> dict:
        """Parse HTML content using BeautifulSoup and extract selected elements."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            if not selectors:
                # Return basic page info if no selectors specified
                return {
                    "title": soup.title.string if soup.title else None,
                    "text": soup.get_text(strip=True),
                    "links": [{"text": a.text, "href": a.get("href")} for a in soup.find_all("a", href=True)]
                }
            
            # Extract content for each selector
            result = {}
            for selector in selectors:
                elements = soup.select(selector)
                result[selector] = [
                    {
                        "text": el.get_text(strip=True),
                        "html": str(el),
                        "attributes": el.attrs
                    }
                    for el in elements
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            raise ValueError(f"Error parsing HTML: {e}")

    async def async_execute(
        self,
        website: str,
        selectors: Optional[str] = None,
        delay: str = "1.0",
        timeout: str = "30.0",
        max_retries: str = "3"
    ) -> dict:
        """
        Execute the web scraping task asynchronously.
        
        Args:
            website: Target website URL
            selectors: Comma-separated CSS selectors
            delay: Delay between requests in seconds
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            
        Returns:
            dict: Extracted content in structured format
        """
        # Convert and validate parameters
        self.config = self.Config(
            website=website,
            selectors=selectors.split(",") if selectors else None,
            delay=float(delay),
            timeout=float(timeout),
            max_retries=int(max_retries)
        )
        
        # Add random delay to prevent rate limiting
        await asyncio.sleep(random.uniform(0.5, self.config.delay))
        
        try:
            async with aiohttp.ClientSession() as session:
                html = await self._make_request(session, self.config.website, self.config.timeout)
                return self._parse_content(html, self.config.selectors)
                
        except Exception as e:
            logger.error(f"Error scraping {website}: {e}")
            raise ValueError(f"Error scraping {website}: {e}")

    def execute(
        self,
        website: str,
        selectors: Optional[str] = None,
        delay: str = "1.0",
        timeout: str = "30.0",
        max_retries: str = "3"
    ) -> dict:
        """Synchronous wrapper for async_execute."""
        return asyncio.run(
            self.async_execute(
                website=website,
                selectors=selectors,
                delay=delay,
                timeout=timeout,
                max_retries=max_retries
            )
        )


if __name__ == "__main__":
    # Example usage
    scraper = WebScraperTool()
    
    # Example 1: Basic scraping
    result = scraper.execute(website="https://www.viveris.fr/")
    print("Basic scraping result:", result)
    
    # Example 2: Scraping with selectors
    result = scraper.execute(
        website="https://www.viveris.fr/",
        selectors="h1,p,a.link"
    )
    print("\nScraping with selectors:", result)
