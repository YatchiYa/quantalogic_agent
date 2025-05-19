"""Web Scraper LLM Tool for extracting and analyzing data from websites with AI-powered insights."""

import asyncio
from typing import Optional, Dict, Any, List

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.website_search.web_scraper_tool import WebScraperTool
from quantalogic.tools.llm_tool import LLMTool

class WebScraperLLMTool(Tool):
    """Advanced web scraping tool with LLM-powered analysis capabilities."""

    class Config(BaseModel):
        url: str = Field(..., description="Target website URL to scrape and analyze")
        query: str = Field(..., description="Question to ask about the website content")
        max_retries: int = Field(
            default=3,
            description="Maximum number of retry attempts",
            ge=1,
            le=5
        )
        timeout: float = Field(
            default=60.0,
            description="Request timeout in seconds",
            ge=5.0,
            le=120.0
        )

        @field_validator("url")
        def validate_url(cls, v: str) -> str:
            if not v.startswith(("http://", "https://")):
                raise ValueError("URL must start with http:// or https://")
            return v

    name: str = "web_scraper_llm"
    description: str = (
        "Advanced web scraping tool with LLM-powered analysis. "
        "Extracts content from websites and uses AI to answer specific questions about the content. "
        "Can perform tasks like feature extraction, content summarization, and detailed analysis."
    )
    arguments: list = [
        ToolArgument(
            name="url",
            arg_type="string",
            description="Target website URL to scrape and analyze",
            required=True,
            example="https://example.com"
        ),
        ToolArgument(
            name="query",
            arg_type="string",
            description="Question to ask about the website content (e.g., 'What are the main features?')",
            required=True,
            example="What are the main features?"
        ),
        ToolArgument(
            name="max_retries",
            arg_type="int",
            description="Maximum number of retry attempts (1-5)",
            required=False,
            default="3"
        ),
        ToolArgument(
            name="timeout",
            arg_type="float",
            description="Request timeout in seconds (5-120)",
            required=False,
            default="60.0"
        )
    ]

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        **kwargs
    ):
        """Initialize the WebScraperLLMTool.
        
        Args:
            model_name: Name of the LLM model to use for analysis
            **kwargs: Additional arguments passed to Tool
        """
        super().__init__(**kwargs)
        self.scraper = WebScraperTool()
        self.llm = LLMTool(model_name=model_name)

    async def _extract_website_content(self, url: str, timeout: float, max_retries: int) -> Dict[str, Any]:
        """Extract content from the website using WebScraperTool."""
        try:
            content = await self.scraper.async_execute(
                website=url,
                timeout=str(timeout),
                max_retries=str(max_retries)
            )
            return content
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            raise ValueError(f"Error extracting content from {url}: {e}")

    async def _analyze_content_with_llm(self, content: Dict[str, Any], query: str) -> str:
        """Analyze website content using LLM to answer the specific query."""
        try:
            # Create a structured context from the content
            context = self._format_content_for_llm(content)
            
            # Create a system prompt for the LLM
            system_prompt = (
                "You are an expert web content analyzer. Your task is to analyze website content "
                "and provide accurate, detailed answers to questions about that content. "
                "Base your analysis ONLY on the provided website content. "
                "If the answer cannot be determined from the content, say so clearly. "
                "Provide specific references to sections of the content where possible."
            )
            
            # Create a prompt with the context and query
            prompt = f"""
WEBSITE CONTENT:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer to the user's question based solely on the website content provided above.
If the content doesn't contain information to answer the question, state that clearly.
"""
            
            # Get LLM response
            response = await self.llm.async_execute(
                system_prompt=system_prompt,
                prompt=prompt,
                temperature="0.3"  # Lower temperature for more factual responses
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing content with LLM: {e}")
            raise ValueError(f"Error analyzing content with LLM: {e}")

    def _format_content_for_llm(self, content: Dict[str, Any]) -> str:
        """Format the scraped content into a structured text for the LLM."""
        formatted_text = []
        
        # Add title if available
        if content.get("title"):
            formatted_text.append(f"TITLE: {content['title']}")
            formatted_text.append("")
        
        # Add main text content
        if content.get("text"):
            formatted_text.append("MAIN CONTENT:")
            formatted_text.append(content["text"])
            formatted_text.append("")
        
        # Add links if available
        if content.get("links") and len(content["links"]) > 0:
            formatted_text.append("LINKS:")
            for i, link in enumerate(content["links"][:20], 1):  # Limit to 20 links to avoid token overflow
                formatted_text.append(f"{i}. {link.get('text', 'No text')} - {link.get('href', 'No URL')}")
            formatted_text.append("")
        
        # Add selector-specific content if available
        for selector, elements in content.items():
            if selector not in ["title", "text", "links"] and isinstance(elements, list):
                formatted_text.append(f"SELECTOR '{selector}':")
                for i, element in enumerate(elements, 1):
                    if isinstance(element, dict) and "text" in element:
                        formatted_text.append(f"{i}. {element['text']}")
                formatted_text.append("")
        
        return "\n".join(formatted_text)

    async def async_execute(
        self,
        url: str,
        query: str,
        max_retries: str = "3",
        timeout: str = "60.0"
    ) -> Dict[str, Any]:
        """
        Execute the web scraping and LLM analysis task asynchronously.
        
        Args:
            url: Target website URL
            query: Question to ask about the website content
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            
        Returns:
            dict: Analysis results with extracted content and LLM answer
        """
        # Convert and validate parameters
        self.config = self.Config(
            url=url,
            query=query,
            max_retries=int(max_retries),
            timeout=float(timeout)
        )
        
        try:
            # Step 1: Extract website content
            logger.info(f"Extracting content from {url}")
            content = await self._extract_website_content(
                url=self.config.url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
            
            # Step 2: Analyze content with LLM
            logger.info(f"Analyzing content with LLM for query: {query}")
            answer = await self._analyze_content_with_llm(content, self.config.query)
            
            # Return results
            return {
                "url": self.config.url,
                "query": self.config.query,
                "answer": answer,
                "content_summary": {
                    "title": content.get("title"),
                    "text_length": len(content.get("text", "")),
                    "link_count": len(content.get("links", [])),
                }
            }
                
        except Exception as e:
            logger.error(f"Error in WebScraperLLMTool: {e}")
            raise ValueError(f"Error in WebScraperLLMTool: {e}")

    def execute(
        self,
        url: str,
        query: str,
        max_retries: str = "3",
        timeout: str = "60.0"
    ) -> Dict[str, Any]:
        """Synchronous wrapper for async_execute."""
        return asyncio.run(
            self.async_execute(
                url=url,
                query=query,
                max_retries=max_retries,
                timeout=timeout
            )
        )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize tool
        tool = WebScraperLLMTool()
        
        # Example 1: Basic feature analysis
        result = await tool.async_execute(
            url="https://www.viveris.fr/",
            query="What are the main features or services offered by this company?"
        )
        print("\nQuery:", result["query"])
        print("Answer:", result["answer"])
        
        # Example 2: Detailed audit
        # result = await tool.async_execute(
        #     url="https://www.viveris.fr/",
        #     query="Give a full audit of the webpage including its structure, content quality, and main topics."
        # )
        # print("\nQuery:", result["query"])
        # print("Answer:", result["answer"])
        
        # # Example 3: Content extraction
        # result = await tool.async_execute(
        #     url="https://www.viveris.fr/",
        #     query="Extract and summarize the main articles or content sections on this page."
        # )
        # print("\nQuery:", result["query"])
        # print("Answer:", result["answer"])
    
    asyncio.run(main())
