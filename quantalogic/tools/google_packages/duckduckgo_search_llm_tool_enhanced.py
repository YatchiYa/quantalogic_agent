"""DuckDuckGo Search LLM Tool for retrieving and analyzing search results with AI-powered insights."""

import asyncio
import json
from typing import List, Dict, Any, Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools import DuckDuckGoSearchTool
from quantalogic.tools.website_search.web_scraper_tool import WebScraperTool
from quantalogic.tools.llm_tool import LLMTool


class DuckDuckGoSearchLLMTool(Tool):
    """Advanced search tool with LLM-powered analysis capabilities."""

    class Config(BaseModel):
        query: str = Field(..., description="The search query to execute")
        question: str = Field(..., description="Question to ask about the search results")
        max_results: int = Field(
            default=5,
            description="Maximum number of search results to analyze (1-10)",
            ge=1,
            le=10
        )
        search_type: str = Field(
            default="text",
            description="Type of search to perform (text, news)"
        )
        region: str = Field(
            default="wt-wt",
            description="Region for search results"
        )
        safesearch: str = Field(
            default="moderate",
            description="Safesearch level"
        )
        timeout: float = Field(
            default=60.0,
            description="Request timeout in seconds",
            ge=5.0,
            le=120.0
        )
        output_format: str = Field(
            default="standard",
            description="Format of the output (standard, article, technical)"
        )

        @field_validator("search_type")
        def validate_search_type(cls, v: str) -> str:
            if v not in ["text", "news"]:
                raise ValueError("search_type must be one of: text, news")
            return v
            
        @field_validator("safesearch")
        def validate_safesearch(cls, v: str) -> str:
            if v not in ["on", "moderate", "off"]:
                raise ValueError("safesearch must be one of: on, moderate, off")
            return v
            
        @field_validator("output_format")
        def validate_output_format(cls, v: str) -> str:
            if v not in ["standard", "article", "technical"]:
                raise ValueError("output_format must be one of: standard, article, technical")
            return v

    name: str = "duckduckgo_search_llm"
    description: str = (
        "Advanced search tool with LLM-powered analysis. "
        "Retrieves search results from DuckDuckGo, optionally scrapes the content, "
        "and uses AI to answer specific questions about the results. "
        "Can perform tasks like information synthesis, fact verification, and content analysis. "
        "Supports different output formats including detailed articles and technical reports."
    )
    arguments: list = [
        ToolArgument(
            name="query",
            arg_type="string",
            description="The search query to execute",
            required=True,
            example="artificial intelligence developments"
        ),
        ToolArgument(
            name="question",
            arg_type="string",
            description="Question to ask about the search results (e.g., 'What are the key facts?')",
            required=True,
            example="What are the key facts about this topic?"
        ),
        ToolArgument(
            name="max_results",
            arg_type="int",
            description="Maximum number of search results to analyze (1-10)",
            required=False,
            default="5"
        ),
        ToolArgument(
            name="search_type",
            arg_type="string",
            description="Type of search to perform (text, news)",
            required=False,
            default="text"
        ),
        ToolArgument(
            name="region",
            arg_type="string",
            description="Region for search results (e.g., 'wt-wt', 'us-en')",
            required=False,
            default="wt-wt"
        ),
        ToolArgument(
            name="safesearch",
            arg_type="string",
            description="Safesearch level ('on', 'moderate', 'off')",
            required=False,
            default="moderate"
        ),
        ToolArgument(
            name="scrape_content",
            arg_type="boolean",
            description="Whether to scrape the actual content of search results",
            required=False,
            default="true"
        ),
        ToolArgument(
            name="timeout",
            arg_type="float",
            description="Request timeout in seconds (5-120)",
            required=False,
            default="60.0"
        ),
        ToolArgument(
            name="output_format",
            arg_type="string",
            description="Format of the output (standard, article, technical)",
            required=False,
            default="standard"
        )
    ]

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        **kwargs
    ):
        """Initialize the DuckDuckGoSearchLLMTool.
        
        Args:
            model_name: Name of the LLM model to use for analysis
            **kwargs: Additional arguments passed to Tool
        """
        super().__init__(**kwargs)
        self.search_tool = DuckDuckGoSearchTool()
        self.scraper = WebScraperTool()
        self.llm = LLMTool(model_name=model_name)

    async def _fetch_search_results(
        self,
        query: str,
        max_results: int,
        search_type: str,
        region: str,
        safesearch: str,
        timelimit: str = None
    ) -> List[Dict[str, Any]]:
        """Fetch search results using DuckDuckGoSearchTool."""
        try:
            # Since DuckDuckGoSearchTool.execute is synchronous, run it in a thread pool
            loop = asyncio.get_running_loop()
            
            # Run the synchronous execute method in a thread pool to avoid blocking the event loop
            result = await loop.run_in_executor(
                None,
                lambda: self.search_tool.execute(
                    query=query,
                    max_results=max_results,
                    search_type=search_type,
                    region=region,
                    safesearch=safesearch,
                    timelimit=timelimit
                )
            )
            
            # Parse the JSON result
            results = json.loads(result)
            return results
            
        except Exception as e:
            logger.error(f"Error fetching search results: {e}")
            raise ValueError(f"Error fetching search results: {e}")

    async def _scrape_result_content(
        self, 
        url: str, 
        timeout: float,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Scrape the content of a search result using WebScraperTool."""
        try:
            # Use the web scraper to get the full content
            content = await self.scraper.async_execute(
                website=url,
                timeout=str(timeout),
                max_retries=str(max_retries)
            )
            return content
        except Exception as e:
            logger.warning(f"Error scraping content from {url}: {e}")
            # Return a minimal content structure if scraping fails
            return {
                "title": "Failed to retrieve content",
                "text": f"Could not scrape content from {url}. Error: {str(e)}",
                "links": []
            }

    async def _scrape_multiple_results(
        self, 
        results: List[Dict[str, Any]], 
        timeout: float,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """Scrape content from multiple search results concurrently."""
        tasks = []
        for result in results:
            url = result.get("href") or result.get("url")
            if url:
                tasks.append(self._scrape_result_content(url, timeout, max_retries))
            
        # Execute all scraping tasks concurrently
        scraped_contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine result metadata with scraped content
        enriched_results = []
        for i, (result, content) in enumerate(zip(results, scraped_contents)):
            result_copy = result.copy()
            if isinstance(content, Exception):
                logger.warning(f"Error scraping result {i+1}: {content}")
                # Add minimal content for failed scrapes
                result_copy["scraped_content"] = {
                    "title": result.get("title", "Unknown title"),
                    "text": f"Failed to scrape content. Error: {str(content)}",
                    "links": []
                }
            else:
                result_copy["scraped_content"] = content
            
            enriched_results.append(result_copy)
            
        return enriched_results

    async def _analyze_results_with_llm(
        self, 
        results: List[Dict[str, Any]], 
        query: str, 
        question: str,
        scraped: bool = False,
        output_format: str = "standard"
    ) -> str:
        """Analyze search results using LLM to answer the specific question."""
        try:
            # Create a structured context from the results
            context = self._format_results_for_llm(results, query, scraped)
            
            # Set temperature based on output format
            temperature = self._get_temperature_for_format(output_format)
            
            # Create a system prompt for the LLM based on output format
            system_prompt = self._get_system_prompt_for_format(output_format)
            
            # Create a prompt with the context and question
            prompt = f"""
SEARCH RESULTS FOR "{query}":
{context}

USER QUESTION: {question}

{self._get_instruction_for_format(output_format)}
"""
            
            # Get LLM response
            response = await self.llm.async_execute(
                system_prompt=system_prompt,
                prompt=prompt,
                temperature=temperature
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing results with LLM: {e}")
            raise ValueError(f"Error analyzing results with LLM: {e}")

    def _format_results_for_llm(self, results: List[Dict[str, Any]], query: str, scraped: bool) -> str:
        """Format the search results into a structured text for the LLM."""
        formatted_text = [f"SEARCH QUERY: {query}", ""]
        
        for i, result in enumerate(results, 1):
            # Add result header with metadata
            formatted_text.append(f"RESULT {i}:")
            formatted_text.append(f"TITLE: {result.get('title', 'No title')}")
            
            # Add URL
            url = result.get('href') or result.get('url')
            if url:
                formatted_text.append(f"URL: {url}")
            
            # Add source/domain if available
            source = result.get('source') or self._extract_domain(url)
            if source:
                formatted_text.append(f"SOURCE: {source}")
            
            # Add date if available
            if 'published' in result:
                formatted_text.append(f"DATE: {result.get('published', 'No date')}")
            
            formatted_text.append("")
            
            # Add description/snippet if available
            if 'body' in result or 'snippet' in result or 'description' in result:
                desc = result.get('body') or result.get('snippet') or result.get('description', '')
                formatted_text.append("DESCRIPTION:")
                formatted_text.append(desc)
                formatted_text.append("")
            
            # Add scraped content if available
            if scraped and "scraped_content" in result:
                scraped_content = result.get("scraped_content", {})
                
                # Add main text content from scraping
                if scraped_content.get("text"):
                    text = scraped_content.get("text", "")
                    # Truncate to ~1500 characters to avoid token limits
                    if len(text) > 1500:
                        text = text[:1500] + "..."
                    
                    formatted_text.append("CONTENT:")
                    formatted_text.append(text)
                    formatted_text.append("")
            
            # Add separator between results
            formatted_text.append("-" * 50)
            formatted_text.append("")
        
        return "\n".join(formatted_text)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if not url:
            return ""
        
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return domain
        except:
            # Simple fallback extraction
            if "://" in url:
                domain = url.split("://")[1].split("/")[0]
                return domain
            return ""
            
    def _get_temperature_for_format(self, output_format: str) -> str:
        """Get the appropriate temperature setting based on output format."""
        if output_format == "article":
            return "0.4"  # Slightly higher temperature for more creative article writing
        elif output_format == "technical":
            return "0.1"  # Lower temperature for more detailed, factual analysis
        else:  # standard
            return "0.3"  # Balanced temperature
    
    def _get_system_prompt_for_format(self, output_format: str) -> str:
        """Get the appropriate system prompt based on output format."""
        if output_format == "article":
            return (
                "You are an expert journalist and content creator specializing in writing comprehensive, "
                "well-structured articles. Your task is to analyze search results and create a "
                "professional, engaging, and informative piece that answers the user's question. "
                "Base your analysis ONLY on the provided search results. "
                "If the answer cannot be determined from the results, say so clearly. "
                "Provide specific references to sources where possible. "
                "Your writing should be clear, engaging, and follow journalistic best practices."
            )
        elif output_format == "technical":
            return (
                "You are an expert technical analyst specializing in creating detailed, technically accurate "
                "reports. Your task is to analyze search results and provide a comprehensive technical analysis "
                "that answers the user's question. "
                "Base your analysis ONLY on the provided search results. "
                "If the answer cannot be determined from the results, say so clearly. "
                "Provide specific references to sources where possible. "
                "Your report should be detailed, precise, and technically sound."
            )
        else:  # standard
            return (
                "You are an expert research analyst and fact-checker. Your task is to analyze search results "
                "and provide accurate, detailed answers to questions about them. "
                "Base your analysis ONLY on the provided search results. "
                "If the answer cannot be determined from the results, say so clearly. "
                "Provide specific references to sources where possible. "
                "Compare and contrast information across sources when relevant."
            )
    
    def _get_instruction_for_format(self, output_format: str) -> str:
        """Get the appropriate instruction based on output format."""
        if output_format == "article":
            return (
                "Please write a comprehensive, well-structured article answering the user's question based on the search results provided. "
                "Your article should include:\n"
                "1. An engaging headline\n"
                "2. A compelling introduction that hooks the reader\n"
                "3. Well-organized body sections with appropriate subheadings\n"
                "4. Detailed explanations and analysis of key points\n"
                "5. Proper citations and references to sources\n"
                "6. A conclusion that summarizes the main points and provides perspective\n\n"
                "Make the article informative, engaging, and professionally written. "
                "Use a journalistic style with clear, accessible language. "
                "Include specific details, examples, and quotes from the sources where relevant."
            )
        elif output_format == "technical":
            return (
                "Please write a detailed technical report answering the user's question based on the search results provided. "
                "Your report should include:\n"
                "1. An executive summary of key findings\n"
                "2. Detailed technical analysis with proper terminology\n"
                "3. Comprehensive examination of methodologies, technologies, or approaches\n"
                "4. Critical evaluation of strengths, limitations, and implications\n"
                "5. Precise citations and references to sources\n"
                "6. Technical recommendations or future directions\n\n"
                "Use appropriate technical language and structure. "
                "Ensure all claims are supported by evidence from the sources. "
                "Include relevant technical details, specifications, and methodologies."
            )
        else:  # standard
            return (
                "Please provide a comprehensive answer to the user's question based solely on the search results provided. "
                "If the results don't contain information to answer the question, state that clearly. "
                "When appropriate, compare how different sources report on the same topic. "
                "Include key details while remaining concise and focused. "
                "Reference specific sources where relevant."
            )

    async def async_execute(
        self,
        query: str,
        question: str,
        max_results: str = "5",
        search_type: str = "text",
        region: str = "wt-wt",
        safesearch: str = "moderate",
        scrape_content: str = "true",
        timeout: str = "60.0",
        output_format: str = "standard"
    ) -> Dict[str, Any]:
        """
        Execute the search and LLM analysis task asynchronously.
        
        Args:
            query: Search query
            question: Question to ask about the search results
            max_results: Maximum number of search results to analyze
            search_type: Type of search to perform (text, news)
            region: Region for search results
            safesearch: Safesearch level
            scrape_content: Whether to scrape the actual content of search results
            timeout: Request timeout in seconds
            output_format: Format of the output (standard, article, technical)
            
        Returns:
            dict: Analysis results with search results and LLM answer
        """
        # Convert and validate parameters
        self.config = self.Config(
            query=query,
            question=question,
            max_results=int(max_results),
            search_type=search_type,
            region=region,
            safesearch=safesearch,
            timeout=float(timeout),
            output_format=output_format
        )
        
        # Convert scrape_content to boolean
        should_scrape = scrape_content.lower() == "true"
        
        try:
            # Step 1: Fetch search results
            logger.info(f"Fetching search results for query: {query}")
            results = await self._fetch_search_results(
                query=self.config.query,
                max_results=self.config.max_results,
                search_type=self.config.search_type,
                region=self.config.region,
                safesearch=self.config.safesearch
            )
            
            # Step 2: Optionally scrape content from search results
            if should_scrape:
                logger.info(f"Scraping content from {len(results)} search results")
                enriched_results = await self._scrape_multiple_results(
                    results=results,
                    timeout=self.config.timeout,
                    max_retries=3
                )
            else:
                enriched_results = results
            
            # Step 3: Analyze results with LLM
            logger.info(f"Analyzing search results with LLM for question: {question}")
            answer = await self._analyze_results_with_llm(
                enriched_results, 
                self.config.query, 
                self.config.question,
                should_scrape,
                self.config.output_format
            )
            
            # Prepare result summaries for the response
            result_summaries = []
            for result in enriched_results:
                summary = {
                    "title": result.get("title", "No title"),
                    "url": result.get("href") or result.get("url", "No URL"),
                    "source": result.get("source") or self._extract_domain(result.get("href") or result.get("url", "")),
                }
                
                # Add description if available
                if "body" in result or "snippet" in result or "description" in result:
                    summary["description"] = result.get("body") or result.get("snippet") or result.get("description", "")
                
                # Add content summary if scraped
                if should_scrape and "scraped_content" in result:
                    content = result["scraped_content"]
                    summary["content_summary"] = {
                        "title": content.get("title"),
                        "text_length": len(content.get("text", "")),
                        "link_count": len(content.get("links", [])),
                    }
                
                result_summaries.append(summary)
            
            # Return results
            return {
                "query": self.config.query,
                "question": self.config.question,
                "answer": answer,
                "results": result_summaries,
                "result_count": len(enriched_results),
                "search_type": self.config.search_type,
                "region": self.config.region,
                "content_scraped": should_scrape,
                "output_format": self.config.output_format
            }
                
        except Exception as e:
            logger.error(f"Error in DuckDuckGoSearchLLMTool: {e}")
            raise ValueError(f"Error in DuckDuckGoSearchLLMTool: {e}")

    def execute(
        self,
        query: str,
        question: str,
        max_results: str = "5",
        search_type: str = "text",
        region: str = "wt-wt",
        safesearch: str = "moderate",
        scrape_content: str = "true",
        timeout: str = "60.0",
        output_format: str = "standard"
    ) -> Dict[str, Any]:
        """Synchronous wrapper for async_execute."""
        return asyncio.run(
            self.async_execute(
                query=query,
                question=question,
                max_results=max_results,
                search_type=search_type,
                region=region,
                safesearch=safesearch,
                scrape_content=scrape_content,
                timeout=timeout,
                output_format=output_format
            )
        )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize tool
        tool = DuckDuckGoSearchLLMTool()
        
        try:
            # Example: Article format with content scraping
            result = await tool.async_execute(
                query="quantum computing advancements 2025",
                question="What are the most significant recent advancements in quantum computing?",
                max_results="5",
                scrape_content="true",
                output_format="article"
            )
            print("\nQuery:", result["query"])
            print("Question:", result["question"])
            print("Output Format:", result["output_format"])
            print("Content Scraped:", result["content_scraped"])
            print("Result Count:", result["result_count"])
            print("Answer:", result["answer"])
            
            # Print results analyzed
            if result["results"]:
                print("\nResults analyzed:")
                for i, res in enumerate(result["results"], 1):
                    print(f"{i}. {res['title']}")
                    print(f"   Source: {res['source']}")
                    print(f"   URL: {res['url']}")
                    if "content_summary" in res:
                        print(f"   Content: {res['content_summary']['text_length']} chars, {res['content_summary']['link_count']} links")
                    print()
                
        except Exception as e:
            logger.error(f"Error in example: {e}")
            print(f"Error: {e}")
    
    asyncio.run(main())
