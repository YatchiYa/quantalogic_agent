"""Enhanced Linkup Tool that combines web search with content scraping and LLM analysis."""

import asyncio
import os
from typing import Any, Optional, List, Dict

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.linkup_tool import LinkupTool
from quantalogic.tools.website_search.web_scraper_tool import WebScraperTool
from quantalogic.tools.llm_tool import LLMTool


class LinkupEnhancedTool(Tool):
    """Advanced web search tool with content scraping and LLM-powered analysis capabilities."""

    class Config(BaseModel):
        query: str = Field(..., description="The search query to perform")
        question: str = Field(..., description="Question to ask about the search results")
        depth: str = Field(
            default="deep",
            description="Search depth (standard or deep)"
        )
        analysis_depth: str = Field(
            default="standard",
            description="Depth of LLM analysis (quick, standard, or deep)"
        )
        scrape_sources: bool = Field(
            default=True,
            description="Whether to scrape the content of the sources"
        )
        max_sources_to_scrape: int = Field(
            default=3,
            description="Maximum number of sources to scrape",
            ge=1,
            le=5
        )
        output_format: str = Field(
            default="standard",
            description="Format of the output (standard, article, technical)"
        )

        @field_validator("depth")
        def validate_depth(cls, v: str) -> str:
            if v not in ["standard", "deep"]:
                raise ValueError("depth must be one of: standard, deep")
            return v
            
        @field_validator("analysis_depth")
        def validate_analysis_depth(cls, v: str) -> str:
            if v not in ["quick", "standard", "deep"]:
                raise ValueError("analysis_depth must be one of: quick, standard, deep")
            return v
            
        @field_validator("output_format")
        def validate_output_format(cls, v: str) -> str:
            if v not in ["standard", "article", "technical"]:
                raise ValueError("output_format must be one of: standard, article, technical")
            return v

    name: str = "linkup_enhanced"
    description: str = (
        "Advanced web search tool with content scraping and LLM-powered analysis. "
        "Performs web searches using Linkup API, scrapes the actual content of the sources, "
        "and uses AI to answer specific questions about the results. "
        "Supports different levels of search depth and analysis depth for both quick answers and deep insights. "
        "Can generate responses in various formats including detailed articles."
    )
    arguments: list = [
        ToolArgument(
            name="query",
            arg_type="string",
            description="The search query to perform",
            required=True,
            example="Latest advancements in quantum computing"
        ),
        ToolArgument(
            name="question",
            arg_type="string",
            description="Question to ask about the search results (e.g., 'What are the key developments?')",
            required=True,
            example="What are the key developments in this field?"
        ),
        ToolArgument(
            name="depth",
            arg_type="string",
            description="Search depth (standard or deep)",
            required=False,
            default="deep"
        ),
        ToolArgument(
            name="analysis_depth",
            arg_type="string",
            description="Depth of LLM analysis (quick, standard, or deep)",
            required=False,
            default="standard"
        ),
        ToolArgument(
            name="scrape_sources",
            arg_type="boolean",
            description="Whether to scrape the content of the sources",
            required=False,
            default="true"
        ),
        ToolArgument(
            name="max_sources_to_scrape",
            arg_type="int",
            description="Maximum number of sources to scrape (1-5)",
            required=False,
            default="3"
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
        """Initialize the LinkupEnhancedTool.
        
        Args:
            model_name: Name of the LLM model to use for analysis
            **kwargs: Additional arguments passed to Tool
        """
        super().__init__(**kwargs)
        self.linkup_tool = LinkupTool()
        self.scraper = WebScraperTool()
        self.llm = LLMTool(model_name=model_name)

    async def _fetch_search_results(
        self,
        query: str,
        depth: str
    ) -> Any:
        """Fetch search results using LinkupTool."""
        try:
            # Since LinkupTool.execute is synchronous, run it in a thread pool
            loop = asyncio.get_running_loop()
            
            # Run the synchronous execute method in a thread pool to avoid blocking the event loop
            result = await loop.run_in_executor(
                None,
                lambda: self.linkup_tool.execute(
                    query=query,
                    depth=depth,
                    output_type="sourcedAnswer"  # Always use sourcedAnswer for LLM analysis
                )
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching search results: {e}")
            raise ValueError(f"Error fetching search results: {e}")

    async def _scrape_source_content(
        self, 
        url: str, 
        timeout: float = 60.0,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Scrape the content of a source using WebScraperTool."""
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

    async def _scrape_multiple_sources(
        self, 
        sources: List[Any], 
        max_sources: int = 3,
        timeout: float = 60.0,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """Scrape content from multiple sources concurrently."""
        # Limit the number of sources to scrape
        sources_to_scrape = sources[:max_sources]
        
        tasks = []
        for source in sources_to_scrape:
            if hasattr(source, "url") and source.url:
                tasks.append(self._scrape_source_content(source.url, timeout, max_retries))
            
        # Execute all scraping tasks concurrently
        scraped_contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine source metadata with scraped content
        enriched_sources = []
        for i, (source, content) in enumerate(zip(sources_to_scrape, scraped_contents)):
            source_dict = {
                "title": source.title if hasattr(source, "title") else "No title",
                "url": source.url if hasattr(source, "url") else "No URL",
            }
            
            if isinstance(content, Exception):
                logger.warning(f"Error scraping source {i+1}: {content}")
                # Add minimal content for failed scrapes
                source_dict["scraped_content"] = {
                    "title": source_dict["title"],
                    "text": f"Failed to scrape content. Error: {str(content)}",
                    "links": []
                }
            else:
                source_dict["scraped_content"] = content
            
            # Add original content from the source
            if hasattr(source, "content") and source.content:
                source_dict["original_content"] = source.content
            
            enriched_sources.append(source_dict)
            
        return enriched_sources

    async def _analyze_results_with_llm(
        self, 
        results: Any, 
        enriched_sources: List[Dict[str, Any]],
        query: str, 
        question: str,
        analysis_depth: str,
        output_format: str
    ) -> str:
        """Analyze search results using LLM to answer the specific question."""
        try:
            # Create a structured context from the results and enriched sources
            context = self._format_results_for_llm(results, enriched_sources, query)
            
            # Set temperature based on analysis depth
            temperature = self._get_temperature_for_depth(analysis_depth)
            
            # Create a system prompt for the LLM based on analysis depth and output format
            system_prompt = self._get_system_prompt_for_depth_and_format(analysis_depth, output_format)
            
            # Create a prompt with the context and question
            prompt = f"""
SEARCH QUERY: "{query}"

SEARCH RESULTS AND SCRAPED CONTENT:
{context}

USER QUESTION: {question}

{self._get_instruction_for_depth_and_format(analysis_depth, output_format)}
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

    def _format_results_for_llm(
        self, 
        results: Any, 
        enriched_sources: List[Dict[str, Any]],
        query: str
    ) -> str:
        """Format the search results and scraped content into a structured text for the LLM."""
        formatted_text = []
        
        # Add the answer if available
        if hasattr(results, "answer") and results.answer:
            formatted_text.append("LINKUP ANSWER:")
            formatted_text.append(results.answer)
            formatted_text.append("")
        
        # Add sources with scraped content
        if enriched_sources:
            formatted_text.append("SOURCES WITH SCRAPED CONTENT:")
            for i, source in enumerate(enriched_sources, 1):
                formatted_text.append(f"SOURCE {i}:")
                formatted_text.append(f"TITLE: {source.get('title', 'No title')}")
                formatted_text.append(f"URL: {source.get('url', 'No URL')}")
                formatted_text.append("")
                
                # Add scraped content if available
                if "scraped_content" in source:
                    scraped = source["scraped_content"]
                    
                    if "title" in scraped and scraped["title"]:
                        formatted_text.append(f"SCRAPED TITLE: {scraped['title']}")
                    
                    if "text" in scraped and scraped["text"]:
                        text = scraped["text"]
                        # Truncate long content
                        if len(text) > 2000:
                            text = text[:2000] + "..."
                        formatted_text.append("SCRAPED CONTENT:")
                        formatted_text.append(text)
                        formatted_text.append("")
                
                # Add original content if available and no scraped content
                elif "original_content" in source:
                    content = source["original_content"]
                    # Truncate long content
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    formatted_text.append("ORIGINAL CONTENT:")
                    formatted_text.append(content)
                    formatted_text.append("")
                
                formatted_text.append("-" * 50)
                formatted_text.append("")
        
        # Add remaining sources if any
        elif hasattr(results, "sources") and results.sources:
            formatted_text.append("SOURCES:")
            for i, source in enumerate(results.sources, 1):
                formatted_text.append(f"SOURCE {i}:")
                
                if hasattr(source, "title") and source.title:
                    formatted_text.append(f"TITLE: {source.title}")
                
                if hasattr(source, "url") and source.url:
                    formatted_text.append(f"URL: {source.url}")
                
                if hasattr(source, "content") and source.content:
                    content = source.content
                    # Truncate long content
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    formatted_text.append(f"CONTENT: {content}")
                
                formatted_text.append("")
        
        return "\n".join(formatted_text)
    
    def _get_temperature_for_depth(self, analysis_depth: str) -> str:
        """Get the appropriate temperature setting based on analysis depth."""
        if analysis_depth == "quick":
            return "0.7"  # Higher temperature for more concise, creative responses
        elif analysis_depth == "deep":
            return "0.1"  # Lower temperature for more detailed, factual analysis
        else:  # standard
            return "0.3"  # Balanced temperature
    
    def _get_system_prompt_for_depth_and_format(self, analysis_depth: str, output_format: str) -> str:
        """Get the appropriate system prompt based on analysis depth and output format."""
        base_prompt = "You are an expert research analyst and information synthesizer. "
        
        if output_format == "article":
            base_prompt = "You are an expert journalist and content creator specializing in writing comprehensive, well-structured articles. "
        elif output_format == "technical":
            base_prompt = "You are an expert technical writer specializing in creating detailed, technically accurate content with proper citations. "
        
        if analysis_depth == "quick":
            return base_prompt + (
                "Provide concise, to-the-point answers that capture the essential information. "
                "Focus on the most important facts and insights. "
                "Be brief but accurate."
            )
        elif analysis_depth == "deep":
            return base_prompt + (
                "Provide comprehensive, detailed analysis with nuanced insights. "
                "Explore multiple perspectives and consider implications. "
                "Include specific references to sources and compare information across sources. "
                "Organize your response with clear structure and depth."
            )
        else:  # standard
            return base_prompt + (
                "Provide a balanced analysis that covers the key points while remaining concise. "
                "Include important details and reference sources where relevant. "
                "Focus on accuracy and clarity."
            )
    
    def _get_instruction_for_depth_and_format(self, analysis_depth: str, output_format: str) -> str:
        """Get the appropriate instruction based on analysis depth and output format."""
        if output_format == "article":
            return (
                "Please write a comprehensive, well-structured article answering the user's question based on the search results and scraped content. "
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
                "Please write a detailed technical analysis answering the user's question based on the search results and scraped content. "
                "Your response should include:\n"
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
        elif analysis_depth == "quick":
            return (
                "Please provide a brief, concise answer to the user's question based on the search results and scraped content. "
                "Keep your response under 100 words and focus only on the most essential information."
            )
        elif analysis_depth == "deep":
            return (
                "Please provide a comprehensive, detailed answer to the user's question based on the search results and scraped content. "
                "Include nuanced analysis, compare information across sources, and organize your response with clear structure. "
                "Consider different perspectives and implications where relevant."
            )
        else:  # standard
            return (
                "Please provide a balanced answer to the user's question based on the search results and scraped content. "
                "Include key details while remaining concise and focused. "
                "Reference specific sources where relevant."
            )

    async def async_execute(
        self,
        query: str,
        question: str,
        depth: str = "deep",
        analysis_depth: str = "standard",
        scrape_sources: str = "true",
        max_sources_to_scrape: str = "3",
        output_format: str = "standard"
    ) -> dict:
        """
        Execute the search, scraping, and LLM analysis task asynchronously.
        
        Args:
            query: Search query
            question: Question to ask about the search results
            depth: Search depth (standard or deep)
            analysis_depth: Depth of LLM analysis (quick, standard, or deep)
            scrape_sources: Whether to scrape the content of the sources
            max_sources_to_scrape: Maximum number of sources to scrape
            output_format: Format of the output (standard, article, technical)
            
        Returns:
            dict: Analysis results with search results, scraped content, and LLM answer
        """
        # Convert and validate parameters
        should_scrape = scrape_sources.lower() == "true"
        max_sources = int(max_sources_to_scrape)
        
        self.config = self.Config(
            query=query,
            question=question,
            depth=depth,
            analysis_depth=analysis_depth,
            scrape_sources=should_scrape,
            max_sources_to_scrape=max_sources,
            output_format=output_format
        )
        
        try:
            # Step 1: Fetch search results
            logger.info(f"Fetching search results for query: {query}")
            results = await self._fetch_search_results(
                query=self.config.query,
                depth=self.config.depth
            )
            
            # Step 2: Scrape source content if requested
            enriched_sources = []
            if should_scrape and hasattr(results, "sources") and results.sources:
                logger.info(f"Scraping content from up to {max_sources} sources")
                enriched_sources = await self._scrape_multiple_sources(
                    sources=results.sources,
                    max_sources=max_sources,
                    timeout=60.0,
                    max_retries=3
                )
            
            # Step 3: Analyze results with LLM
            logger.info(f"Analyzing search results and scraped content with LLM for question: {question}")
            answer = await self._analyze_results_with_llm(
                results, 
                enriched_sources,
                self.config.query, 
                self.config.question,
                self.config.analysis_depth,
                self.config.output_format
            )
            
            # Prepare source summaries for the response
            source_summaries = []
            for source in enriched_sources:
                summary = {
                    "title": source.get("title", "No title"),
                    "url": source.get("url", "No URL"),
                }
                
                # Add content summary if scraped
                if "scraped_content" in source:
                    content = source["scraped_content"]
                    summary["content_summary"] = {
                        "title": content.get("title"),
                        "text_length": len(content.get("text", "")),
                        "link_count": len(content.get("links", [])),
                    }
                
                source_summaries.append(summary)
            
            # Return results
            return {
                "query": self.config.query,
                "question": self.config.question,
                "answer": answer,
                "linkup_answer": results.answer if hasattr(results, "answer") else "",
                "sources": source_summaries,
                "sources_count": len(source_summaries) if source_summaries else (
                    len(results.sources) if hasattr(results, "sources") else 0
                ),
                "sources_scraped": should_scrape,
                "depth": self.config.depth,
                "analysis_depth": self.config.analysis_depth,
                "output_format": self.config.output_format
            }
                
        except Exception as e:
            logger.error(f"Error in LinkupEnhancedTool: {e}")
            raise ValueError(f"Error in LinkupEnhancedTool: {e}")

    def execute(
        self,
        query: str,
        question: str,
        depth: str = "deep",
        analysis_depth: str = "standard",
        scrape_sources: str = "true",
        max_sources_to_scrape: str = "3",
        output_format: str = "standard"
    ) -> dict:
        """Synchronous wrapper for async_execute."""
        return asyncio.run(
            self.async_execute(
                query=query,
                question=question,
                depth=depth,
                analysis_depth=analysis_depth,
                scrape_sources=scrape_sources,
                max_sources_to_scrape=max_sources_to_scrape,
                output_format=output_format
            )
        )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize tool
        tool = LinkupEnhancedTool()
        
        try:
            # Example: Analysis with source scraping and article format
            result = await tool.async_execute(
                query="Latest advancements in quantum computing",
                question="What are the most significant recent breakthroughs?",
                analysis_depth="standard",
                scrape_sources="true",
                max_sources_to_scrape="5",
                output_format="article"  # Generate a well-structured article
            )
            print("\nQuery:", result["query"])
            print("Question:", result["question"])
            print("Analysis Depth:", result["analysis_depth"])
            print("Output Format:", result["output_format"])
            print("Sources Scraped:", result["sources_scraped"])
            print("Sources Count:", result["sources_count"])
            print("Answer:", result["answer"])
            
            # Print sources analyzed
            if result["sources"]:
                print("\nSources analyzed:")
                for i, source in enumerate(result["sources"], 1):
                    print(f"{i}. {source['title']}")
                    print(f"   URL: {source['url']}")
                    if "content_summary" in source:
                        print(f"   Content: {source['content_summary']['text_length']} chars, {source['content_summary']['link_count']} links")
                    print()
                
        except Exception as e:
            logger.error(f"Error in example: {e}")
            print(f"Error: {e}")
    
    asyncio.run(main())
