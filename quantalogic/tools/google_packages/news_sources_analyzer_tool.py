"""News Sources Analyzer Tool for retrieving news articles and analyzing their content with LLM."""

import asyncio
import json
from typing import List, Dict, Any, Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.google_packages.google_news_tool import GoogleNewsTool
from quantalogic.tools.website_search.web_scraper_tool import WebScraperTool
from quantalogic.tools.llm_tool import LLMTool


class NewsSourcesAnalyzerTool(Tool):
    """Advanced tool for retrieving news articles and analyzing their content with LLM."""

    class Config(BaseModel):
        query: str = Field(..., description="The news search query")
        question: str = Field(..., description="Question to ask about the news sources content")
        language: str = Field(
            default="en",
            description="Language code (e.g., 'en' for English)"
        )
        period: str = Field(
            default="1m",
            description="Time period (1h, 1d, 7d, 1m)"
        )
        max_results: int = Field(
            default=5,
            description="Maximum number of news sources to analyze (1-10)",
            ge=1,
            le=10
        )
        country: str = Field(
            default="US",
            description="Country code for news sources"
        )
        timeout: float = Field(
            default=60.0,
            description="Request timeout in seconds",
            ge=5.0,
            le=120.0
        )

    name: str = "news_sources_analyzer"
    description: str = (
        "Advanced tool for retrieving news articles and analyzing their content with LLM. "
        "Searches for news articles using Google News, scrapes the actual article content, "
        "and uses AI to answer specific questions about the content. "
        "Can perform tasks like deep analysis of news sources, fact verification, and content comparison."
    )
    arguments: list = [
        ToolArgument(
            name="query",
            arg_type="string",
            description="The news search query",
            required=True,
            example="artificial intelligence developments"
        ),
        ToolArgument(
            name="question",
            arg_type="string",
            description="Question to ask about the news sources content (e.g., 'What are the key facts?')",
            required=True,
            example="What are the key facts reported across these sources?"
        ),
        ToolArgument(
            name="language",
            arg_type="string",
            description="Language code (e.g., 'en' for English)",
            required=False,
            default="en"
        ),
        ToolArgument(
            name="period",
            arg_type="string",
            description="Time period (1h, 1d, 7d, 1m)",
            required=False,
            default="1m"
        ),
        ToolArgument(
            name="max_results",
            arg_type="int",
            description="Maximum number of news sources to analyze (1-10)",
            required=False,
            default="5"
        ),
        ToolArgument(
            name="country",
            arg_type="string",
            description="Country code for news sources",
            required=False,
            default="US"
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
        """Initialize the NewsSourcesAnalyzerTool.
        
        Args:
            model_name: Name of the LLM model to use for analysis
            **kwargs: Additional arguments passed to Tool
        """
        super().__init__(**kwargs)
        self.news_tool = GoogleNewsTool()
        self.scraper = WebScraperTool()
        self.llm = LLMTool(model_name=model_name)

    async def _fetch_news_articles(
        self,
        query: str,
        language: str,
        period: str,
        max_results: int,
        country: str
    ) -> List[Dict[str, Any]]:
        """Fetch news articles using GoogleNewsTool."""
        try:
            # Since GoogleNewsTool.execute uses run_until_complete which conflicts with async context,
            # we need to run it in a separate thread to avoid event loop conflicts
            loop = asyncio.get_running_loop()
            
            # Run the synchronous execute method in a thread pool to avoid blocking the event loop
            result = await loop.run_in_executor(
                None,
                lambda: self.news_tool.execute(
                    query=query,
                    language=language,
                    period=period,
                    max_results=max_results,
                    country=country,
                    analyze=False  # We'll do our own analysis with the scraper
                )
            )
            
            # Parse the JSON result
            articles = json.loads(result)
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news articles: {e}")
            raise ValueError(f"Error fetching news articles: {e}")

    async def _scrape_article_content(
        self, 
        url: str, 
        timeout: float,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Scrape the content of a news article using WebScraperTool."""
        try:
            # Use the web scraper to get the full content
            content = await self.scraper.async_execute(
                website=url,
                timeout=str(timeout),
                max_retries=str(max_retries)
            )
            return content
        except Exception as e:
            logger.warning(f"Error scraping article content from {url}: {e}")
            # Return a minimal content structure if scraping fails
            return {
                "title": "Failed to retrieve content",
                "text": f"Could not scrape content from {url}. Error: {str(e)}",
                "links": []
            }

    async def _scrape_multiple_articles(
        self, 
        articles: List[Dict[str, Any]], 
        timeout: float,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """Scrape content from multiple articles concurrently."""
        tasks = []
        for article in articles:
            url = article.get("url")
            if url:
                tasks.append(self._scrape_article_content(url, timeout, max_retries))
            
        # Execute all scraping tasks concurrently
        scraped_contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine article metadata with scraped content
        enriched_articles = []
        for i, (article, content) in enumerate(zip(articles, scraped_contents)):
            if isinstance(content, Exception):
                logger.warning(f"Error scraping article {i+1}: {content}")
                # Add minimal content for failed scrapes
                article["scraped_content"] = {
                    "title": article.get("title", "Unknown title"),
                    "text": f"Failed to scrape content. Error: {str(content)}",
                    "links": []
                }
            else:
                article["scraped_content"] = content
            
            enriched_articles.append(article)
            
        return enriched_articles

    async def _analyze_articles_with_llm(
        self, 
        articles: List[Dict[str, Any]], 
        query: str, 
        question: str
    ) -> str:
        """Analyze news articles using LLM to answer the specific question."""
        try:
            # Create a structured context from the articles
            context = self._format_articles_for_llm(articles, query)
            
            # Create a system prompt for the LLM
            system_prompt = (
                "You are an expert news analyst and fact-checker. Your task is to analyze news articles "
                "and provide accurate, detailed answers to questions about their content. "
                "Base your analysis ONLY on the provided news sources. "
                "If the answer cannot be determined from the sources, say so clearly. "
                "Provide specific references to articles where possible. "
                "Compare and contrast information across sources when relevant."
            )
            
            # Create a prompt with the context and question
            prompt = f"""
NEWS ARTICLES ABOUT "{query}":
{context}

USER QUESTION: {question}

Please provide a comprehensive answer to the user's question based solely on the news sources provided above.
If the sources don't contain information to answer the question, state that clearly.
When appropriate, compare how different sources report on the same topic.
"""
            
            # Get LLM response
            response = await self.llm.async_execute(
                system_prompt=system_prompt,
                prompt=prompt,
                temperature="0.3"  # Lower temperature for more factual responses
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing articles with LLM: {e}")
            raise ValueError(f"Error analyzing articles with LLM: {e}")

    def _format_articles_for_llm(self, articles: List[Dict[str, Any]], query: str) -> str:
        """Format the news articles and their scraped content into a structured text for the LLM."""
        formatted_text = [f"SEARCH QUERY: {query}", ""]
        
        for i, article in enumerate(articles, 1):
            # Add article header with metadata
            formatted_text.append(f"SOURCE {i}: {article.get('source', 'Unknown source')}")
            formatted_text.append(f"TITLE: {article.get('title', 'No title')}")
            formatted_text.append(f"DATE: {article.get('date', 'No date')}")
            formatted_text.append(f"URL: {article.get('url', 'No URL')}")
            formatted_text.append("")
            
            # Add scraped content if available
            scraped_content = article.get("scraped_content", {})
            
            # Add main text content from scraping
            if scraped_content.get("text"):
                text = scraped_content.get("text", "")
                # Truncate to ~2000 characters to avoid token limits
                if len(text) > 2000:
                    text = text[:2000] + "..."
                
                formatted_text.append("CONTENT:")
                formatted_text.append(text)
                formatted_text.append("")
            
            # Add summary if available from the original article
            elif article.get('summary') and len(article.get('summary', '')) > 50:
                formatted_text.append("SUMMARY:")
                formatted_text.append(article.get('summary', ''))
                formatted_text.append("")
            
            # Add keywords if available
            if article.get('keywords'):
                formatted_text.append("KEYWORDS: " + ", ".join(article.get('keywords', [])))
                formatted_text.append("")
            
            # Add separator between articles
            formatted_text.append("-" * 50)
            formatted_text.append("")
        
        return "\n".join(formatted_text)

    async def async_execute(
        self,
        query: str,
        question: str,
        language: str = "en",
        period: str = "1m",
        max_results: str = "5",
        country: str = "US",
        timeout: str = "60.0"
    ) -> Dict[str, Any]:
        """
        Execute the news retrieval, scraping, and LLM analysis task asynchronously.
        
        Args:
            query: News search query
            question: Question to ask about the news sources content
            language: Language code
            period: Time period for news
            max_results: Maximum number of news sources to analyze
            country: Country code for news sources
            timeout: Request timeout in seconds
            
        Returns:
            dict: Analysis results with news articles and LLM answer
        """
        # Convert and validate parameters
        self.config = self.Config(
            query=query,
            question=question,
            language=language,
            period=period,
            max_results=int(max_results),
            country=country,
            timeout=float(timeout)
        )
        
        try:
            # Step 1: Fetch news articles
            logger.info(f"Fetching news articles for query: {query}")
            articles = await self._fetch_news_articles(
                query=self.config.query,
                language=self.config.language,
                period=self.config.period,
                max_results=self.config.max_results,
                country=self.config.country
            )
            
            # Step 2: Scrape content from each article
            logger.info(f"Scraping content from {len(articles)} news sources")
            enriched_articles = await self._scrape_multiple_articles(
                articles=articles,
                timeout=self.config.timeout,
                max_retries=3
            )
            
            # Step 3: Analyze articles with LLM
            logger.info(f"Analyzing news sources with LLM for question: {question}")
            answer = await self._analyze_articles_with_llm(
                enriched_articles, 
                self.config.query, 
                self.config.question
            )
            
            # Prepare article summaries for the response
            article_summaries = []
            for article in enriched_articles:
                summary = {
                    "title": article.get("title", "No title"),
                    "source": article.get("source", "Unknown source"),
                    "date": article.get("date", "No date"),
                    "url": article.get("url", "No URL"),
                }
                
                # Add content summary if available
                if "scraped_content" in article:
                    content = article["scraped_content"]
                    summary["content_summary"] = {
                        "title": content.get("title"),
                        "text_length": len(content.get("text", "")),
                        "link_count": len(content.get("links", [])),
                    }
                
                article_summaries.append(summary)
            
            # Return results
            return {
                "query": self.config.query,
                "question": self.config.question,
                "answer": answer,
                "articles": article_summaries,
                "article_count": len(enriched_articles),
                "period": self.config.period,
                "language": self.config.language,
                "country": self.config.country
            }
                
        except Exception as e:
            logger.error(f"Error in NewsSourcesAnalyzerTool: {e}")
            raise ValueError(f"Error in NewsSourcesAnalyzerTool: {e}")

    def execute(
        self,
        query: str,
        question: str,
        language: str = "en",
        period: str = "1m",
        max_results: str = "5",
        country: str = "US",
        timeout: str = "60.0"
    ) -> Dict[str, Any]:
        """Synchronous wrapper for async_execute."""
        return asyncio.run(
            self.async_execute(
                query=query,
                question=question,
                language=language,
                period=period,
                max_results=max_results,
                country=country,
                timeout=timeout
            )
        )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize tool
        tool = NewsSourcesAnalyzerTool()
        
        try:
            # Example: Deep analysis of news sources
            result = await tool.async_execute(
                query="artificial intelligence ethics",
                question="What are the main ethical concerns about AI discussed in these articles?",
                max_results="3"  # Limit to 3 sources for faster execution
            )
            print("\nQuery:", result["query"])
            print("Question:", result["question"])
            print("Answer:", result["answer"])
            print(f"Based on {result['article_count']} news sources")
            
            # Print sources analyzed
            print("\nSources analyzed:")
            for i, article in enumerate(result["articles"], 1):
                print(f"{i}. {article['title']} - {article['source']} ({article['date']})")
                print(f"   URL: {article['url']}")
                if "content_summary" in article:
                    print(f"   Content: {article['content_summary']['text_length']} chars, {article['content_summary']['link_count']} links")
                print()
                
        except Exception as e:
            logger.error(f"Error in example: {e}")
            print(f"Error: {e}")
    
    asyncio.run(main())
