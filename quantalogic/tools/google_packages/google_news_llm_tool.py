"""Google News LLM Tool for retrieving and analyzing news articles with AI-powered insights."""

import asyncio
import json
from typing import List, Dict, Any, Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.google_packages.google_news_tool import GoogleNewsTool
from quantalogic.tools.llm_tool import LLMTool
from quantalogic.tools.website_search.web_scraper_tool import WebScraperTool


class GoogleNewsLLMTool(Tool):
    """Advanced news retrieval and analysis tool with LLM-powered insights."""

    class Config(BaseModel):
        query: str = Field(..., description="The news search query")
        question: str = Field(..., description="Question to ask about the news articles")
        language: str = Field(
            default="en",
            description="Language code (e.g., 'en' for English)"
        )
        period: str = Field(
            default="1m",
            description="Time period (1h, 1d, 7d, 1m)"
        )
        max_results: int = Field(
            default=10,
            description="Maximum number of results (1-30)",
            ge=1,
            le=30
        )
        country: str = Field(
            default="US",
            description="Country code for news sources"
        )
        scrape_articles: bool = Field(
            default=True,
            description="Whether to scrape the full content of articles"
        )
        max_articles_to_scrape: int = Field(
            default=3,
            description="Maximum number of articles to scrape",
            ge=1,
            le=5
        )
        output_format: str = Field(
            default="standard",
            description="Format of the output (standard, article, technical)"
        )

        @field_validator("output_format")
        def validate_output_format(cls, v: str) -> str:
            if v not in ["standard", "article", "technical"]:
                raise ValueError("output_format must be one of: standard, article, technical")
            return v

    name: str = "google_news_llm"
    description: str = (
        "Advanced news retrieval and analysis tool with LLM-powered insights. "
        "Retrieves news articles from Google News, optionally scrapes the full content, "
        "and uses AI to answer specific questions about them. "
        "Can perform tasks like trend analysis, content summarization, and detailed analysis. "
        "Supports different output formats including detailed articles and technical reports."
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
            description="Question to ask about the news articles (e.g., 'What are the key trends?')",
            required=True,
            example="What are the key trends in this topic?"
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
            description="Maximum number of results (1-30)",
            required=False,
            default="10"
        ),
        ToolArgument(
            name="country",
            arg_type="string",
            description="Country code for news sources",
            required=False,
            default="US"
        ),
        ToolArgument(
            name="scrape_articles",
            arg_type="boolean",
            description="Whether to scrape the full content of articles",
            required=False,
            default="true"
        ),
        ToolArgument(
            name="max_articles_to_scrape",
            arg_type="int",
            description="Maximum number of articles to scrape (1-5)",
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
        """Initialize the GoogleNewsLLMTool.
        
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
                    analyze=True  # Always analyze to get full content
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
        timeout: float = 60.0,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Scrape the full content of an article using WebScraperTool."""
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

    async def _scrape_multiple_articles(
        self, 
        articles: List[Dict[str, Any]], 
        max_articles: int = 3,
        timeout: float = 60.0,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """Scrape content from multiple articles concurrently."""
        # Limit the number of articles to scrape
        articles_to_scrape = articles[:max_articles]
        
        tasks = []
        for article in articles_to_scrape:
            # Check for both 'link' and 'url' keys to be more robust
            article_url = article.get('link') or article.get('url')
            if article_url:
                tasks.append(self._scrape_article_content(article_url, timeout, max_retries))
            
        # Execute all scraping tasks concurrently
        scraped_contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine article metadata with scraped content
        enriched_articles = []
        for i, (article, content) in enumerate(zip(articles_to_scrape, scraped_contents)):
            article_copy = article.copy()
            
            if isinstance(content, Exception):
                logger.warning(f"Error scraping article {i+1}: {content}")
                # Add minimal content for failed scrapes
                article_copy["scraped_content"] = {
                    "title": article.get('title', 'No title'),
                    "text": f"Failed to scrape content. Error: {str(content)}",
                    "links": []
                }
            else:
                article_copy["scraped_content"] = content
            
            enriched_articles.append(article_copy)
            
        return enriched_articles

    async def _analyze_articles_with_llm(
        self, 
        articles: List[Dict[str, Any]], 
        enriched_articles: List[Dict[str, Any]],
        query: str, 
        question: str,
        output_format: str
    ) -> str:
        """Analyze news articles using LLM to answer the specific question."""
        try:
            # Create a structured context from the articles and enriched articles
            context = self._format_articles_for_llm(articles, enriched_articles, query)
            
            # Set temperature based on output format
            temperature = self._get_temperature_for_format(output_format)
            
            # Create a system prompt for the LLM based on output format
            system_prompt = self._get_system_prompt_for_format(output_format)
            
            # Create a prompt with the context and question
            prompt = f"""
NEWS ARTICLES ABOUT "{query}":
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
            logger.error(f"Error analyzing articles with LLM: {e}")
            raise ValueError(f"Error analyzing articles with LLM: {e}")

    def _format_articles_for_llm(
        self, 
        articles: List[Dict[str, Any]], 
        enriched_articles: List[Dict[str, Any]],
        query: str
    ) -> str:
        """Format the news articles into a structured text for the LLM."""
        formatted_text = [f"SEARCH QUERY: {query}", ""]
        
        # First add the enriched articles with scraped content
        if enriched_articles:
            formatted_text.append("ARTICLES WITH FULL CONTENT:")
            for i, article in enumerate(enriched_articles, 1):
                # Add article header with metadata
                formatted_text.append(f"ARTICLE {i} (FULL CONTENT):")
                formatted_text.append(f"TITLE: {article.get('title', 'No title')}")
                formatted_text.append(f"SOURCE: {article.get('source', 'Unknown source')}")
                formatted_text.append(f"DATE: {article.get('date', 'No date')}")
                # Check for both 'link' and 'url' keys
                article_url = article.get('link') or article.get('url') or 'No URL'
                formatted_text.append(f"URL: {article_url}")
                formatted_text.append("")
                
                # Add scraped content if available
                if "scraped_content" in article:
                    scraped = article["scraped_content"]
                    
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
                
                # Add summary if available
                if article.get('summary') and len(article.get('summary', '')) > 50:
                    formatted_text.append("SUMMARY:")
                    formatted_text.append(article.get('summary', ''))
                    formatted_text.append("")
                
                # Add keywords if available
                if article.get('keywords'):
                    formatted_text.append("KEYWORDS: " + ", ".join(article.get('keywords', [])))
                    formatted_text.append("")
                
                # Add sentiment if available
                if article.get('sentiment'):
                    sentiment = article.get('sentiment', {})
                    sentiment_str = f"SENTIMENT: Positive: {sentiment.get('pos', 0):.2f}, Negative: {sentiment.get('neg', 0):.2f}, Neutral: {sentiment.get('neu', 0):.2f}"
                    formatted_text.append(sentiment_str)
                    formatted_text.append("")
                
                # Add separator between articles
                formatted_text.append("-" * 40)
                formatted_text.append("")
        
        # Add remaining articles (without scraped content)
        # Skip articles that were already included with scraped content
        enriched_urls = []
        for article in enriched_articles:
            article_url = article.get('link') or article.get('url')
            if article_url:
                enriched_urls.append(article_url)
                
        remaining_articles = []
        for article in articles:
            article_url = article.get('link') or article.get('url')
            if article_url and article_url not in enriched_urls:
                remaining_articles.append(article)
        
        if remaining_articles:
            formatted_text.append("ADDITIONAL ARTICLES (SUMMARY ONLY):")
            for i, article in enumerate(remaining_articles, len(enriched_articles) + 1):
                # Add article header with metadata
                formatted_text.append(f"ARTICLE {i}:")
                formatted_text.append(f"TITLE: {article.get('title', 'No title')}")
                formatted_text.append(f"SOURCE: {article.get('source', 'Unknown source')}")
                formatted_text.append(f"DATE: {article.get('date', 'No date')}")
                # Check for both 'link' and 'url' keys
                article_url = article.get('link') or article.get('url') or 'No URL'
                formatted_text.append(f"URL: {article_url}")
                formatted_text.append("")
                
                # Add summary if available, otherwise use the first part of full text
                if article.get('summary') and len(article.get('summary', '')) > 50:
                    formatted_text.append("SUMMARY:")
                    formatted_text.append(article.get('summary', ''))
                    formatted_text.append("")
                
                # Add full text if available (truncated to avoid token limits)
                full_text = article.get('full_text', '')
                if full_text:
                    # Truncate to ~1000 characters to avoid token limits
                    if len(full_text) > 1000:
                        full_text = full_text[:1000] + "..."
                    
                    formatted_text.append("CONTENT:")
                    formatted_text.append(full_text)
                    formatted_text.append("")
                
                # Add keywords if available
                if article.get('keywords'):
                    formatted_text.append("KEYWORDS: " + ", ".join(article.get('keywords', [])))
                    formatted_text.append("")
                
                # Add sentiment if available
                if article.get('sentiment'):
                    sentiment = article.get('sentiment', {})
                    sentiment_str = f"SENTIMENT: Positive: {sentiment.get('pos', 0):.2f}, Negative: {sentiment.get('neg', 0):.2f}, Neutral: {sentiment.get('neu', 0):.2f}"
                    formatted_text.append(sentiment_str)
                    formatted_text.append("")
                
                # Add separator between articles
                formatted_text.append("-" * 40)
                formatted_text.append("")
        
        return "\n".join(formatted_text)
    
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
                "well-structured news articles. Your task is to analyze news articles and create a "
                "professional, engaging, and informative piece that answers the user's question. "
                "Base your analysis ONLY on the provided news articles. "
                "If the answer cannot be determined from the articles, say so clearly. "
                "Provide specific references to sources where relevant. "
                "Your writing should be clear, engaging, and follow journalistic best practices."
            )
        elif output_format == "technical":
            return (
                "You are an expert technical analyst specializing in creating detailed, technically accurate "
                "reports. Your task is to analyze news articles and provide a comprehensive technical analysis "
                "that answers the user's question. "
                "Base your analysis ONLY on the provided news articles. "
                "If the answer cannot be determined from the articles, say so clearly. "
                "Provide specific references to sources where possible. "
                "Your report should be detailed, precise, and technically sound."
            )
        else:  # standard
            return (
                "You are an expert news analyst. Your task is to analyze news articles "
                "and provide accurate, detailed answers to questions about them. "
                "Base your analysis ONLY on the provided news articles. "
                "If the answer cannot be determined from the articles, say so clearly. "
                "Provide specific references to articles where possible."
            )
    
    def _get_instruction_for_format(self, output_format: str) -> str:
        """Get the appropriate instruction based on output format."""
        if output_format == "article":
            return (
                "Please write a comprehensive, well-structured news article answering the user's question based on the news articles provided. "
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
                "Please write a detailed technical report answering the user's question based on the news articles provided. "
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
                "Please provide a comprehensive answer to the user's question based solely on the news articles provided. "
                "If the articles don't contain information to answer the question, state that clearly. "
                "Include key details while remaining concise and focused. "
                "Reference specific sources where relevant."
            )

    async def async_execute(
        self,
        query: str,
        question: str,
        language: str = "en",
        period: str = "1m",
        max_results: str = "10",
        country: str = "US",
        scrape_articles: str = "true",
        max_articles_to_scrape: str = "3",
        output_format: str = "standard"
    ) -> Dict[str, Any]:
        """
        Execute the news retrieval and LLM analysis task asynchronously.
        
        Args:
            query: News search query
            question: Question to ask about the news articles
            language: Language code
            period: Time period for news
            max_results: Maximum number of results
            country: Country code for news sources
            scrape_articles: Whether to scrape the full content of articles
            max_articles_to_scrape: Maximum number of articles to scrape
            output_format: Format of the output (standard, article, technical)
            
        Returns:
            dict: Analysis results with news articles and LLM answer
        """
        # Convert and validate parameters
        should_scrape = scrape_articles.lower() == "true"
        max_articles = int(max_articles_to_scrape)
        
        self.config = self.Config(
            query=query,
            question=question,
            language=language,
            period=period,
            max_results=int(max_results),
            country=country,
            scrape_articles=should_scrape,
            max_articles_to_scrape=max_articles,
            output_format=output_format
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
            
            # Step 2: Scrape article content if requested
            enriched_articles = []
            if should_scrape and articles:
                logger.info(f"Scraping content from up to {max_articles} articles")
                enriched_articles = await self._scrape_multiple_articles(
                    articles=articles,
                    max_articles=max_articles,
                    timeout=60.0,
                    max_retries=3
                )
            
            # Step 3: Analyze articles with LLM
            logger.info(f"Analyzing articles with LLM for question: {question}")
            answer = await self._analyze_articles_with_llm(
                articles, 
                enriched_articles,
                self.config.query, 
                self.config.question,
                self.config.output_format
            )
            
            # Prepare article summaries for the response
            article_summaries = []
            for article in enriched_articles:
                summary = {
                    "title": article.get("title", "No title"),
                    "source": article.get("source", "Unknown source"),
                    "date": article.get("date", "No date"),
                    "link": article.get("link") or article.get("url") or "No URL",
                }
                
                # Add content summary if scraped
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
                "articles": article_summaries if article_summaries else articles[:5],
                "article_count": len(articles),
                "articles_scraped": should_scrape,
                "scraped_count": len(enriched_articles),
                "period": self.config.period,
                "language": self.config.language,
                "country": self.config.country,
                "output_format": self.config.output_format
            }
                
        except Exception as e:
            logger.error(f"Error in GoogleNewsLLMTool: {e}")
            raise ValueError(f"Error in GoogleNewsLLMTool: {e}")

    def execute(
        self,
        query: str,
        question: str,
        language: str = "en",
        period: str = "1m",
        max_results: str = "10",
        country: str = "US",
        scrape_articles: str = "true",
        max_articles_to_scrape: str = "3",
        output_format: str = "standard"
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
                scrape_articles=scrape_articles,
                max_articles_to_scrape=max_articles_to_scrape,
                output_format=output_format
            )
        )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize tool
        tool = GoogleNewsLLMTool()
        
        try:
            # Example: Article format with scraping
            result = await tool.async_execute(
                query="artificial intelligence",
                question="What are the main trends and developments in AI over the past month?",
                max_results="15",
                scrape_articles="true",
                max_articles_to_scrape="5",
                output_format="article"
            )
            print("\nQuery:", result["query"])
            print("Question:", result["question"])
            print("Output Format:", result["output_format"])
            print("Articles Scraped:", result["articles_scraped"])
            print("Article Count:", result["article_count"])
            print("Scraped Count:", result["scraped_count"])
            print("Answer:", result["answer"])
            
            # Print articles analyzed
            if result["articles"]:
                print("\nArticles analyzed:")
                for i, article in enumerate(result["articles"], 1):
                    print(f"{i}. {article['title']}")
                    print(f"   Source: {article['source']}")
                    print(f"   Date: {article['date']}")
                    print(f"   URL: {article['link']}")
                    if "content_summary" in article:
                        print(f"   Content: {article['content_summary']['text_length']} chars, {article['content_summary']['link_count']} links")
                    print()
                
        except Exception as e:
            logger.error(f"Error in example: {e}")
            print(f"Error: {e}")
    
    asyncio.run(main())
