"""Linkup LLM Tool for performing web searches with AI-powered analysis capabilities."""

import asyncio
import os
from typing import Any, Optional, Literal

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.linkup_tool import LinkupTool
from quantalogic.tools.llm_tool import LLMTool


class LinkupLLMTool(Tool):
    """Advanced web search tool with LLM-powered analysis capabilities."""

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

    name: str = "linkup_llm"
    description: str = (
        "Advanced web search tool with LLM-powered analysis. "
        "Performs web searches using Linkup API and uses AI to answer specific questions about the results. "
        "Supports different levels of search depth and analysis depth for both quick answers and deep insights."
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
        )
    ]

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        **kwargs
    ):
        """Initialize the LinkupLLMTool.
        
        Args:
            model_name: Name of the LLM model to use for analysis
            **kwargs: Additional arguments passed to Tool
        """
        super().__init__(**kwargs)
        self.linkup_tool = LinkupTool()
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

    async def _analyze_results_with_llm(
        self, 
        results: Any, 
        query: str, 
        question: str,
        analysis_depth: str
    ) -> str:
        """Analyze search results using LLM to answer the specific question."""
        try:
            # Create a structured context from the results
            context = self._format_results_for_llm(results, query)
            
            # Set temperature and token limits based on analysis depth
            temperature = self._get_temperature_for_depth(analysis_depth)
            
            # Create a system prompt for the LLM based on analysis depth
            system_prompt = self._get_system_prompt_for_depth(analysis_depth)
            
            # Create a prompt with the context and question
            prompt = f"""
SEARCH QUERY: "{query}"

SEARCH RESULTS:
{context}

USER QUESTION: {question}

{self._get_instruction_for_depth(analysis_depth)}
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

    def _format_results_for_llm(self, results: Any, query: str) -> str:
        """Format the search results into a structured text for the LLM."""
        formatted_text = []
        
        # Add the answer if available
        if hasattr(results, "answer") and results.answer:
            formatted_text.append("ANSWER:")
            formatted_text.append(results.answer)
            formatted_text.append("")
        
        # Add sources if available
        if hasattr(results, "sources") and results.sources:
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
    
    def _get_system_prompt_for_depth(self, analysis_depth: str) -> str:
        """Get the appropriate system prompt based on analysis depth."""
        base_prompt = "You are an expert research analyst and information synthesizer. "
        
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
    
    def _get_instruction_for_depth(self, analysis_depth: str) -> str:
        """Get the appropriate instruction based on analysis depth."""
        if analysis_depth == "quick":
            return (
                "Please provide a brief, concise answer to the user's question based on the search results. "
                "Keep your response under 100 words and focus only on the most essential information."
            )
        elif analysis_depth == "deep":
            return (
                "Please provide a comprehensive, detailed answer to the user's question based on the search results. "
                "Include nuanced analysis, compare information across sources, and organize your response with clear structure. "
                "Consider different perspectives and implications where relevant."
            )
        else:  # standard
            return (
                "Please provide a balanced answer to the user's question based on the search results. "
                "Include key details while remaining concise and focused. "
                "Reference specific sources where relevant."
            )

    async def async_execute(
        self,
        query: str,
        question: str,
        depth: str = "deep",
        analysis_depth: str = "standard"
    ) -> dict:
        """
        Execute the search and LLM analysis task asynchronously.
        
        Args:
            query: Search query
            question: Question to ask about the search results
            depth: Search depth (standard or deep)
            analysis_depth: Depth of LLM analysis (quick, standard, or deep)
            
        Returns:
            dict: Analysis results with search results and LLM answer
        """
        # Convert and validate parameters
        self.config = self.Config(
            query=query,
            question=question,
            depth=depth,
            analysis_depth=analysis_depth
        )
        
        try:
            # Step 1: Fetch search results
            logger.info(f"Fetching search results for query: {query}")
            results = await self._fetch_search_results(
                query=self.config.query,
                depth=self.config.depth
            )
            
            # Step 2: Analyze results with LLM
            logger.info(f"Analyzing search results with LLM for question: {question}")
            answer = await self._analyze_results_with_llm(
                results, 
                self.config.query, 
                self.config.question,
                self.config.analysis_depth
            )
            
            # Return results
            return {
                "query": self.config.query,
                "question": self.config.question,
                "answer": answer,
                "linkup_answer": results.answer if hasattr(results, "answer") else "",
                "sources_count": len(results.sources) if hasattr(results, "sources") else 0,
                "depth": self.config.depth,
                "analysis_depth": self.config.analysis_depth
            }
                
        except Exception as e:
            logger.error(f"Error in LinkupLLMTool: {e}")
            raise ValueError(f"Error in LinkupLLMTool: {e}")

    def execute(
        self,
        query: str,
        question: str,
        depth: str = "deep",
        analysis_depth: str = "standard"
    ) -> dict:
        """Synchronous wrapper for async_execute."""
        return asyncio.run(
            self.async_execute(
                query=query,
                question=question,
                depth=depth,
                analysis_depth=analysis_depth
            )
        )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize tool
        tool = LinkupLLMTool()
        
        try:
            # Example 1: Quick analysis
            result = await tool.async_execute(
                query="Latest advancements in quantum computing",
                question="What are the most significant recent breakthroughs?",
                analysis_depth="standard"
            )
            print("\nQuery:", result["query"])
            print("Question:", result["question"])
            print("Analysis Depth:", result["analysis_depth"])
            print("Answer:", result["answer"])
            
            # Example 2: Deep analysis
            result = await tool.async_execute(
                query="Climate change mitigation strategies",
                question="What are the most promising approaches to reducing carbon emissions?",
                analysis_depth="deep"
            )
            print("\nQuery:", result["query"])
            print("Question:", result["question"])
            print("Analysis Depth:", result["analysis_depth"])
            print("Answer:", result["answer"])
                
        except Exception as e:
            logger.error(f"Error in example: {e}")
            print(f"Error: {e}")
    
    asyncio.run(main())
