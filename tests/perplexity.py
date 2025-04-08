"""Simple test script for the Perplexity Deep Search Tool."""

import os
from quantalogic.tools.perplexity_requests_tool import PerplexityRequestsTool
from loguru import logger

def test_without_streaming():
    """Test Perplexity search without streaming.""" 
    
    # Create tool instance
    tool = PerplexityRequestsTool()
    
    # Test queries with different models
    queries = [
        ("How many stars are in the universe?", "sonar-pro"), 
        # ("Explain dark matter", "sonar-small-chat")
    ]
    
    for query, model in queries:
        print(f"\nTesting {model} model:")
        print("Query:", query)
        print("-" * 50)
        
        try:
            result = tool.execute(
                query=query,
                model=model,
                stream=True,
                include_sources=True
            )
            print("Content:", result["content"])
            if "sources" in result and result["sources"]:
                print("\nSources:")
                for source in result["sources"]:
                    print(f"- {source}")
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("-" * 50)

def test_with_streaming():
    """Test Perplexity search with streaming."""
    # Create tool instance
    tool = PerplexityDeepSearchTool()
    
    print("\nTesting streaming with sonar-pro:")
    query = "What is the theory of everything in physics?"
    print("Query:", query)
    print("-" * 50)
    
    try:
        result = tool.execute(
            query=query,
            model="sonar-pro",
            stream=True,
            include_sources=True
        )
        print("Content:", result["content"])
        if "sources" in result and result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("-" * 50)

if __name__ == "__main__":
    # Test without streaming
    # test_without_streaming()
    
    # Test with streaming
    test_with_streaming()