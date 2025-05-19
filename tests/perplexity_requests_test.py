"""Test script for the Perplexity Requests Tool with enhanced features."""

import os
import json
from quantalogic.tools.perplexity_requests_tool import PerplexityRequestsTool
from loguru import logger


def test_basic_search():
    """Test basic search functionality with different models."""
    
    # Create tool instance
    tool = PerplexityRequestsTool()
    
    # Test queries with different models and parameters
    test_cases = [
        {
            "query": "What are the latest developments in quantum computing?",
            "model": "sonar-pro",
            "temperature": 0.2,
            "max_tokens": 500,
            "stream": True,
            "return_related_questions": True
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting {case['model']} model:")
        print("Query:", case["query"])
        print("-" * 50)
        
        try:
            result = tool.execute(**case)
            
            print("Content:", result["content"])
            
            if "related_questions" in result:
                print("\nRelated Questions:")
                for question in result["related_questions"]:
                    print(f"- {question}")
                    
            if "sources" in result:
                print("\nSources:")
                for source in result["sources"]:
                    print(f"- {source}")
                    
            if "usage" in result:
                print("\nToken Usage:")
                print(result["usage"])
                
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("-" * 50)


def test_advanced_features():
    """Test advanced features like domain filtering and search recency."""
    
    tool = PerplexityRequestsTool()
    
    # Test with minimal parameters first
    test_case = {
        "query": "Latest research on artificial intelligence",
        "model": "sonar-pro"
    }
    
    print("\nTesting basic query:")
    print("Query:", test_case["query"])
    print("Model:", test_case["model"])
    print("-" * 50)
    
    try:
        # Debug: Print the payload that will be sent
        payload = tool._prepare_payload(**test_case)
        print("Request Payload:")
        print(json.dumps(payload, indent=2))
        print("-" * 50)
        
        result = tool.execute(**test_case)
        
        print("Content:", result["content"])
        
        if "related_questions" in result:
            print("\nRelated Questions:")
            for question in result["related_questions"]:
                print(f"- {question}")
                
        if "sources" in result:
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")
                
        if "usage" in result:
            print("\nToken Usage:")
            print(result["usage"])
            
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("-" * 50)


def test_streaming():
    """Test streaming functionality."""
    
    tool = PerplexityRequestsTool()
    
    test_case = {
        "query": "Explain the concept of quantum entanglement",
        "model": "sonar-pro",
        "stream": True
    }
    
    print("\nTesting streaming:")
    print("Query:", test_case["query"])
    print("-" * 50)
    
    try:
        # Debug: Print the payload that will be sent
        payload = tool._prepare_payload(**test_case)
        print("Request Payload:")
        print(json.dumps(payload, indent=2))
        print("-" * 50)
        
        full_response = ""
        for chunk in tool.execute(**test_case):
            if "content" in chunk and chunk["content"]:
                content = chunk["content"]
                print(content, end="", flush=True)
                full_response += content
        print("\nFull response length:", len(full_response))
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "-" * 50)


if __name__ == "__main__":
    # Test basic search functionality
    # test_basic_search()
    
    # Test advanced features
    test_advanced_features()
    
    # Test streaming
    # test_streaming()
