"""Test script for the Linkup Tool."""

from loguru import logger
from quantalogic.tools.linkup_tool import LinkupTool

def test_microsoft_revenue():
    """Test the sourced answer functionality with Microsoft revenue query."""
    tool = LinkupTool()
    
    try:
        logger.info("Testing Microsoft revenue query...")
        response = tool.execute(
            query="What is Microsoft's 2024 revenue?",
            depth="deep",
            output_type="sourcedAnswer"
        )
        print("\n=== Microsoft Revenue Query ===")
        print("Answer:", response.answer)
        print("\nSources:")
        for i, source in enumerate(response.sources, 1):
            print(f"\n{i}. {source.name}")
            print(f"   URL: {source.url}")
            print(f"   Snippet: {source.snippet}...")
        print("\n")
        
    except Exception as e:
        logger.error(f"Error in Microsoft revenue test: {str(e)}")

def test_search_results():
    """Test the search results functionality."""
    tool = LinkupTool()
    
    try:
        logger.info("Testing search results mode...")
        response = tool.execute(
            query="Latest developments in quantum computing 2024",
            depth="standard",
            output_type="searchResults"
        )
        print("\n=== Search Results Test ===")
        print(response)
        print("\n")
        
    except Exception as e:
        logger.error(f"Error in search results test: {str(e)}")

def main():
    """Run all tests."""
    print("Starting Linkup Tool Tests...")
    print("=" * 50)
    
    # Test Microsoft revenue query
    test_microsoft_revenue() 

if __name__ == "__main__":
    main()
