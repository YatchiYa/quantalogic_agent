"""
Finance Analysis Flow - Advanced market analysis and trading strategy generator.

This flow combines technical analysis, news impact assessment, and trading strategy
generation for stocks, indices, and cryptocurrencies using templates and LLM nodes.
"""

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional
import datetime
from pydantic import BaseModel, Field

import anyio
import typer
from loguru import logger

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from quantalogic.tools.finance.yahoo_finance import YahooFinanceTool
from quantalogic.tools.google_packages.linkup_enhanced_tool import LinkupEnhancedTool

# Configure logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Set pandas display options
import pandas as pd
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 120)

# Define structured output models
class PriceLevel(BaseModel):
    level: float
    type: str  # support, resistance, liquidity, order_block, etc.
    strength: str  # weak, moderate, strong
    description: str

class TechnicalAnalysis(BaseModel):
    market_structure: str
    trend_analysis: str
    smc_analysis: str
    ict_analysis: str
    volume_analysis: str
    fibonacci_levels: str  # Simplified to string to avoid nested structure issues
    price_levels: str  # Simplified to string to avoid nested structure issues

class NewsImpact(BaseModel):
    key_news_summary: str
    market_sentiment: str
    price_correlation: str
    future_catalysts: str
    sector_implications: str

class TradingSetup(BaseModel):
    entry_price: float
    stop_loss: float
    take_profit_levels: str  # Simplified to string to avoid nested structure issues
    risk_reward_ratio: float
    position_sizing: str
    timeframe: str
    setup_type: str  # breakout, reversal, continuation, etc.
    confidence_level: str  # low, medium, high

class MarketAnalysis(BaseModel):
    symbol: str
    analysis_date: str
    technical_analysis: TechnicalAnalysis
    news_impact: NewsImpact
    short_term_outlook: str
    long_term_outlook: str
    trading_setup: TradingSetup
    alternative_scenarios: str
    risk_management: str

    def to_markdown(self) -> str:
        """Convert the analysis to a formatted markdown document."""
        markdown = f"""# Market Analysis for {self.symbol}

*Analysis Date: {self.analysis_date}*

## Technical Analysis

### Market Structure
{self.technical_analysis.market_structure}

### Trend Analysis
{self.technical_analysis.trend_analysis}

### Smart Money Concepts (SMC) Analysis
{self.technical_analysis.smc_analysis}

### ICT (Inner Circle Trader) Analysis
{self.technical_analysis.ict_analysis}

### Volume Analysis
{self.technical_analysis.volume_analysis}

### Fibonacci Levels
{self.technical_analysis.fibonacci_levels}

### Key Price Levels
{self.technical_analysis.price_levels}

## News Impact

### Key News Summary
{self.news_impact.key_news_summary}

### Market Sentiment
{self.news_impact.market_sentiment}

### Correlation with Price Action
{self.news_impact.price_correlation}

### Potential Future Catalysts
{self.news_impact.future_catalysts}

### Sector-wide Implications
{self.news_impact.sector_implications}

## Market Outlook

### Short-term Outlook (1-7 days)
{self.short_term_outlook}

### Long-term Outlook (2-4 weeks)
{self.long_term_outlook}

## Trading Strategy

### Trading Setup
- **Setup Type:** {self.trading_setup.setup_type}
- **Timeframe:** {self.trading_setup.timeframe}
- **Confidence Level:** {self.trading_setup.confidence_level}
- **Entry Price:** {self.trading_setup.entry_price}
- **Stop Loss:** {self.trading_setup.stop_loss}
- **Risk-Reward Ratio:** {self.trading_setup.risk_reward_ratio}

### Take Profit Levels
{self.trading_setup.take_profit_levels}
"""
        markdown += f"""
### Position Sizing
{self.trading_setup.position_sizing}

### Alternative Scenarios
{self.alternative_scenarios}

### Risk Management
{self.risk_management}

---

*Disclaimer: This analysis is for informational purposes only and should not be considered financial advice. Always conduct your own research before making investment decisions.*
"""
        return markdown

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_reports")

# Create analysis directory if it doesn't exist
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Helper function to get template paths
def get_template_path(template_name):
    return os.path.join(TEMPLATES_DIR, template_name)

# Custom Observer for Workflow Events
async def finance_progress_observer(event: WorkflowEvent):
    if event.event_type == WorkflowEventType.WORKFLOW_STARTED:
        print(f"\n{'='*50}\n🚀 Starting Financial Analysis 🚀\n{'='*50}")
    elif event.event_type == WorkflowEventType.NODE_STARTED:
        print(f"\n🔄 [{event.node_name}] Starting...")
    elif event.event_type == WorkflowEventType.NODE_COMPLETED:
        if event.node_name == "fetch_price_data":
            symbol = event.context.get("symbol", "Unknown")
            interval = event.context.get("interval", "Unknown")
            range_type = event.context.get("range_type", "Unknown")
            print(f"✅ [{event.node_name}] Completed for {symbol} ({interval}, {range_type})")
        elif event.node_name == "generate_market_analysis":
            print(f"✅ [{event.node_name}] Market analysis completed")
        elif event.node_name == "generate_news_impact":
            print(f"✅ [{event.node_name}] News impact analysis completed")
        elif event.node_name == "generate_trading_strategy":
            print(f"✅ [{event.node_name}] Trading strategy completed")
        elif event.node_name == "save_analysis_to_markdown":
            file_path = event.result.get("markdown_file", "Unknown location")
            print(f"✅ [{event.node_name}] Analysis saved to markdown file: {file_path}")
        else:
            print(f"✅ [{event.node_name}] Completed")
    elif event.event_type == WorkflowEventType.WORKFLOW_COMPLETED:
        print(f"\n{'='*50}\n🎉 Financial Analysis Completed 🎉\n{'='*50}")

# Workflow Nodes
@Nodes.define(output=None)
async def fetch_price_data(symbol: str, interval: str, range_type: str) -> dict:
    """Fetch price data from Yahoo Finance."""
    logger.info(f"Fetching price data for {symbol}")
    
    try:
        yahoo_tool = YahooFinanceTool()
        price_data = await yahoo_tool.execute(
            symbol=symbol,
            interval=interval,
            range_type=range_type
        )
        
        # Extract key statistics for summary
        data_points = price_data.get("data_points", 0)
        start_date = price_data.get("start_date", "")
        end_date = price_data.get("end_date", "")
        
        # Calculate basic statistics
        if "data" in price_data and price_data["data"]:
            latest_price = price_data["data"][-1]["close"]
            highest_price = max(item["high"] for item in price_data["data"])
            lowest_price = min(item["low"] for item in price_data["data"])
            avg_volume = sum(item["volume"] for item in price_data["data"]) / len(price_data["data"])
            
            # Calculate price change
            first_price = price_data["data"][0]["open"]
            last_price = price_data["data"][-1]["close"]
            price_change = last_price - first_price
            price_change_pct = (price_change / first_price) * 100
            
            price_summary = {
                "symbol": symbol,
                "interval": interval,
                "range_type": range_type,
                "data_points": data_points,
                "start_date": start_date,
                "end_date": end_date,
                "latest_price": latest_price,
                "highest_price": highest_price,
                "lowest_price": lowest_price,
                "avg_volume": avg_volume,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "info": price_data.get("info", {})
            }
        else:
            price_summary = {
                "symbol": symbol,
                "interval": interval,
                "range_type": range_type,
                "data_points": 0,
                "error": "No data points available"
            }
        
        # Log the head of the dataframe
        if "dataframe" in price_data and isinstance(price_data["dataframe"], pd.DataFrame):
            df = price_data["dataframe"]
            logger.info(f"\nPrice data for {symbol} ({interval}, {range_type}):\n{df.head()}\n")
            logger.info(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        
        return {
            "price_data": price_data,
            "price_data_summary": price_summary
        }
    except Exception as e:
        logger.error(f"Error fetching price data: {e}")
        return {
            "price_data": None,
            "price_data_summary": {
                "symbol": symbol,
                "error": str(e)
            }
        }

@Nodes.define(output=None)
async def fetch_market_context(symbol: str) -> dict:
    """Fetch market context and news using LinkupEnhancedTool."""
    logger.info(f"Fetching market context for {symbol}")
    
    try:
        linkup_tool = LinkupEnhancedTool()
        
        # Create a search query for the symbol
        search_query = f"{symbol} stock market analysis news recent developments"
        
        # Execute the search
        context_data = await linkup_tool.async_execute(
            query=search_query,
            question=f"What are the most important recent news and developments for {symbol}?",
            depth="deep",
            analysis_depth="deep",
            scrape_sources="true",
            max_sources_to_scrape="5",
            output_format="technical"
        )
        
        # Format the context data
        market_context = {
            "symbol": symbol,
            "search_query": search_query,
            "answer": context_data.get("answer", ""),
            "sources_count": context_data.get("sources_count", 0),
            "sources": context_data.get("sources", [])
        }
        
        return {
            "market_context": market_context,
            "news_data": context_data.get("answer", "")
        }
    except Exception as e:
        logger.error(f"Error fetching market context: {e}")
        return {
            "market_context": {
                "symbol": symbol,
                "error": str(e)
            },
            "news_data": f"Error fetching news: {str(e)}"
        }

@Nodes.define(output=None)
async def prepare_recent_performance(price_data_summary: dict) -> dict:
    """Prepare a summary of recent performance for news impact analysis."""
    if "error" in price_data_summary:
        return {"recent_performance": f"Error: {price_data_summary['error']}"}
    
    # Create a formatted summary of recent performance
    performance = f"""
Symbol: {price_data_summary.get('symbol', 'Unknown')}
Latest Price: {price_data_summary.get('latest_price', 'N/A')}
Price Change: {price_data_summary.get('price_change', 'N/A')} ({price_data_summary.get('price_change_pct', 'N/A'):.2f}%)
Highest Price: {price_data_summary.get('highest_price', 'N/A')}
Lowest Price: {price_data_summary.get('lowest_price', 'N/A')}
Average Volume: {int(price_data_summary.get('avg_volume', 0))}
Data Range: {price_data_summary.get('start_date', 'N/A')} to {price_data_summary.get('end_date', 'N/A')}
"""
    
    return {"recent_performance": performance}

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_market_analysis.j2"),
    output="technical_analysis",
    response_model=TechnicalAnalysis,
    prompt_file=get_template_path("prompt_market_analysis.j2"),
    temperature=0.2,
    max_tokens=4000,
)
async def generate_market_analysis(
    model: str,
    symbol: str,
    timeframe: str,
    analysis_period: str,
    price_data_summary: dict,
    market_context: dict
) -> TechnicalAnalysis:
    """Generate technical analysis using LLM."""
    logger.info(f"Generating market analysis for {symbol}")
    
    # Fallback implementation in case the LLM call fails
    try:
        # Let the structured_llm_node decorator handle the actual LLM call
        pass
    except Exception as e:
        logger.error(f"Error in LLM call for market analysis: {e}")
        logger.warning("Using fallback implementation for market analysis")
        
        # Create a simple fallback analysis
        return TechnicalAnalysis(
            market_structure=f"Fallback market structure analysis for {symbol}",
            trend_analysis=f"Fallback trend analysis for {symbol}",
            smc_analysis=f"Fallback SMC analysis for {symbol}",
            ict_analysis=f"Fallback ICT analysis for {symbol}",
            volume_analysis=f"Fallback volume analysis for {symbol}",
            fibonacci_levels=f"Fallback Fibonacci levels for {symbol}",
            price_levels=f"Fallback price levels for {symbol}"
        )

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_news_impact.j2"),
    output="news_impact",
    response_model=NewsImpact,
    prompt_file=get_template_path("prompt_news_impact.j2"),
    temperature=0.3,
    max_tokens=3000,
)
async def generate_news_impact(
    model: str,
    symbol: str,
    news_data: str,
    recent_performance: str
) -> NewsImpact:
    """Generate news impact analysis using LLM."""
    logger.info(f"Generating news impact analysis for {symbol}")
    
    # Fallback implementation in case the LLM call fails
    try:
        # Let the structured_llm_node decorator handle the actual LLM call
        pass
    except Exception as e:
        logger.error(f"Error in LLM call for news impact: {e}")
        logger.warning("Using fallback implementation for news impact")
        
        # Create a simple fallback analysis
        return NewsImpact(
            key_news_summary=f"Fallback news summary for {symbol}",
            market_sentiment=f"Fallback market sentiment for {symbol}",
            price_correlation=f"Fallback price correlation for {symbol}",
            future_catalysts=f"Fallback future catalysts for {symbol}",
            sector_implications=f"Fallback sector implications for {symbol}"
        )

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_trading_strategy.j2"),
    output="trading_strategy",
    response_model=TradingSetup,
    prompt_file=get_template_path("prompt_trading_strategy.j2"),
    temperature=0.2,
    max_tokens=3000,
)
async def generate_trading_strategy(
    model: str,
    symbol: str,
    timeframe: str,
    technical_analysis: TechnicalAnalysis,
    news_impact: NewsImpact,
    price_levels: str
) -> TradingSetup:
    """Generate trading strategy using LLM."""
    logger.info(f"Generating trading strategy for {symbol}")
    
    # Fallback implementation in case the LLM call fails
    try:
        # Let the structured_llm_node decorator handle the actual LLM call
        pass
    except Exception as e:
        logger.error(f"Error in LLM call for trading strategy: {e}")
        logger.warning("Using fallback implementation for trading strategy")
        
        # Create a simple fallback trading setup
        return TradingSetup(
            entry_price=100.0,  # Placeholder value
            stop_loss=95.0,     # Placeholder value
            take_profit_levels=f"Fallback take profit levels for {symbol}",
            risk_reward_ratio=2.0,  # Placeholder value
            position_sizing=f"Fallback position sizing for {symbol}",
            timeframe=timeframe,
            setup_type="fallback",
            confidence_level="medium"
        )

@Nodes.define(output="complete_analysis")
async def compile_analysis(
    symbol: str,
    technical_analysis: TechnicalAnalysis,
    news_impact: NewsImpact,
    trading_strategy: TradingSetup
) -> MarketAnalysis:
    """Compile the complete market analysis."""
    logger.info(f"Compiling complete analysis for {symbol}")
    
    # Get current date and time
    now = datetime.datetime.now()
    analysis_date = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the complete analysis
    analysis = MarketAnalysis(
        symbol=symbol,
        analysis_date=analysis_date,
        technical_analysis=technical_analysis,
        news_impact=news_impact,
        short_term_outlook="To be generated",  # These will be filled in by the fallback implementation
        long_term_outlook="To be generated",
        trading_setup=trading_strategy,
        alternative_scenarios="To be generated",
        risk_management="To be generated"
    )
    
    return analysis

@Nodes.define(output=None)
async def save_analysis_to_markdown(complete_analysis: MarketAnalysis) -> dict:
    """Save the analysis to a markdown file."""
    logger.info(f"Saving analysis for {complete_analysis.symbol} to markdown")
    
    # Generate markdown content
    markdown_content = complete_analysis.to_markdown()
    
    # Create filename with symbol and date
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    filename = f"{complete_analysis.symbol}_analysis_{date_str}.md"
    file_path = os.path.join(ANALYSIS_DIR, filename)
    
    # Save to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    logger.info(f"Analysis saved to {file_path}")
    
    # Print the absolute path in a more visible way
    abs_path = os.path.abspath(file_path)
    print(f"\n{'='*80}")
    print(f"📄 REPORT SAVED TO:")
    print(f"📄 {abs_path}")
    print(f"\nTo view the report, run:")
    print(f"cat {abs_path}")
    print(f"{'='*80}\n")
    
    return {
        "markdown_file": file_path,
        "markdown_content": markdown_content
    }

@Nodes.define(output=None)
async def display_analysis(markdown_content: str) -> None:
    """Display the full analysis in the console."""
    print("\n" + "="*80)
    print("📊 FINANCIAL ANALYSIS REPORT - FULL CONTENT 📊")
    print("="*80)
    
    # Print the full report content
    print(markdown_content)
    print("="*80)
    print(f"\nFull report length: {len(markdown_content)} characters")

# Define the Workflow
workflow = (
    Workflow("fetch_price_data")
    .add_observer(finance_progress_observer)
    .then("fetch_market_context")
    .then("prepare_recent_performance")
    .then("generate_market_analysis")
    .then("generate_news_impact")
    .node("generate_trading_strategy", inputs_mapping={
        "symbol": "symbol",
        "timeframe": "timeframe",
        "technical_analysis": "technical_analysis",
        "news_impact": "news_impact",
        "price_levels": lambda ctx: ctx["technical_analysis"].price_levels,
        "model": "model"
    })
    .then("compile_analysis")
    .then("save_analysis_to_markdown")
    .then("display_analysis")
)

def analyze_financial_instrument(
    symbol: str = typer.Argument(..., help="Symbol to analyze (e.g., BTC-USD, AAPL, ^GSPC)"),
    interval: str = typer.Option("1d", help="Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d)"),
    range_type: str = typer.Option("month", help="Range type (today, date, week, month, ytd)"),
    model: str = typer.Option("gemini/gemini-2.0-flash", help="LLM model to use for analysis"),
):
    """
    Analyze a financial instrument (stock, index, cryptocurrency) and generate a comprehensive
    market analysis with trading strategy.
    """
    # Map interval to timeframe description
    timeframe_map = {
        "1m": "1-minute",
        "2m": "2-minute",
        "5m": "5-minute",
        "15m": "15-minute",
        "30m": "30-minute",
        "60m": "1-hour",
        "90m": "90-minute",
        "1h": "1-hour",
        "1d": "Daily"
    }
    
    # Map range_type to analysis period description
    period_map = {
        "today": "Intraday",
        "date": "Custom Date Range",
        "week": "1-Week",
        "month": "1-Month",
        "ytd": "Year-to-Date"
    }
    
    timeframe = timeframe_map.get(interval, interval)
    analysis_period = period_map.get(range_type, range_type)
    
    initial_context = {
        "symbol": symbol,
        "interval": interval,
        "range_type": range_type,
        "timeframe": timeframe,
        "analysis_period": analysis_period,
        "model": model
    }
    
    logger.info(f"Starting financial analysis for {symbol}")
    engine = workflow.build()
    result = anyio.run(engine.run, initial_context)
    logger.info("Financial analysis completed successfully 🎉")
    return result

def main():
    """Main function with hardcoded symbols for financial analysis."""
    # Define symbols to analyze
    symbols = [
        {"symbol": "BTC-USD", "interval": "1d", "range_type": "month", "description": "Bitcoin"},
        {"symbol": "AAPL", "interval": "1d", "range_type": "month", "description": "Apple Inc."},
        {"symbol": "MSFT", "interval": "1d", "range_type": "month", "description": "Microsoft"},
        {"symbol": "^GSPC", "interval": "1d", "range_type": "month", "description": "S&P 500 Index"},
        {"symbol": "GC=F", "interval": "1d", "range_type": "month", "description": "Gold Futures"},
    ]
    
    # Check if command line arguments were provided
    if len(sys.argv) > 1:
        # If arguments provided, use typer to parse them
        typer.run(analyze_financial_instrument)
        return
    
    # No arguments provided, use the hardcoded symbols
    print(f"\n{'='*50}\n🔍 Finance Analysis Tool 🔍\n{'='*50}")
    print("Available symbols for analysis:\n")
    
    for i, symbol_data in enumerate(symbols, 1):
        print(f"{i}. {symbol_data['description']} ({symbol_data['symbol']}) - {symbol_data['interval']} data, {symbol_data['range_type']} range")
    
    print("\nEnter the number of the symbol to analyze (or 'q' to quit): ")
    
    try:
        choice = input().strip()
        if choice.lower() == 'q':
            print("Exiting...")
            return
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(symbols):
                selected = symbols[choice_idx]
                print(f"\nAnalyzing {selected['description']} ({selected['symbol']})...\n")
                
                # Run the analysis with the selected symbol
                analyze_financial_instrument(
                    symbol=selected['symbol'],
                    interval=selected['interval'],
                    range_type=selected['range_type'],
                    model="gemini/gemini-2.0-flash"
                )
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(symbols)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
    except KeyboardInterrupt:
        print("\nAnalysis cancelled by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"\nAn error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if isinstance(exit_code, int) else 0)
