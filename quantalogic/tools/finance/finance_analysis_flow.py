#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru>=0.7.2",
#     "litellm>=1.0.0",
#     "pydantic>=2.0.0",
#     "asyncio",
#     "jinja2>=3.1.0",
#     "quantalogic",
#     "instructor>=0.5.2",
#     "typer>=0.9.0",
#     "rich>=13.0.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "matplotlib>=3.7.0",
#     "ta>=0.10.0",  # Technical analysis library
#     "yfinance>=0.2.0",
# ]
# ///

import asyncio
import os
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from quantalogic.flow.flow import Nodes, Workflow
from quantalogic.tools.finance.yahoo_finance import YahooFinanceTool
from quantalogic.tools.google_packages.linkup_enhanced_tool import LinkupEnhancedTool

console = Console()

# Constants
ANALYSIS_DIR = Path("finance_analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)

# Define Pydantic models for structured output
class MarketData(BaseModel):
    """Market data for financial instrument."""
    symbol: str = Field(description="Symbol of the financial instrument")
    name: str = Field(description="Name of the financial instrument")
    current_price: float = Field(description="Current price")
    open_price: float = Field(description="Opening price")
    high_price: float = Field(description="Highest price in the period")
    low_price: float = Field(description="Lowest price in the period")
    volume: int = Field(description="Trading volume")
    change_percent: float = Field(description="Price change percentage")
    time_period: str = Field(description="Time period of the analysis")
    data_points: int = Field(description="Number of data points")

class TechnicalIndicators(BaseModel):
    """Technical indicators for the financial instrument."""
    rsi: float = Field(description="Relative Strength Index")
    macd: Dict[str, float] = Field(description="Moving Average Convergence Divergence")
    bollinger_bands: Dict[str, List[float]] = Field(description="Bollinger Bands")
    moving_averages: Dict[str, float] = Field(description="Various moving averages")
    support_levels: List[float] = Field(description="Identified support levels")
    resistance_levels: List[float] = Field(description="Identified resistance levels")
    volume_profile: Dict[str, Any] = Field(description="Volume profile analysis")

class NewsAnalysis(BaseModel):
    """News and sentiment analysis."""
    sentiment_score: float = Field(description="Overall sentiment score (-1 to 1)")
    key_events: List[Dict[str, Any]] = Field(description="Key market events")
    market_sentiment: str = Field(description="Market sentiment description")
    news_summary: str = Field(description="Summary of relevant news")
    impact_analysis: str = Field(description="Analysis of news impact on price")

class PricePatterns(BaseModel):
    """Identified chart patterns."""
    identified_patterns: List[Dict[str, Any]] = Field(description="List of identified patterns")
    pattern_strength: Dict[str, float] = Field(description="Strength of each pattern")
    pattern_targets: Dict[str, Dict[str, float]] = Field(description="Price targets for patterns")
    pattern_analysis: str = Field(description="Analysis of pattern implications")

class TradingRecommendation(BaseModel):
    """Trading recommendation based on analysis."""
    direction: Literal["buy", "sell", "hold"] = Field(description="Trading direction")
    entry_points: List[float] = Field(description="Recommended entry points")
    stop_loss: List[float] = Field(description="Recommended stop loss levels")
    take_profit: List[float] = Field(description="Recommended take profit levels")
    risk_reward_ratio: float = Field(description="Risk to reward ratio")
    confidence: float = Field(description="Confidence level (0-1)")
    timeframe: str = Field(description="Recommended trading timeframe")
    strategy: str = Field(description="Recommended trading strategy")
    rationale: str = Field(description="Rationale for the recommendation")

class FinanceAnalysis(BaseModel):
    """Complete finance analysis."""
    market_data: MarketData = Field(description="Market data")
    technical_indicators: TechnicalIndicators = Field(description="Technical indicators")
    news_analysis: NewsAnalysis = Field(description="News analysis")
    price_patterns: PricePatterns = Field(description="Price patterns")
    trading_recommendation: TradingRecommendation = Field(description="Trading recommendation")
    executive_summary: str = Field(description="Executive summary of the analysis")
    technical_analysis: str = Field(description="Detailed technical analysis")
    fundamental_analysis: str = Field(description="Fundamental analysis")
    market_outlook: str = Field(description="Market outlook")
    risk_assessment: str = Field(description="Risk assessment")
    
    def to_markdown(self) -> str:
        """Convert the analysis to markdown format."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = f"""# Financial Analysis Report: {self.market_data.symbol}
*Generated on: {current_time}*

## Executive Summary
{self.executive_summary}

## Market Data
- **Symbol**: {self.market_data.symbol}
- **Name**: {self.market_data.name}
- **Current Price**: ${self.market_data.current_price:.2f}
- **Change**: {self.market_data.change_percent:.2f}%
- **Volume**: {self.market_data.volume:,}
- **Period**: {self.market_data.time_period}

## Technical Analysis
{self.technical_analysis}

### Key Technical Indicators
- **RSI**: {self.technical_indicators.rsi:.2f}
- **MACD**: Signal: {self.technical_indicators.macd["signal"]:.2f}, Line: {self.technical_indicators.macd["line"]:.2f}
- **Moving Averages**:
  - MA(50): ${self.technical_indicators.moving_averages.get("MA50", 0):.2f}
  - MA(200): ${self.technical_indicators.moving_averages.get("MA200", 0):.2f}

### Support & Resistance Levels
- **Support Levels**: {", ".join([f"${level:.2f}" for level in self.technical_indicators.support_levels])}
- **Resistance Levels**: {", ".join([f"${level:.2f}" for level in self.technical_indicators.resistance_levels])}

## Pattern Analysis
{self.price_patterns.pattern_analysis}

### Identified Patterns
"""
        
        # Add patterns
        for pattern in self.price_patterns.identified_patterns:
            md += f"- **{pattern['name']}**: {pattern['description']}\n"
        
        md += f"""
## News & Sentiment Analysis
{self.news_analysis.news_summary}

### Market Sentiment
{self.news_analysis.market_sentiment}

### Impact Analysis
{self.news_analysis.impact_analysis}

## Fundamental Analysis
{self.fundamental_analysis}

## Market Outlook
{self.market_outlook}

## Risk Assessment
{self.risk_assessment}

## Trading Recommendation
- **Direction**: {self.trading_recommendation.direction.upper()}
- **Entry Points**: {", ".join([f"${point:.2f}" for point in self.trading_recommendation.entry_points])}
- **Stop Loss**: {", ".join([f"${sl:.2f}" for sl in self.trading_recommendation.stop_loss])}
- **Take Profit**: {", ".join([f"${tp:.2f}" for tp in self.trading_recommendation.take_profit])}
- **Risk/Reward**: {self.trading_recommendation.risk_reward_ratio:.2f}
- **Confidence**: {self.trading_recommendation.confidence*100:.0f}%
- **Timeframe**: {self.trading_recommendation.timeframe}
- **Strategy**: {self.trading_recommendation.strategy}

### Rationale
{self.trading_recommendation.rationale}

---
*Disclaimer: This analysis is for informational purposes only and does not constitute investment advice. Always conduct your own research before making investment decisions.*
"""
        return md

# Node: Market Data Collection
@Nodes.define(output="market_data")
async def fetch_market_data(
    symbol: str,
    interval: str = "1d",
    range_type: str = "month"
) -> MarketData:
    """Fetch market data for the specified symbol."""
    try:
        # Use Yahoo Finance Tool
        tool = YahooFinanceTool()
        data = await tool.execute(
            symbol=symbol,
            interval=interval,
            range_type=range_type
        )
        
        # Check if there's an error in the response
        if "error" in data:
            raise ValueError(f"Error from Yahoo Finance API: {data['error']}")
        
        # Extract data from the dataframe
        df = data["dataframe"]
        
        if len(df) == 0:
            raise ValueError(f"No data returned for symbol {symbol}")
        
        # Get first and last data points
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        
        # Calculate change percentage
        change_percent = ((last_row["Close"] - first_row["Open"]) / first_row["Open"]) * 100
        
        # Create MarketData object
        market_data = MarketData(
            symbol=symbol,
            name=data["info"].get("longName", "") or data["info"].get("shortName", "") or symbol,
            current_price=float(last_row["Close"]),
            open_price=float(first_row["Open"]),
            high_price=float(df["High"].max()),
            low_price=float(df["Low"].min()),
            volume=int(df["Volume"].sum()),
            change_percent=float(change_percent),
            time_period=f"{data['start_date']} to {data['end_date']}",
            data_points=len(df)
        )
        
        logger.info(f"Fetched market data for {symbol}: {market_data.current_price:.2f} ({market_data.change_percent:.2f}%)")
        return market_data
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        raise

# Node: Technical Indicators Calculation
@Nodes.define(output="technical_indicators")
async def calculate_technical_indicators(
    symbol: str,
    market_data: MarketData,
    interval: str = "1d",
    range_type: str = "month"
) -> TechnicalIndicators:
    """Calculate technical indicators for the financial instrument."""
    try:
        # Fetch historical data for calculations
        tool = YahooFinanceTool()
        data = await tool.execute(
            symbol=symbol,
            interval=interval,
            range_type=range_type
        )
        
        # Check if there's an error in the response
        if "error" in data:
            raise ValueError(f"Error from Yahoo Finance API: {data['error']}")
        
        # Convert to pandas DataFrame for technical analysis
        df = data["dataframe"]
        
        if len(df) == 0:
            raise ValueError(f"No data returned for symbol {symbol}")
        
        # Import technical analysis libraries
        from ta import momentum, trend, volatility, volume
        
        # Calculate RSI
        rsi = momentum.RSIIndicator(close=df["Close"], window=14).rsi().iloc[-1]
        
        # Calculate MACD
        macd = trend.MACD(close=df["Close"])
        macd_line = macd.macd().iloc[-1]
        macd_signal = macd.macd_signal().iloc[-1]
        macd_hist = macd.macd_diff().iloc[-1]
        
        # Calculate Bollinger Bands
        bollinger = volatility.BollingerBands(close=df["Close"])
        bb_high = bollinger.bollinger_hband().iloc[-1]
        bb_mid = bollinger.bollinger_mavg().iloc[-1]
        bb_low = bollinger.bollinger_lband().iloc[-1]
        
        # Calculate Moving Averages
        ma50 = df["Close"].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
        ma200 = df["Close"].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
        
        # Identify support and resistance levels using pivot points
        # This is a simplified implementation - would be more sophisticated in production
        # Here we're using a simplified approach
        high = df["High"].iloc[-20:].max()
        low = df["Low"].iloc[-20:].min()
        close = df["Close"].iloc[-1]
        
        # Calculate pivot point
        pivot = (high + low + close) / 3
        
        # Calculate support and resistance levels
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        
        support_levels = [s1, s2]
        resistance_levels = [r1, r2]
        
        # Volume profile - simplified
        volume_profile = {
            "average_volume": float(df["Volume"].mean()),
            "volume_trend": "increasing" if df["Volume"].iloc[-1] > df["Volume"].mean() else "decreasing",
            "price_volume_correlation": 0.75  # Simplified
        }
        
        # Create TechnicalIndicators object
        indicators = TechnicalIndicators(
            rsi=float(rsi),
            macd={
                "line": float(macd_line), 
                "signal": float(macd_signal), 
                "histogram": float(macd_hist)
            },
            bollinger_bands={
                "high": [float(bb_high)], 
                "mid": [float(bb_mid)], 
                "low": [float(bb_low)]
            },
            moving_averages={
                "MA50": float(ma50) if ma50 is not None else 0, 
                "MA200": float(ma200) if ma200 is not None else 0
            },
            support_levels=[float(s) for s in support_levels],
            resistance_levels=[float(r) for r in resistance_levels],
            volume_profile=volume_profile
        )
        
        logger.info(f"Calculated technical indicators for {symbol}: RSI={indicators.rsi:.2f}")
        return indicators
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        raise

# Node: News Analysis
@Nodes.define(output="news_analysis")
async def analyze_news(
    symbol: str,
    market_data: MarketData
) -> NewsAnalysis:
    """Analyze news and sentiment for the financial instrument."""
    try:
        # Use LinkupEnhancedTool for news analysis
        tool = LinkupEnhancedTool()
        
        # Create search query based on symbol and name
        query = f"latest news {market_data.name} {symbol} financial analysis"
        question = f"What is the current market sentiment for {market_data.name} ({symbol}) and what are the key events affecting its price?"
        
        # Execute the tool
        result = await tool.async_execute(
            query=query,
            question=question,
            depth="deep",
            analysis_depth="deep",
            scrape_sources="true",
            max_sources_to_scrape="5",
            output_format="technical"
        )
        
        # Extract sentiment score - this would be more sophisticated in production
        # Here we're using a simplified approach
        sentiment_words = {
            "very bullish": 0.9, "bullish": 0.7, "positive": 0.5, "optimistic": 0.3,
            "neutral": 0.0,
            "pessimistic": -0.3, "negative": -0.5, "bearish": -0.7, "very bearish": -0.9
        }
        
        sentiment_score = 0.0
        for word, score in sentiment_words.items():
            if word in result["answer"].lower():
                sentiment_score = score
                break
        
        # Extract key events - simplified implementation
        key_events = []
        lines = result["answer"].split("\n")
        for line in lines:
            if "announced" in line.lower() or "reported" in line.lower() or "launched" in line.lower():
                key_events.append({"description": line.strip(), "impact": "high"})
        
        # Create NewsAnalysis object
        news = NewsAnalysis(
            sentiment_score=sentiment_score,
            key_events=key_events[:5] if key_events else [{"description": "No significant events found", "impact": "low"}],
            market_sentiment="Bullish" if sentiment_score > 0.3 else "Bearish" if sentiment_score < -0.3 else "Neutral",
            news_summary="\n".join(lines[:10]) if lines else "No significant news found",
            impact_analysis="\n".join([line for line in lines if "impact" in line.lower() or "affect" in line.lower() or "influence" in line.lower()])
        )
        
        logger.info(f"Analyzed news for {symbol}: Sentiment={news.market_sentiment}")
        return news
    except Exception as e:
        logger.error(f"Error analyzing news: {e}")
        # Provide fallback news analysis if the tool fails
        return NewsAnalysis(
            sentiment_score=0.0,
            key_events=[{"description": "Unable to fetch news data", "impact": "unknown"}],
            market_sentiment="Neutral",
            news_summary="Unable to fetch news data due to an error.",
            impact_analysis="No impact analysis available due to data retrieval error."
        )

# Node: Pattern Recognition
@Nodes.define(output="price_patterns")
async def identify_price_patterns(
    symbol: str,
    market_data: MarketData,
    technical_indicators: TechnicalIndicators
) -> PricePatterns:
    """Identify chart patterns in the price data."""
    try:
        # Fetch historical data for pattern recognition
        tool = YahooFinanceTool()
        data = await tool.execute(
            symbol=symbol,
            interval="1d",
            range_type="month"
        )
        
        # Check if there's an error in the response
        if "error" in data:
            raise ValueError(f"Error from Yahoo Finance API: {data['error']}")
        
        # Convert to pandas DataFrame
        df = data["dataframe"]
        
        if len(df) == 0:
            raise ValueError(f"No data returned for symbol {symbol}")
        
        # This would be a sophisticated pattern recognition algorithm in production
        # Here we're using a simplified approach for demonstration
        
        # Check for potential patterns based on price action and indicators
        patterns = []
        pattern_strength = {}
        pattern_targets = {}
        
        # Example pattern check: Bullish trend with RSI recovery
        if (df["Close"].iloc[-1] > df["Close"].iloc[-5] and 
            technical_indicators.rsi > 50 and 
            technical_indicators.macd["histogram"] > 0):
            
            patterns.append({
                "name": "Bullish Trend Continuation",
                "description": "Price in uptrend with strengthening RSI and positive MACD histogram",
                "confirmation": 0.75
            })
            pattern_strength["Bullish Trend Continuation"] = 0.75
            pattern_targets["Bullish Trend Continuation"] = {
                "target1": market_data.current_price * 1.05,
                "target2": market_data.current_price * 1.10,
                "stop": market_data.current_price * 0.97
            }
        
        # Example pattern check: Bearish divergence
        if (df["Close"].iloc[-1] > df["Close"].iloc[-5] and 
            technical_indicators.rsi < 50 and
            technical_indicators.macd["histogram"] < 0):
            
            patterns.append({
                "name": "Bearish RSI Divergence",
                "description": "Price making higher highs but RSI showing weakness with negative MACD histogram",
                "confirmation": 0.65
            })
            pattern_strength["Bearish RSI Divergence"] = 0.65
            pattern_targets["Bearish RSI Divergence"] = {
                "target1": market_data.current_price * 0.95,
                "target2": market_data.current_price * 0.90,
                "stop": market_data.current_price * 1.03
            }
        
        # Example pattern check: Support bounce
        if any(abs(market_data.low_price - support) / support < 0.02 for support in technical_indicators.support_levels):
            patterns.append({
                "name": "Support Bounce",
                "description": "Price bounced off a key support level",
                "confirmation": 0.70
            })
            pattern_strength["Support Bounce"] = 0.70
            pattern_targets["Support Bounce"] = {
                "target1": market_data.current_price * 1.03,
                "target2": market_data.current_price * 1.07,
                "stop": market_data.current_price * 0.98
            }
        
        # Example pattern check: Resistance rejection
        if any(abs(market_data.high_price - resistance) / resistance < 0.02 for resistance in technical_indicators.resistance_levels):
            patterns.append({
                "name": "Resistance Rejection",
                "description": "Price rejected at a key resistance level",
                "confirmation": 0.70
            })
            pattern_strength["Resistance Rejection"] = 0.70
            pattern_targets["Resistance Rejection"] = {
                "target1": market_data.current_price * 0.97,
                "target2": market_data.current_price * 0.93,
                "stop": market_data.current_price * 1.02
            }
        
        # Create pattern analysis text
        if patterns:
            pattern_analysis = f"Analysis identified {len(patterns)} significant patterns. "
            for pattern in patterns:
                pattern_analysis += f"The {pattern['name']} pattern suggests {pattern['description']} with {pattern['confirmation']*100:.0f}% confidence. "
            
            if "Bullish" in pattern_analysis and "Bearish" in pattern_analysis:
                pattern_analysis += "The conflicting patterns suggest a period of consolidation or volatility ahead."
            elif "Bullish" in pattern_analysis:
                pattern_analysis += "The bullish patterns suggest potential upside in the near term."
            elif "Bearish" in pattern_analysis:
                pattern_analysis += "The bearish patterns suggest caution and potential downside risk."
        else:
            pattern_analysis = "No significant chart patterns identified in the current price action."
            
            # Add a default pattern if none found
            patterns.append({
                "name": "Neutral Consolidation",
                "description": "Price moving in a sideways range with no clear direction",
                "confirmation": 0.50
            })
            pattern_strength["Neutral Consolidation"] = 0.50
            pattern_targets["Neutral Consolidation"] = {
                "target1": market_data.current_price * 1.02,
                "target2": market_data.current_price * 0.98,
                "stop": market_data.current_price * 0.95
            }
        
        # Create PricePatterns object
        price_patterns = PricePatterns(
            identified_patterns=patterns,
            pattern_strength=pattern_strength,
            pattern_targets=pattern_targets,
            pattern_analysis=pattern_analysis
        )
        
        logger.info(f"Identified {len(patterns)} price patterns for {symbol}")
        return price_patterns
    except Exception as e:
        logger.error(f"Error identifying price patterns: {e}")
        # Return a default pattern if there's an error
        return PricePatterns(
            identified_patterns=[{
                "name": "Analysis Error",
                "description": "Unable to identify patterns due to an error",
                "confirmation": 0.0
            }],
            pattern_strength={"Analysis Error": 0.0},
            pattern_targets={"Analysis Error": {
                "target1": market_data.current_price,
                "target2": market_data.current_price,
                "stop": market_data.current_price
            }},
            pattern_analysis="Pattern analysis failed due to an error in data processing."
        )

# Node: LLM Analysis
@Nodes.structured_llm_node(
    system_prompt="""You are an expert financial analyst with 20 years of experience in trading and market analysis.
    Your expertise spans technical analysis, fundamental analysis, and market psychology.
    You specialize in identifying trading opportunities across multiple asset classes including stocks, cryptocurrencies, commodities, and forex.
    Provide professional, detailed analysis with specific price levels, clear recommendations, and thorough rationale.""",
    output="finance_analysis",
    response_model=FinanceAnalysis,
    max_tokens=4000,
    prompt_template="""
Analyze the following financial data for {{market_data.symbol}} ({{market_data.name}}):

## Market Data
{{market_data}}

## Technical Indicators
{{technical_indicators}}

## News Analysis
{{news_analysis}}

## Price Patterns
{{price_patterns}}

Based on the above data, provide a comprehensive financial analysis including:

1. Technical Analysis - Interpret the technical indicators, price action, and chart patterns
2. Fundamental Analysis - Assess the impact of news, events, and market sentiment
3. Market Outlook - Project short and long-term price movements
4. Risk Assessment - Identify key risks and volatility expectations
5. Trading Recommendation - Provide specific entry points, stop loss, take profit levels with rationale

Your analysis should be detailed, professional, and actionable. Include specific price levels and clear reasoning.
"""
)
async def generate_finance_analysis(
    market_data: MarketData,
    technical_indicators: TechnicalIndicators,
    news_analysis: NewsAnalysis,
    price_patterns: PricePatterns,
    model: str
) -> FinanceAnalysis:
    """Generate comprehensive finance analysis using LLM."""
    logger.debug(f"generate_finance_analysis called with model: {model}")
    pass

# Node: Save Analysis to Markdown
@Nodes.define(output="analysis_saved")
async def save_analysis_to_markdown(
    finance_analysis: FinanceAnalysis,
    symbol: str
) -> str:
    """Save the analysis to a markdown file."""
    try:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{timestamp}_analysis.md"
        filepath = ANALYSIS_DIR / filename
        
        # Convert analysis to markdown
        markdown_content = finance_analysis.to_markdown()
        
        # Save to file
        with open(filepath, "w") as f:
            f.write(markdown_content)
        
        logger.info(f"Analysis saved to {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving analysis to markdown: {e}")
        raise

# Workflow Definition
def create_finance_analysis_workflow() -> Workflow:
    """Create a workflow for finance analysis."""
    workflow = (
        Workflow("fetch_market_data")
        .then("calculate_technical_indicators")
        .then("analyze_news")
        .then("identify_price_patterns")
        .then("generate_finance_analysis")
        .then("save_analysis_to_markdown")
    )
    
    # Add input mappings
    workflow.node_input_mappings = {
        "fetch_market_data": {
            "symbol": "symbol",
            "interval": "interval",
            "range_type": "range_type"
        },
        "calculate_technical_indicators": {
            "symbol": "symbol",
            "interval": "interval",
            "range_type": "range_type"
        },
        "analyze_news": {
            "symbol": "symbol"
        },
        "identify_price_patterns": {
            "symbol": "symbol"
        },
        "generate_finance_analysis": {
            "model": "analysis_model"
        },
        "save_analysis_to_markdown": {
            "symbol": "symbol"
        }
    }
    
    return workflow

# Main Analysis Function
async def analyze_financial_instrument(
    symbol: str,
    interval: str = "1d",
    range_type: str = "month",
    analysis_model: str = "gemini/gemini-2.0-flash"
) -> FinanceAnalysis:
    """Analyze a financial instrument and generate a comprehensive report."""
    
    # Ensure symbol is not empty
    if not symbol or symbol.strip() == "":
        raise ValueError("Symbol cannot be empty")
    
    logger.info(f"Starting financial analysis for {symbol}")
    
    initial_context = {
        "symbol": symbol,
        "interval": interval,
        "range_type": range_type,
        "analysis_model": analysis_model
    }
    
    try:
        workflow = create_finance_analysis_workflow()
        engine = workflow.build()
        
        result = await engine.run(initial_context)
        
        if not isinstance(result.get("finance_analysis"), FinanceAnalysis):
            raise ValueError("Workflow did not produce a valid finance analysis")
        
        logger.info(f"Financial analysis for {symbol} completed successfully")
        return result["finance_analysis"]
        
    except Exception as e:
        logger.error(f"Error analyzing financial instrument {symbol}: {e}")
        raise

# Specialized Analysis Templates
async def analyze_cryptocurrency(
    symbol: str,
    interval: str = "1d",
    range_type: str = "month",
    analysis_model: str = "gemini/gemini-2.0-flash"
) -> FinanceAnalysis:
    """Specialized analysis for cryptocurrencies with additional crypto-specific metrics."""
    # This would include additional crypto-specific analysis in a production environment
    # For now, we'll use the base analysis
    return await analyze_financial_instrument(symbol, interval, range_type, analysis_model)

async def analyze_stock(
    symbol: str,
    interval: str = "1d",
    range_type: str = "month",
    analysis_model: str = "gemini/gemini-2.0-flash"
) -> FinanceAnalysis:
    """Specialized analysis for stocks with additional stock-specific metrics."""
    # This would include additional stock-specific analysis in a production environment
    # For now, we'll use the base analysis
    return await analyze_financial_instrument(symbol, interval, range_type, analysis_model)

async def analyze_commodity(
    symbol: str,
    interval: str = "1d",
    range_type: str = "month",
    analysis_model: str = "gemini/gemini-2.0-flash"
) -> FinanceAnalysis:
    """Specialized analysis for commodities with additional commodity-specific metrics."""
    # This would include additional commodity-specific analysis in a production environment
    # For now, we'll use the base analysis
    return await analyze_financial_instrument(symbol, interval, range_type, analysis_model)

async def analyze_forex(
    symbol: str,
    interval: str = "1d",
    range_type: str = "month",
    analysis_model: str = "gemini/gemini-2.0-flash"
) -> FinanceAnalysis:
    """Specialized analysis for forex pairs with additional forex-specific metrics."""
    # This would include additional forex-specific analysis in a production environment
    # For now, we'll use the base analysis
    return await analyze_financial_instrument(symbol, interval, range_type, analysis_model)

# Add main function for testing
async def main():
    """Test function for the finance analysis workflow."""
    # Test with different financial instruments
    symbols = {
        "crypto": "BTC-USD",  # Bitcoin
        "stock": "AAPL",      # Apple stock
        "commodity": "GC=F",  # Gold Futures
        "forex": "EURUSD=X"   # EUR/USD forex pair
    }
    
    try:
        # Test with Bitcoin
        console.print("\n[bold blue]Testing with Bitcoin...[/]")
        btc_analysis = await analyze_cryptocurrency(
            symbol=symbols["crypto"],
            interval="1d",
            range_type="month"
        )
        
        # Display the analysis
        md_path = await save_analysis_to_markdown(btc_analysis, symbols["crypto"])
        console.print(f"[bold green]✓ Bitcoin analysis saved to {md_path}[/]")
        
        # Display summary in console
        console.print(Panel(Markdown(f"""
## {btc_analysis.market_data.name} ({btc_analysis.market_data.symbol}) Analysis Summary

**Current Price**: ${btc_analysis.market_data.current_price:.2f} ({btc_analysis.market_data.change_percent:.2f}%)

**Recommendation**: {btc_analysis.trading_recommendation.direction.upper()}

**Key Levels**:
- Entry: ${btc_analysis.trading_recommendation.entry_points[0]:.2f}
- Stop: ${btc_analysis.trading_recommendation.stop_loss[0]:.2f}
- Target: ${btc_analysis.trading_recommendation.take_profit[0]:.2f}

**Sentiment**: {btc_analysis.news_analysis.market_sentiment}
        """)), title="Bitcoin Analysis Summary")
        
    except Exception as e:
        console.print(f"[bold red]Error during testing:[/] {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
