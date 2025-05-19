#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru",
#     "litellm",
#     "pydantic>=2.0",
#     "anyio",
#     "quantalogic>=0.35",
#     "jinja2",
#     "typer",
#     "pyperclip",
#     "instructor",
#     "yfinance",
#     "pandas",
#     "numpy",
#     "plotly",
#     "ta",
#     "matplotlib"
# ]
# ///

import os
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime, timedelta

import anyio
import asyncio
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from loguru import logger
from pydantic import BaseModel, Field

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from quantalogic.tools.finance.yahoo_finance import YahooFinanceTool
from quantalogic.tools.google_packages.google_news_tool import GoogleNewsTool
from quantalogic.tools.finance.llm_node import FinanceLLMTool
from quantalogic.tools.finance.trading_decision import TradingDecisionTool

# Configure logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Define structured output models
class TechnicalIndicators(BaseModel):
    """Technical indicators for market analysis."""
    rsi: List[float] = Field(description="Relative Strength Index values")
    macd: Dict[str, List[float]] = Field(description="MACD indicator values")
    bollinger_bands: Dict[str, List[float]] = Field(description="Bollinger Bands values")
    support_resistance: Dict[str, List[float]] = Field(description="Support and resistance levels")
    volume_profile: Dict[str, float] = Field(description="Volume profile analysis")

class MarketAnalysis(BaseModel):
    symbol: str
    timeframe: str
    trend: str
    key_levels: List[float]
    support_zones: List[float]
    resistance_zones: List[float]
    volume_analysis: Dict
    sentiment: str
    risk_level: str
    technical_indicators: TechnicalIndicators

class TradingStrategy(BaseModel):
    symbol: str
    position: str  # long/short/neutral
    entry_price: Decimal
    stop_loss: Decimal
    take_profit: List[Decimal]
    timeframe: str
    risk_reward: float
    confidence: float
    notes: List[str]

# Custom Observer for Workflow Events
async def trading_progress_observer(event: WorkflowEvent):
    if event.event_type == WorkflowEventType.WORKFLOW_STARTED:
        print(f"\n{'='*50}\n🚀 Starting Trading Analysis 🚀\n{'='*50}")
    elif event.event_type == WorkflowEventType.NODE_STARTED:
        print(f"\n🔄 [{event.node_name}] Starting...")
    elif event.event_type == WorkflowEventType.NODE_COMPLETED:
        print(f"✅ [{event.node_name}] Completed")
    elif event.event_type == WorkflowEventType.WORKFLOW_COMPLETED:
        print(f"\n{'='*50}\n🎉 Trading Analysis Finished 🎉\n{'='*50}")
    elif event.event_type == WorkflowEventType.TRANSITION_EVALUATED:
        logger.debug(f"Transition evaluated: {event.transition_from} -> {event.transition_to}")

# Workflow Nodes
@Nodes.define(output=None)
async def fetch_market_data(symbol: str, interval: str = "1h", range_type: str = "week") -> dict:
    """Fetch market data from Yahoo Finance."""
    yahoo_tool = YahooFinanceTool()
    data = await yahoo_tool.execute(symbol=symbol, interval=interval, range_type=range_type)
    logger.info(f"Fetched market data for {symbol}")
    return {"market_data": data}

@Nodes.define(output=None)
async def fetch_news_sentiment(symbol: str) -> dict:
    """Fetch relevant news and sentiment."""
    try:
        news_tool = GoogleNewsTool()
        # Run the news fetch in a thread pool to avoid event loop issues
        loop = asyncio.get_running_loop()
        news = await loop.run_in_executor(
            None,
            lambda: asyncio.run(news_tool.execute(
                query=f"{symbol} stock market",
                max_results=10
            ))
        )
        logger.info(f"Fetched news for {symbol}")
        return {"news_data": news}
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return {"news_data": {"error": str(e), "articles": []}}

@Nodes.define(output=None)
async def analyze_technical_indicators(market_data: dict) -> dict:
    """Generate technical analysis and charts."""
    try:
        # Convert market data to DataFrame
        data = market_data.get('data', [])
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Rename columns to match ta library expectations
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Calculate technical indicators
        indicators = {}
        
        # RSI
        indicators['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        indicators['macd'] = {
            'macd': macd.macd(),
            'signal': macd.macd_signal(),
            'histogram': macd.macd_diff()
        }
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        indicators['bollinger_bands'] = {
            'high': bollinger.bollinger_hband(),
            'mid': bollinger.bollinger_mavg(),
            'low': bollinger.bollinger_lband()
        }
        
        # Support/Resistance using price action
        def find_support_resistance(data, window=20):
            highs = []
            lows = []
            for i in range(window, len(data) - window):
                if all(data[i] > data[i-j] for j in range(1, window)) and \
                   all(data[i] > data[i+j] for j in range(1, window)):
                    highs.append(float(data[i]))
                if all(data[i] < data[i-j] for j in range(1, window)) and \
                   all(data[i] < data[i+j] for j in range(1, window)):
                    lows.append(float(data[i]))
            return {'resistance': highs[-3:] if highs else [], 'support': lows[-3:] if lows else []}
        
        indicators['support_resistance'] = find_support_resistance(df['Close'].values)
        
        # Volume Profile
        def calculate_volume_profile(data, price_levels=10):
            price_range = np.linspace(data['Low'].min(), data['High'].max(), price_levels)
            volume_profile = {}
            for i in range(len(price_range)-1):
                mask = (data['Close'] >= price_range[i]) & (data['Close'] < price_range[i+1])
                volume_profile[f"{float(price_range[i]):.2f}-{float(price_range[i+1]):.2f}"] = float(data.loc[mask, 'Volume'].sum())
            return volume_profile
        
        indicators['volume_profile'] = calculate_volume_profile(df)
        
        # Generate charts
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Price & BB', 'RSI', 'MACD', 'Volume'))
        
        # Candlestick chart with Bollinger Bands
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=indicators['bollinger_bands']['high'],
            name='BB Upper',
            line=dict(color='gray', dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=indicators['bollinger_bands']['low'],
            name='BB Lower',
            line=dict(color='gray', dash='dash'),
            fill='tonexty'
        ), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=df.index,
            y=indicators['rsi'],
            name='RSI'
        ), row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(
            x=df.index,
            y=indicators['macd']['macd'],
            name='MACD'
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=indicators['macd']['signal'],
            name='Signal'
        ), row=3, col=1)
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=indicators['macd']['histogram'],
            name='Histogram'
        ), row=3, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume'
        ), row=4, col=1)
        
        # Save charts
        charts_dir = "trading_analysis/charts"
        os.makedirs(charts_dir, exist_ok=True)
        fig.write_html(f"{charts_dir}/technical_analysis.html")
        
        logger.info("Generated technical analysis and charts")
        return {
            "technical_analysis": {
                "indicators": {
                    'rsi': float(indicators['rsi'].iloc[-1]) if not indicators['rsi'].empty else None,
                    'macd': {
                        'macd': float(indicators['macd']['macd'].iloc[-1]) if not indicators['macd']['macd'].empty else None,
                        'signal': float(indicators['macd']['signal'].iloc[-1]) if not indicators['macd']['signal'].empty else None,
                        'histogram': float(indicators['macd']['histogram'].iloc[-1]) if not indicators['macd']['histogram'].empty else None
                    },
                    'support_resistance': indicators['support_resistance'],
                    'volume_profile': indicators['volume_profile']
                },
                "charts_path": f"{charts_dir}/technical_analysis.html"
            }
        }
    except Exception as e:
        logger.error(f"Error in technical analysis: {str(e)}")
        return {"technical_analysis": {"error": str(e)}}

@Nodes.define(output=None)
async def analyze_market_structure(
    model: str,
    symbol: str,
    market_data: dict,
    news_data: dict,
    technical_analysis: dict,
    timeframes: List[str] = ["4h", "1h", "15m"]
) -> dict:
    """Analyze market structure using LLM."""
    try:
        llm_tool = FinanceLLMTool(model_name=model)
        
        # Format news data
        news_articles = news_data.get('articles', [])
        news_summary = "\n".join([
            f"- {article.get('title', 'No title')}: {article.get('description', 'No description')}"
            for article in news_articles
        ]) if news_articles else "No recent news available."
        
        # Prepare market data
        data_list = market_data.get('data', [])
        latest_price = data_list[-1].get('close', 'N/A') if data_list else 'N/A'
        volume_24h = sum(d.get('volume', 0) for d in data_list[-24:]) if data_list else 0
        price_low = min((d.get('low', float('inf')) for d in data_list), default=float('inf'))
        price_high = max((d.get('high', 0) for d in data_list), default=0)
        
        # Format technical indicators
        indicators = technical_analysis.get('indicators', {})
        
        # Get latest indicator values
        latest_rsi = indicators.get('rsi')
        latest_macd = indicators.get('macd', {})
        support_resistance = indicators.get('support_resistance', {})
        volume_profile = indicators.get('volume_profile', {})
        
        # Format technical analysis summary
        technical_summary = f"""
Technical Indicators Summary:
1. RSI (14): {latest_rsi:.2f if isinstance(latest_rsi, (int, float)) else 'N/A'}
   - Interpretation: {
        'Oversold' if isinstance(latest_rsi, (int, float)) and latest_rsi < 30
        else 'Overbought' if isinstance(latest_rsi, (int, float)) and latest_rsi > 70
        else 'Neutral' if isinstance(latest_rsi, (int, float))
        else 'N/A'
    }

2. MACD:
   - MACD Line: {latest_macd.get('macd', 'N/A'):.2f if isinstance(latest_macd.get('macd'), (int, float)) else 'N/A'}
   - Signal Line: {latest_macd.get('signal', 'N/A'):.2f if isinstance(latest_macd.get('signal'), (int, float)) else 'N/A'}
   - Histogram: {latest_macd.get('histogram', 'N/A'):.2f if isinstance(latest_macd.get('histogram'), (int, float)) else 'N/A'}
   - Signal: {
        'Bullish' if isinstance(latest_macd.get('histogram'), (int, float)) and latest_macd.get('histogram') > 0
        else 'Bearish' if isinstance(latest_macd.get('histogram'), (int, float)) and latest_macd.get('histogram') < 0
        else 'Neutral' if isinstance(latest_macd.get('histogram'), (int, float))
        else 'N/A'
    }

3. Key Levels:
   Support Levels: {', '.join(f'${level:.2f}' for level in support_resistance.get('support', []))}
   Resistance Levels: {', '.join(f'${level:.2f}' for level in support_resistance.get('resistance', []))}

4. Volume Profile Analysis:
{chr(10).join(f'   ${price_range}: {volume:,.0f}' for price_range, volume in volume_profile.items())}
"""
        
        # Prepare comprehensive market context
        market_context = f"""
Market Data Summary for {symbol}:
Latest Price: ${latest_price if latest_price != 'N/A' else 'N/A'}
24h Volume: {volume_24h:,.0f}
Price Range: ${price_low:,.2f} - ${price_high:,.2f}

{technical_summary}

Recent News:
{news_summary}
"""
        
        # Analyze market structure
        analysis = await llm_tool.analyze_market_structure(
            symbol=symbol,
            timeframes=timeframes,
            analysis_type="comprehensive"
        )
        
        # Ensure analysis is a string
        analysis_str = str(analysis) if analysis is not None else "No analysis available"
        
        # Combine all analysis
        enhanced_analysis = f"""
{market_context}

LLM Analysis:
{analysis_str}

Charts have been saved to: {technical_analysis.get('charts_path', 'N/A')}
"""
        
        logger.info(f"Completed market structure analysis for {symbol}")
        return {"market_analysis": enhanced_analysis}
    except Exception as e:
        logger.error(f"Error in market analysis: {str(e)}")
        return {"market_analysis": f"Error: {str(e)}"}

@Nodes.define(output=None)
async def generate_trading_strategy(
    model: str,
    symbol: str,
    market_analysis: str,
    risk_profile: str = "moderate"
) -> dict:
    """Generate trading strategy using LLM."""
    try:
        decision_tool = TradingDecisionTool(model_name=model)
        strategy = await decision_tool.get_trade_decision(
            symbol=symbol,
            market_context=market_analysis,
            risk_profile=risk_profile
        )
        
        # Ensure strategy is a string
        strategy_str = str(strategy) if strategy is not None else "No strategy available"
        
        logger.info(f"Generated trading strategy for {symbol}")
        return {"trading_strategy": strategy_str}
    except Exception as e:
        logger.error(f"Error generating trading strategy: {str(e)}")
        return {"trading_strategy": f"Error: {str(e)}"}

@Nodes.define(output=None)
async def save_analysis(
    symbol: str,
    market_data: dict,
    news_data: dict,
    market_analysis: str,
    trading_strategy: str,
    output_dir: str
) -> None:
    """Save analysis results to files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        market_data_file = os.path.join(output_dir, f"{symbol}_{timestamp}_market_data.json")
        with open(market_data_file, 'w') as f:
            f.write(str(market_data))
        
        news_data_file = os.path.join(output_dir, f"{symbol}_{timestamp}_news.json")
        with open(news_data_file, 'w') as f:
            f.write(str(news_data))
        
        analysis_file = os.path.join(output_dir, f"{symbol}_{timestamp}_analysis.txt")
        with open(analysis_file, 'w') as f:
            f.write(f"Market Analysis:\n{market_analysis}\n\nTrading Strategy:\n{trading_strategy}")
        
        logger.info(f"Saved analysis results to {output_dir}")
    except Exception as e:
        logger.error(f"Error saving analysis: {str(e)}")

# Define error handling at the node execution level
@Nodes.define(output=None)
async def handle_node_error(error: Exception, context: dict) -> dict:
    """Generic error handler for workflow nodes."""
    node_name = context.get("current_node", "unknown")
    logger.error(f"Error in node {node_name}: {str(error)}")
    
    # Return appropriate fallback data based on node
    if node_name == "fetch_news_sentiment":
        return {"news_data": {"error": str(error), "articles": []}}
    elif node_name == "analyze_market_structure":
        return {"market_analysis": f"Error: {str(error)}"}
    elif node_name == "generate_trading_strategy":
        return {"trading_strategy": f"Error: {str(error)}"}
    else:
        return {"error": str(error)}

# Define the Workflow with error handling through node definitions
workflow = (
    Workflow("fetch_market_data")
    .add_observer(trading_progress_observer)
    .then("analyze_technical_indicators")
    .then("fetch_news_sentiment")
    .then("analyze_market_structure")
    .then("generate_trading_strategy")
    .then("save_analysis")
)

async def analyze_trading(
    symbol: str,
    model: str = "gemini/gemini-2.0-flash",
    interval: str = "1h",
    range_type: str = "week",
    risk_profile: str = "moderate",
    output_dir: str = "trading_analysis",
) -> None:
    """Generate comprehensive trading analysis and strategy."""
    try:
        initial_context = {
            "symbol": symbol,
            "model": model,
            "interval": interval,
            "range_type": range_type,
            "risk_profile": risk_profile,
            "output_dir": output_dir,
        }
        logger.info(f"Starting trading analysis for {symbol}")
        engine = workflow.build()
        result = await engine.run(initial_context)
        logger.info("Trading analysis completed successfully 🎉")
        return result
    except Exception as e:
        logger.error(f"Error in trading analysis: {str(e)}")
        raise

async def main():
    """Main entry point for trading analysis."""
    try:
        symbols = ["BTC-USD"]  # Add more symbols as needed
        for symbol in symbols:
            try:
                await analyze_trading(
                    symbol=symbol,
                    interval="1h",
                    range_type="week",
                    risk_profile="moderate"
                )
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")

if __name__ == "__main__":
    # Create and set a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
