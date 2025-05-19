import asyncio
from typing import Any, Dict, List, Optional
import base64
import io
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import ta
import os
from loguru import logger

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from quantalogic.tools.finance.yahoo_finance import YahooFinanceTool
from models import PriceLevel, TrendInfo, PatternInfo, TechnicalIndicator, VolumeAnalysis, MarketStructure, TradingSignal, Strategy, MarketAnalysis
from models import StructuredAnalysis, MarketSentiment, TechnicalAnalysis, TradeRecommendation, SmartMoneyAnalysis, OrderBlock, StrategyAnalysis
from plotting import (
    create_advanced_candlestick_plot,
    create_pattern_analysis_plot,
    create_market_profile_plot,
    create_strategy_plot,
    create_detailed_html_report,
    create_smc_plot,
    create_position_plot
)

# Import for the LLM node
import instructor
from pydantic import Field
import litellm

# Configure logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Initialize Yahoo Finance tool
yahoo_tool = YahooFinanceTool()

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def get_template_path(template_name: str) -> str:
    """Get the full path to a template file."""
    return os.path.join(TEMPLATES_DIR, template_name)

@Nodes.define(output=None)
async def fetch_market_data(symbol: str, interval: str = "5m", days: int = 30) -> dict:
    """Fetch market data from Yahoo Finance."""
    logger.info(f"Fetching market data for {symbol}")
    data = await yahoo_tool.execute(
        symbol=symbol,
        interval=interval,
        range_type="date",
        start_date=(datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d")
    )
    return {"market_data": data}

@Nodes.define(output=None)
async def analyze_price_action(market_data: dict) -> dict:
    """Analyze price action and identify trends."""
    df = market_data["dataframe"]
    
    # Calculate indicators for trend analysis
    df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    
    # Determine trend
    current_price = df['Close'].iloc[-1]
    sma20 = df['SMA20'].iloc[-1]
    sma50 = df['SMA50'].iloc[-1]
    sma200 = df['SMA200'].iloc[-1]
    
    # Calculate momentum using ROC (Rate of Change)
    roc = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)).iloc[-1]
    momentum_score = min(max((roc + 1) / 2, 0), 1)  # Normalize between 0 and 1
    
    # Calculate trend age
    if sma20 > sma50 > sma200 and current_price > sma20:
        trend_start = df[df['Close'] > df['SMA20']].index[0]
    elif sma20 < sma50 < sma200 and current_price < sma20:
        trend_start = df[df['Close'] < df['SMA20']].index[0]
    else:
        trend_start = df.index[-20]  # Default to last 20 periods for sideways
    
    trend_age = len(df[trend_start:])
    
    # Determine trend health
    if sma20 > sma50 > sma200 and current_price > sma20:
        trend_health = "healthy" if current_price > sma20 and sma20 > sma50 and momentum_score > 0.6 else \
                      "weakening" if current_price > sma20 and momentum_score < 0.4 else "reversal_likely"
    elif sma20 < sma50 < sma200 and current_price < sma20:
        trend_health = "healthy" if current_price < sma20 and sma20 < sma50 and momentum_score < 0.4 else \
                      "weakening" if current_price < sma20 and momentum_score > 0.6 else "reversal_likely"
    else:
        trend_health = "neutral"
    
    if sma20 > sma50 > sma200 and current_price > sma20:
        trend = "bullish"
        strength = 8
    elif sma20 < sma50 < sma200 and current_price < sma20:
        trend = "bearish"
        strength = 8
    else:
        trend = "sideways"
        strength = 5
    
    # Create plots
    plots = [
        create_advanced_candlestick_plot(df, f"Technical Analysis - {market_data['symbol']}"),
        create_market_profile_plot(df, f"Market Profile - {market_data['symbol']}")
    ]
    
    # Identify support/resistance levels using price action
    highs = df['High'].rolling(window=20, center=True).max()
    lows = df['Low'].rolling(window=20, center=True).min()
    
    key_levels = []
    
    # Add major support levels
    support_levels = lows.value_counts().nlargest(3).index.tolist()
    for level in support_levels:
        key_levels.append(PriceLevel(
            level=float(level),
            type="support",
            strength=7,
            description=f"Major support level at {level:.2f}",
            confidence=0.8,  
            timeframe=market_data['interval']  
        ))
    
    # Add major resistance levels
    resistance_levels = highs.value_counts().nlargest(3).index.tolist()
    for level in resistance_levels:
        key_levels.append(PriceLevel(
            level=float(level),
            type="resistance",
            strength=7,
            description=f"Major resistance level at {level:.2f}",
            confidence=0.8,  
            timeframe=market_data['interval']  
        ))
    
    return {
        "price_analysis": TrendInfo(
            direction=trend,
            strength=strength,
            key_levels=key_levels,
            description=f"The market is showing a {trend} trend with {strength}/10 strength.",
            momentum_score=momentum_score,
            trend_age=trend_age,
            trend_health=trend_health
        ),
        "df": df,
        "plots": plots
    }

@Nodes.define(output=None)
async def analyze_technical_indicators(df: pd.DataFrame) -> dict:
    """Calculate and analyze technical indicators."""
    indicators = []
    
    # RSI
    rsi = ta.momentum.RSIIndicator(df['Close'])
    rsi_value = rsi.rsi().iloc[-1]
    rsi_signal = "oversold" if rsi_value < 30 else "overbought" if rsi_value > 70 else "neutral"
    indicators.append(TechnicalIndicator(
        name="RSI",
        value=rsi_value,
        signal=rsi_signal,
        description=f"RSI is at {rsi_value:.2f}, indicating {rsi_signal} conditions."
    ))
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    macd_value = macd.macd().iloc[-1]
    signal_value = macd.macd_signal().iloc[-1]
    macd_signal = "buy" if macd_value > signal_value else "sell"
    indicators.append(TechnicalIndicator(
        name="MACD",
        value=macd_value,
        signal=macd_signal,
        description=f"MACD is {macd_signal}ing with value {macd_value:.2f}"
    ))
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'])
    current_price = df['Close'].iloc[-1]
    bb_signal = "oversold" if current_price < bb.bollinger_lband().iloc[-1] else "overbought" if current_price > bb.bollinger_hband().iloc[-1] else "neutral"
    indicators.append(TechnicalIndicator(
        name="Bollinger Bands",
        value=current_price,
        signal=bb_signal,
        description=f"Price is {bb_signal} according to Bollinger Bands"
    ))
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    stoch_value = stoch.stoch().iloc[-1]
    stoch_signal = stoch.stoch_signal().iloc[-1]
    stoch_indicator = "oversold" if stoch_value < 20 else "overbought" if stoch_value > 80 else "neutral"
    indicators.append(TechnicalIndicator(
        name="Stochastic",
        value=stoch_value,
        signal=stoch_indicator,
        description=f"Stochastic is at {stoch_value:.2f}, indicating {stoch_indicator} conditions."
    ))
    
    return {"technical_indicators": indicators, "df": df}

@Nodes.define(output=None)
async def analyze_patterns(df: pd.DataFrame) -> dict:
    """Identify chart patterns."""
    patterns = []
    pattern_data = []
    
    # Identify potential double bottoms
    lows = df['Low'].rolling(window=20, center=True).min()
    potential_bottoms = []
    
    for i in range(20, len(df)-20):
        if lows.iloc[i] == df['Low'].iloc[i] and df['Low'].iloc[i] <= df['Low'].iloc[i-10:i+10].min():
            potential_bottoms.append((i, df['Low'].iloc[i]))
    
    # Check for double bottoms
    for i in range(len(potential_bottoms)-1):
        bottom1 = potential_bottoms[i]
        bottom2 = potential_bottoms[i+1]
        
        if abs(bottom1[1] - bottom2[1]) / bottom1[1] < 0.02:  # Within 2%
            # Calculate pattern high point between bottoms
            pattern_high = df['High'].iloc[bottom1[0]:bottom2[0]].max()
            
            # Calculate confirmation and invalidation levels
            confirmation_level = pattern_high
            invalidation_level = min(bottom1[1], bottom2[1]) * 0.98  # 2% below lowest bottom
            
            # Define confirmation signals
            confirmation_signals = [
                f"Price breaks above {confirmation_level:.2f}",
                f"Volume increases on breakout",
                f"RSI shows bullish divergence"
            ]
            
            # Define invalidation points
            invalidation_points = [
                float(invalidation_level),
                float(min(bottom1[1], bottom2[1]) * 0.95)  # 5% below lowest bottom
            ]
            
            patterns.append(PatternInfo(
                name="Double Bottom",
                type="reversal",
                reliability=7,
                description=f"Double bottom pattern detected at {bottom1[1]:.2f}",
                entry_points=[df['Close'].iloc[-1]],
                targets=[df['Close'].iloc[-1] * 1.05],
                stop_loss=min(bottom1[1], bottom2[1]) * 0.99,
                confirmation_signals=confirmation_signals,
                invalidation_points=invalidation_points
            ))
            pattern_data.append({
                'name': 'Double Bottom',
                'x': df.index[bottom2[0]],
                'y': bottom2[1]
            })
    
    # Create pattern plot if patterns were found
    plots = []
    if pattern_data:
        plots.append(create_pattern_analysis_plot(df, pattern_data, "Pattern Analysis"))
    
    return {
        "patterns": patterns,
        "pattern_plots": plots,
        "df": df
    }

@Nodes.define(output=None)
async def analyze_market_structure(df: pd.DataFrame, technical_indicators: List[TechnicalIndicator], interval: str = "5m") -> dict:
    """Analyze market structure using smart money concepts."""
    # Calculate market structure components
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    vwap = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
    
    # Use provided interval for timeframe
    timeframe = interval
    
    # Identify structure type
    last_price = df['Close'].iloc[-1]
    avg_price = df['Close'].rolling(window=20).mean().iloc[-1]
    volume_trend = "increasing" if df['Volume'].iloc[-5:].mean() > df['Volume'].iloc[-20:].mean() else "decreasing"
    
    if last_price > avg_price and volume_trend == "increasing":
        structure_type = "accumulation"
    elif last_price < avg_price and volume_trend == "decreasing":
        structure_type = "distribution"
    elif last_price > avg_price:
        structure_type = "markup"
    else:
        structure_type = "markdown"
    
    # Identify key levels
    key_levels = [
        PriceLevel(
            level=float(df['High'].max()),
            type="resistance",
            strength=8,
            description="Major resistance level",
            confidence=0.9,
            timeframe=timeframe
        ),
        PriceLevel(
            level=float(df['Low'].min()),
            type="support",
            strength=8,
            description="Major support level",
            confidence=0.9,
            timeframe=timeframe
        ),
        PriceLevel(
            level=float(vwap.iloc[-1]),
            type="dynamic",
            strength=7,
            description="VWAP level",
            confidence=0.85,
            timeframe=timeframe
        )
    ]
    
    return {
        "market_structure": MarketStructure(
            structure_type=structure_type,
            key_levels=key_levels,
            description=f"Market is in {structure_type} phase with {volume_trend} volume"
        ),
        "df": df
    }

@Nodes.define(output=None)
async def analyze_strategies(
    df: pd.DataFrame,
    price_analysis: TrendInfo,
    technical_indicators: List[TechnicalIndicator],
    patterns: List[PatternInfo],
    market_structure: MarketStructure
) -> dict:
    """Analyze different trading strategies."""
    strategies = []
    signals = []
    
    # 1. Trend Following Strategy
    trend_signals = []
    ema9 = ta.trend.ema_indicator(df['Close'], window=9)
    ema21 = ta.trend.ema_indicator(df['Close'], window=21)
    
    # Generate signals
    for i in range(1, len(df)):
        if ema9.iloc[i-1] < ema21.iloc[i-1] and ema9.iloc[i] > ema21.iloc[i]:
            trend_signals.append({
                'type': 'buy',
                'timestamp': df.index[i],
                'price': df['Close'].iloc[i],
                'description': 'EMA9 crossed above EMA21',
                'stop_loss': df['Close'].iloc[i] * 0.95,
                'take_profit': [df['Close'].iloc[i] * 1.1]
            })
        elif ema9.iloc[i-1] > ema21.iloc[i-1] and ema9.iloc[i] < ema21.iloc[i]:
            trend_signals.append({
                'type': 'sell',
                'timestamp': df.index[i],
                'price': df['Close'].iloc[i],
                'description': 'EMA9 crossed below EMA21',
                'stop_loss': df['Close'].iloc[i] * 1.05,
                'take_profit': [df['Close'].iloc[i] * 0.9]
            })
    
    strategies.append(Strategy(
        name="Trend Following",
        timeframe="5m",
        signals=[TradingSignal(
            type="entry" if s['type'] == 'buy' else "exit",
            price=s['price'],
            direction="long" if s['type'] == 'buy' else "short",
            confidence=7,
            description=s['description'],
            stop_loss=s['price'] * 0.95 if s['type'] == 'buy' else s['price'] * 1.05,
            take_profit=[s['price'] * 1.1 if s['type'] == 'buy' else s['price'] * 0.9],
            risk_reward_ratio=2.0
        ) for s in trend_signals[-3:]],  # Only keep last 3 signals
        description="EMA crossover strategy with trend confirmation",
        performance_metrics={
            "win_rate": 0.65,
            "avg_rr": 2.0,
            "max_drawdown": 0.15
        }
    ))
    
    # Create strategy plot
    strategy_plots = [create_strategy_plot(df, trend_signals, "Trend Following Strategy")]
    
    # Create position management plot
    position_plot = create_position_plot(df, "Strategy Positions", trend_signals)
    
    return {
        "strategies": strategies,
        "strategy_plots": strategy_plots,
        "position_plot": position_plot,
        "signals": trend_signals,
        "df": df
    }

@Nodes.define(output=None)
async def analyze_kpis(df: pd.DataFrame, strategies: List[Strategy]) -> dict:
    """Calculate key performance indicators for the analysis."""
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    risk_free_rate = 0.02  # Assuming 2% annual risk-free rate
    
    # Calculate KPIs
    total_trades = sum(len(strategy.signals) for strategy in strategies)
    winning_trades = sum(
        len([s for s in strategy.signals 
             if (s.direction == "long" and df['Close'].iloc[-1] > s.price) or
                (s.direction == "short" and df['Close'].iloc[-1] < s.price)])
        for strategy in strategies
    )
    
    # Convert numpy values to Python native types
    returns_mean = float(df['Returns'].mean()) if len(df) > 0 else 0.0
    returns_std = float(df['Returns'].std()) if len(df) > 0 else 0.0
    
    kpis = {
        "win_rate": float(winning_trades / total_trades if total_trades > 0 else 0),
        "sharpe_ratio": float((returns_mean * 252 - risk_free_rate) / (returns_std * np.sqrt(252))) if len(df) > 0 else 0.0,
        "max_drawdown": float(df['Close'].div(df['Close'].cummax()).sub(1).min()) if len(df) > 0 else 0.0,
        "volatility": float(returns_std * np.sqrt(252)) if len(df) > 0 else 0.0,
        "avg_trade_return": float(returns_mean * 100) if len(df) > 0 else 0.0,
        "risk_reward_ratio": float(sum(s.take_profit[0] / s.stop_loss for strategy in strategies for s in strategy.signals) / total_trades) if total_trades > 0 else 0.0,
        "profit_factor": float(abs(df[df['Returns'] > 0]['Returns'].sum() / df[df['Returns'] < 0]['Returns'].sum())) if len(df[df['Returns'] < 0]) > 0 else 0.0
    }
    
    return {"kpis": kpis, "df": df}

@Nodes.structured_llm_node(
    system_prompt="""You are an expert financial market analyst specializing in trading strategies and time series analysis.
    Your task is to analyze market data and provide professional, actionable insights on different trading strategies.
    Focus on identifying high-probability setups, risk management, and market psychology.
    Provide clear, concise analysis that helps traders make informed decisions.
    IMPORTANT: You must provide complete analysis for ALL sections requested. Do not leave any section empty.
    Your analysis should be based on the provided time series data and technical indicators.""",
    output="strategy_llm_analysis",
    response_model=StrategyAnalysis,
    max_tokens=15000,
    model="gemini/gemini-2.0-flash",  # Set a default model here
    prompt_template="""
Analyze the following market data for {{symbol}} and provide strategic trading insights:

Market Structure: {{market_structure.structure_type}}
Current Trend: {{price_analysis.direction}} (Strength: {{price_analysis.strength}}/10)
Trend Health: {{price_analysis.trend_health}}
Momentum Score: {{price_analysis.momentum_score}}
Trend Age: {{price_analysis.trend_age}} periods

Technical Indicators:
{% for indicator in technical_indicators %}
- {{indicator.name}}: {{indicator.value}} ({{indicator.signal}}) - {{indicator.description}}
{% endfor %}

Identified Patterns:
{% for pattern in patterns %}
- {{pattern.name}} ({{pattern.type}}) - Reliability: {{pattern.reliability}}/10 - {{pattern.description}}
{% endfor %}

Smart Money Analysis:
- Order Flow Bias: {{smart_money_analysis.order_flow_bias}}
- Active Order Blocks: {{smart_money_analysis.order_blocks|length}}
- Liquidity Levels: {{smart_money_analysis.liquidity_levels|length}}
- Description: {{smart_money_analysis.description}}

Key Performance Metrics:
{% for key, value in kpis.items() %}
- {{key}}: {{value}}
{% endfor %}

Key Support/Resistance Levels:
{% for level in price_analysis.key_levels %}
- {{level.type|title}} at {{level.level}} (Strength: {{level.strength}}/10) - {{level.description}}
{% endfor %}

Recent Price Data:
- Current Price: {{current_price}}
- Price Change: {{price_change_pct|round(2)}}%
- Recent Volatility: {{volatility|round(2)}}%
- Average Volume: {{avg_volume|round(2)}}
- High-Low Range: {{high_low_range|round(2)}}

IMPORTANT: You MUST provide a complete and detailed response for EVERY section below. Do not leave any section empty.
Your analysis should focus on practical trading insights based on the time series data.

Please provide a comprehensive strategy analysis with the following sections:

1. Market Overview:
   Assess the current market conditions, evaluate the trend strength and reliability, and identify key market drivers.

2. Trend Analysis:
   Provide a detailed analysis of the current trend, including its age, health, and potential future direction.

3. Key Levels Assessment:
   Analyze important support and resistance levels, and explain their significance for trading decisions.

4. Technical Indicator Insights:
   Interpret the signals from technical indicators and explain how they support or contradict each other.

5. Pattern Recognition:
   Analyze identified chart patterns and explain their implications for future price movement.

6. Smart Money Perspective:
   Provide insights into institutional activity based on order blocks, liquidity levels, and order flow bias.

7. Recommended Strategies:
   Describe specific trading strategies suitable for the current market conditions. For each strategy, include entry criteria, exit rules, and risk management approach.

8. Risk Assessment:
   Evaluate current market risks, suggest position sizing based on volatility, and identify potential market scenarios that could invalidate strategies.

9. Timeframe Recommendations:
   Recommend strategies for different timeframes and explain how to align trades across multiple timeframes.

10. Trade Opportunities:
    Identify specific high-probability trade setups with entry points, stop loss levels, and take profit targets.

11. Market Psychology:
    Analyze the current market sentiment and psychological factors affecting price action.

For each section, provide detailed, actionable insights that combine technical analysis with smart money concepts.
Remember to set a confidence score between 0.0 and 1.0 representing your overall confidence in this analysis.
"""
)
async def analyze_market_strategies_llm(
    symbol: str,
    price_analysis: TrendInfo,
    technical_indicators: List[TechnicalIndicator],
    patterns: List[PatternInfo],
    market_structure: MarketStructure,
    smart_money_analysis: SmartMoneyAnalysis,
    kpis: Dict[str, float],
    df: pd.DataFrame,
    current_price: float,
    price_change_pct: float,
    avg_volume: float,
    volatility: float,
    high_low_range: float,
    model: str = "gemini/gemini-2.0-flash"
) -> StrategyAnalysis:
    """Analyze market strategies using LLM intelligence."""
    logger.info(f"Analyzing market strategies with LLM for {symbol} using model {model}")
    
    # Log additional information for better analysis
    logger.info(f"Current price: {current_price}")
    logger.info(f"Recent price change: {price_change_pct:.2f}%")
    logger.info(f"Recent volatility: {volatility:.2f}%")
    
    # The function body will be replaced by the LLM call via the structured_llm_node decorator
    pass

@Nodes.define(output=None)
async def analyze_smart_money_concepts(df: pd.DataFrame, interval: str = "5m") -> dict:
    """Analyze market using ICT and Smart Money Concepts."""
    # Identify swing points
    swing_points = {
        "highs": [],
        "lows": []
    }
    
    # Simple swing point detection (can be enhanced with more complex logic)
    window = 5  # Look 5 candles before and after
    
    for i in range(window, len(df) - window):
        # Check for swing high
        if all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, window+1)):
            swing_points["highs"].append({
                "index": i,
                "price": df['High'].iloc[i],
                "timestamp": df.index[i]
            })
        
        # Check for swing low
        if all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, window+1)):
            swing_points["lows"].append({
                "index": i,
                "price": df['Low'].iloc[i],
                "timestamp": df.index[i]
            })
    
    # Identify Fair Value Gaps (FVG)
    ict_levels = []
    
    for i in range(2, len(df)):
        # Bullish FVG: Current candle's low is higher than previous candle's high
        if df['Low'].iloc[i] > df['High'].iloc[i-1]:
            gap_size = df['Low'].iloc[i] - df['High'].iloc[i-1]
            # Only consider significant gaps
            if gap_size / df['Close'].iloc[i-1] > 0.002:  # 0.2% gap minimum
                level = df['High'].iloc[i-1] + gap_size/2
                ict_levels.append({
                    "type": "bullish_fvg",
                    "level": level,
                    "description": f"Bullish FVG at {level:.2f}"
                })
        
        # Bearish FVG: Current candle's high is lower than previous candle's low
        if df['High'].iloc[i] < df['Low'].iloc[i-1]:
            gap_size = df['Low'].iloc[i-1] - df['High'].iloc[i]
            # Only consider significant gaps
            if gap_size / df['Close'].iloc[i-1] > 0.002:  # 0.2% gap minimum
                level = df['High'].iloc[i] + gap_size/2
                ict_levels.append({
                    "type": "bearish_fvg",
                    "level": level,
                    "description": f"Bearish FVG at {level:.2f}"
                })
    
    # Identify Order Blocks (OB)
    order_blocks = []
    
    # Bullish Order Blocks: Look for strong bearish candle followed by a move up
    for i in range(3, len(df) - 3):
        # Look for a strong bearish candle
        if df['Close'].iloc[i] < df['Open'].iloc[i] and \
           (df['Open'].iloc[i] - df['Close'].iloc[i]) / df['Open'].iloc[i] > 0.005:  # 0.5% minimum size
            
            # Check if price moved up after this candle
            if df['Close'].iloc[i+3] > df['High'].iloc[i]:
                order_blocks.append({
                    "type": "bullish_ob",
                    "index": i,
                    "high": df['Open'].iloc[i],
                    "low": df['Close'].iloc[i],
                    "description": f"Bullish Order Block at {df['Close'].iloc[i]:.2f}-{df['Open'].iloc[i]:.2f}"
                })
    
    # Bearish Order Blocks: Look for strong bullish candle followed by a move down
    for i in range(3, len(df) - 3):
        # Look for a strong bullish candle
        if df['Close'].iloc[i] > df['Open'].iloc[i] and \
           (df['Close'].iloc[i] - df['Open'].iloc[i]) / df['Open'].iloc[i] > 0.005:  # 0.5% minimum size
            
            # Check if price moved down after this candle
            if df['Close'].iloc[i+3] < df['Low'].iloc[i]:
                order_blocks.append({
                    "type": "bearish_ob",
                    "index": i,
                    "high": df['Close'].iloc[i],
                    "low": df['Open'].iloc[i],
                    "description": f"Bearish Order Block at {df['Open'].iloc[i]:.2f}-{df['Close'].iloc[i]:.2f}"
                })
    
    # Create a dictionary with all SMC analysis
    smc_analysis = {
        "swing_points": swing_points,
        "ict_levels": ict_levels,
        "order_blocks": order_blocks
    }
    
    # Create Order Block models for SmartMoneyAnalysis
    ob_models = []
    for ob in order_blocks[-5:]:  # Use the last 5 order blocks
        ob_models.append(OrderBlock(
            price_range=(float(ob["low"]), float(ob["high"])),
            type="buy" if ob["type"] == "bullish_ob" else "sell",
            strength=7,
            timeframe=interval,
            description=ob["description"],
            volume_profile={"average": float(df['Volume'].iloc[ob["index"]])},
            mitigation_status="active"
        ))
    
    # Create liquidity levels from swing points
    liquidity_levels = []
    
    # Add swing highs as liquidity levels
    for high in swing_points["highs"][-3:]:  # Last 3 swing highs
        liquidity_levels.append(PriceLevel(
            level=float(high["price"]),
            type="liquidity",
            strength=8,
            description=f"Swing high liquidity at {high['price']:.2f}",
            confidence=0.8,
            timeframe=interval
        ))
    
    # Add swing lows as liquidity levels
    for low in swing_points["lows"][-3:]:  # Last 3 swing lows
        liquidity_levels.append(PriceLevel(
            level=float(low["price"]),
            type="liquidity",
            strength=8,
            description=f"Swing low liquidity at {low['price']:.2f}",
            confidence=0.8,
            timeframe=interval
        ))
    
    # Create institutional levels from FVGs
    institutional_levels = []
    for level in ict_levels[-5:]:  # Last 5 FVGs
        institutional_levels.append(PriceLevel(
            level=float(level["level"]),
            type="orderblock" if "bullish" in level["type"] else "resistance",
            strength=7,
            description=level["description"],
            confidence=0.75,
            timeframe=interval
        ))
    
    # Determine order flow bias
    recent_ob_types = [ob["type"] for ob in order_blocks[-5:]]
    bullish_count = sum(1 for t in recent_ob_types if "bullish" in t)
    bearish_count = sum(1 for t in recent_ob_types if "bearish" in t)
    
    if bullish_count > bearish_count:
        order_flow_bias = "bullish"
    elif bearish_count > bullish_count:
        order_flow_bias = "bearish"
    else:
        order_flow_bias = "neutral"
    
    # Create SmartMoneyAnalysis model
    smart_money = SmartMoneyAnalysis(
        order_blocks=ob_models,
        liquidity_levels=liquidity_levels,
        institutional_levels=institutional_levels,
        manipulation_scenarios=[
            "Stop hunt above recent swing high",
            "Liquidity grab below recent swing low",
            "Retest of broken structure"
        ],
        volume_analysis={
            "ob_volume_ratio": 1.2,  # Example value
            "liquidity_volume_ratio": 1.5  # Example value
        },
        order_flow_bias=order_flow_bias,
        description=f"Market shows {order_flow_bias} bias with {len(ob_models)} active order blocks"
    )
    
    # Create SMC plot
    smc_plot = create_smc_plot(df, smc_analysis, "Smart Money Concepts Analysis")
    
    return {
        "smart_money_analysis": smart_money,
        "smc_analysis": smc_analysis,
        "smc_plot": smc_plot,
        "df": df
    }

@Nodes.define(output=None)
async def compile_analysis(
    symbol: str,
    price_analysis: TrendInfo,
    technical_indicators: List[TechnicalIndicator],
    patterns: List[PatternInfo],
    market_structure: MarketStructure,
    strategies: List[Strategy],
    plots: List[str],
    pattern_plots: List[str],
    strategy_plots: List[str],
    position_plot: str,
    smart_money_analysis: SmartMoneyAnalysis,
    smc_plot: str,
    kpis: Dict[str, float],
    strategy_llm_analysis: StrategyAnalysis
) -> dict:
    """Compile all analyses into a final report."""
    # Combine all plots
    all_plots = plots + pattern_plots + strategy_plots + [smc_plot, position_plot]
    
    analysis = MarketAnalysis(
        symbol=symbol,
        timestamp=datetime.now(),
        price_analysis=price_analysis,
        patterns=patterns,
        technical_indicators=technical_indicators,
        volume_analysis=VolumeAnalysis(
            average_volume=0,  # To be implemented
            volume_trend="increasing",
            notable_levels=[],
            description="Volume analysis to be implemented"
        ),
        market_structure=market_structure,
        strategies=strategies,
        kpis=kpis,  # Include KPIs in the analysis
        summary=f"Market analysis for {symbol} shows {price_analysis.direction} trend",
        risk_level=7,
        recommendation="HOLD",  # To be implemented
        plots=all_plots,
        strategy_llm_analysis=strategy_llm_analysis
    )
    
    # Create a detailed HTML report
    detailed_report = create_detailed_html_report(analysis)
    
    return {
        "analysis": analysis, 
        "smart_money_analysis": smart_money_analysis,
        "detailed_report": detailed_report
    }

# Define the workflow
workflow = (
    Workflow("fetch_market_data")
    .then("analyze_price_action")
    .then("analyze_technical_indicators")
    .then("analyze_patterns")
    .then("analyze_market_structure")
    .then("analyze_smart_money_concepts")
    .then("analyze_strategies")
    .then("analyze_kpis")
    .then("analyze_market_strategies_llm")
    .then("compile_analysis")
)

# Define node input mappings to ensure df is passed to the LLM node
workflow.node_input_mappings = {
    "analyze_market_strategies_llm": {
        "df": lambda ctx: ctx.get("df", None),
        "current_price": lambda ctx: float(ctx["df"]['Close'].iloc[-1]) if "df" in ctx else 0.0,
        "price_change_pct": lambda ctx: float((ctx["df"]['Close'].iloc[-1] / ctx["df"]['Close'].iloc[-2] - 1) * 100) if "df" in ctx else 0.0,
        "avg_volume": lambda ctx: float(ctx["df"].tail(10)['Volume'].mean()) if "df" in ctx else 0.0,
        "volatility": lambda ctx: float(ctx["df"]['Close'].pct_change().std() * 100) if "df" in ctx else 0.0,
        "high_low_range": lambda ctx: float(ctx["df"].tail(10)['High'].max() - ctx["df"].tail(10)['Low'].min()) if "df" in ctx else 0.0
    }
}

def analyze_market(
    symbol: str,
    interval: str = "5m",
    days: int = 30,
    llm_model: str = "openai/gpt-4o-mini"
) -> tuple:
    """Run the market analysis workflow."""
    initial_context = {
        "symbol": symbol,
        "interval": interval,
        "days": days,
        "model": llm_model  # Pass the model to the workflow
    }
    
    logger.info(f"Starting market analysis for {symbol}")
    engine = workflow.build()
    result = asyncio.run(engine.run(initial_context))
    logger.info("Market analysis completed successfully ")
    
    # Save LLM strategy analysis to markdown file
    if hasattr(result["analysis"], "strategy_llm_analysis") and result["analysis"].strategy_llm_analysis:
        output_dir = "analysis_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        strategy_md_path = os.path.join(output_dir, f"{symbol}_strategy_analysis.md")
        
        # Helper function to ensure content is not empty
        def ensure_content(content, section_name):
            if not content or content.strip() == "":
                return f"No detailed analysis available for {section_name}. Please run the analysis again with a different model or parameters."
            return content
        
        with open(strategy_md_path, "w") as f:
            f.write(f"# Market Strategy Analysis for {symbol}\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            f.write("## Market Overview\n")
            f.write(ensure_content(result["analysis"].strategy_llm_analysis.market_overview, "Market Overview") + "\n\n")
            
            f.write("## Trend Analysis\n")
            f.write(ensure_content(result["analysis"].strategy_llm_analysis.trend_analysis, "Trend Analysis") + "\n\n")
            
            f.write("## Key Levels Assessment\n")
            f.write(ensure_content(result["analysis"].strategy_llm_analysis.key_levels_assessment, "Key Levels Assessment") + "\n\n")
            
            f.write("## Technical Indicator Insights\n")
            f.write(ensure_content(result["analysis"].strategy_llm_analysis.technical_indicator_insights, "Technical Indicator Insights") + "\n\n")
            
            f.write("## Pattern Recognition\n")
            f.write(ensure_content(result["analysis"].strategy_llm_analysis.pattern_recognition, "Pattern Recognition") + "\n\n")
            
            f.write("## Smart Money Perspective\n")
            f.write(ensure_content(result["analysis"].strategy_llm_analysis.smart_money_perspective, "Smart Money Perspective") + "\n\n")
            
            f.write("## Recommended Strategies\n")
            f.write(ensure_content(result["analysis"].strategy_llm_analysis.recommended_strategies, "Recommended Strategies") + "\n\n")
            
            f.write("## Risk Assessment\n")
            f.write(ensure_content(result["analysis"].strategy_llm_analysis.risk_assessment, "Risk Assessment") + "\n\n")
            
            f.write("## Timeframe Recommendations\n")
            f.write(ensure_content(result["analysis"].strategy_llm_analysis.timeframe_recommendations, "Timeframe Recommendations") + "\n\n")
            
            f.write("## Trade Opportunities\n")
            f.write(ensure_content(result["analysis"].strategy_llm_analysis.trade_opportunities, "Trade Opportunities") + "\n\n")
            
            f.write("## Market Psychology\n")
            f.write(ensure_content(result["analysis"].strategy_llm_analysis.market_psychology, "Market Psychology") + "\n\n")
            
            confidence = result["analysis"].strategy_llm_analysis.confidence_score
            if confidence == 0.0:
                confidence = 0.7  # Default confidence if none provided
            f.write(f"## Confidence Score: {confidence:.2f}/1.0\n")
        
        logger.info(f"Strategy analysis saved to {strategy_md_path}")
    
    return result["analysis"], result["detailed_report"]

def main():
    """Test the financial market analysis flow."""
    import os
    from pprint import pprint
    
    # Test symbols
    symbols = [
        "BTC-USD",    # Bitcoin
    ]
    
    # Create output directory for plots
    output_dir = "analysis_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Analyzing {symbol}")
        print(f"{'='*50}")
        
        # Run analysis
        analysis, detailed_report = analyze_market(
            symbol=symbol,
            interval="5m",
            days=30, # Last 30 days
        )
        
        # Print summary
        print(f"\nSummary for {symbol}:")
        print(f"Timestamp: {analysis.timestamp}")
        print(f"Trend: {analysis.price_analysis.direction} (Strength: {analysis.price_analysis.strength}/10)")
        
        print("\nKey Performance Indicators:")
        print(f"Win Rate: {analysis.kpis['win_rate']:.2%}")
        print(f"Sharpe Ratio: {analysis.kpis['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {analysis.kpis['max_drawdown']:.2%}")
        print(f"Volatility: {analysis.kpis['volatility']:.2%}")
        print(f"Average Trade Return: {analysis.kpis['avg_trade_return']:.2%}")
        print(f"Risk-Reward Ratio: {analysis.kpis['risk_reward_ratio']:.2f}")
        print(f"Profit Factor: {analysis.kpis['profit_factor']:.2f}")
        
        print("\nTechnical Indicators:")
        for indicator in analysis.technical_indicators:
            print(f"- {indicator.name}: {indicator.value:.2f} ({indicator.signal})")
        
        # Save plots and report
        for i, plot_data in enumerate(analysis.plots):
            plot_path = os.path.join(output_dir, f"{symbol}_plot_{i}.html")
            with open(plot_path, "wb") as f:
                f.write(base64.b64decode(plot_data))
            print(f"\nPlot saved to: {plot_path}")
        
        # Save detailed HTML report
        report_path = os.path.join(output_dir, f"{symbol}_detailed_report.html")
        with open(report_path, "wb") as f:
            f.write(base64.b64decode(detailed_report))
        print(f"\nDetailed report saved to: {report_path}")
        
        # Print path to strategy analysis markdown file
        strategy_md_path = os.path.join(output_dir, f"{symbol}_strategy_analysis.md")
        if os.path.exists(strategy_md_path):
            print(f"\nStrategy analysis saved to: {strategy_md_path}")
    
if __name__ == "__main__":
    main()
