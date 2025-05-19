"""Market Analysis Flow for financial data.

This flow analyzes financial data using different strategies and provides
trading pattern recommendations based on the analysis.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import json

import anyio
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from quantalogic.tools.finance.yahoo_finance import YahooFinanceTool

# Configure logging
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Define structured output models
class TechnicalIndicators(BaseModel):
    """Technical indicators for market analysis."""
    moving_averages: Dict[str, Optional[float]]
    rsi: Optional[float] = None
    macd: Dict[str, Optional[float]]
    bollinger_bands: Dict[str, Optional[float]]
    support_resistance: Dict[str, List[float]]
    # Advanced indicators
    stochastic: Dict[str, Optional[float]] = Field(default_factory=dict)
    atr: Optional[float] = None
    adx: Optional[float] = None
    obv: Optional[float] = None
    volume_profile: Dict[str, List[float]] = Field(default_factory=dict)
    # Patterns
    candlestick_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    chart_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    # SMC and ICT concepts
    liquidity_levels: Dict[str, List[float]] = Field(default_factory=dict)
    order_blocks: List[Dict[str, Any]] = Field(default_factory=list)
    fair_value_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    imbalances: List[Dict[str, Any]] = Field(default_factory=list)

class FundamentalAnalysis(BaseModel):
    """Fundamental analysis results."""
    market_sentiment: str
    trend_strength: float
    volatility_assessment: str
    volume_analysis: str
    key_levels: List[float] = Field(default_factory=list)
    # Market structure
    market_structure: str  # bullish, bearish, ranging, consolidation
    market_phase: str  # accumulation, markup, distribution, markdown
    # Multi-timeframe analysis
    timeframe_alignment: Dict[str, str] = Field(default_factory=dict)  # alignment across timeframes
    # Institutional analysis
    institutional_levels: List[Dict[str, Any]] = Field(default_factory=list)
    smart_money_movements: List[Dict[str, Any]] = Field(default_factory=list)
    # Market context
    news_impact: Dict[str, Any] = Field(default_factory=dict)
    seasonal_patterns: List[Dict[str, Any]] = Field(default_factory=dict)
    correlated_assets: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

class TradingRecommendation(BaseModel):
    """Trading recommendation based on analysis."""
    action: str  # buy, sell, hold
    confidence: float
    timeframe: str
    entry_points: List[float]
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    reasoning: str
    # Enhanced trading details
    entry_type: str = "market"  # market, limit, stop
    position_sizing: Dict[str, Any] = Field(default_factory=dict)
    # Multiple timeframe strategies
    timeframe_specific_strategies: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Trade management
    exit_strategies: List[Dict[str, Any]] = Field(default_factory=list)
    trade_management: Dict[str, Any] = Field(default_factory=dict)
    # Advanced patterns
    pattern_based_entries: List[Dict[str, Any]] = Field(default_factory=list)
    # Day trading specifics
    intraday_levels: List[Dict[str, Any]] = Field(default_factory=list)
    session_analysis: Dict[str, Any] = Field(default_factory=dict)

class MarketAnalysisResult(BaseModel):
    """Complete market analysis result."""
    symbol: str
    timestamp: str
    price_data: Dict[str, Any]
    technical_indicators: TechnicalIndicators
    fundamental_analysis: FundamentalAnalysis
    trading_recommendation: TradingRecommendation
    strategy_performance: Dict[str, Dict[str, float]]

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Helper function to get template paths
def get_template_path(template_name):
    return os.path.join(TEMPLATES_DIR, template_name)

# Custom Observer for Workflow Events
async def market_analysis_progress_observer(event: WorkflowEvent):
    if event.event_type == WorkflowEventType.WORKFLOW_STARTED:
        print(f"\n{'='*50}\n🚀 Starting Market Analysis 🚀\n{'='*50}")
    elif event.event_type == WorkflowEventType.NODE_STARTED:
        print(f"\n🔄 [{event.node_name}] Starting...")
    elif event.event_type == WorkflowEventType.NODE_COMPLETED:
        if event.node_name == "analyze_with_strategy" and event.result is not None:
            strategy_name = event.context.get("current_strategy", "Unknown")
            print(f"✅ [{event.node_name}] Strategy '{strategy_name}' analysis completed")
        else:
            print(f"✅ [{event.node_name}] Completed")
    elif event.event_type == WorkflowEventType.WORKFLOW_COMPLETED:
        print(f"\n{'='*50}\n🎉 Market Analysis Finished 🎉\n{'='*50}")
    elif event.event_type == WorkflowEventType.TRANSITION_EVALUATED:
        logger.debug(f"Transition evaluated: {event.transition_from} -> {event.transition_to}")

# Technical Analysis Functions
def calculate_moving_averages(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate various moving averages."""
    close_prices = df['Close']
    return {
        'SMA_5': close_prices.rolling(window=5).mean().iloc[-1] if len(close_prices) >= 5 else None,
        'SMA_10': close_prices.rolling(window=10).mean().iloc[-1] if len(close_prices) >= 10 else None,
        'SMA_20': close_prices.rolling(window=20).mean().iloc[-1] if len(close_prices) >= 20 else None,
        'SMA_50': close_prices.rolling(window=50).mean().iloc[-1] if len(close_prices) >= 50 else None,
        'SMA_200': close_prices.rolling(window=200).mean().iloc[-1] if len(close_prices) >= 200 else None,
        'EMA_12': close_prices.ewm(span=12, adjust=False).mean().iloc[-1] if len(close_prices) >= 12 else None,
        'EMA_26': close_prices.ewm(span=26, adjust=False).mean().iloc[-1] if len(close_prices) >= 26 else None,
        'EMA_50': close_prices.ewm(span=50, adjust=False).mean().iloc[-1] if len(close_prices) >= 50 else None,
        'EMA_200': close_prices.ewm(span=200, adjust=False).mean().iloc[-1] if len(close_prices) >= 200 else None,
        'VWAP': calculate_vwap(df).iloc[-1] if len(df) > 0 else None
    }

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    if 'Volume' not in df.columns or len(df) == 0:
        return pd.Series([None] * len(df))
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> float:
    """Calculate Relative Strength Index."""
    if len(df) < window + 1:
        return None
    
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_macd(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    close_prices = df['Close']
    if len(close_prices) < 26:
        return {'macd': None, 'signal': None, 'histogram': None}
    
    ema_12 = close_prices.ewm(span=12, adjust=False).mean()
    ema_26 = close_prices.ewm(span=26, adjust=False).mean()
    
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line.iloc[-1],
        'signal': signal_line.iloc[-1],
        'histogram': histogram.iloc[-1]
    }

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> Dict[str, float]:
    """Calculate Bollinger Bands."""
    if len(df) < window:
        return {'upper': None, 'middle': None, 'lower': None}
    
    close_prices = df['Close']
    sma = close_prices.rolling(window=window).mean()
    std = close_prices.rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return {
        'upper': upper_band.iloc[-1],
        'middle': sma.iloc[-1],
        'lower': lower_band.iloc[-1]
    }

def find_support_resistance(df: pd.DataFrame, window: int = 10) -> Dict[str, List[float]]:
    """Find support and resistance levels."""
    if len(df) < window * 2:
        return {'support': [], 'resistance': []}
    
    # Simple implementation - using local mins and maxes
    highs = df['High'].rolling(window=window, center=True).max()
    lows = df['Low'].rolling(window=window, center=True).min()
    
    # Find potential resistance levels (local highs)
    resistance_levels = []
    for i in range(window, len(highs) - window):
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
            resistance_levels.append(highs.iloc[i])
    
    # Find potential support levels (local lows)
    support_levels = []
    for i in range(window, len(lows) - window):
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
            support_levels.append(lows.iloc[i])
    
    # Take only the most recent levels (up to 3)
    return {
        'support': sorted(support_levels[-3:]) if support_levels else [],
        'resistance': sorted(resistance_levels[-3:]) if resistance_levels else []
    }

def calculate_stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> Dict[str, float]:
    """Calculate Stochastic Oscillator."""
    if len(df) < k_window:
        return {'k': None, 'd': None}
    
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_window).mean()
    
    return {
        'k': k.iloc[-1],
        'd': d.iloc[-1]
    }

def calculate_atr(df: pd.DataFrame, window: int = 14) -> float:
    """Calculate Average True Range."""
    if len(df) < window + 1:
        return None
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    return atr.iloc[-1]

def calculate_adx(df: pd.DataFrame, window: int = 14) -> float:
    """Calculate Average Directional Index."""
    if len(df) < window * 2:
        return None
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    # Smoothed TR and DM
    atr = tr.rolling(window=window).mean()
    pos_di = 100 * (pos_dm.rolling(window=window).mean() / atr)
    neg_di = 100 * (neg_dm.rolling(window=window).mean() / atr)
    
    # ADX
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
    adx = dx.rolling(window=window).mean()
    
    return adx.iloc[-1]

def calculate_obv(df: pd.DataFrame) -> float:
    """Calculate On-Balance Volume."""
    if len(df) < 2:
        return None
    
    close = df['Close']
    volume = df['Volume']
    
    obv = pd.Series(index=df.index)
    obv.iloc[0] = 0
    
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv.iloc[-1]

def identify_candlestick_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify common candlestick patterns."""
    if len(df) < 5:
        return []
    
    patterns = []
    
    # Get OHLC data for the last few candles
    recent_df = df.tail(5)
    
    # Doji pattern
    for i in range(len(recent_df)):
        candle = recent_df.iloc[i]
        body_size = abs(candle['Open'] - candle['Close'])
        total_size = candle['High'] - candle['Low']
        
        if total_size > 0 and body_size / total_size < 0.1:
            patterns.append({
                'pattern': 'Doji',
                'position': i,
                'price': candle['Close'],
                'significance': 'moderate'
            })
    
    # Hammer pattern
    for i in range(len(recent_df)):
        candle = recent_df.iloc[i]
        body_size = abs(candle['Open'] - candle['Close'])
        lower_wick = min(candle['Open'], candle['Close']) - candle['Low']
        upper_wick = candle['High'] - max(candle['Open'], candle['Close'])
        
        if body_size > 0 and lower_wick > 2 * body_size and upper_wick < 0.1 * body_size:
            patterns.append({
                'pattern': 'Hammer',
                'position': i,
                'price': candle['Close'],
                'significance': 'high'
            })
    
    # Engulfing pattern
    for i in range(1, len(recent_df)):
        prev_candle = recent_df.iloc[i-1]
        curr_candle = recent_df.iloc[i]
        
        prev_body_low = min(prev_candle['Open'], prev_candle['Close'])
        prev_body_high = max(prev_candle['Open'], prev_candle['Close'])
        curr_body_low = min(curr_candle['Open'], curr_candle['Close'])
        curr_body_high = max(curr_candle['Open'], curr_candle['Close'])
        
        # Bullish engulfing
        if (prev_candle['Close'] < prev_candle['Open'] and  # Previous candle is bearish
            curr_candle['Close'] > curr_candle['Open'] and  # Current candle is bullish
            curr_body_low < prev_body_low and 
            curr_body_high > prev_body_high):
            patterns.append({
                'pattern': 'Bullish Engulfing',
                'position': i,
                'price': curr_candle['Close'],
                'significance': 'high'
            })
        
        # Bearish engulfing
        if (prev_candle['Close'] > prev_candle['Open'] and  # Previous candle is bullish
            curr_candle['Close'] < curr_candle['Open'] and  # Current candle is bearish
            curr_body_low < prev_body_low and 
            curr_body_high > prev_body_high):
            patterns.append({
                'pattern': 'Bearish Engulfing',
                'position': i,
                'price': curr_candle['Close'],
                'significance': 'high'
            })
    
    return patterns

def identify_chart_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify chart patterns like head and shoulders, double tops, etc."""
    if len(df) < 30:
        return []
    
    patterns = []
    
    # Get OHLC data for the last few candles
    recent_df = df.tail(30)
    
    # Simplified pattern detection
    # This is a placeholder for more sophisticated pattern recognition
    # In a real implementation, you would use more advanced algorithms
    
    # Double Top detection (very simplified)
    highs = df['High'].rolling(window=5, center=True).max()
    for i in range(10, len(highs) - 10):
        if (highs.iloc[i] > highs.iloc[i-5:i].max() and 
            highs.iloc[i] > highs.iloc[i+1:i+6].max() and
            abs(highs.iloc[i] - highs.iloc[i-10:i+10].max()) < 0.01 * highs.iloc[i]):
            
            # Look for another peak of similar height
            for j in range(i + 5, min(i + 20, len(highs) - 5)):
                if (abs(highs.iloc[j] - highs.iloc[i]) < 0.01 * highs.iloc[i] and
                    highs.iloc[j] > highs.iloc[j-5:j].max() and
                    highs.iloc[j] > highs.iloc[j+1:j+6].max()):
                    patterns.append({
                        'pattern': 'Double Top',
                        'position': j,
                        'price': df['Close'].iloc[j],
                        'significance': 'high'
                    })
                    break
    
    return patterns

def identify_smc_concepts(df: pd.DataFrame) -> Dict[str, Any]:
    """Identify Smart Money Concepts (SMC) like order blocks, fair value gaps, etc."""
    if len(df) < 10:
        return {
            'order_blocks': [],
            'fair_value_gaps': [],
            'imbalances': []
        }
    
    order_blocks = []
    fair_value_gaps = []
    imbalances = []
    
    # Order Blocks (simplified detection)
    for i in range(2, len(df) - 1):
        # Bullish order block
        if (df['Close'].iloc[i] > df['Open'].iloc[i] and  # Bullish candle
            df['Close'].iloc[i+1] < df['Open'].iloc[i+1] and  # Next candle bearish
            df['Low'].iloc[i+1] < df['Low'].iloc[i]):  # Next candle breaks the low
            
            order_blocks.append({
                'type': 'Bullish',
                'position': i,
                'price_range': [df['Low'].iloc[i], df['Open'].iloc[i]],
                'strength': 'medium'
            })
        
        # Bearish order block
        if (df['Close'].iloc[i] < df['Open'].iloc[i] and  # Bearish candle
            df['Close'].iloc[i+1] > df['Open'].iloc[i+1] and  # Next candle bullish
            df['High'].iloc[i+1] > df['High'].iloc[i]):  # Next candle breaks the high
            
            order_blocks.append({
                'type': 'Bearish',
                'position': i,
                'price_range': [df['Close'].iloc[i], df['High'].iloc[i]],
                'strength': 'medium'
            })
    
    # Fair Value Gaps (simplified detection)
    for i in range(1, len(df) - 1):
        # Bullish FVG
        if df['Low'].iloc[i+1] > df['High'].iloc[i-1]:
            fair_value_gaps.append({
                'type': 'Bullish',
                'position': i,
                'price_range': [df['High'].iloc[i-1], df['Low'].iloc[i+1]],
                'size': df['Low'].iloc[i+1] - df['High'].iloc[i-1]
            })
        
        # Bearish FVG
        if df['High'].iloc[i+1] < df['Low'].iloc[i-1]:
            fair_value_gaps.append({
                'type': 'Bearish',
                'position': i,
                'price_range': [df['Low'].iloc[i-1], df['High'].iloc[i+1]],
                'size': df['Low'].iloc[i-1] - df['High'].iloc[i+1]
            })
    
    # Imbalances (simplified detection)
    for i in range(2, len(df) - 2):
        # Supply imbalance
        if (df['Low'].iloc[i] > df['High'].iloc[i-1] and
            df['Low'].iloc[i+1] > df['High'].iloc[i-1]):
            imbalances.append({
                'type': 'Supply',
                'position': i,
                'price_range': [df['High'].iloc[i-1], df['Low'].iloc[i]],
                'significance': 'high'
            })
        
        # Demand imbalance
        if (df['High'].iloc[i] < df['Low'].iloc[i-1] and
            df['High'].iloc[i+1] < df['Low'].iloc[i-1]):
            imbalances.append({
                'type': 'Demand',
                'position': i,
                'price_range': [df['High'].iloc[i], df['Low'].iloc[i-1]],
                'significance': 'high'
            })
    
    return {
        'order_blocks': order_blocks,
        'fair_value_gaps': fair_value_gaps,
        'imbalances': imbalances
    }

def identify_ict_concepts(df: pd.DataFrame) -> Dict[str, Any]:
    """Identify Institutional Candle Theory (ICT) concepts like liquidity levels."""
    if len(df) < 20:
        return {
            'liquidity_levels': {'buy_side': [], 'sell_side': []},
            'breaker_blocks': []
        }
    
    # Liquidity levels (simplified)
    buy_side_liquidity = []
    sell_side_liquidity = []
    
    # Look for clusters of lows (buy-side liquidity)
    lows = df['Low'].values
    for i in range(5, len(lows) - 5):
        if all(lows[i] < lows[i-j] for j in range(1, 4)) and all(lows[i] < lows[i+j] for j in range(1, 4)):
            buy_side_liquidity.append(lows[i])
    
    # Look for clusters of highs (sell-side liquidity)
    highs = df['High'].values
    for i in range(5, len(highs) - 5):
        if all(highs[i] > highs[i-j] for j in range(1, 4)) and all(highs[i] > highs[i+j] for j in range(1, 4)):
            sell_side_liquidity.append(highs[i])
    
    # Breaker blocks (simplified)
    breaker_blocks = []
    for i in range(2, len(df) - 2):
        # Bullish breaker
        if (df['Close'].iloc[i] > df['Open'].iloc[i] and  # Bullish candle
            df['Low'].iloc[i+1:i+3].min() < df['Low'].iloc[i] and  # Price breaks below
            df['Close'].iloc[i+1:i+3].max() > df['High'].iloc[i]):  # Then price breaks above
            
            breaker_blocks.append({
                'type': 'Bullish Breaker',
                'position': i,
                'price_range': [df['Low'].iloc[i], df['High'].iloc[i]],
                'strength': 'high'
            })
        
        # Bearish breaker
        if (df['Close'].iloc[i] < df['Open'].iloc[i] and  # Bearish candle
            df['High'].iloc[i+1:i+3].max() > df['High'].iloc[i] and  # Price breaks above
            df['Close'].iloc[i+1:i+3].min() < df['Low'].iloc[i]):  # Then price breaks below
            
            breaker_blocks.append({
                'type': 'Bearish Breaker',
                'position': i,
                'price_range': [df['Low'].iloc[i], df['High'].iloc[i]],
                'strength': 'high'
            })
    
    return {
        'liquidity_levels': {
            'buy_side': buy_side_liquidity,
            'sell_side': sell_side_liquidity
        },
        'breaker_blocks': breaker_blocks
    }

# Workflow Nodes
@Nodes.define(output=None)
async def fetch_multi_timeframe_data(symbol: str, primary_interval: str, range_type: str, 
                                   start_date: Optional[str] = None, 
                                   end_date: Optional[str] = None) -> dict:
    """Fetch market data for multiple timeframes."""
    yahoo_tool = YahooFinanceTool()
    
    # Define timeframes to analyze
    timeframes = {
        'short_term': '5m' if primary_interval in ['1m', '2m', '5m'] else '1h',
        'medium_term': '1h' if primary_interval in ['1m', '2m', '5m', '15m', '30m'] else '1d',
        'long_term': '1d'
    }
    
    # Fetch data for each timeframe
    multi_timeframe_data = {}
    multi_timeframe_indicators = {}
    
    for tf_name, tf_interval in timeframes.items():
        try:
            data = await yahoo_tool.execute(
                symbol=symbol,
                interval=tf_interval,
                range_type=range_type,
                start_date=start_date,
                end_date=end_date
            )
            
            if "error" in data:
                logger.warning(f"Error fetching {tf_name} timeframe data: {data['error']}")
                continue
            
            multi_timeframe_data[tf_name] = data
            
            # Calculate indicators for this timeframe
            if 'dataframe' in data and len(data['dataframe']) > 0:
                indicators = {
                    "moving_averages": calculate_moving_averages(data['dataframe']),
                    "rsi": calculate_rsi(data['dataframe']),
                    "macd": calculate_macd(data['dataframe']),
                    "bollinger_bands": calculate_bollinger_bands(data['dataframe']),
                    "support_resistance": find_support_resistance(data['dataframe'])
                }
                multi_timeframe_indicators[tf_name] = indicators
            
            logger.info(f"Successfully fetched {data['data_points']} data points for {symbol} on {tf_name} timeframe")
        except Exception as e:
            logger.error(f"Error fetching {tf_name} timeframe: {str(e)}")
            multi_timeframe_data[tf_name] = {"error": str(e)}
    
    # Get primary timeframe data
    primary_data = await yahoo_tool.execute(
        symbol=symbol,
        interval=primary_interval,
        range_type=range_type,
        start_date=start_date,
        end_date=end_date
    )
    
    if "error" in primary_data:
        logger.error(f"Error fetching primary data: {primary_data['error']}")
        raise ValueError(f"Failed to fetch market data: {primary_data['error']}")
    
    logger.info(f"Successfully fetched {primary_data['data_points']} data points for {symbol} on primary timeframe")
    
    return {
        "market_data": primary_data,
        "symbol": symbol,
        "dataframe": primary_data["dataframe"],
        "multi_timeframe_data": multi_timeframe_data,
        "multi_timeframe_indicators": multi_timeframe_indicators
    }

@Nodes.define(output=None)
async def calculate_technical_indicators(dataframe: pd.DataFrame) -> dict:
    """Calculate technical indicators from market data."""
    if len(dataframe) < 5:
        logger.warning("Not enough data points for technical analysis")
        return {"technical_indicators": None}
    
    indicators = {
        "moving_averages": calculate_moving_averages(dataframe),
        "rsi": calculate_rsi(dataframe),
        "macd": calculate_macd(dataframe),
        "bollinger_bands": calculate_bollinger_bands(dataframe),
        "support_resistance": find_support_resistance(dataframe),
        "stochastic": calculate_stochastic(dataframe),
        "atr": calculate_atr(dataframe),
        "adx": calculate_adx(dataframe),
        "obv": calculate_obv(dataframe),
        "candlestick_patterns": identify_candlestick_patterns(dataframe),
        "chart_patterns": identify_chart_patterns(dataframe),
        "smc_concepts": identify_smc_concepts(dataframe),
        "ict_concepts": identify_ict_concepts(dataframe)
    }
    
    logger.info("Technical indicators calculated successfully")
    return {"technical_indicators": indicators}

@Nodes.llm_node(
    prompt_template=get_template_path("fundamental_analysis_prompt.txt"),
    output="fundamental_analysis",
    temperature=0.7,
)
async def analyze_fundamentals(model: str, market_data: Dict[str, Any], technical_indicators: Dict[str, Any]) -> str:
    """Analyze market fundamentals using LLM."""
    logger.debug("Analyzing market fundamentals")
    pass

@Nodes.llm_node(
    prompt_template=get_template_path("market_narrative_prompt.txt"),
    output="market_narrative",
    temperature=0.3,
)
async def analyze_market_narrative(
    model: str,
    symbol: str,
    market_data: Dict[str, Any],
    technical_indicators: Dict[str, Any],
    fundamental_analysis: str
) -> str:
    """
    Analyze the market narrative and storytelling aspects of price action.
    
    This node uses LLM to interpret the "story" behind the price movements,
    identifying key narrative shifts, market psychology, and potential future
    narrative developments that could impact price.
    """
    logger.debug("Analyzing market narrative")
    pass

@Nodes.llm_node(
    prompt_template=get_template_path("sentiment_analysis_prompt.txt"),
    output="sentiment_signals",
    temperature=0.3,
)
async def analyze_sentiment_signals(
    model: str,
    symbol: str,
    market_data: Dict[str, Any],
    technical_indicators: Dict[str, Any],
    fundamental_analysis: str,
    market_narrative: str
) -> str:
    """
    Analyze sentiment signals from various sources.
    
    This node uses LLM to analyze sentiment from:
    1. Social media mentions and trends
    2. News sentiment analysis
    3. Options market sentiment (put/call ratios)
    4. Retail vs institutional positioning
    5. Analyst consensus and changes
    """
    logger.debug("Analyzing sentiment signals")
    pass

@Nodes.define(output=None)
async def initialize_strategies() -> dict:
    """Initialize the list of trading strategies to analyze."""
    strategies = [
        "trend_following",
        "mean_reversion",
        "breakout",
        "momentum",
        "volatility_based"
    ]
    
    logger.info(f"Initialized {len(strategies)} trading strategies for analysis")
    return {
        "strategies": strategies,
        "strategy_results": {},
        "current_strategy_index": 0
    }

@Nodes.define(output="current_strategy")
async def select_next_strategy(strategies: List[str], current_strategy_index: int) -> str:
    """Select the next strategy to analyze."""
    if current_strategy_index >= len(strategies):
        logger.error("No more strategies to analyze")
        raise ValueError("Strategy index out of bounds")
    
    strategy = strategies[current_strategy_index]
    logger.info(f"Selected strategy: {strategy}")
    return strategy

@Nodes.llm_node(
    prompt_template=get_template_path("strategy_analysis_prompt.txt"),
    output="strategy_analysis",
    temperature=0.3,
)
async def analyze_with_strategy(
    model: str, 
    symbol: str,
    market_data: Dict[str, Any], 
    technical_indicators: Dict[str, Any],
    fundamental_analysis: str,
    market_narrative: str,
    sentiment_signals: str,
    current_strategy: str
) -> str:
    """Analyze market data using a specific trading strategy."""
    logger.debug(f"Analyzing with strategy: {current_strategy}")
    pass

@Nodes.define(output="current_strategy_index")
async def store_strategy_result(
    strategy_results: Dict[str, Any],
    current_strategy: str,
    strategy_analysis: str,
    current_strategy_index: int
) -> int:
    """Store the result of the current strategy analysis and increment the index."""
    strategy_results[current_strategy] = strategy_analysis
    logger.info(f"Stored results for strategy: {current_strategy}")
    return current_strategy_index + 1

@Nodes.llm_node(
    prompt_template=get_template_path("combine_strategy_results_prompt.txt"),
    output="final_recommendation",
    temperature=0.4,
)
async def combine_strategy_results(
    model: str,
    symbol: str,
    market_data: Dict[str, Any],
    technical_indicators: Dict[str, Any],
    fundamental_analysis: str,
    market_narrative: str,
    sentiment_signals: str,
    strategy_results: Dict[str, str]
) -> str:
    """Combine results from all strategies into a final recommendation."""
    logger.debug("Combining strategy results")
    pass

@Nodes.llm_node(
    prompt_template=get_template_path("risk_scenario_analysis_prompt.txt"),
    output="risk_scenario_analysis",
    temperature=0.4,
)
async def generate_risk_scenario_analysis(
    model: str,
    symbol: str,
    market_data: Dict[str, Any],
    technical_indicators: Dict[str, Any],
    fundamental_analysis: str,
    market_narrative: str,
    sentiment_signals: str,
    analysis_result: MarketAnalysisResult
) -> str:
    """
    Generate detailed risk scenario analysis.
    
    This node uses LLM to:
    1. Identify potential black swan events
    2. Model different market scenarios (bullish, bearish, sideways)
    3. Estimate probability and impact of each scenario
    4. Provide risk mitigation strategies for each scenario
    5. Analyze correlation risks with broader market conditions
    """
    logger.debug("Generating risk scenario analysis")
    pass

@Nodes.llm_node(
    prompt_template=get_template_path("narrative_explanation_prompt.txt"),
    output="narrative_explanation",
    temperature=0.4,
)
async def generate_narrative_explanation(
    model: str,
    symbol: str,
    market_data: Dict[str, Any],
    technical_indicators: Dict[str, Any],
    fundamental_analysis: str,
    market_narrative: str,
    sentiment_signals: str,
    risk_scenario_analysis: str,
    analysis_result: MarketAnalysisResult
) -> str:
    """
    Generate a detailed narrative explanation of the analysis.
    
    This node uses LLM to create a comprehensive, story-driven explanation of:
    1. The current market situation in plain language
    2. Key factors driving price action
    3. How different analysis components fit together
    4. Potential future developments and their implications
    5. Clear reasoning behind the trading recommendation
    """
    logger.debug("Generating narrative explanation")
    pass

@Nodes.define(output="analysis_result")
async def compile_final_analysis(
    symbol: str,
    market_data: Dict[str, Any],
    technical_indicators: Dict[str, Any],
    fundamental_analysis: str,
    market_narrative: str,
    sentiment_signals: str,
    final_recommendation: str,
    strategy_results: Dict[str, str]
) -> MarketAnalysisResult:
    """Compile the final market analysis result."""
    import json
    import re
    
    # Helper function to extract JSON from text that might contain other content
    def extract_json(text):
        if not text:
            return None
        # Try to find JSON content between curly braces
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None
    
    # Parse JSON strings from LLM outputs
    fundamental_analysis_dict = extract_json(fundamental_analysis)
    final_recommendation_dict = extract_json(final_recommendation)
    market_narrative_dict = extract_json(market_narrative)
    sentiment_signals_dict = extract_json(sentiment_signals)
    
    # Create fallback objects if parsing fails
    if not fundamental_analysis_dict:
        logger.error(f"Error parsing fundamental analysis JSON from LLM output")
        fundamental_analysis_dict = {
            "market_sentiment": "neutral",
            "trend_strength": 0.5,
            "volatility_assessment": "moderate",
            "volume_analysis": "average volume",
            "key_levels": [],
            "market_structure": "ranging",
            "market_phase": "accumulation",
            "timeframe_alignment": {},
            "institutional_levels": [],
            "smart_money_movements": [],
            "news_impact": {},
            "seasonal_patterns": [],
            "correlated_assets": {}
        }
    
    if not final_recommendation_dict:
        logger.error(f"Error parsing final recommendation JSON from LLM output")
        final_recommendation_dict = {
            "action": "hold",
            "confidence": 0.5,
            "timeframe": "medium-term",
            "entry_points": [float(market_data["data"][-1]["close"]) if market_data["data"] else 0.0],
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "risk_reward_ratio": 1.0,
            "reasoning": "Unable to parse LLM output"
        }
    
    # Enhance fundamental analysis with narrative and sentiment if available
    if market_narrative_dict:
        fundamental_analysis_dict["market_narrative"] = market_narrative_dict.get("narrative", "No narrative available")
        fundamental_analysis_dict["narrative_key_points"] = market_narrative_dict.get("key_points", [])
        fundamental_analysis_dict["narrative_confidence"] = market_narrative_dict.get("confidence", 0.5)
    
    if sentiment_signals_dict:
        fundamental_analysis_dict["sentiment_analysis"] = sentiment_signals_dict.get("sentiment_analysis", "No sentiment analysis available")
        fundamental_analysis_dict["sentiment_sources"] = sentiment_signals_dict.get("sentiment_sources", {})
        fundamental_analysis_dict["sentiment_score"] = sentiment_signals_dict.get("sentiment_score", 0.0)
    
    # Calculate strategy performance metrics
    strategy_performance = {}
    for strategy_name, recommendation in strategy_results.items():
        recommendation_dict = extract_json(recommendation)
        if recommendation_dict:
            strategy_performance[strategy_name] = {
                "confidence": recommendation_dict.get("confidence", 0.5),
                "risk_reward": recommendation_dict.get("risk_reward_ratio", 1.0),
                "alignment_with_trend": 1.0 if recommendation_dict.get("action") == final_recommendation_dict.get("action") else 0.0
            }
        else:
            # Fallback if parsing fails
            strategy_performance[strategy_name] = {
                "confidence": 0.5,
                "risk_reward": 1.0,
                "alignment_with_trend": 0.5
            }
    
    # Get latest price data
    latest_data = market_data["data"][-1] if market_data["data"] else {}
    
    # Create the final analysis result
    result = MarketAnalysisResult(
        symbol=symbol,
        timestamp=datetime.now().isoformat(),
        price_data={
            "latest": latest_data,
            "interval": market_data["interval"],
            "range": f"{market_data['start_date']} to {market_data['end_date']}"
        },
        technical_indicators=TechnicalIndicators(**technical_indicators),
        fundamental_analysis=FundamentalAnalysis(**fundamental_analysis_dict),
        trading_recommendation=TradingRecommendation(**final_recommendation_dict),
        strategy_performance=strategy_performance
    )
    
    logger.info(f"Compiled final analysis for {symbol}")
    return result

@Nodes.define(output=None)
async def save_analysis_result(analysis_result: MarketAnalysisResult, output_path: Optional[str] = None) -> None:
    """Save the analysis result to a file."""
    if not output_path:
        # Generate default output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"market_analysis_{analysis_result.symbol}_{timestamp}.json"
    
    # Convert to dict for JSON serialization
    result_dict = analysis_result.model_dump()
    
    # Remove dataframe from result as it's not JSON serializable
    if "dataframe" in result_dict["price_data"]:
        del result_dict["price_data"]["dataframe"]
    
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    
    logger.info(f"Saved analysis result to {output_path}")
    return {"output_path": output_path}

@Nodes.define(output=None)
async def display_analysis_summary(analysis_result: MarketAnalysisResult) -> None:
    """Display a summary of the analysis result."""
    recommendation = analysis_result.trading_recommendation
    
    print("\n" + "="*50)
    print(f"📊 Market Analysis Summary for {analysis_result.symbol} 📊")
    print("="*50)
    
    print(f"\n🔍 Technical Analysis:")
    # Handle potential None values in RSI
    if analysis_result.technical_indicators.rsi is not None:
        print(f"  RSI: {analysis_result.technical_indicators.rsi:.2f}")
    else:
        print(f"  RSI: Not enough data")
    
    # Handle potential None values in MACD
    macd = analysis_result.technical_indicators.macd
    if macd.get('macd') is not None:
        print(f"  MACD: {macd['macd']:.2f}")
    else:
        print(f"  MACD: Not enough data")
    
    # Handle potential None values in Bollinger Bands
    bb = analysis_result.technical_indicators.bollinger_bands
    if bb.get('middle') is not None and bb.get('upper') is not None and bb.get('lower') is not None:
        print(f"  Bollinger Bands: {bb['middle']:.2f} " +
              f"(Upper: {bb['upper']:.2f}, " +
              f"Lower: {bb['lower']:.2f})")
    else:
        print(f"  Bollinger Bands: Not enough data")
    
    # Show moving averages
    ma = analysis_result.technical_indicators.moving_averages
    print(f"  Moving Averages:")
    for key, value in ma.items():
        if value is not None:
            print(f"    {key}: {value:.2f}")
    
    print(f"\n📝 Fundamental Analysis:")
    print(f"  Market Sentiment: {analysis_result.fundamental_analysis.market_sentiment}")
    print(f"  Trend Strength: {analysis_result.fundamental_analysis.trend_strength:.2f}")
    print(f"  Volatility: {analysis_result.fundamental_analysis.volatility_assessment}")
    print(f"  Volume Analysis: {analysis_result.fundamental_analysis.volume_analysis}")
    
    # Show key levels
    if analysis_result.fundamental_analysis.key_levels:
        print(f"  Key Levels: {', '.join([str(level) for level in analysis_result.fundamental_analysis.key_levels])}")
    
    print(f"  Market Structure: {analysis_result.fundamental_analysis.market_structure}")
    print(f"  Market Phase: {analysis_result.fundamental_analysis.market_phase}")
    
    # Show timeframe alignment
    if analysis_result.fundamental_analysis.timeframe_alignment:
        print(f"  Timeframe Alignment:")
        for timeframe, alignment in analysis_result.fundamental_analysis.timeframe_alignment.items():
            print(f"    {timeframe}: {alignment}")
    
    # Show institutional levels
    if analysis_result.fundamental_analysis.institutional_levels:
        print(f"  Institutional Levels:")
        for level in analysis_result.fundamental_analysis.institutional_levels:
            print(f"    {level}")
    
    # Show smart money movements
    if analysis_result.fundamental_analysis.smart_money_movements:
        print(f"  Smart Money Movements:")
        for movement in analysis_result.fundamental_analysis.smart_money_movements:
            print(f"    {movement}")
    
    # Show news impact
    if analysis_result.fundamental_analysis.news_impact:
        print(f"  News Impact:")
        for news, impact in analysis_result.fundamental_analysis.news_impact.items():
            print(f"    {news}: {impact}")
    
    # Show seasonal patterns
    if analysis_result.fundamental_analysis.seasonal_patterns:
        print(f"  Seasonal Patterns:")
        for pattern in analysis_result.fundamental_analysis.seasonal_patterns:
            print(f"    {pattern}")
    
    # Show correlated assets
    if analysis_result.fundamental_analysis.correlated_assets:
        print(f"  Correlated Assets:")
        for asset, correlation in analysis_result.fundamental_analysis.correlated_assets.items():
            print(f"    {asset}: {correlation}")
    
    print(f"\n🎯 Trading Recommendation:")
    print(f"  Action: {recommendation.action.upper()}")
    print(f"  Confidence: {recommendation.confidence:.2f}")
    print(f"  Timeframe: {recommendation.timeframe}")
    
    # Show entry points
    if recommendation.entry_points:
        print(f"  Entry Points: {', '.join([str(round(p, 2)) for p in recommendation.entry_points])}")
    
    print(f"  Stop Loss: {recommendation.stop_loss:.2f}")
    print(f"  Take Profit: {recommendation.take_profit:.2f}")
    print(f"  Risk/Reward Ratio: {recommendation.risk_reward_ratio:.2f}")
    
    print(f"\n💡 Reasoning:")
    print(f"  {recommendation.reasoning}")
    
    # Show strategy performance
    print(f"\n📈 Strategy Performance:")
    for strategy, metrics in analysis_result.strategy_performance.items():
        if isinstance(metrics, dict):
            confidence = metrics.get("confidence", 0)
            risk_reward = metrics.get("risk_reward", 0)
            print(f"  {strategy}: Confidence {confidence:.2f}, Risk/Reward {risk_reward:.2f}")
        else:
            print(f"  {strategy}: {metrics:.2f}")
    
    print("\n" + "="*50)
    print(f"Analysis completed at {analysis_result.timestamp}")
    print("="*50)

# Define the Workflow with explicit transitions
multi_timeframe_workflow = (
    Workflow("fetch_multi_timeframe_data")
    .add_observer(market_analysis_progress_observer)
    .then("calculate_technical_indicators")
    .then("analyze_fundamentals")
    .then("analyze_market_narrative")  # New LLM node for market narrative
    .then("analyze_sentiment_signals")  # New LLM node for sentiment analysis
    .then("initialize_strategies")
    .then("select_next_strategy")
    .then("analyze_with_strategy")
    .node("analyze_with_strategy", inputs_mapping={
        "symbol": "symbol",
        "market_data": "market_data",
        "technical_indicators": "technical_indicators",
        "fundamental_analysis": "fundamental_analysis",
        "market_narrative": "market_narrative",  # Add market narrative to strategy analysis
        "sentiment_signals": "sentiment_signals",  # Add sentiment signals to strategy analysis
        "current_strategy": "current_strategy",
        "model": "model"
    })
    .then("store_strategy_result")
    .branch([
        # Continue with next strategy if available
        ("select_next_strategy", lambda ctx: ctx.get("current_strategy_index", 0) < len(ctx.get("strategies", []))),
        # Move to combining results if all strategies are analyzed
        ("combine_strategy_results", lambda ctx: ctx.get("current_strategy_index", 0) >= len(ctx.get("strategies", [])))
    ])
    
    # Define the path for continuing with another strategy
    .node("select_next_strategy", inputs_mapping={
        "strategies": "strategies",
        "current_strategy_index": "current_strategy_index"
    })
    
    # Define the path for combining results and finalizing
    .node("combine_strategy_results", inputs_mapping={
        "symbol": "symbol",
        "market_data": "market_data",
        "technical_indicators": "technical_indicators",
        "fundamental_analysis": "fundamental_analysis",
        "market_narrative": "market_narrative",  # Add market narrative to final analysis
        "sentiment_signals": "sentiment_signals",  # Add sentiment signals to final analysis
        "strategy_results": "strategy_results",
        "model": "model"
    })
    .then("compile_final_analysis")
    .then("generate_risk_scenario_analysis")  # New LLM node for risk scenario analysis
    .then("generate_narrative_explanation")  # New LLM node for narrative explanation
    .then("save_analysis_result")
    .then("display_analysis_summary")
)

@Nodes.define(output=None)
async def generate_detailed_report(
    symbol: str,
    market_data: Dict[str, Any],
    technical_indicators: Dict[str, Any],
    fundamental_analysis: Dict[str, Any],
    trading_recommendation: Dict[str, Any],
    multi_timeframe_indicators: Dict[str, Dict[str, Any]],
    multi_timeframe_data: Dict[str, Dict[str, Any]],
    market_narrative: str,
    sentiment_signals: str,
    risk_scenario_analysis: str,
    narrative_explanation: str,
    output_path: Optional[str] = None
) -> dict:
    """Generate a detailed market analysis report in markdown format."""
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"market_analysis_report_{symbol}_{timestamp}.md"
    
    # Create the report content
    report = []
    
    # Title and header
    report.append(f"# Comprehensive Market Analysis Report: {symbol}")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Symbol:** {symbol}")
    report.append(f"**Timeframe:** {market_data['interval']}")
    report.append(f"**Period:** {market_data['start_date']} to {market_data['end_date']}")
    report.append("\n---\n")
    
    # Executive Summary
    report.append("## 📊 Executive Summary")
    
    # Get the latest price data
    latest_data = {}
    if "dataframe" in market_data and len(market_data["dataframe"]) > 0:
        latest_data = market_data["dataframe"].iloc[-1].to_dict()
    
    # Extract key metrics for the summary
    current_price = latest_data.get("close", 0)
    day_change = latest_data.get("close", 0) - latest_data.get("open", 0)
    day_change_pct = (day_change / latest_data.get("open", 1)) * 100 if latest_data.get("open") else 0
    
    # Add price summary
    report.append(f"**Current Price:** ${current_price:.2f}")
    report.append(f"**Day Change:** ${day_change:.2f} ({day_change_pct:.2f}%)")
    report.append("")
    
    # Add trading recommendation summary
    action = trading_recommendation.get("action", "No action")
    confidence = trading_recommendation.get("confidence", 0)
    timeframe = trading_recommendation.get("timeframe", "Unknown")
    
    # Create visual indicator for recommendation
    if action.lower() == "buy":
        action_indicator = "🟢 BUY"
    elif action.lower() == "sell":
        action_indicator = "🔴 SELL"
    else:
        action_indicator = "⚪ HOLD/NEUTRAL"
    
    report.append(f"**Trading Recommendation:** {action_indicator}")
    report.append(f"**Confidence Level:** {confidence:.2f}/1.0")
    report.append(f"**Recommended Timeframe:** {timeframe}")
    report.append("")
    
    # Add risk-reward summary
    risk_reward = trading_recommendation.get("risk_reward_ratio", 0)
    stop_loss = trading_recommendation.get("stop_loss", 0)
    take_profit = trading_recommendation.get("take_profit", 0)
    
    report.append(f"**Risk-Reward Ratio:** {risk_reward:.2f}")
    report.append(f"**Stop Loss:** ${stop_loss:.2f}")
    report.append(f"**Take Profit:** ${take_profit:.2f}")
    report.append("")
    
    # Add market sentiment summary
    market_sentiment = fundamental_analysis.get("market_sentiment", "Unknown")
    trend_strength = fundamental_analysis.get("trend_strength", 0)
    
    # Truncate market sentiment to first 150 characters for summary
    sentiment_summary = market_sentiment[:150] + "..." if len(market_sentiment) > 150 else market_sentiment
    
    report.append(f"**Market Sentiment:** {sentiment_summary}")
    report.append(f"**Trend Strength:** {trend_strength:.2f}")
    report.append("")
    
    report.append("\n---\n")
    
    # Price Analysis
    report.append("## 📈 Price Analysis")
    
    # Get support and resistance levels
    support_resistance = technical_indicators.get("support_resistance", {})
    support_levels = support_resistance.get("support", [])
    resistance_levels = support_resistance.get("resistance", [])
    
    # Add support and resistance levels with visual indicators
    if support_levels:
        report.append("### Support Levels")
        for level in sorted(support_levels):
            # Calculate distance from current price
            distance = ((level - current_price) / current_price) * 100
            report.append(f"- 🔵 ${level:.2f} ({distance:.2f}% from current price)")
    
    if resistance_levels:
        report.append("\n### Resistance Levels")
        for level in sorted(resistance_levels):
            # Calculate distance from current price
            distance = ((level - current_price) / current_price) * 100
            report.append(f"- 🔴 ${level:.2f} ({distance:.2f}% from current price)")
    
    # Add key price levels from fundamental analysis
    key_levels = fundamental_analysis.get("key_levels", [])
    if key_levels:
        report.append("\n### Key Price Levels")
        for level in sorted(key_levels):
            # Determine if level is above or below current price
            if level > current_price:
                report.append(f"- 🔼 ${level:.2f} (Resistance)")
            else:
                report.append(f"- 🔽 ${level:.2f} (Support)")
    
    # Add institutional levels if available
    institutional_levels = fundamental_analysis.get("institutional_levels", [])
    if institutional_levels:
        report.append("\n### Institutional Levels")
        for level_info in institutional_levels:
            level_type = level_info.get("type", "Unknown")
            price = level_info.get("price", 0)
            description = level_info.get("description", "")
            
            # Truncate description if too long
            desc_summary = description[:100] + "..." if len(description) > 100 else description
            
            report.append(f"- **{level_type} at ${price:.2f}**: {desc_summary}")
    
    # Add smart money movements if available
    smart_money = fundamental_analysis.get("smart_money_movements", [])
    if smart_money:
        report.append("\n### Smart Money Movements")
        for movement in smart_money:
            movement_type = movement.get("type", "Unknown")
            price_range = movement.get("price_range", [0, 0])
            description = movement.get("description", "")
            
            # Format price range
            price_range_str = f"${price_range[0]:.2f} - ${price_range[1]:.2f}" if len(price_range) >= 2 else "Unknown"
            
            # Truncate description if too long
            desc_summary = description[:100] + "..." if len(description) > 100 else description
            
            report.append(f"- **{movement_type} ({price_range_str})**: {desc_summary}")
    
    # Add price structure analysis
    market_structure = fundamental_analysis.get("market_structure", "")
    if market_structure:
        report.append("\n### Market Structure Analysis")
        report.append(market_structure)
    
    report.append("\n---\n")
    
    # Technical Analysis
    report.append("## 📊 Technical Analysis")
    
    # Moving Averages
    ma = technical_indicators["moving_averages"]
    report.append("### Moving Averages")
    for period, value in ma.items():
        if value is not None:
            # Determine if price is above or below MA
            if current_price > value:
                report.append(f"- **{period.upper()}:** {value:.2f} (Price is **ABOVE** ⬆️)")
            else:
                report.append(f"- **{period.upper()}:** {value:.2f} (Price is **BELOW** ⬇️)")
    
    # RSI
    if technical_indicators.get("rsi") is not None:
        rsi = technical_indicators["rsi"]
        report.append("\n### Relative Strength Index (RSI)")
        
        # Add RSI interpretation
        rsi_status = "NEUTRAL"
        if rsi > 70:
            rsi_status = "OVERBOUGHT 🔴"
        elif rsi < 30:
            rsi_status = "OVERSOLD 🟢"
        
        report.append(f"**RSI Value:** {rsi:.2f} ({rsi_status})")
    
    # MACD
    macd = technical_indicators["macd"]
    if macd["macd"] is not None:
        report.append("\n### MACD")
        
        # Determine MACD signal
        macd_signal = "NEUTRAL"
        if macd["macd"] > macd["signal"]:
            macd_signal = "BULLISH 🟢"
        elif macd["macd"] < macd["signal"]:
            macd_signal = "BEARISH 🔴"
        
        report.append(f"**MACD Line:** {macd['macd']:.4f}")
        report.append(f"**Signal Line:** {macd['signal']:.4f}")
        report.append(f"**Histogram:** {macd['histogram']:.4f}")
        report.append(f"**Signal:** {macd_signal}")
    
    # Stochastic
    stoch = technical_indicators.get("stochastic", {})
    if stoch.get("k") is not None:
        report.append("\n### Stochastic Oscillator")
        
        # Add stochastic interpretation
        stoch_status = "NEUTRAL"
        if stoch["k"] > 80:
            stoch_status = "OVERBOUGHT 🔴"
        elif stoch["k"] < 20:
            stoch_status = "OVERSOLD 🟢"
        
        report.append(f"**Stochastic %K:** {stoch['k']:.2f} ({stoch_status})")
        report.append(f"**Stochastic %D:** {stoch['d']:.2f}")
    
    # ADX
    if technical_indicators.get("adx") is not None:
        report.append("\n### Average Directional Index (ADX)")
        
        # Add ADX interpretation
        adx_status = "NEUTRAL"
        if technical_indicators["adx"] > 25:
            adx_status = "STRONG TREND 🔥"
        elif technical_indicators["adx"] < 20:
            adx_status = "WEAK TREND ❄️"
        
        report.append(f"**ADX Value:** {technical_indicators['adx']:.2f} ({adx_status})")
    
    # Bollinger Bands
    bb = technical_indicators["bollinger_bands"]
    if bb["middle"] is not None:
        report.append("\n### Bollinger Bands")
        
        # Determine position based on available data
        bb_position = "Middle Band"  # Default position
        if current_price > bb["upper"]:
            bb_position = "Upper Band"
        elif current_price < bb["lower"]:
            bb_position = "Lower Band"
        
        report.append(f"**Bollinger Middle Band:** {bb['middle']:.2f}")
        report.append(f"**Bollinger Upper Band:** {bb['upper']:.2f}")
        report.append(f"**Bollinger Lower Band:** {bb['lower']:.2f}")
        report.append(f"**Price Position:** Near {bb_position}")
    
    # ATR
    if technical_indicators.get("atr") is not None:
        report.append("\n### Average True Range (ATR)")
        report.append(f"**ATR Value:** {technical_indicators['atr']:.2f}")
    
    # OBV
    if technical_indicators.get("obv") is not None:
        report.append("\n### On-Balance Volume (OBV)")
        report.append(f"**OBV Value:** {technical_indicators['obv']:.2f}")
    
    # Candlestick Patterns
    if technical_indicators.get("candlestick_patterns"):
        report.append("\n### Candlestick Patterns")
        for pattern in technical_indicators["candlestick_patterns"]:
            report.append(f"- **{pattern['pattern']}** at price {pattern['price']:.2f} (Significance: {pattern['significance']})")
    
    # Chart Patterns
    if technical_indicators.get("chart_patterns"):
        report.append("\n### Chart Patterns")
        for pattern in technical_indicators["chart_patterns"]:
            report.append(f"- **{pattern['pattern']}** at price {pattern['price']:.2f} (Significance: {pattern['significance']})")
    
    report.append("\n---\n")
    
    # Smart Money Concepts (SMC) Analysis
    smc_concepts = technical_indicators.get("smc_concepts", {})
    if smc_concepts:
        report.append("## 📊 Smart Money Concepts (SMC) Analysis")
        
        # Order Blocks
        if smc_concepts.get("order_blocks"):
            report.append("### Order Blocks")
            for block in smc_concepts["order_blocks"]:
                report.append(f"- **{block['type']} Order Block:** Price range {block['price_range'][0]:.2f} - {block['price_range'][1]:.2f} (Strength: {block['strength']})")
        
        # Fair Value Gaps
        if smc_concepts.get("fair_value_gaps"):
            report.append("### Fair Value Gaps")
            for gap in smc_concepts["fair_value_gaps"]:
                report.append(f"- **{gap['type']} FVG:** Price range {gap['price_range'][0]:.2f} - {gap['price_range'][1]:.2f} (Size: {gap['size']:.2f})")
        
        # Imbalances
        if smc_concepts.get("imbalances"):
            report.append("### Imbalances")
            for imb in smc_concepts["imbalances"]:
                report.append(f"- **{imb['type']} Imbalance:** Price range {imb['price_range'][0]:.2f} - {imb['price_range'][1]:.2f} (Significance: {imb['significance']})")
        
        report.append("")
        report.append("\n---\n")
    
    # ICT Concepts Analysis
    ict_concepts = technical_indicators.get("ict_concepts", {})
    if ict_concepts:
        report.append("## 📊 Institutional Candle Theory (ICT) Analysis")
        
        # Liquidity levels
        liquidity = ict_concepts.get("liquidity_levels", {})
        if liquidity.get("buy_side") or liquidity.get("sell_side"):
            report.append("### Liquidity Levels")
            
            if liquidity.get("buy_side"):
                report.append(f"**Buy-Side Liquidity Levels:** {', '.join([str(round(level, 2)) for level in liquidity['buy_side']])}")
            
            if liquidity.get("sell_side"):
                report.append(f"**Sell-Side Liquidity Levels:** {', '.join([str(round(level, 2)) for level in liquidity['sell_side']])}")
        
        # Breaker blocks
        if ict_concepts.get("breaker_blocks"):
            report.append("### Breaker Blocks")
            for block in ict_concepts["breaker_blocks"]:
                report.append(f"- **{block['type']}:** Price range {block['price_range'][0]:.2f} - {block['price_range'][1]:.2f} (Strength: {block['strength']})")
        
        report.append("")
        report.append("\n---\n")
    
    # Multi-Timeframe Analysis
    report.append("## 📊 Multi-Timeframe Analysis")
    
    # Timeframe Alignment
    if fundamental_analysis.get("timeframe_alignment"):
        report.append("### Timeframe Alignment")
        for tf, alignment in fundamental_analysis["timeframe_alignment"].items():
            report.append(f"**{tf}:** {alignment}")
        report.append("")
    
    # Detailed Multi-Timeframe Analysis
    if multi_timeframe_indicators:
        report.append("### Detailed Multi-Timeframe Analysis")
        
        for tf_name, indicators in multi_timeframe_indicators.items():
            report.append(f"#### {tf_name.replace('_', ' ').title()} Timeframe")
            
            # RSI
            if indicators.get("rsi") is not None:
                rsi_value = indicators["rsi"]
                rsi_interpretation = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
                report.append(f"**RSI:** {rsi_value:.2f} - {rsi_interpretation}")
            
            # MACD
            macd = indicators.get("macd", {})
            if macd.get("macd") is not None:
                macd_signal = "Bullish" if macd["macd"] > macd["signal"] else "Bearish"
                report.append(f"**MACD Signal:** {macd_signal}")
            
            # Bollinger Bands
            bb = indicators.get("bollinger_bands", {})
            if bb.get("middle") is not None:
                report.append(f"**Bollinger Middle Band:** {bb.get('middle', 0):.2f}")
                report.append(f"**Bollinger Upper Band:** {bb.get('upper', 0):.2f}")
                report.append(f"**Bollinger Lower Band:** {bb.get('lower', 0):.2f}")
                
                # Determine position based on available data
                bb_position = "Middle Band"  # Default position
                report.append(f"**Price Position:** Near {bb_position}")
            
            # Support/Resistance
            sr = indicators.get("support_resistance", {})
            if sr.get("support") or sr.get("resistance"):
                levels = []
                if sr.get("support"):
                    levels.append(f"Support at {', '.join([str(round(level, 2)) for level in sr['support']])}")
                if sr.get("resistance"):
                    levels.append(f"Resistance at {', '.join([str(round(level, 2)) for level in sr['resistance']])}")
                report.append(f"**Key Levels:** {'; '.join(levels)}")
            
            report.append("")
    
    report.append("\n---\n")
    
    # Market Context
    report.append("## 📊 Market Context")
    
    # News Impact
    if fundamental_analysis.get("news_impact"):
        report.append("### News Impact")
        for news, impact in fundamental_analysis["news_impact"].items():
            report.append(f"- **{news}:** {impact}")
        report.append("")
    
    # Seasonal Patterns
    if fundamental_analysis.get("seasonal_patterns"):
        report.append("### Seasonal Patterns")
        for pattern in fundamental_analysis["seasonal_patterns"]:
            report.append(f"- {pattern}")
        report.append("")
    
    # Correlated Assets
    if fundamental_analysis.get("correlated_assets"):
        report.append("### Correlated Assets")
        for asset, correlation in fundamental_analysis["correlated_assets"].items():
            report.append(f"- **{asset}:** {correlation}")
        report.append("")
    
    # Market Narrative
    report.append("## 📊 Market Narrative")
    report.append(market_narrative)
    report.append("")
    
    # Sentiment Signals
    report.append("## 📊 Sentiment Signals")
    report.append(sentiment_signals)
    report.append("")
    
    # Risk Scenario Analysis
    report.append("## 📊 Risk Scenario Analysis")
    report.append(risk_scenario_analysis)
    report.append("")
    
    # Narrative Explanation
    report.append("## 📊 Narrative Explanation")
    report.append(narrative_explanation)
    report.append("")
    
    # Trading Recommendation
    report.append("## 📊 Trading Recommendation")
    
    report.append(f"**Action:** {trading_recommendation['action'].upper()}")
    report.append(f"**Confidence:** {trading_recommendation['confidence']:.2f}/1.0")
    report.append(f"**Timeframe:** {trading_recommendation['timeframe']}")
    
    # Entry Points
    if trading_recommendation.get("entry_points"):
        report.append(f"**Entry Points:** {', '.join([str(round(p, 2)) for p in trading_recommendation['entry_points']])}")
    
    # Entry Type
    if trading_recommendation.get("entry_type"):
        report.append(f"**Entry Type:** {trading_recommendation['entry_type']}")
    
    # Risk Management
    report.append(f"**Stop Loss:** {trading_recommendation['stop_loss']:.2f}")
    report.append(f"**Take Profit:** {trading_recommendation['take_profit']:.2f}")
    report.append(f"**Risk/Reward Ratio:** {trading_recommendation['risk_reward_ratio']:.2f}")
    
    # Position Sizing
    if trading_recommendation.get("position_sizing"):
        report.append("### Position Sizing")
        for key, value in trading_recommendation["position_sizing"].items():
            report.append(f"- **{key}:** {value}")
    
    # Exit Strategies
    if trading_recommendation.get("exit_strategies"):
        report.append("### Exit Strategies")
        for strategy in trading_recommendation["exit_strategies"]:
            report.append(f"- {strategy}")
    
    # Trade Management
    if trading_recommendation.get("trade_management"):
        report.append("### Trade Management")
        for key, value in trading_recommendation["trade_management"].items():
            report.append(f"- **{key}:** {value}")
    
    # Reasoning
    report.append("### Analysis Reasoning")
    report.append(trading_recommendation["reasoning"])
    
    report.append("\n---\n")
    
    # Timeframe-Specific Strategies
    if trading_recommendation.get("timeframe_specific_strategies"):
        report.append("## 📊 Timeframe-Specific Strategies")
        
        for tf, strategy in trading_recommendation["timeframe_specific_strategies"].items():
            report.append(f"### {tf.replace('_', ' ').title()} Timeframe Strategy")
            for key, value in strategy.items():
                if isinstance(value, dict):
                    report.append(f"**{key}:**")
                    for subkey, subvalue in value.items():
                        report.append(f"  - {subkey}: {subvalue}")
                else:
                    report.append(f"**{key}:** {value}")
            report.append("")
    
    # Day Trading Specifics
    if trading_recommendation.get("intraday_levels") or trading_recommendation.get("session_analysis"):
        report.append("## 📊 Day Trading Specifics")
        
        # Intraday Levels
        if trading_recommendation.get("intraday_levels"):
            report.append("### Intraday Levels")
            for level in trading_recommendation["intraday_levels"]:
                report.append(f"- **{level.get('type', 'Level')}:** {level.get('price', 'N/A')} ({level.get('description', '')})")
        
        # Session Analysis
        if trading_recommendation.get("session_analysis"):
            report.append("### Session Analysis")
            for key, value in trading_recommendation["session_analysis"].items():
                report.append(f"- **{key}:** {value}")
    
    report.append("\n---\n")
    
    # Disclaimer
    report.append("## Disclaimer")
    report.append("*This analysis is for informational purposes only and does not constitute investment advice. Trading involves risk, and past performance is not indicative of future results. Always conduct your own research before making investment decisions.*")
    
    # Write the report to file
    with open(output_path, "w") as f:
        f.write("\n".join(report))
    
    logger.info(f"Detailed market analysis report saved to {output_path}")
    return {"report_path": output_path}

def analyze_market(
    symbol: str,
    interval: str = "1h",
    range_type: str = "week",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model: str = "openai/gpt-4o-mini",
    output_path: Optional[str] = None,
    generate_report: bool = True,
    task_id: str = "default",
    _handle_event: Optional[Any] = None
):
    """
    Analyze market data for a given symbol using multiple trading strategies.
    
    Args:
        symbol: Stock or cryptocurrency symbol (e.g., AAPL, BTC-USD)
        interval: Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d)
        range_type: Type of date range (today, date, week, month, ytd)
        start_date: Start date in YYYY-MM-DD format (only for range_type='date')
        end_date: End date in YYYY-MM-DD format (only for range_type='date')
        model: LLM model to use for analysis
        output_path: Path to save the analysis result (optional)
        generate_report: Whether to generate a detailed markdown report
        task_id: Task identifier for event handling
        _handle_event: Event handler function
    
    Returns:
        The final market analysis result
    """
    # Update workflow to use multi-timeframe analysis
    multi_timeframe_workflow = (
        Workflow("fetch_multi_timeframe_data")
        .add_observer(market_analysis_progress_observer)
        .then("calculate_technical_indicators")
        .then("analyze_fundamentals")
        .then("analyze_market_narrative")  # New LLM node for market narrative
        .then("analyze_sentiment_signals")  # New LLM node for sentiment analysis
        .then("initialize_strategies")
        .then("select_next_strategy")
        .then("analyze_with_strategy")
        .node("analyze_with_strategy", inputs_mapping={
            "symbol": "symbol",
            "market_data": "market_data",
            "technical_indicators": "technical_indicators",
            "fundamental_analysis": "fundamental_analysis",
            "market_narrative": "market_narrative",  # Add market narrative to strategy analysis
            "sentiment_signals": "sentiment_signals",  # Add sentiment signals to strategy analysis
            "current_strategy": "current_strategy",
            "model": "model"
        })
        .then("store_strategy_result")
        .branch([
            # Continue with next strategy if available
            ("select_next_strategy", lambda ctx: ctx.get("current_strategy_index", 0) < len(ctx.get("strategies", []))),
            # Move to combining results if all strategies are analyzed
            ("combine_strategy_results", lambda ctx: ctx.get("current_strategy_index", 0) >= len(ctx.get("strategies", [])))
        ])
        
        # Define the path for continuing with another strategy
        .node("select_next_strategy", inputs_mapping={
            "strategies": "strategies",
            "current_strategy_index": "current_strategy_index"
        })
        
        # Define the path for combining results and finalizing
        .node("combine_strategy_results", inputs_mapping={
            "symbol": "symbol",
            "market_data": "market_data",
            "technical_indicators": "technical_indicators",
            "fundamental_analysis": "fundamental_analysis",
            "market_narrative": "market_narrative",  # Add market narrative to final analysis
            "sentiment_signals": "sentiment_signals",  # Add sentiment signals to final analysis
            "strategy_results": "strategy_results",
            "model": "model"
        })
        .then("compile_final_analysis")
        .then("generate_risk_scenario_analysis")  # New LLM node for risk scenario analysis
        .then("generate_narrative_explanation")  # New LLM node for narrative explanation
        .then("save_analysis_result")
        .then("display_analysis_summary")
    )
    
    # Add report generation if requested
    if generate_report:
        multi_timeframe_workflow = multi_timeframe_workflow.then("generate_detailed_report")
        multi_timeframe_workflow = multi_timeframe_workflow.node("generate_detailed_report", inputs_mapping={
            "symbol": "symbol",
            "market_data": "market_data",
            "technical_indicators": "technical_indicators",
            "fundamental_analysis": lambda ctx: ctx["analysis_result"].fundamental_analysis.model_dump(),
            "trading_recommendation": lambda ctx: ctx["analysis_result"].trading_recommendation.model_dump(),
            "multi_timeframe_indicators": lambda ctx: ctx.get("multi_timeframe_indicators", {}),
            "multi_timeframe_data": lambda ctx: ctx.get("multi_timeframe_data", {}),
            "market_narrative": "market_narrative",  # Add market narrative to report
            "sentiment_signals": "sentiment_signals",  # Add sentiment signals to report
            "risk_scenario_analysis": "risk_scenario_analysis",  # Add risk analysis to report
            "narrative_explanation": "narrative_explanation",  # Add narrative explanation to report
            "output_path": "output_path"
        })
    
    initial_context = {
        "symbol": symbol,
        "primary_interval": interval,
        "range_type": range_type,
        "start_date": start_date,
        "end_date": end_date,
        "model": model,
        "output_path": output_path
    }

    logger.info(f"Starting market analysis for {symbol}")
    engine = multi_timeframe_workflow.build()
    
    result = anyio.run(engine.run, initial_context)
    logger.info("Market analysis completed successfully 🎉")
    return result

if __name__ == "__main__":
    # Example usage
    analyze_market(
        symbol="BTC-USD",
        interval="1h",
        range_type="week"
    )
