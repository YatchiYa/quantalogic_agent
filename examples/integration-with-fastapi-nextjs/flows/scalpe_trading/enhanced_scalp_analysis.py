#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru>=0.7.2",
#     "litellm>=1.0.0",
#     "pydantic>=2.0.0",
#     "anyio",
#     "quantalogic>=0.35",
#     "jinja2>=3.1.0",
#     "typer>=0.9.0",
#     "yfinance>=0.2.28",
#     "pandas>=2.0.0",
#     "pytz",
#     "numpy>=1.24.0",
#     "ta>=0.10.2",
#     "plotly>=5.14.0",
#     "kaleido>=0.2.1",
#     "rich>=13.0.0",
#     "pyperclip>=1.8.2"
# ]
# ///

"""
Enhanced Scalp Trading Analysis Flow

This module provides comprehensive scalp trading analysis with multi-timeframe support,
technical indicators, and detailed trading recommendations for beginners.
"""

import asyncio
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from pydantic import BaseModel, Field, field_validator, model_validator

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from quantalogic.tools.finance.yahoo_finance import YahooFinanceTool
# from ..service import event_observer

# Initialize console for rich output
console = Console()

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Helper function to get template paths
def get_template_path(template_name):
    return os.path.join(TEMPLATES_DIR, template_name)

# Initialize Typer app
app = typer.Typer(help="Scalp trading analysis and recommendations")

# Constants for timeframes
TIMEFRAMES = {
    "scalping": ["1m", "5m", "15m"],
    "intraday": ["15m", "30m", "1h"],
    "swing": ["1h", "4h", "1d"],
}

# Constants for technical analysis
TECHNICAL_PARAMS = {
    "ma_periods": [9, 20, 50, 200],
    "ema_periods": [8, 21, 55, 89, 200],
    "rsi_period": 14,
    "macd_params": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger_params": {"period": 20, "std_dev": 2},
    "atr_period": 14,
    "volume_ma_period": 20,
}

# Pydantic Models
class FinancialSymbol(BaseModel):
    """Financial symbol with metadata."""
    symbol: str
    name: str = ""
    type: str = ""  # stock, crypto, forex, etc.
    
class TimeframeRequest(BaseModel):
    """Request for a specific timeframe."""
    interval: str
    range_type: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
class MultiTimeframeRequest(BaseModel):
    """Request for multiple timeframes."""
    symbols: List[FinancialSymbol]
    timeframes: List[TimeframeRequest]
    
class FinancialDataPoint(BaseModel):
    """Single financial data point with OHLCV data."""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    @model_validator(mode='after')
    def round_values(self):
        """Round numeric values to appropriate precision."""
        self.open = round(self.open, 4)
        self.high = round(self.high, 4)
        self.low = round(self.low, 4)
        self.close = round(self.close, 4)
        return self
    
class FinancialDataSeries(BaseModel):
    """Financial data series for a symbol and timeframe."""
    symbol: str
    name: str
    interval: str
    range_type: str
    start_date: str
    end_date: str
    data_points: int
    data: List[FinancialDataPoint]
    currency: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    
class FinancialDataCollection(BaseModel):
    """Collection of financial data series across symbols and timeframes."""
    series: List[FinancialDataSeries]
    timestamp: str
    
class PriceLevel(BaseModel):
    """Price level with type and strength."""
    price: float
    level_type: str  # support, resistance, order_block, fair_value_gap
    strength: int  # 1-10, with 10 being strongest
    timeframe: str  # timeframe where this level was identified
    description: str  # description of the level
    
class VolumeAnalysis(BaseModel):
    """Volume analysis for a symbol."""
    symbol: str
    average_volume: float
    volume_trend: str  # increasing, decreasing, flat
    volume_spikes: List[str]  # timestamps of volume spikes
    volume_divergence: bool  # True if price/volume divergence detected
    
class TechnicalIndicators(BaseModel):
    """Technical indicators for a symbol and timeframe."""
    symbol: str
    timeframe: str
    moving_averages: Dict[str, Union[float, List[Optional[float]]]]  # Now supports both single values and arrays
    exponential_mas: Dict[str, Union[float, List[Optional[float]]]]  # Now supports both single values and arrays
    rsi: Optional[Union[float, List[Optional[float]]]] = None  # Can be a single value or array
    macd: Optional[Dict[str, float]] = None
    bollinger_bands: Optional[Dict[str, float]] = None
    atr: Optional[float] = None
    volume_analysis: Optional[VolumeAnalysis] = None
    
    @model_validator(mode='after')
    def ensure_compatible_types(self):
        """Ensure that we handle both array-based and single-value indicators.
        This allows backward compatibility with existing code."""
        # For moving averages, if we have a list with a single value, extract just that value
        for key, value in self.moving_averages.items():
            if isinstance(value, list) and len(value) == 1:
                self.moving_averages[key] = value[0]
                
        # Same for exponential MAs
        for key, value in self.exponential_mas.items():
            if isinstance(value, list) and len(value) == 1:
                self.exponential_mas[key] = value[0]
                
        # For RSI
        if isinstance(self.rsi, list) and len(self.rsi) == 1:
            self.rsi = self.rsi[0]
            
        return self
    
class TechnicalAnalysis(BaseModel):
    """Collection of technical indicators across symbols and timeframes."""
    indicators: List[TechnicalIndicators]
    timestamp: str
    
class MarketStructureAnalysis(BaseModel):
    """Analysis of market structure for a symbol."""
    symbol: str
    trend: str  # bullish, bearish, neutral
    key_levels: List[PriceLevel]
    swing_points: List[Dict[str, Any]]  # high/low swing points
    order_blocks: List[Dict[str, Any]]  # order blocks
    fair_value_gaps: List[Dict[str, Any]]  # fair value gaps
    
class EntryStrategy(BaseModel):
    """Entry strategy for a trading opportunity."""
    symbol: str
    entry_price: float
    entry_type: str  # limit, market, stop
    entry_timeframe: str
    entry_trigger: str  # what triggers the entry
    entry_rationale: str  # why this entry point
    alternative_entries: List[Dict[str, Any]]  # alternative entry points
    
class TakeProfitLevel(BaseModel):
    """Take profit level with price and rationale."""
    price: float
    rationale: str

class ExitStrategy(BaseModel):
    """Exit strategy for a trading opportunity."""
    symbol: str
    stop_loss: float
    take_profit_levels: List[TakeProfitLevel]  # multiple take profit levels
    time_based_exit: Optional[str] = None  # time-based exit criteria
    exit_rationale: str  # why these exit points
    
class PositionSizing(BaseModel):
    """Position sizing recommendations."""
    symbol: str
    risk_percentage: float  # % of account to risk
    position_size: str  # description of position size
    risk_reward_ratio: float
    max_loss_amount: str  # description of max loss
    
class ScalpTradingOpportunity(BaseModel):
    """Complete scalp trading opportunity."""
    symbol: str
    direction: str  # long or short
    timeframe: str
    entry_strategy: EntryStrategy
    exit_strategy: ExitStrategy
    position_sizing: PositionSizing
    expected_duration: str  # expected trade duration
    confidence_level: int  # 1-10, with 10 being highest
    key_considerations: List[str]  # important factors to consider
    
class FinancialAnalysis(BaseModel):
    """Comprehensive financial analysis."""
    timestamp: str
    market_context: str
    market_structure_analysis: str
    technical_analysis_summary: str
    volume_analysis: str
    key_levels_analysis: str
    multi_timeframe_analysis: str
    
class TradingAnalysis(BaseModel):
    """Trading analysis and recommendations."""
    timestamp: str
    market_outlook: str
    trading_opportunities: List[ScalpTradingOpportunity]
    risk_management_guidelines: str
    educational_insights: str
    
class ScalpTradingReport(BaseModel):
    """Complete scalp trading report."""
    timestamp: str
    financial_analysis: FinancialAnalysis
    trading_analysis: TradingAnalysis
    markdown_report: str

# Helper Functions
def get_price_range(data_points):
    """Calculate the min and max values from a list of data points."""
    if not data_points:
        return (0, 0)
    low_values = [point.low for point in data_points]
    high_values = [point.high for point in data_points]
    return (min(low_values), max(high_values))

def create_interactive_chart(data_series, technical_indicators=None, key_levels=None, trading_opportunities=None):
    """Create an interactive candlestick chart with technical indicators and trading signals.
    
    Args:
        data_series: FinancialDataSeries object containing price data
        technical_indicators: Optional TechnicalIndicators object with indicator values
        key_levels: Optional list of PriceLevel objects
        trading_opportunities: Optional list of ScalpTradingOpportunity objects
        
    Returns:
        Plotly figure object that can be displayed in a browser
    """
    # Convert data to format suitable for plotting
    timestamps = [pd.to_datetime(point.timestamp) for point in data_series.data]
    
    # Create subplots: 1 for price, 1 for volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, subplot_titles=(f'{data_series.symbol} - {data_series.interval}', 'Volume'),
                       row_heights=[0.8, 0.2])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=timestamps,
            open=[point.open for point in data_series.data],
            high=[point.high for point in data_series.data],
            low=[point.low for point in data_series.data],
            close=[point.close for point in data_series.data],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=[point.volume for point in data_series.data],
            name="Volume",
            marker_color='rgba(0, 150, 255, 0.6)'
        ),
        row=2, col=1
    )
    
    # Add technical indicators if provided
    if technical_indicators:
        # Add moving averages
        try:
            for ma_name, ma_value in technical_indicators.moving_averages.items():
                if ma_name.startswith('MA'):
                    period = int(ma_name[2:])
                    # Calculate MA for all data points
                    closes = [point.close for point in data_series.data]
                    ma_values = calculate_moving_averages(closes, [period])[f'MA{period}']
                    
                    # Ensure we have valid data for plotting
                    if ma_values and len(ma_values) == len(timestamps):
                        fig.add_trace(
                            go.Scatter(
                                x=timestamps,
                                y=ma_values,
                                mode='lines',
                                name=f'{ma_name}',
                                line=dict(width=1.5),
                                connectgaps=True  # Connect gaps from None values
                            ),
                            row=1, col=1
                        )
        except Exception as e:
            logger.error(f"Error plotting moving averages: {e}")
            # Continue without MAs if there's an error
        
        # Add RSI if available
        if technical_indicators.rsi is not None:
            try:
                closes = [point.close for point in data_series.data]
                # Calculate RSI values for all data points
                rsi_values = []
                for i in range(len(closes)):
                    if i < TECHNICAL_PARAMS['rsi_period']:
                        # Not enough data for RSI calculation at beginning
                        rsi_values.append(None)
                    else:
                        # Calculate RSI for this window
                        window = closes[i-TECHNICAL_PARAMS['rsi_period']:i+1]
                        rsi = calculate_rsi(window, TECHNICAL_PARAMS['rsi_period'])
                        rsi_values.append(rsi)
                
                # Create a new row for RSI
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=rsi_values,
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=1.5),
                        connectgaps=True  # Connect gaps from None values
                    ),
                    row=1, col=1
                )
            except Exception as e:
                logger.error(f"Error plotting RSI: {e}")
                # Continue without RSI if there's an error
            
            # Add RSI reference lines (30 and 70)
            fig.add_hline(y=30, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    
    # Add key price levels if provided
    if key_levels:
        for level in key_levels:
            fig.add_hline(
                y=level.price,
                line_dash="dash",
                line_color="rgba(255, 165, 0, 0.7)",
                annotation_text=f"{level.level_type}: {level.price}",
                annotation_position="right",
                row=1, col=1
            )
    
    # Add trading opportunities if provided
    if trading_opportunities:
        for opportunity in trading_opportunities:
            if opportunity.symbol == data_series.symbol:
                # Use the last timestamp for plotting trading signals
                signal_x = timestamps[-1] if timestamps else pd.Timestamp.now()
                
                try:
                    # Add entry point
                    entry_price = float(opportunity.entry_strategy.entry_price)
                    fig.add_trace(
                        go.Scatter(
                            x=[signal_x],
                            y=[entry_price],
                            mode='markers',
                            name='Entry',
                            marker=dict(
                                symbol='triangle-up' if opportunity.direction.lower() == 'long' or opportunity.direction.lower() == 'buy' else 'triangle-down',
                                size=12,
                                color='green' if opportunity.direction.lower() == 'long' or opportunity.direction.lower() == 'buy' else 'red',
                            )
                        ),
                        row=1, col=1
                    )
                    
                    # Add stop loss
                    stop_loss = float(opportunity.exit_strategy.stop_loss)
                    fig.add_trace(
                        go.Scatter(
                            x=[signal_x],
                            y=[stop_loss],
                            mode='markers',
                            name='Stop Loss',
                            marker=dict(symbol='x', size=10, color='red')
                        ),
                        row=1, col=1
                    )
                    
                    # Add take profit levels
                    for i, tp in enumerate(opportunity.exit_strategy.take_profit_levels):
                        tp_price = float(tp.price)
                        fig.add_trace(
                            go.Scatter(
                                x=[signal_x],
                                y=[tp_price],
                                mode='markers',
                                name=f'TP{i+1}',
                                marker=dict(symbol='diamond', size=10, color='green')
                            ),
                            row=1, col=1
                        )
                except (ValueError, TypeError) as e:
                    logger.error(f"Error plotting trading signals: {e}")
                    # Continue with chart generation even if trading signals can't be plotted
    
    # Update layout
    fig.update_layout(
        title=f'{data_series.symbol} ({data_series.name}) - {data_series.interval} Timeframe',
        xaxis_title='Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=800,
        width=1200,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=85, b=50),
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def calculate_moving_averages(closes, periods):
    """Calculate moving averages for specified periods.
    
    Args:
        closes: List of closing prices
        periods: List of periods to calculate MAs for
        
    Returns:
        Dictionary with MA values for each period, where values are arrays
    """
    moving_averages = {}
    
    for period in periods:
        # Initialize array with None values for periods where MA can't be calculated
        ma_values = [None] * (period - 1)
        
        # Calculate MA for each window once we have enough data points
        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1:i + 1]
            ma = sum(window) / period
            ma_values.append(round(ma, 2))
            
        moving_averages[f"MA{period}"] = ma_values
        
    return moving_averages

def calculate_exponential_ma(closes, period):
    """Calculate exponential moving average."""
    if len(closes) < period:
        return None
    
    # Convert to numpy array for calculations
    closes_array = np.array(closes)
    
    # Calculate multiplier
    multiplier = 2 / (period + 1)
    
    # Initialize EMA with SMA
    ema = np.mean(closes_array[:period])
    
    # Calculate EMA for remaining periods
    for i in range(period, len(closes_array)):
        ema = (closes_array[i] * multiplier) + (ema * (1 - multiplier))
    
    return round(ema, 2)

def calculate_rsi(closes, period=14):
    """Calculate Relative Strength Index.
    
    Returns an array of RSI values for the entire price series.
    """
    if len(closes) < period + 1:
        return None
    
    # Calculate price changes
    deltas = np.diff(closes)
    
    # Create arrays for gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Initialize arrays for RSI calculation
    avg_gains = np.zeros_like(closes)
    avg_losses = np.zeros_like(closes)
    rsi_values = np.zeros_like(closes)
    
    # Fill with None for the first period elements where RSI can't be calculated
    rsi_values[:period] = None
    
    # Calculate first average gain and loss
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])
    
    # Calculate first RSI value
    if avg_losses[period] == 0:
        rsi_values[period] = 100
    else:
        rs = avg_gains[period] / avg_losses[period]
        rsi_values[period] = 100 - (100 / (1 + rs))
    
    # Calculate subsequent RSI values using smoothing
    for i in range(period + 1, len(closes)):
        avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i-1]) / period
        avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i-1]) / period
        
        if avg_losses[i] == 0:
            rsi_values[i] = 100
        else:
            rs = avg_gains[i] / avg_losses[i]
            rsi_values[i] = 100 - (100 / (1 + rs))
    
    # Round values and convert to list
    rsi_values = [None if np.isnan(x) else round(float(x), 2) for x in rsi_values]
    
    return rsi_values

def calculate_atr(highs, lows, closes, period=14):
    """Calculate Average True Range."""
    if len(highs) < period + 1:
        return None
    
    # Convert to numpy arrays
    highs_array = np.array(highs)
    lows_array = np.array(lows)
    closes_array = np.array(closes)
    
    # Calculate true ranges
    tr1 = np.abs(highs_array[1:] - lows_array[1:])
    tr2 = np.abs(highs_array[1:] - closes_array[:-1])
    tr3 = np.abs(lows_array[1:] - closes_array[:-1])
    
    # Combine true ranges
    tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
    
    # Calculate ATR
    atr = np.mean(tr[:period])
    
    return round(atr, 4)

# Nodes
@Nodes.define(output="multi_timeframe_request")
async def prepare_multi_timeframe_request(
    symbols_list: List[str], 
    timeframe_type: str = "scalping"
) -> MultiTimeframeRequest:
    """Prepare the multi-timeframe request with the provided symbols and timeframe type."""
    # Log the raw input for debugging
    logger.debug(f"Raw symbols_list: {symbols_list}")
    
    # Ensure we have valid symbols
    if not symbols_list:
        logger.error("No symbols provided")
        raise ValueError("No symbols provided")
    
    # Create symbol objects
    symbols = []
    for symbol in symbols_list:
        # Skip any command-line artifacts that might have been passed
        if symbol.startswith('-') or symbol in ['analyze']:
            logger.debug(f"Skipping invalid symbol: {symbol}")
            continue
        symbols.append(FinancialSymbol(symbol=symbol))
    
    # Ensure we have at least one valid symbol
    if not symbols:
        logger.error("No valid symbols found in the input list")
        raise ValueError("No valid symbols found in the input list")
    
    # Get timeframes based on type
    intervals = TIMEFRAMES.get(timeframe_type, TIMEFRAMES["scalping"])
    
    # Create timeframe requests
    timeframes = []
    for interval in intervals:
        # Determine appropriate range type and dates based on interval
        if interval in ["1m", "2m", "5m"]:
            range_type = "date"
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        elif interval in ["15m", "30m", "60m", "90m"]:
            range_type = "week"
            start_date = None
            end_date = None
        else:
            range_type = "month"
            start_date = None
            end_date = None
        
        timeframes.append(TimeframeRequest(
            interval=interval,
            range_type=range_type,
            start_date=start_date,
            end_date=end_date
        ))
    
    logger.info(f"Preparing multi-timeframe request for {len(symbols)} symbols with {timeframe_type} timeframes")
    return MultiTimeframeRequest(
        symbols=symbols,
        timeframes=timeframes
    )

@Nodes.define(output="multi_timeframe_data")
async def fetch_multi_timeframe_data(request: MultiTimeframeRequest) -> Dict[str, FinancialDataCollection]:
    """Fetch financial data for multiple timeframes."""
    # Validate input
    if request is None:
        logger.error("Request is None in fetch_multi_timeframe_data")
        raise ValueError("Request is None in fetch_multi_timeframe_data")
    
    if not hasattr(request, 'timeframes') or not request.timeframes:
        logger.error("No timeframes in request")
        raise ValueError("No timeframes in request")
    
    if not hasattr(request, 'symbols') or not request.symbols:
        logger.error("No symbols in request")
        raise ValueError("No symbols in request")
    
    yahoo_tool = YahooFinanceTool()
    timeframe_data = {}
    
    # Log the request details for debugging
    logger.debug(f"Request symbols: {[s.symbol for s in request.symbols]}")
    logger.debug(f"Request timeframes: {[t.interval for t in request.timeframes]}")
    
    for timeframe_req in request.timeframes:
        series_list = []
        
        for symbol_obj in request.symbols:
            symbol = symbol_obj.symbol
            logger.info(f"Fetching data for {symbol} with {timeframe_req.interval} interval")
            
            try:
                # Fetch data using the Yahoo Finance tool
                data = await yahoo_tool.execute(
                    symbol=symbol,
                    interval=timeframe_req.interval,
                    range_type=timeframe_req.range_type,
                    start_date=timeframe_req.start_date,
                    end_date=timeframe_req.end_date
                )
                
                # Process the data
                data_points = []
                for point in data.get("data", []):
                    data_points.append(FinancialDataPoint(
                        timestamp=point["timestamp"],
                        open=point["open"],
                        high=point["high"],
                        low=point["low"],
                        close=point["close"],
                        volume=point["volume"]
                    ))
                
                # Create a data series for this symbol
                series = FinancialDataSeries(
                    symbol=symbol,
                    name=data.get("info", {}).get("shortName", symbol),
                    interval=timeframe_req.interval,
                    range_type=timeframe_req.range_type,
                    start_date=data.get("start_date", ""),
                    end_date=data.get("end_date", ""),
                    data_points=len(data_points),
                    data=data_points,
                    currency=data.get("info", {}).get("currency"),
                    exchange=data.get("info", {}).get("exchange"),
                    sector=data.get("info", {}).get("sector"),
                    industry=data.get("info", {}).get("industry")
                )
                
                series_list.append(series)
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol} with {timeframe_req.interval} interval: {e}")
        
        # Create collection for this timeframe
        collection = FinancialDataCollection(
            series=series_list,
            timestamp=datetime.now().isoformat()
        )
        
        timeframe_data[timeframe_req.interval] = collection
    
    logger.info(f"Fetched data for {len(request.symbols)} symbols across {len(request.timeframes)} timeframes")
    return timeframe_data

@Nodes.define(output="multi_timeframe_technical_analysis")
async def calculate_multi_timeframe_technical_indicators(
    multi_timeframe_data: Dict[str, FinancialDataCollection]
) -> Dict[str, TechnicalAnalysis]:
    """Calculate technical indicators for each symbol across multiple timeframes."""
    multi_timeframe_analysis = {}
    
    for timeframe, data_collection in multi_timeframe_data.items():
        indicators_list = []
        
        for series in data_collection.series:
            # Extract price data
            closes = [point.close for point in series.data]
            highs = [point.high for point in series.data]
            lows = [point.low for point in series.data]
            volumes = [point.volume for point in series.data]
            
            # Calculate moving averages - returns arrays for each MA period
            moving_averages = calculate_moving_averages(closes, TECHNICAL_PARAMS["ma_periods"])
            
            # Calculate exponential moving averages
            exponential_mas = {}
            for period in TECHNICAL_PARAMS["ema_periods"]:
                ema = calculate_exponential_ma(closes, period)
                if ema is not None:
                    # Convert single value to float if needed
                    if not isinstance(ema, list):
                        exponential_mas[f"EMA{period}"] = float(ema)
                    else:
                        exponential_mas[f"EMA{period}"] = ema
            
            # Calculate RSI
            rsi = calculate_rsi(closes, TECHNICAL_PARAMS["rsi_period"])
            
            # Calculate MACD
            macd = None
            if len(closes) >= TECHNICAL_PARAMS["macd_params"]["slow"] + 1:
                fast_ema = calculate_exponential_ma(closes, TECHNICAL_PARAMS["macd_params"]["fast"])
                slow_ema = calculate_exponential_ma(closes, TECHNICAL_PARAMS["macd_params"]["slow"])
                
                if fast_ema is not None and slow_ema is not None:
                    macd_line = fast_ema - slow_ema
                    
                    # Use the last (slow) periods of MACD line to calculate signal line
                    signal_line = None
                    if len(closes) >= TECHNICAL_PARAMS["macd_params"]["slow"] + TECHNICAL_PARAMS["macd_params"]["signal"]:
                        # This is simplified; in practice, you'd calculate EMA of the MACD line
                        signal_line = sum([macd_line]) / TECHNICAL_PARAMS["macd_params"]["signal"]
                    
                    macd = {
                        "macd_line": round(macd_line, 2),
                        "signal_line": round(signal_line, 2) if signal_line else None,
                        "histogram": round(macd_line - signal_line, 2) if signal_line else None
                    }
            
            # Calculate Bollinger Bands
            bollinger_bands = None
            if len(closes) >= TECHNICAL_PARAMS["bollinger_params"]["period"]:
                period = TECHNICAL_PARAMS["bollinger_params"]["period"]
                std_dev = TECHNICAL_PARAMS["bollinger_params"]["std_dev"]
                
                # Calculate middle band (SMA)
                middle_band = sum(closes[-period:]) / period
                
                # Calculate standard deviation
                variance = sum([(x - middle_band) ** 2 for x in closes[-period:]]) / period
                std = variance ** 0.5
                
                # Calculate upper and lower bands
                upper_band = middle_band + (std * std_dev)
                lower_band = middle_band - (std * std_dev)
                
                bollinger_bands = {
                    "upper_band": round(upper_band, 2),
                    "middle_band": round(middle_band, 2),
                    "lower_band": round(lower_band, 2),
                    "bandwidth": round((upper_band - lower_band) / middle_band, 2)
                }
            
            # Calculate ATR
            atr = calculate_atr(highs, lows, closes, TECHNICAL_PARAMS["atr_period"])
            
            # Calculate Volume Analysis
            volume_analysis = None
            if len(volumes) >= TECHNICAL_PARAMS["volume_ma_period"]:
                avg_volume = sum(volumes[-TECHNICAL_PARAMS["volume_ma_period"]:]) / TECHNICAL_PARAMS["volume_ma_period"]
                
                # Determine volume trend
                recent_avg = sum(volumes[-5:]) / 5
                if recent_avg > avg_volume * 1.2:
                    volume_trend = "increasing"
                elif recent_avg < avg_volume * 0.8:
                    volume_trend = "decreasing"
                else:
                    volume_trend = "flat"
                
                # Identify volume spikes
                volume_spikes = []
                for i, vol in enumerate(volumes[-10:]):
                    if vol > avg_volume * 2:
                        timestamp = series.data[-(10-i)].timestamp
                        volume_spikes.append(timestamp)
                
                # Check for price/volume divergence
                price_trend = "up" if closes[-1] > closes[-5] else "down"
                volume_trend_direction = "up" if recent_avg > avg_volume else "down"
                volume_divergence = price_trend != volume_trend_direction
                
                volume_analysis = VolumeAnalysis(
                    symbol=series.symbol,
                    average_volume=round(avg_volume, 0),
                    volume_trend=volume_trend,
                    volume_spikes=volume_spikes,
                    volume_divergence=volume_divergence
                )
            
            # Create technical indicators object
            indicators = TechnicalIndicators(
                symbol=series.symbol,
                timeframe=timeframe,
                moving_averages=moving_averages,
                exponential_mas=exponential_mas,
                rsi=rsi,
                macd=macd,
                bollinger_bands=bollinger_bands,
                atr=atr,
                volume_analysis=volume_analysis
            )
            
            indicators_list.append(indicators)
        
        # Create technical analysis for this timeframe
        technical_analysis = TechnicalAnalysis(
            indicators=indicators_list,
            timestamp=datetime.now().isoformat()
        )
        
        multi_timeframe_analysis[timeframe] = technical_analysis
    
    logger.info(f"Calculated technical indicators across {len(multi_timeframe_analysis)} timeframes")
    return multi_timeframe_analysis

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_analyze_financial_data.j2"),
    output="financial_analysis",
    response_model=FinancialAnalysis,
    prompt_file=get_template_path("prompt_analyze_financial_data.j2"),
    max_tokens=7000,
    model="gpt-4o-mini"  # Specifying a default model
)
async def analyze_financial_data(
    multi_timeframe_data: Dict[str, FinancialDataCollection], 
    multi_timeframe_technical_analysis: Dict[str, TechnicalAnalysis],
    get_price_range,
    model: str = "gpt-4o-mini"  # Default model parameter
) -> FinancialAnalysis:
    """Analyze financial data across multiple timeframes."""
    logger.debug(f"analyze_financial_data called with model: {model}")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_generate_trading_recommendations.j2"),
    output="trading_analysis",
    response_model=TradingAnalysis,
    prompt_file=get_template_path("prompt_generate_trading_recommendations.j2"),
    max_tokens=7000,
    model="gpt-4o-mini"  # Specifying a default model
)
async def generate_trading_recommendations(
    multi_timeframe_data: Dict[str, FinancialDataCollection],
    multi_timeframe_technical_analysis: Dict[str, TechnicalAnalysis],
    financial_analysis: FinancialAnalysis,
    get_price_range,
    model: str = "gpt-4o-mini"  # Default model parameter
) -> TradingAnalysis:
    """Generate trading recommendations based on financial analysis."""
    logger.debug(f"generate_trading_recommendations called with model: {model}")
    pass

@Nodes.define(output="scalp_trading_report")
async def generate_scalp_trading_report(
    financial_analysis: FinancialAnalysis,
    trading_analysis: TradingAnalysis,
    multi_timeframe_data: Dict[str, FinancialDataCollection],
    multi_timeframe_technical_analysis: Dict[str, TechnicalAnalysis]
) -> ScalpTradingReport:
    """Generate a comprehensive scalp trading report."""
    # Create markdown report
    markdown = f"""
# Scalp Trading Analysis Report
*Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Market Analysis
{financial_analysis.market_context}

### Market Structure
{financial_analysis.market_structure_analysis}

### Technical Analysis
{financial_analysis.technical_analysis_summary}

### Volume Analysis
{financial_analysis.volume_analysis}

### Key Levels
{financial_analysis.key_levels_analysis}

### Multi-Timeframe Analysis
{financial_analysis.multi_timeframe_analysis}

## Trading Recommendations

### Market Outlook
{trading_analysis.market_outlook}

### Trading Opportunities
"""
    
    # Add each trading opportunity
    for i, opportunity in enumerate(trading_analysis.trading_opportunities, 1):
        markdown += f"""
#### Opportunity {i}: {opportunity.symbol} {opportunity.direction.upper()}
- **Timeframe:** {opportunity.timeframe}
- **Confidence:** {opportunity.confidence_level}/10
- **Expected Duration:** {opportunity.expected_duration}

**Entry Strategy:**
- Entry Price: ${opportunity.entry_strategy.entry_price}
- Entry Type: {opportunity.entry_strategy.entry_type}
- Trigger: {opportunity.entry_strategy.entry_trigger}
- Rationale: {opportunity.entry_strategy.entry_rationale}

**Exit Strategy:**
- Stop Loss: ${opportunity.exit_strategy.stop_loss}
- Take Profit Levels:
"""
        
        for j, tp in enumerate(opportunity.exit_strategy.take_profit_levels, 1):
            markdown += f"  - TP{j}: ${tp.price} ({tp.rationale})\n"
        
        markdown += f"""
- Exit Rationale: {opportunity.exit_strategy.exit_rationale}

**Position Sizing:**
- Risk: {opportunity.position_sizing.risk_percentage}% of account
- Position Size: {opportunity.position_sizing.position_size}
- Risk:Reward: {opportunity.position_sizing.risk_reward_ratio}
- Max Loss: {opportunity.position_sizing.max_loss_amount}

**Key Considerations:**
"""
        
        for consideration in opportunity.key_considerations:
            markdown += f"- {consideration}\n"
    
    # Add risk management guidelines
    markdown += f"""
### Risk Management Guidelines
{trading_analysis.risk_management_guidelines}

### Educational Insights
{trading_analysis.educational_insights}
"""
    
    # Generate interactive charts for each symbol and timeframe
    # Create reports directory if it doesn't exist
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    # Create charts subdirectory within reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    charts_dir = report_dir / f"charts_{timestamp}"
    charts_dir.mkdir(exist_ok=True)
    
    chart_paths = []
    
    try:
        # Extract trading opportunities for chart annotations
        trading_opportunities = trading_analysis.trading_opportunities
        
        # Generate charts for each timeframe and symbol
        for timeframe, data_collection in multi_timeframe_data.items():
            # Find corresponding technical indicators
            tech_indicators = multi_timeframe_technical_analysis.get(timeframe)
            
            for series in data_collection.series:
                # Skip if no data points
                if not series.data or len(series.data) < 2:
                    logger.warning(f"Not enough data points for {series.symbol} in {timeframe} timeframe")
                    continue
                    
                try:
                    # Find corresponding technical indicators for this symbol
                    series_indicators = None
                    if tech_indicators:
                        for indicator in tech_indicators.indicators:
                            if indicator.symbol == series.symbol and indicator.timeframe == timeframe:
                                series_indicators = indicator
                                break
                    
                    # Create interactive chart
                    fig = create_interactive_chart(
                        data_series=series,
                        technical_indicators=series_indicators,
                        trading_opportunities=trading_opportunities
                    )
                    
                    # Save chart to HTML file
                    chart_filename = f"{series.symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    chart_path = charts_dir / chart_filename
                    fig.write_html(str(chart_path))
                    
                    # Add chart path to list
                    chart_paths.append(str(chart_path))
                    
                    # Add chart link to markdown report - use relative path for better portability
                    relative_path = f"charts_{timestamp}/{chart_filename}"
                    markdown += f"\n### Interactive Chart: [{series.symbol} - {timeframe}]({relative_path})\n"
                    
                    logger.info(f"Generated interactive chart for {series.symbol} in {timeframe} timeframe")
                except Exception as e:
                    logger.error(f"Error generating chart for {series.symbol} in {timeframe} timeframe: {e}")
    except Exception as e:
        logger.error(f"Error in chart generation process: {e}")
        # Continue with report generation even if charts fail
    
    # Create the report
    report = ScalpTradingReport(
        timestamp=datetime.now().isoformat(),
        financial_analysis=financial_analysis,
        trading_analysis=trading_analysis,
        markdown_report=markdown
    )
    
    logger.info(f"Generated comprehensive scalp trading report with {len(chart_paths)} interactive charts")
    return report

# Create the workflow
def create_scalp_trading_workflow() -> Workflow:
    """Create a workflow for scalp trading analysis."""
    workflow = (
        Workflow("prepare_multi_timeframe_request")
        .then("fetch_multi_timeframe_data")
        .then("calculate_multi_timeframe_technical_indicators")
        .then("analyze_financial_data")
        .then("generate_trading_recommendations")
        .then("generate_scalp_trading_report")
    )
    
    # Add input mappings with explicit data flow
    workflow.node_input_mappings = {
        "fetch_multi_timeframe_data": {
            "request": "multi_timeframe_request"
        },
        "calculate_multi_timeframe_technical_indicators": {
            "multi_timeframe_data": "multi_timeframe_data"
        },
        "analyze_financial_data": {
            "multi_timeframe_data": "multi_timeframe_data",
            "multi_timeframe_technical_analysis": "multi_timeframe_technical_analysis",
            "get_price_range": lambda _: get_price_range,
            "model": "analysis_model"  # Pass the model parameter
        },
        "generate_trading_recommendations": {
            "multi_timeframe_data": "multi_timeframe_data",
            "multi_timeframe_technical_analysis": "multi_timeframe_technical_analysis",
            "financial_analysis": "financial_analysis",
            "get_price_range": lambda _: get_price_range,
            "model": "trading_model"  # Pass the model parameter
        },
        "generate_scalp_trading_report": {
            "financial_analysis": "financial_analysis",
            "trading_analysis": "trading_analysis",
            "multi_timeframe_data": "multi_timeframe_data",
            "multi_timeframe_technical_analysis": "multi_timeframe_technical_analysis"
        }
    }
    
    return workflow

async def analyze_for_scalp_trading(
    symbols: List[str],
    timeframe_type: str = "scalping",
    analysis_model: str = "gpt-4o-mini",
    trading_model: str = "gpt-4o-mini",
    task_id: str = "default"
) -> ScalpTradingReport:
    """Analyze financial data for scalp trading opportunities."""
    
    if not symbols:
        raise ValueError("At least one symbol must be provided")
    
    if timeframe_type not in TIMEFRAMES:
        raise ValueError(f"timeframe_type must be one of: {', '.join(TIMEFRAMES.keys())}")
    
    logger.info(f"Starting scalp trading analysis for {len(symbols)} symbols with {timeframe_type} timeframes")
    logger.info(f"Using models - Analysis: {analysis_model}, Trading: {trading_model}")
    
    initial_context = {
        "symbols_list": symbols,
        "timeframe_type": timeframe_type,
        "analysis_model": analysis_model,
        "trading_model": trading_model
    }
    
    try:
        workflow = create_scalp_trading_workflow()
        engine = workflow.build()
        
        result = await engine.run(initial_context)
        
        if not isinstance(result.get("scalp_trading_report"), ScalpTradingReport):
            raise ValueError("Workflow did not produce a valid scalp trading report")
        
        logger.info("Scalp trading analysis completed successfully")
        
        # Display the report
        console.print("\n[bold blue]Scalp Trading Analysis Completed[/]")
        console.print(Panel(Markdown(result["scalp_trading_report"].markdown_report), 
                            title="Scalp Trading Report"))
        
        # Save the report to a file - use the same timestamp from the report object
        report_timestamp = result["scalp_trading_report"].timestamp
        # Convert ISO format to filename-friendly format
        report_datetime = datetime.fromisoformat(report_timestamp.replace('Z', '+00:00'))
        timestamp = report_datetime.strftime("%Y%m%d_%H%M%S")
        
        filename = f"scalp_trading_report_{timestamp}.md"
        report_path = Path("reports") / filename
        
        with open(report_path, "w") as f:
            f.write(result["scalp_trading_report"].markdown_report)
        
        console.print(f"[bold green]Report saved to:[/] {report_path}")
        
        return result["scalp_trading_report"]
        
    except Exception as e:
        logger.error(f"Error in scalp trading analysis: {e}")
        raise

@app.command()
def analyze(
    symbols: Annotated[List[str], typer.Argument(help="List of symbols to analyze")],
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="Timeframe type (scalping, intraday, swing)")] = "scalping",
    analysis_model: Annotated[str, typer.Option("--analysis-model", help="Model to use for financial analysis")] = "gpt-4o-mini",
    trading_model: Annotated[str, typer.Option("--trading-model", help="Model to use for trading recommendations")] = "gpt-4o-mini"
):
    """Analyze symbols for scalp trading opportunities."""
    try:
        # Fix for command line parsing - ensure symbols are correctly parsed
        # When --timeframe is used, Typer might incorrectly include it in symbols
        filtered_symbols = [s for s in symbols if s not in ["--timeframe", "-t", "--analysis-model", "--trading-model"] 
                           and not s.startswith("--") and s != "analyze"]
        
        if not filtered_symbols:
            console.print("[bold red]Error:[/] No valid symbols provided")
            raise typer.Exit(code=1)
            
        console.print(f"[bold blue]Analyzing symbols:[/] {', '.join(filtered_symbols)}")
        console.print(f"[bold blue]Timeframe:[/] {timeframe}")
        console.print(f"[bold blue]Models:[/] Analysis: {analysis_model}, Trading: {trading_model}")
        
        asyncio.run(analyze_for_scalp_trading(
            symbols=filtered_symbols, 
            timeframe_type=timeframe,
            analysis_model=analysis_model,
            trading_model=trading_model
        ))
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
