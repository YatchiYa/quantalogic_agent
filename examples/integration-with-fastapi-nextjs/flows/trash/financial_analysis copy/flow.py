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
from models import StructuredAnalysis, MarketSentiment, TechnicalAnalysis, TradeRecommendation
from plotting import (
    create_advanced_candlestick_plot,
    create_pattern_analysis_plot,
    create_market_profile_plot,
    create_strategy_plot,
    create_detailed_html_report,
    create_smc_plot,
    create_position_plot
)

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
    
    # Get smart money analysis
    smc_analysis = await analyze_smart_money(df)
    
    # Create plots with enhanced visualization
    plots = [
        create_advanced_candlestick_plot(df, f"Technical Analysis - {market_data['symbol']}"),
        create_market_profile_plot(df, f"Market Profile - {market_data['symbol']}"),
        create_smc_plot(df, smc_analysis, f"Smart Money Analysis - {market_data['symbol']}"),
        create_position_plot(df, market_data['symbol'])
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
        "plots": plots,
        "smc_analysis": smc_analysis
    }

@Nodes.define(output=None)
async def analyze_smart_money(df: pd.DataFrame) -> dict:
    """Analyze market structure using ICT and SMC concepts."""
    
    # Cache commonly used values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    # Identify key ICT levels
    ict_levels = []
    
    # Find fair value gaps (FVG)
    for i in range(2, len(df)):
        if lows[i] > highs[i-2]:  # Bullish FVG
            ict_levels.append({
                'type': 'bullish_fvg',
                'level': (lows[i] + highs[i-2]) / 2,
                'strength': 8,
                'description': f'Bullish Fair Value Gap at {((lows[i] + highs[i-2]) / 2):.2f}'
            })
        elif highs[i] < lows[i-2]:  # Bearish FVG
            ict_levels.append({
                'type': 'bearish_fvg',
                'level': (highs[i] + lows[i-2]) / 2,
                'strength': 8,
                'description': f'Bearish Fair Value Gap at {((highs[i] + lows[i-2]) / 2):.2f}'
            })
    
    # Find liquidity levels (stop runs)
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(df)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append({
                'price': highs[i],
                'index': i
            })
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append({
                'price': lows[i],
                'index': i
            })
    
    # Identify SMC order blocks
    order_blocks = []
    
    for i in range(3, len(df)):
        # Bullish order block
        if closes[i] > closes[i-1] * 1.005:  # Strong move up
            ob_high = max(highs[i-3:i])
            ob_low = min(lows[i-3:i])
            order_blocks.append({
                'type': 'bullish_ob',
                'high': ob_high,
                'low': ob_low,
                'index': i-3,
                'strength': 9,
                'description': f'Bullish Order Block {ob_low:.2f}-{ob_high:.2f}'
            })
        # Bearish order block
        elif closes[i] < closes[i-1] * 0.995:  # Strong move down
            ob_high = max(highs[i-3:i])
            ob_low = min(lows[i-3:i])
            order_blocks.append({
                'type': 'bearish_ob',
                'high': ob_high,
                'low': ob_low,
                'index': i-3,
                'strength': 9,
                'description': f'Bearish Order Block {ob_low:.2f}-{ob_high:.2f}'
            })
    
    return {
        "ict_levels": ict_levels,
        "swing_points": {"highs": swing_highs, "lows": swing_lows},
        "order_blocks": order_blocks
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
    
    return {"technical_indicators": indicators}

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
        "pattern_plots": plots
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
        )
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
                'description': 'EMA9 crossed above EMA21'
            })
        elif ema9.iloc[i-1] > ema21.iloc[i-1] and ema9.iloc[i] < ema21.iloc[i]:
            trend_signals.append({
                'type': 'sell',
                'timestamp': df.index[i],
                'price': df['Close'].iloc[i],
                'description': 'EMA9 crossed below EMA21'
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
    
    return {
        "strategies": strategies,
        "strategy_plots": strategy_plots
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
    
    return {"kpis": kpis}

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
    kpis: Dict[str, float]
) -> dict:
    """Compile all analyses into a final report."""
    # Combine all plots
    all_plots = plots + pattern_plots + strategy_plots
    
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
        plots=all_plots
    )
    
    return {"analysis": analysis}

# Define the workflow
workflow = (
    Workflow("fetch_market_data")
    .then("analyze_price_action")
    .then("analyze_technical_indicators")
    .then("analyze_patterns")
    .then("analyze_market_structure")
    .then("analyze_strategies")
    .then("analyze_kpis")
    .then("compile_analysis")
)

def analyze_market(
    symbol: str,
    interval: str = "5m",
    days: int = 30,
) -> MarketAnalysis:
    """Run the market analysis workflow."""
    initial_context = {
        "symbol": symbol,
        "interval": interval,
        "days": days
    }
    
    logger.info(f"Starting market analysis for {symbol}")
    engine = workflow.build()
    result = asyncio.run(engine.run(initial_context))
    logger.info("Market analysis completed successfully ")
    return result["analysis"]

def main():
    """Test the financial market analysis flow."""
    import os
    from pprint import pprint
    from plotting import create_detailed_html_report
    
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
        analysis = analyze_market(
            symbol=symbol,
            interval="5m",
            days=30 # Last 7 days
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
        
        # Generate and save detailed HTML report
        detailed_report = create_detailed_html_report(analysis)
        report_path = os.path.join(output_dir, f"{symbol}_detailed_report.html")
        with open(report_path, "wb") as f:
            f.write(base64.b64decode(detailed_report))
        print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    main()
