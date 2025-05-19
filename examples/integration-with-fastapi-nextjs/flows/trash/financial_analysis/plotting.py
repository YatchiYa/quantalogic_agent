"""Advanced plotting module for financial analysis."""
import base64
import io
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from ta.trend import EMAIndicator, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

def create_advanced_candlestick_plot(df: pd.DataFrame, title: str) -> str:
    """Create an advanced candlestick chart with multiple indicators."""
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=(
            'Price & Indicators',
            'Volume',
            'RSI & Stochastic',
            'MACD'
        )
    )

    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            showlegend=True
        ),
        row=1, col=1
    )

    # Add EMAs
    ema_periods = [9, 21, 50, 200]
    ema_colors = ['yellow', 'orange', 'blue', 'red']
    
    for period, color in zip(ema_periods, ema_colors):
        ema = EMAIndicator(df['Close'], period).ema_indicator()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ema,
                name=f'EMA{period}',
                line=dict(color=color, width=1),
                showlegend=True
            ),
            row=1, col=1
        )

    # Add Bollinger Bands
    bb = BollingerBands(df['Close'])
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=bb.bollinger_hband(),
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=bb.bollinger_lband(),
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            showlegend=True
        ),
        row=1, col=1
    )

    # Add Ichimoku Cloud
    ichimoku = IchimokuIndicator(df['High'], df['Low'])
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=ichimoku.ichimoku_conversion_line(),
            name='Conversion Line',
            line=dict(color='blue', width=1),
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=ichimoku.ichimoku_base_line(),
            name='Base Line',
            line=dict(color='red', width=1),
            showlegend=True
        ),
        row=1, col=1
    )

    # Add volume
    colors = ['red' if row['Open'] > row['Close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=True
        ),
        row=2, col=1
    )

    # Add VWAP
    vwap = VolumeWeightedAveragePrice(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume']
    ).volume_weighted_average_price()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=vwap,
            name='VWAP',
            line=dict(color='purple', width=1),
            showlegend=True
        ),
        row=2, col=1
    )

    # Add RSI
    rsi = RSIIndicator(df['Close']).rsi()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=rsi,
            name='RSI',
            line=dict(color='blue', width=1),
            showlegend=True
        ),
        row=3, col=1
    )

    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # Add Stochastic
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=stoch.stoch(),
            name='%K',
            line=dict(color='orange', width=1),
            showlegend=True
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=stoch.stoch_signal(),
            name='%D',
            line=dict(color='blue', width=1),
            showlegend=True
        ),
        row=3, col=1
    )

    # Add MACD
    macd = ta.trend.MACD(df['Close'])
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=macd.macd(),
            name='MACD',
            line=dict(color='blue', width=1),
            showlegend=True
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=macd.macd_signal(),
            name='Signal',
            line=dict(color='orange', width=1),
            showlegend=True
        ),
        row=4, col=1
    )
    
    # Add MACD histogram
    colors = ['red' if val < 0 else 'green' for val in macd.macd_diff()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=macd.macd_diff(),
            name='MACD Hist',
            marker_color=colors,
            showlegend=True
        ),
        row=4, col=1
    )

    # Update layout
    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=1200,
        xaxis_rangeslider_visible=False,
        margin=dict(t=30, l=0, r=0, b=0)
    )

    # Update Y-axes labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI/Stoch", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    # Convert to base64
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_str = buffer.getvalue()
    return base64.b64encode(html_str.encode()).decode()

def create_pattern_analysis_plot(df: pd.DataFrame, patterns: List[dict], title: str) -> str:
    """Create a chart highlighting detected patterns."""
    fig = go.Figure()

    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        )
    )

    # Add pattern annotations
    for pattern in patterns:
        fig.add_annotation(
            x=pattern['x'],
            y=pattern['y'],
            text=pattern['name'],
            showarrow=True,
            arrowhead=1,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=0,
            ay=-40
        )

    # Update layout
    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )

    # Convert to base64
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_str = buffer.getvalue()
    return base64.b64encode(html_str.encode()).decode()

def create_market_profile_plot(df: pd.DataFrame, title: str) -> str:
    """Create a market profile (volume profile) visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        specs=[[{"type": "candlestick"}, {"type": "bar"}]]
    )

    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # Calculate volume profile
    price_bins = np.linspace(df['Low'].min(), df['High'].max(), 50)
    volume_profile = []
    
    for i in range(len(price_bins)-1):
        mask = (df['Low'] >= price_bins[i]) & (df['High'] <= price_bins[i+1])
        volume_profile.append(df.loc[mask, 'Volume'].sum())

    # Add volume profile
    fig.add_trace(
        go.Bar(
            x=volume_profile,
            y=price_bins[:-1],
            orientation='h',
            name='Volume Profile',
            marker_color='rgba(128,128,128,0.5)'
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    # Convert to base64
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_str = buffer.getvalue()
    return base64.b64encode(html_str.encode()).decode()

def create_strategy_plot(df: pd.DataFrame, signals: List[dict], title: str) -> str:
    """Create an enhanced plot showing strategy signals and KPIs."""
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"type": "indicator"}, {"type": "indicator"}],
        ],
        row_heights=[0.5, 0.3, 0.2],
        vertical_spacing=0.08,
        subplot_titles=('Price & Signals', 'Cumulative Returns', 'Key Metrics')
    )

    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # Add EMAs for trend context
    for period, color in [(9, 'yellow'), (21, 'orange')]:
        ema = ta.trend.ema_indicator(df['Close'], window=period)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ema,
                name=f'EMA{period}',
                line=dict(color=color, width=1),
                showlegend=True
            ),
            row=1, col=1
        )

    # Add signals
    buy_signals = [s for s in signals if s['type'] == 'buy']
    sell_signals = [s for s in signals if s['type'] == 'sell']

    # Plot buy signals
    if buy_signals:
        fig.add_trace(
            go.Scatter(
                x=[s['timestamp'] for s in buy_signals],
                y=[s['price'] for s in buy_signals],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='green',
                    line=dict(width=2, color='white')
                ),
                name='Buy Signals',
                text=[s['description'] for s in buy_signals],
                hoverinfo='text+y'
            ),
            row=1, col=1
        )

    # Plot sell signals
    if sell_signals:
        fig.add_trace(
            go.Scatter(
                x=[s['timestamp'] for s in sell_signals],
                y=[s['price'] for s in sell_signals],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='red',
                    line=dict(width=2, color='white')
                ),
                name='Sell Signals',
                text=[s['description'] for s in sell_signals],
                hoverinfo='text+y'
            ),
            row=1, col=1
        )

    # Calculate and plot cumulative returns
    df['Returns'] = df['Close'].pct_change()
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Cumulative_Returns'],
            name='Cumulative Returns',
            line=dict(color='cyan', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 255, 0.1)'
        ),
        row=2, col=1
    )

    # Add win rate indicator
    win_rate = len([s for s in signals 
                   if (s['type'] == 'buy' and df['Close'].iloc[-1] > s['price']) or
                      (s['type'] == 'sell' and df['Close'].iloc[-1] < s['price'])]) / len(signals) if signals else 0

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=win_rate * 100,
            title={'text': "Win Rate %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "cyan"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "green"}
                ]
            }
        ),
        row=3, col=1
    )

    # Add profit factor indicator
    profit_factor = abs(
        df[df['Returns'] > 0]['Returns'].sum() / 
        df[df['Returns'] < 0]['Returns'].sum()
    ) if len(df[df['Returns'] < 0]) > 0 else 0

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=min(profit_factor * 100, 200),  # Cap at 200 for visualization
            title={'text': "Profit Factor"},
            gauge={
                'axis': {'range': [0, 200]},
                'bar': {'color': "cyan"},
                'steps': [
                    {'range': [0, 100], 'color': "red"},
                    {'range': [100, 150], 'color': "yellow"},
                    {'range': [150, 200], 'color': "green"}
                ]
            }
        ),
        row=3, col=2
    )

    # Update layout
    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=1200,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(t=30, l=0, r=0, b=0)
    )

    # Update axes labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Returns", row=2, col=1)

    # Convert to base64
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_str = buffer.getvalue()
    return base64.b64encode(html_str.encode()).decode()

def create_detailed_html_report(analysis: 'MarketAnalysis') -> str:
    """Create a detailed HTML report with all analysis information."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Market Analysis Report - {symbol}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #1a1a1a;
                color: #ffffff;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .section {{
                background-color: #2d2d2d;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            .kpi-card {{
                background-color: #3d3d3d;
                padding: 15px;
                border-radius: 6px;
                text-align: center;
            }}
            .kpi-value {{
                font-size: 24px;
                font-weight: bold;
                color: #00ffff;
            }}
            .kpi-label {{
                font-size: 14px;
                color: #999;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #444;
            }}
            th {{
                background-color: #3d3d3d;
            }}
            .signal {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
            }}
            .signal-buy {{
                background-color: #1b5e20;
                color: white;
            }}
            .signal-sell {{
                background-color: #b71c1c;
                color: white;
            }}
            .signal-neutral {{
                background-color: #666;
                color: white;
            }}
            h1, h2 {{
                color: #00ffff;
            }}
            .pattern {{
                border-left: 4px solid #00ffff;
                padding-left: 15px;
                margin-bottom: 15px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Market Analysis Report - {symbol}</h1>
            <p>Generated on: {timestamp}</p>
            
            <div class="section">
                <h2>Key Performance Indicators</h2>
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-value">{win_rate:.1%}</div>
                        <div class="kpi-label">Win Rate</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{sharpe:.2f}</div>
                        <div class="kpi-label">Sharpe Ratio</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{max_dd:.1%}</div>
                        <div class="kpi-label">Max Drawdown</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{profit_factor:.2f}</div>
                        <div class="kpi-label">Profit Factor</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{volatility:.1%}</div>
                        <div class="kpi-label">Volatility</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{avg_return:.2%}</div>
                        <div class="kpi-label">Avg Trade Return</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Market Overview</h2>
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-value">{trend}</div>
                        <div class="kpi-label">Trend Direction</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{trend_strength}/10</div>
                        <div class="kpi-label">Trend Strength</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{risk_level}/10</div>
                        <div class="kpi-label">Risk Level</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{recommendation}</div>
                        <div class="kpi-label">Recommendation</div>
                    </div>
                </div>
                <p>{summary}</p>
            </div>

            <div class="section">
                <h2>Technical Indicators</h2>
                <table>
                    <tr>
                        <th>Indicator</th>
                        <th>Value</th>
                        <th>Signal</th>
                        <th>Description</th>
                    </tr>
                    {technical_indicators}
                </table>
            </div>


            <div class="section">
                <h2>Market Structure</h2>
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-value">{structure_type}</div>
                        <div class="kpi-label">Structure Type</div>
                    </div>
                </div>
                <h3>Key Levels</h3>
                <table>
                    <tr>
                        <th>Level</th>
                        <th>Type</th>
                        <th>Strength</th>
                        <th>Description</th>
                    </tr>
                    {key_levels}
                </table>
            </div>

            <div class="section">
                <h2>Trading Strategies</h2>
                {strategies}
            </div>
        </div>
    </body>
    </html>
    """

    # Format technical indicators
    technical_indicators_html = ""
    for indicator in analysis.technical_indicators:
        signal_class = f"signal signal-{indicator.signal.lower()}"
        technical_indicators_html += f"""
        <tr>
            <td>{indicator.name}</td>
            <td>{indicator.value:.2f}</td>
            <td><span class="{signal_class}">{indicator.signal}</span></td>
            <td>{indicator.description}</td>
        </tr>
        """

    # Format patterns
    patterns_html = ""
    for pattern in analysis.patterns:
        patterns_html += f"""
        <div class="pattern">
            <h3>{pattern.name} ({pattern.type})</h3>
            <p>Reliability: {pattern.reliability}/10</p>
            <p>{pattern.description}</p>
            <p>Entry Points: {', '.join(f'{ep:.2f}' for ep in pattern.entry_points)}</p>
            <p>Targets: {', '.join(f'{t:.2f}' for t in pattern.targets)}</p>
            <p>Stop Loss: {pattern.stop_loss:.2f}</p>
        </div>
        """

    # Format key levels
    key_levels_html = ""
    for level in analysis.market_structure.key_levels:
        key_levels_html += f"""
        <tr>
            <td>{level.level:.2f}</td>
            <td>{level.type}</td>
            <td>{level.strength}/10</td>
            <td>{level.description}</td>
        </tr>
        """

    # Format strategies
    strategies_html = ""
    for strategy in analysis.strategies:
        signals_html = ""
        for signal in strategy.signals:
            signal_class = f"signal signal-{'buy' if signal.direction == 'long' else 'sell'}"
            signals_html += f"""
            <tr>
                <td><span class="{signal_class}">{signal.type.upper()} {signal.direction}</span></td>
                <td>{signal.price:.2f}</td>
                <td>{signal.confidence}/10</td>
                <td>{signal.stop_loss:.2f}</td>
                <td>{', '.join(f'{tp:.2f}' for tp in signal.take_profit)}</td>
                <td>{signal.risk_reward_ratio:.2f}</td>
                <td>{signal.description}</td>
            </tr>
            """

        strategies_html += f"""
        <div class="strategy">
            <h3>{strategy.name} ({strategy.timeframe})</h3>
            <p>{strategy.description}</p>
            <table>
                <tr>
                    <th>Signal</th>
                    <th>Price</th>
                    <th>Confidence</th>
                    <th>Stop Loss</th>
                    <th>Take Profit</th>
                    <th>RR Ratio</th>
                    <th>Description</th>
                </tr>
                {signals_html}
            </table>
        </div>
        """

    # Fill the template
    html_content = html_template.format(
        symbol=analysis.symbol,
        timestamp=analysis.timestamp,
        win_rate=analysis.kpis['win_rate'],
        sharpe=analysis.kpis['sharpe_ratio'],
        max_dd=analysis.kpis['max_drawdown'],
        profit_factor=analysis.kpis['profit_factor'],
        volatility=analysis.kpis['volatility'],
        avg_return=analysis.kpis['avg_trade_return'],
        trend=analysis.price_analysis.direction,
        trend_strength=analysis.price_analysis.strength,
        risk_level=analysis.risk_level,
        recommendation=analysis.recommendation,
        summary=analysis.summary,
        technical_indicators=technical_indicators_html,
        patterns=patterns_html,
        structure_type=analysis.market_structure.structure_type,
        key_levels=key_levels_html,
        strategies=strategies_html
    )

    return base64.b64encode(html_content.encode()).decode()


def create_smc_plot(df: pd.DataFrame, smc_analysis: dict, title: str) -> str:
    """Create a plot showing ICT and SMC analysis and return as base64 encoded HTML."""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Add Fair Value Gaps
    for fvg in smc_analysis['ict_levels']:
        fig.add_hline(
            y=fvg['level'],
            line=dict(
                color='rgba(255,165,0,0.5)' if fvg['type'] == 'bullish_fvg' else 'rgba(138,43,226,0.5)',
                width=2,
                dash='dash'
            ),
            annotation_text=fvg['description']
        )
    
    # Add Order Blocks
    for ob in smc_analysis['order_blocks'][-5:]:  # Show last 5 order blocks
        fig.add_shape(
            type="rect",
            x0=df.index[ob['index']],
            x1=df.index[ob['index']+3],
            y0=ob['low'],
            y1=ob['high'],
            fillcolor='rgba(0,255,0,0.2)' if ob['type'] == 'bullish_ob' else 'rgba(255,0,0,0.2)',
            line=dict(width=0),
        )
    
    # Add swing points
    for high in smc_analysis['swing_points']['highs'][-3:]:  # Show last 3 swing highs
        fig.add_trace(go.Scatter(
            x=[df.index[high['index']]],
            y=[high['price']],
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='red'),
            name='Swing High'
        ))
    
    for low in smc_analysis['swing_points']['lows'][-3:]:  # Show last 3 swing lows
        fig.add_trace(go.Scatter(
            x=[df.index[low['index']]],
            y=[low['price']],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            name='Swing Low'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True,
        template="plotly_dark"
    )
    
    # Convert to base64
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_str = buffer.getvalue()
    return base64.b64encode(html_str.encode()).decode()

def create_position_plot(df: pd.DataFrame, symbol: str, signals: list = None) -> str:
    """Create a plot showing position management with TP and SL levels and return as base64 encoded HTML."""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    if signals:
        # Add position markers for the last 3 signals
        for signal in signals[-3:]:
            marker_color = 'green' if signal['type'] == 'buy' else 'red'
            marker_symbol = 'triangle-up' if signal['type'] == 'buy' else 'triangle-down'
            
            # Entry point
            fig.add_trace(go.Scatter(
                x=[signal['timestamp']],
                y=[signal['price']],
                mode='markers',
                marker=dict(
                    symbol=marker_symbol,
                    size=12,
                    color=marker_color
                ),
                name=f"{signal['type'].capitalize()} Entry"
            ))
            
            # Stop Loss level
            fig.add_shape(
                type="line",
                x0=signal['timestamp'],
                x1=df.index[-1],
                y0=signal['stop_loss'],
                y1=signal['stop_loss'],
                line=dict(
                    color="red",
                    width=1,
                    dash="dash"
                )
            )
            
            # Take Profit levels
            for i, tp in enumerate(signal['take_profit']):
                fig.add_shape(
                    type="line",
                    x0=signal['timestamp'],
                    x1=df.index[-1],
                    y0=tp,
                    y1=tp,
                    line=dict(
                        color="green",
                        width=1,
                        dash="dash"
                    )
                )
    
    fig.update_layout(
        title=f"Position Management - {symbol}",
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True,
        template="plotly_dark"
    )
    
    # Convert to base64
    buffer = io.StringIO()
    fig.write_html(buffer)
    html_str = buffer.getvalue()
    return base64.b64encode(html_str.encode()).decode()