#!/usr/bin/env python3
"""
CoinMarketCap Chart Generator

Fetches cryptocurrency data from CoinMarketCap's API, generates interactive charts,
and saves data to CSV files and news to markdown files.
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import csv
from typing import Dict, Any, List, Tuple
from loguru import logger

# Import our CoinMarketCap API client
from coinmarketcap_api import CoinMarketCapClient

# Import our Trading Analysis module
from trading_analysis import TradingAnalysis


class CryptoChartGenerator:
    """Generates interactive charts and exports data from CoinMarketCap API."""
    
    def __init__(self, output_dir: str = "./data"):
        """
        Initialize the chart generator.
        
        Args:
            output_dir: Directory to save CSV and markdown files
        """
        self.client = CoinMarketCapClient()
        self.output_dir = output_dir
        
        # Create output directories if they don't exist
        self.csv_dir = os.path.join(output_dir, "csv")
        self.news_dir = os.path.join(output_dir, "news")
        self.charts_dir = os.path.join(output_dir, "charts")
        
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.news_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
    
    def fetch_and_process_data(self, 
                              crypto_name: str, 
                              interval: str = "1d", 
                              time_range: str = "1Y") -> Dict[str, Any]:
        """
        Fetch and process data for a cryptocurrency.
        
        Args:
            crypto_name: Name of the cryptocurrency (e.g., 'bitcoin')
            interval: Time interval for data points
            time_range: Time range for the chart
            
        Returns:
            Processed data dictionary
        """
        logger.info(f"Fetching {crypto_name} data with interval={interval}, range={time_range}")
        
        # Fetch data from API
        raw_data = self.client.get_chart_data(crypto_name, interval, time_range)
        processed_data = self.client.parse_chart_data(raw_data)
        
        return processed_data
    
    def save_to_csv(self, 
                   crypto_name: str, 
                   data: Dict[str, Any], 
                   interval: str, 
                   time_range: str) -> str:
        """
        Save time series data to CSV.
        
        Args:
            crypto_name: Name of the cryptocurrency
            data: Processed data dictionary
            interval: Time interval used
            time_range: Time range used
            
        Returns:
            Path to the saved CSV file
        """
        # Create DataFrame from the data
        df = pd.DataFrame({
            'timestamp': data['timestamps'],
            'price': data['prices'],
            'volume': data['volumes'],
            'market_cap': data['market_caps']
        })
        
        # Format the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{crypto_name}_{interval}_{time_range}_{timestamp}.csv"
        filepath = os.path.join(self.csv_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
        
        return filepath
    
    def save_news_to_markdown(self, 
                             crypto_name: str, 
                             data: Dict[str, Any]) -> str:
        """
        Save news/annotations to a markdown file.
        
        Args:
            crypto_name: Name of the cryptocurrency
            data: Processed data dictionary
            
        Returns:
            Path to the saved markdown file
        """
        if not data['annotations']:
            logger.info(f"No news/annotations found for {crypto_name}")
            return ""
        
        # Format the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{crypto_name}_news_{timestamp}.md"
        filepath = os.path.join(self.news_dir, filename)
        
        # Create markdown content
        with open(filepath, 'w') as f:
            f.write(f"# {crypto_name.title()} News and Annotations\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, annotation in enumerate(data['annotations'], 1):
                f.write(f"## {i}. {annotation['type'] or 'News'} - {annotation['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if annotation['title']:
                    f.write(f"**{annotation['title']}**\n\n")
                
                if annotation['description']:
                    f.write(f"{annotation['description']}\n\n")
                
                if annotation['url']:
                    f.write(f"[Read more]({annotation['url']})\n\n")
                
                f.write("---\n\n")
        
        logger.info(f"Saved news to {filepath}")
        return filepath
    
    def calculate_technical_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate technical indicators for the chart.
        
        Args:
            data: Processed data dictionary with prices and timestamps
            
        Returns:
            Dictionary with technical indicators
        """
        # Convert to pandas DataFrame for easier calculation
        df = pd.DataFrame({
            'timestamp': data['timestamps'],
            'price': data['prices'],
            'volume': data['volumes']
        })
        
        # Sort by timestamp to ensure correct calculation
        df = df.sort_values('timestamp')
        
        indicators = {}
        
        # Calculate Moving Averages
        indicators['sma_20'] = df['price'].rolling(window=20).mean().tolist()
        indicators['sma_50'] = df['price'].rolling(window=50).mean().tolist()
        indicators['sma_200'] = df['price'].rolling(window=200).mean().tolist()
        
        # Calculate Exponential Moving Averages
        indicators['ema_12'] = df['price'].ewm(span=12, adjust=False).mean().tolist()
        indicators['ema_26'] = df['price'].ewm(span=26, adjust=False).mean().tolist()
        
        # Calculate MACD
        ema_12 = df['price'].ewm(span=12, adjust=False).mean()
        ema_26 = df['price'].ewm(span=26, adjust=False).mean()
        indicators['macd'] = (ema_12 - ema_26).tolist()
        indicators['macd_signal'] = (ema_12 - ema_26).ewm(span=9, adjust=False).mean().tolist()
        indicators['macd_histogram'] = ((ema_12 - ema_26) - (ema_12 - ema_26).ewm(span=9, adjust=False).mean()).tolist()
        
        # Calculate RSI (14-period)
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        indicators['rsi'] = (100 - (100 / (1 + rs))).fillna(0).tolist()
        
        # Calculate Bollinger Bands (20-period, 2 standard deviations)
        sma_20 = df['price'].rolling(window=20).mean()
        std_20 = df['price'].rolling(window=20).std()
        indicators['bollinger_upper'] = (sma_20 + 2 * std_20).tolist()
        indicators['bollinger_lower'] = (sma_20 - 2 * std_20).tolist()
        
        # Calculate Volume Moving Average
        indicators['volume_sma_20'] = df['volume'].rolling(window=20).mean().tolist()
        
        return indicators
    
    def generate_interactive_chart(self, 
                                  crypto_name: str, 
                                  data: Dict[str, Any], 
                                  interval: str, 
                                  time_range: str) -> str:
        """
        Generate an interactive TradingView-like chart using Plotly.
        
        Args:
            crypto_name: Name of the cryptocurrency
            data: Processed data dictionary
            interval: Time interval used
            time_range: Time range used
            
        Returns:
            Path to the saved HTML file
        """
        if not data['timestamps'] or not data['prices']:
            logger.warning(f"Not enough data to generate chart for {crypto_name}")
            return ""
        
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(data)
        
        # Create figure with subplots for price, indicators, and volume
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.5, 0.2, 0.3],
                           subplot_titles=(f"{crypto_name.title()} Price", "Indicators", "Volume"))
        
        # Add price chart
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=data['prices'],
                mode='lines',
                name=f"{crypto_name.title()} Price",
                line=dict(color='rgb(49, 130, 189)', width=2),
                hovertemplate='%{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add Moving Averages to price chart
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=indicators['sma_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='rgba(255, 165, 0, 0.7)', width=1.5, dash='dot'),
                hovertemplate='%{x}<br>SMA 20: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=indicators['sma_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='rgba(255, 0, 0, 0.7)', width=1.5, dash='dot'),
                hovertemplate='%{x}<br>SMA 50: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=indicators['bollinger_upper'],
                mode='lines',
                name='Bollinger Upper',
                line=dict(color='rgba(0, 128, 0, 0.3)', width=1),
                hovertemplate='%{x}<br>Upper Band: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=indicators['bollinger_lower'],
                mode='lines',
                name='Bollinger Lower',
                line=dict(color='rgba(0, 128, 0, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(0, 128, 0, 0.1)',
                hovertemplate='%{x}<br>Lower Band: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add RSI to indicators subplot
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=indicators['rsi'],
                mode='lines',
                name='RSI (14)',
                line=dict(color='purple', width=1.5),
                hovertemplate='%{x}<br>RSI: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add RSI reference lines at 30 and 70
        fig.add_shape(
            type="line",
            x0=data['timestamps'][0],
            y0=30,
            x1=data['timestamps'][-1],
            y1=30,
            line=dict(color="red", width=1, dash="dash"),
            row=2, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=data['timestamps'][0],
            y0=70,
            x1=data['timestamps'][-1],
            y1=70,
            line=dict(color="red", width=1, dash="dash"),
            row=2, col=1
        )
        
        # Add MACD to indicators subplot
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=indicators['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=1.5),
                hovertemplate='%{x}<br>MACD: %{y:.2f}<extra></extra>',
                visible='legendonly'  # Hide by default, can be toggled
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=indicators['macd_signal'],
                mode='lines',
                name='MACD Signal',
                line=dict(color='red', width=1.5),
                hovertemplate='%{x}<br>Signal: %{y:.2f}<extra></extra>',
                visible='legendonly'  # Hide by default, can be toggled
            ),
            row=2, col=1
        )
        
        # Add volume bar chart
        fig.add_trace(
            go.Bar(
                x=data['timestamps'],
                y=data['volumes'],
                name='Volume',
                marker=dict(
                    color='rgba(58, 71, 80, 0.6)',
                    line=dict(color='rgba(58, 71, 80, 1.0)', width=1)
                ),
                hovertemplate='%{x}<br>Volume: $%{y:,.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add volume moving average
        fig.add_trace(
            go.Scatter(
                x=data['timestamps'],
                y=indicators['volume_sma_20'],
                mode='lines',
                name='Volume SMA (20)',
                line=dict(color='rgba(255, 165, 0, 0.7)', width=1.5),
                hovertemplate='%{x}<br>Vol SMA: $%{y:,.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add vertical lines for news events
        if data['annotations']:
            for annotation in data['annotations']:
                # Add vertical line on the price chart
                fig.add_shape(
                    type="line",
                    x0=annotation['timestamp'],
                    y0=0,
                    x1=annotation['timestamp'],
                    y1=max(data['prices']) * 1.1,
                    line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dot"),
                    row=1, col=1
                )
                
                # Add vertical line on the indicators chart
                fig.add_shape(
                    type="line",
                    x0=annotation['timestamp'],
                    y0=0,
                    x1=annotation['timestamp'],
                    y1=100,  # RSI max value
                    line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dot"),
                    row=2, col=1
                )
                
                # Add vertical line on the volume chart
                fig.add_shape(
                    type="line",
                    x0=annotation['timestamp'],
                    y0=0,
                    x1=annotation['timestamp'],
                    y1=max(data['volumes']) * 1.1,
                    line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dot"),
                    row=3, col=1
                )
                
                # Add annotation with hover text
                fig.add_trace(
                    go.Scatter(
                        x=[annotation['timestamp']],
                        y=[max(data['prices']) * 1.05],
                        mode='markers',
                        marker=dict(symbol='star', size=10, color='red'),
                        name=f"News: {annotation['timestamp'].strftime('%Y-%m-%d')}",
                        text=annotation['title'] or annotation['description'][:50] + '...',
                        hoverinfo='text',
                        hovertemplate='<b>%{text}</b><extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Update layout to make it look more like TradingView with white/gray theme
        fig.update_layout(
            title={
                'text': f"<b>{crypto_name.title()} Interactive Chart</b> ({interval}, {time_range})",
                'font': {'size': 24, 'color': '#444444'}
            },
            xaxis_rangeslider_visible=False,
            template="plotly_white",  # Light theme as requested
            paper_bgcolor='rgba(248, 249, 250, 1)',  # Light gray background
            plot_bgcolor='rgba(248, 249, 250, 1)',   # Light gray background
            height=900,  # Taller chart
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(200, 200, 200, 1)",
                borderwidth=1,
                font=dict(color='#444444')
            ),
            margin=dict(l=50, r=50, t=100, b=50),  # More space at top for buttons
            hovermode="x unified",  # Show all traces for a given x value
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.95)",
                font_size=12,
                font_family="Arial",
                font=dict(color='#444444')
            ),
            # Add buttons for different chart types and timeframes with improved styling
            updatemenus=[
                # Chart type selector
                dict(
                    type="buttons",
                    direction="right",
                    x=0.05,
                    y=1.12,  # Position higher to avoid overlap
                    showactive=True,
                    active=0,
                    bgcolor='rgba(230, 230, 230, 0.9)',
                    bordercolor='rgba(200, 200, 200, 1)',
                    font=dict(color='#444444'),
                    buttons=[
                        dict(label="<b>Price</b>",
                             method="update",
                             args=[{"visible": [True, True, True, True, True, True, False, False, True, True]}]),
                        dict(label="<b>RSI</b>",
                             method="update",
                             args=[{"visible": [False, False, False, False, False, True, False, False, False, False]}]),
                        dict(label="<b>MACD</b>",
                             method="update",
                             args=[{"visible": [False, False, False, False, False, False, True, True, False, False]}]),
                        dict(label="<b>All</b>",
                             method="update",
                             args=[{"visible": [True, True, True, True, True, True, True, True, True, True]}])
                    ]
                ),
                # Indicator toggle buttons
                dict(
                    type="buttons",
                    direction="right",
                    x=0.35,  # Position to the right of chart type selector
                    y=1.12,
                    showactive=True,
                    bgcolor='rgba(230, 230, 230, 0.9)',
                    bordercolor='rgba(200, 200, 200, 1)',
                    font=dict(color='#444444'),
                    buttons=[
                        dict(label="<b>Moving Averages</b>",
                             method="update",
                             args=[{"visible": [True, True, True, False, False, False, False, False, True, True]}]),
                        dict(label="<b>Bollinger Bands</b>",
                             method="update",
                             args=[{"visible": [True, False, False, True, True, False, False, False, True, True]}]),
                        dict(label="<b>No Indicators</b>",
                             method="update",
                             args=[{"visible": [True, False, False, False, False, False, False, False, True, True]}])
                    ]
                )
            ],
            # Add annotations for section labels
            annotations=[
                dict(
                    text="<b>Chart Type:</b>",
                    x=0.01,
                    y=1.12,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="#444444")
                ),
                dict(
                    text="<b>Indicators:</b>",
                    x=0.31,
                    y=1.12,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="#444444")
                )
            ]
        )
        
        # Style the price chart
        fig.update_yaxes(
            title_text="Price (USD)", 
            title_font=dict(color="#444444"),
            row=1, col=1, 
            gridcolor="rgba(200, 200, 200, 0.3)",
            zerolinecolor="rgba(150, 150, 150, 0.5)",
            tickfont=dict(color="#444444")
        )
        
        # Style the indicators chart
        fig.update_yaxes(
            title_text="RSI / MACD", 
            title_font=dict(color="#444444"),
            row=2, col=1, 
            gridcolor="rgba(200, 200, 200, 0.3)",
            zerolinecolor="rgba(150, 150, 150, 0.5)",
            range=[0, 100],  # For RSI
            tickfont=dict(color="#444444")
        )
        
        # Style the volume chart
        fig.update_yaxes(
            title_text="Volume (USD)", 
            title_font=dict(color="#444444"),
            row=3, col=1, 
            gridcolor="rgba(200, 200, 200, 0.3)",
            zerolinecolor="rgba(150, 150, 150, 0.5)",
            tickfont=dict(color="#444444")
        )
        
        # Add range selector to x-axis with improved styling
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor='rgba(230, 230, 230, 0.9)',
                activecolor='rgba(180, 180, 180, 0.9)',
                font=dict(color='#444444', size=12)
            ),
            row=3, col=1,  # Apply to bottom subplot
            tickfont=dict(color="#444444")
        )
        
        # Format the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{crypto_name}_{interval}_{time_range}_{timestamp}.html"
        filepath = os.path.join(self.charts_dir, filename)
        
        # Custom JavaScript for enhanced interactivity with grab-and-move functionality
        custom_js = """
        <script>
        // Wait for Plotly to be fully loaded
        window.addEventListener('load', function() {
            // Add keyboard shortcuts for zooming and panning
            document.addEventListener('keydown', function(e) {
                var gd = document.getElementsByClassName('js-plotly-plot')[0];
                if (!gd) return;
                
                // Zoom in: + or =
                if (e.key === '+' || e.key === '=') {
                    Plotly.relayout(gd, {
                        'xaxis.range': [gd._fullLayout.xaxis.range[0] * 0.8 + gd._fullLayout.xaxis.range[1] * 0.2, 
                                        gd._fullLayout.xaxis.range[0] * 0.2 + gd._fullLayout.xaxis.range[1] * 0.8]
                    });
                }
                
                // Zoom out: -
                if (e.key === '-') {
                    var currentRange = gd._fullLayout.xaxis.range;
                    var rangeSize = currentRange[1] - currentRange[0];
                    Plotly.relayout(gd, {
                        'xaxis.range': [currentRange[0] - rangeSize * 0.3, currentRange[1] + rangeSize * 0.3]
                    });
                }
                
                // Pan left: left arrow
                if (e.key === 'ArrowLeft') {
                    var currentRange = gd._fullLayout.xaxis.range;
                    var rangeSize = currentRange[1] - currentRange[0];
                    var shift = rangeSize * 0.2;
                    Plotly.relayout(gd, {
                        'xaxis.range': [currentRange[0] - shift, currentRange[1] - shift]
                    });
                }
                
                // Pan right: right arrow
                if (e.key === 'ArrowRight') {
                    var currentRange = gd._fullLayout.xaxis.range;
                    var rangeSize = currentRange[1] - currentRange[0];
                    var shift = rangeSize * 0.2;
                    Plotly.relayout(gd, {
                        'xaxis.range': [currentRange[0] + shift, currentRange[1] + shift]
                    });
                }
                
                // Reset zoom: r
                if (e.key === 'r' || e.key === 'R') {
                    Plotly.relayout(gd, {
                        'xaxis.autorange': true,
                        'yaxis.autorange': true
                    });
                }
            });
            
            // Add grab-and-move (panning) functionality
            var plotArea = document.getElementsByClassName('js-plotly-plot')[0];
            if (plotArea) {
                var isDragging = false;
                var startX, startY;
                var startRangeX, startRangeY;
                
                // Change cursor to grab when hovering over the plot
                plotArea.style.cursor = 'grab';
                
                // Mouse down event - start dragging
                plotArea.addEventListener('mousedown', function(e) {
                    // Only activate if not clicking on a button or control
                    if (e.target.tagName !== 'BUTTON' && !e.target.closest('.modebar')) {
                        e.preventDefault();
                        isDragging = true;
                        
                        // Change cursor to grabbing during drag
                        plotArea.style.cursor = 'grabbing';
                        
                        // Record starting position
                        startX = e.clientX;
                        startY = e.clientY;
                        
                        var gd = document.getElementsByClassName('js-plotly-plot')[0];
                        if (gd && gd._fullLayout) {
                            // Store the current axis ranges
                            startRangeX = {
                                xaxis: gd._fullLayout.xaxis.range.slice(),
                                yaxis: gd._fullLayout.yaxis.range.slice()
                            };
                            
                            // Store ranges for all subplots if they exist
                            if (gd._fullLayout.xaxis2) {
                                startRangeX.xaxis2 = gd._fullLayout.xaxis2.range.slice();
                                startRangeX.yaxis2 = gd._fullLayout.yaxis2.range.slice();
                            }
                            
                            if (gd._fullLayout.xaxis3) {
                                startRangeX.xaxis3 = gd._fullLayout.xaxis3.range.slice();
                                startRangeX.yaxis3 = gd._fullLayout.yaxis3.range.slice();
                            }
                        }
                    }
                });
                
                // Mouse move event - update plot position during drag
                document.addEventListener('mousemove', function(e) {
                    if (isDragging) {
                        e.preventDefault();
                        
                        var gd = document.getElementsByClassName('js-plotly-plot')[0];
                        if (!gd || !gd._fullLayout || !startRangeX) return;
                        
                        // Calculate how far we've moved
                        var dx = e.clientX - startX;
                        var dy = e.clientY - startY;
                        
                        // Convert pixel movement to data coordinates
                        var xaxis = gd._fullLayout.xaxis;
                        var yaxis = gd._fullLayout.yaxis;
                        
                        var pixelRatioX = (startRangeX.xaxis[1] - startRangeX.xaxis[0]) / plotArea.clientWidth;
                        var pixelRatioY = (startRangeX.yaxis[1] - startRangeX.yaxis[0]) / plotArea.clientHeight;
                        
                        // Calculate new ranges - move in opposite direction of mouse movement
                        var newRangeX = [
                            startRangeX.xaxis[0] - dx * pixelRatioX,
                            startRangeX.xaxis[1] - dx * pixelRatioX
                        ];
                        
                        // Prepare the relayout object
                        var update = {
                            'xaxis.range': newRangeX
                        };
                        
                        // If we have multiple subplots, move them all together
                        if (startRangeX.xaxis2) {
                            update['xaxis2.range'] = [
                                startRangeX.xaxis2[0] - dx * pixelRatioX,
                                startRangeX.xaxis2[1] - dx * pixelRatioX
                            ];
                        }
                        
                        if (startRangeX.xaxis3) {
                            update['xaxis3.range'] = [
                                startRangeX.xaxis3[0] - dx * pixelRatioX,
                                startRangeX.xaxis3[1] - dx * pixelRatioX
                            ];
                        }
                        
                        // Apply the new ranges
                        Plotly.relayout(gd, update);
                    }
                });
                
                // Mouse up event - stop dragging
                document.addEventListener('mouseup', function(e) {
                    if (isDragging) {
                        isDragging = false;
                        plotArea.style.cursor = 'grab';
                    }
                });
                
                // Mouse leave event - stop dragging if mouse leaves the window
                document.addEventListener('mouseleave', function(e) {
                    if (isDragging) {
                        isDragging = false;
                        plotArea.style.cursor = 'grab';
                    }
                });
            }
            
            // Add custom toolbar with additional controls
            var plotContainer = document.getElementsByClassName('js-plotly-plot')[0];
            if (plotContainer) {
                var toolbar = document.createElement('div');
                toolbar.style.position = 'absolute';
                toolbar.style.top = '10px';
                toolbar.style.right = '10px';
                toolbar.style.zIndex = '1000';
                toolbar.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
                toolbar.style.padding = '5px';
                toolbar.style.borderRadius = '5px';
                toolbar.style.boxShadow = '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)';
                
                // Add zoom in button
                var zoomInBtn = document.createElement('button');
                zoomInBtn.innerHTML = '+'; 
                zoomInBtn.title = 'Zoom In (or press + key)';
                zoomInBtn.style.margin = '0 5px';
                zoomInBtn.style.padding = '5px 10px';
                zoomInBtn.style.cursor = 'pointer';
                zoomInBtn.onclick = function() {
                    var gd = document.getElementsByClassName('js-plotly-plot')[0];
                    Plotly.relayout(gd, {
                        'xaxis.range': [gd._fullLayout.xaxis.range[0] * 0.8 + gd._fullLayout.xaxis.range[1] * 0.2, 
                                        gd._fullLayout.xaxis.range[0] * 0.2 + gd._fullLayout.xaxis.range[1] * 0.8]
                    });
                };
                toolbar.appendChild(zoomInBtn);
                
                // Add zoom out button
                var zoomOutBtn = document.createElement('button');
                zoomOutBtn.innerHTML = '-';
                zoomOutBtn.title = 'Zoom Out (or press - key)';
                zoomOutBtn.style.margin = '0 5px';
                zoomOutBtn.style.padding = '5px 10px';
                zoomOutBtn.style.cursor = 'pointer';
                zoomOutBtn.onclick = function() {
                    var gd = document.getElementsByClassName('js-plotly-plot')[0];
                    var currentRange = gd._fullLayout.xaxis.range;
                    var rangeSize = currentRange[1] - currentRange[0];
                    Plotly.relayout(gd, {
                        'xaxis.range': [currentRange[0] - rangeSize * 0.3, currentRange[1] + rangeSize * 0.3]
                    });
                };
                toolbar.appendChild(zoomOutBtn);
                
                // Add reset button
                var resetBtn = document.createElement('button');
                resetBtn.innerHTML = '&#8635;';
                resetBtn.title = 'Reset View (or press R key)';
                resetBtn.style.margin = '0 5px';
                resetBtn.style.padding = '5px 10px';
                resetBtn.style.cursor = 'pointer';
                resetBtn.onclick = function() {
                    var gd = document.getElementsByClassName('js-plotly-plot')[0];
                    Plotly.relayout(gd, {
                        'xaxis.autorange': true,
                        'yaxis.autorange': true
                    });
                };
                toolbar.appendChild(resetBtn);
                
                // Add help button with keyboard shortcuts
                var helpBtn = document.createElement('button');
                helpBtn.innerHTML = '?';
                helpBtn.title = 'Show Keyboard Shortcuts';
                helpBtn.style.margin = '0 5px';
                helpBtn.style.padding = '5px 10px';
                helpBtn.style.cursor = 'pointer';
                helpBtn.onclick = function() {
                    alert('Chart Controls:\n\n' +
                          'Click and Drag: Pan the chart\n' +
                          '+ or = : Zoom In\n' +
                          '- : Zoom Out\n' +
                          'Left Arrow : Pan Left\n' +
                          'Right Arrow : Pan Right\n' +
                          'R : Reset View\n' +
                          'Ctrl+Scroll: Zoom in/out at cursor');
                };
                toolbar.appendChild(helpBtn);
                
                plotContainer.appendChild(toolbar);
            }
            
            // Add smooth wheel zooming
            var plotArea = document.getElementsByClassName('js-plotly-plot')[0];
            if (plotArea) {
                plotArea.addEventListener('wheel', function(e) {
                    if (e.ctrlKey) {
                        e.preventDefault();
                        var gd = document.getElementsByClassName('js-plotly-plot')[0];
                        var xaxis = gd._fullLayout.xaxis;
                        var yaxis = gd._fullLayout.yaxis;
                        
                        // Get mouse position relative to plot
                        var rect = plotArea.getBoundingClientRect();
                        var x = e.clientX - rect.left;
                        var y = e.clientY - rect.top;
                        
                        // Convert to data coordinates
                        var xInDataCoord = xaxis.p2d(x);
                        var yInDataCoord = yaxis.p2d(y);
                        
                        // Calculate new ranges based on zoom direction
                        var zoomFactor = e.deltaY > 0 ? 1.1 : 0.9;
                        var xRange = xaxis.range;
                        var yRange = yaxis.range;
                        
                        var newXRange = [
                            xInDataCoord - (xInDataCoord - xRange[0]) * zoomFactor,
                            xInDataCoord + (xRange[1] - xInDataCoord) * zoomFactor
                        ];
                        
                        var newYRange = [
                            yInDataCoord - (yInDataCoord - yRange[0]) * zoomFactor,
                            yInDataCoord + (yRange[1] - yInDataCoord) * zoomFactor
                        ];
                        
                        Plotly.relayout(gd, {
                            'xaxis.range': newXRange,
                            'yaxis.range': newYRange
                        });
                    }
                }, { passive: false });
            }
        });
        </script>
        """
        
        # Add custom JavaScript for more interactivity
        with open(filepath, 'w') as f:
            # Save the interactive chart with full HTML and custom JS
            html_content = fig.to_html(
                include_plotlyjs='cdn',
                full_html=True,
                include_mathjax='cdn',
                config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
                    'modeBarButtonsToRemove': ['lasso2d'],
                    'responsive': True
                }
            )
            
            # Insert custom JS before the closing </body> tag
            html_content = html_content.replace('</body>', f'{custom_js}</body>')
            f.write(html_content)
        
        logger.info(f"Saved enhanced interactive chart with custom controls to {filepath}")
        return filepath
    
    def process_crypto(self, 
                      crypto_name: str, 
                      interval: str = "1d", 
                      time_range: str = "1Y") -> Tuple[str, str, str]:
        """
        Process a cryptocurrency: fetch data, generate chart, and save to files.
        
        Args:
            crypto_name: Name of the cryptocurrency
            interval: Time interval for data points
            time_range: Time range for the chart
            
        Returns:
            Tuple of (csv_path, news_path, chart_path)
        """
        # Fetch and process data
        data = self.fetch_and_process_data(crypto_name, interval, time_range)
        
        # Save data to CSV
        csv_path = self.save_to_csv(crypto_name, data, interval, time_range)
        
        # Save news to markdown
        news_path = self.save_news_to_markdown(crypto_name, data)
        
        # Generate interactive chart
        chart_path = self.generate_interactive_chart(crypto_name, data, interval, time_range)
        
        return csv_path, news_path, chart_path


def generate_trading_analysis_chart(crypto_name: str, interval: str, time_range: str) -> str:
    """
    Generate a trading analysis chart with trendlines and ABCD patterns.
    
    Args:
        crypto_name: Name of the cryptocurrency
        interval: Time interval for data points
        time_range: Time range for the chart
        
    Returns:
        Path to the saved HTML file
    """
    # Initialize chart generator and trading analysis
    generator = CryptoChartGenerator()
    analyzer = TradingAnalysis()
    
    # Fetch and process data
    data = generator.fetch_and_process_data(crypto_name, interval, time_range)
    
    # Generate trading analysis chart
    fig = analyzer.generate_trading_chart(crypto_name, data, interval, time_range)
    
    # Save chart to HTML file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{crypto_name}_trading_analysis_{interval}_{time_range}_{timestamp}.html"
    filepath = os.path.join(generator.charts_dir, filename)
    
    # Save the interactive chart
    fig.write_html(
        filepath,
        include_plotlyjs='cdn',
        full_html=True,
        include_mathjax='cdn',
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
            'modeBarButtonsToRemove': ['lasso2d'],
            'responsive': True
        }
    )
    
    logger.info(f"Saved trading analysis chart to {filepath}")
    return filepath


def main():
    """Main function to demonstrate the chart generator and trading analysis."""
    # Initialize chart generator
    generator = CryptoChartGenerator()
    
    # Process Bitcoin data with standard charts
    print("\n=== Processing Bitcoin Data with Standard Charts ===")
    btc_daily_paths = generator.process_crypto("bitcoin", "15m", "1d")
    print(f"Bitcoin Daily Data CSV: {btc_daily_paths[0]}")
    print(f"Bitcoin News: {btc_daily_paths[1]}")
    print(f"Bitcoin Daily Chart: {btc_daily_paths[2]}")
    
    # Generate trading analysis charts with trendlines and ABCD patterns
    print("\n=== Generating Advanced Trading Analysis Charts ===")
    
    # Bitcoin trading analysis
    btc_analysis_chart = generate_trading_analysis_chart("bitcoin", "15m", "1d")
    print(f"Bitcoin Trading Analysis Chart: {btc_analysis_chart}")
    
    # Ethereum trading analysis
    eth_analysis_chart = generate_trading_analysis_chart("ethereum", "15m", "1d")
    print(f"Ethereum Trading Analysis Chart: {eth_analysis_chart}")
    
    # Generate a longer timeframe for trend analysis
    btc_trend_chart = generate_trading_analysis_chart("bitcoin", "1h", "7d")
    print(f"Bitcoin 7-Day Trend Analysis Chart: {btc_trend_chart}")
    
    print("\nAll data processing complete!")
    print(f"CSV files saved to: {generator.csv_dir}")
    print(f"News files saved to: {generator.news_dir}")
    print(f"Chart files saved to: {generator.charts_dir}")


if __name__ == "__main__":
    main()
