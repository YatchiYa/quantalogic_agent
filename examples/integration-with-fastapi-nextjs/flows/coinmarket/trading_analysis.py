#!/usr/bin/env python3
"""
Trading Analysis Module

Provides advanced trading pattern analysis and real-time trading signals
for cryptocurrency data from CoinMarketCap.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from loguru import logger


class TradingAnalysis:
    """Analyzes cryptocurrency data for trading patterns and signals."""
    
    def __init__(self):
        """Initialize the trading analysis module."""
        pass
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for trading analysis.
        
        Args:
            df: DataFrame with 'timestamp', 'price', 'volume' columns
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original
        df_result = df.copy()
        
        # Simple Moving Averages
        df_result['sma_20'] = df_result['price'].rolling(window=20).mean()
        df_result['sma_50'] = df_result['price'].rolling(window=50).mean()
        df_result['sma_200'] = df_result['price'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df_result['ema_12'] = df_result['price'].ewm(span=12, adjust=False).mean()
        df_result['ema_26'] = df_result['price'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df_result['macd'] = df_result['ema_12'] - df_result['ema_26']
        df_result['macd_signal'] = df_result['macd'].ewm(span=9, adjust=False).mean()
        df_result['macd_hist'] = df_result['macd'] - df_result['macd_signal']
        
        # RSI (14-period)
        delta = df_result['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df_result['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (20-period, 2 standard deviations)
        df_result['bb_middle'] = df_result['price'].rolling(window=20).mean()
        std_dev = df_result['price'].rolling(window=20).std()
        df_result['bb_upper'] = df_result['bb_middle'] + 2 * std_dev
        df_result['bb_lower'] = df_result['bb_middle'] - 2 * std_dev
        
        # Average True Range (ATR)
        high_low = df_result['price'].rolling(2).max() - df_result['price'].rolling(2).min()
        high_close = abs(df_result['price'].rolling(2).max() - df_result['price'].shift(1))
        low_close = abs(df_result['price'].rolling(2).min() - df_result['price'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_result['atr'] = true_range.rolling(window=14).mean()
        
        # Volume Moving Average
        df_result['volume_sma'] = df_result['volume'].rolling(window=20).mean()
        
        # Price Rate of Change
        df_result['roc'] = df_result['price'].pct_change(periods=12) * 100
        
        # Calculate price momentum
        df_result['momentum'] = df_result['price'] - df_result['price'].shift(10)
        
        return df_result
    
    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with added trading signals
        """
        df_signals = df.copy()
        
        # Initialize signal columns
        df_signals['signal'] = 0  # 1 for buy, -1 for sell, 0 for hold
        df_signals['signal_strength'] = 0  # 0-100 scale
        df_signals['trend'] = 'neutral'  # 'bullish', 'bearish', or 'neutral'
        
        # Moving Average Crossover Signal
        df_signals['ma_crossover'] = 0
        df_signals.loc[df_signals['sma_20'] > df_signals['sma_50'], 'ma_crossover'] = 1
        df_signals.loc[df_signals['sma_20'] < df_signals['sma_50'], 'ma_crossover'] = -1
        
        # MACD Signal
        df_signals['macd_crossover'] = 0
        df_signals.loc[df_signals['macd'] > df_signals['macd_signal'], 'macd_crossover'] = 1
        df_signals.loc[df_signals['macd'] < df_signals['macd_signal'], 'macd_crossover'] = -1
        
        # RSI Signals
        df_signals['rsi_signal'] = 0
        df_signals.loc[df_signals['rsi'] < 30, 'rsi_signal'] = 1  # Oversold - Buy
        df_signals.loc[df_signals['rsi'] > 70, 'rsi_signal'] = -1  # Overbought - Sell
        
        # Bollinger Band Signals
        df_signals['bb_signal'] = 0
        df_signals.loc[df_signals['price'] < df_signals['bb_lower'], 'bb_signal'] = 1  # Price below lower band - Buy
        df_signals.loc[df_signals['price'] > df_signals['bb_upper'], 'bb_signal'] = -1  # Price above upper band - Sell
        
        # Volume Confirmation
        df_signals['volume_signal'] = 0
        df_signals.loc[df_signals['volume'] > df_signals['volume_sma'] * 1.5, 'volume_signal'] = 1
        
        # Trend Determination
        df_signals.loc[(df_signals['sma_20'] > df_signals['sma_50']) & 
                      (df_signals['sma_50'] > df_signals['sma_200']), 'trend'] = 'bullish'
        df_signals.loc[(df_signals['sma_20'] < df_signals['sma_50']) & 
                      (df_signals['sma_50'] < df_signals['sma_200']), 'trend'] = 'bearish'
        
        # Combine signals for overall signal
        df_signals['signal'] = (df_signals['ma_crossover'] + 
                              df_signals['macd_crossover'] + 
                              df_signals['rsi_signal'] + 
                              df_signals['bb_signal'])
        
        # Normalize signal strength to 0-100 scale
        max_signal = 4  # Maximum possible signal value (sum of all signals)
        df_signals['signal_strength'] = ((df_signals['signal'] + max_signal) / (2 * max_signal)) * 100
        
        # Adjust signal to -1, 0, 1
        df_signals['signal'] = np.sign(df_signals['signal'])
        
        return df_signals
    
    def identify_support_resistance(self, df: pd.DataFrame, window: int = 10) -> Tuple[List[float], List[float]]:
        """
        Identify support and resistance levels.
        
        Args:
            df: DataFrame with price data
            window: Window size for peak detection
            
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        prices = df['price'].values
        
        # Find local maxima and minima
        resistance_indices = []
        support_indices = []
        
        for i in range(window, len(prices) - window):
            if all(prices[i] > prices[i-j] for j in range(1, window+1)) and all(prices[i] > prices[i+j] for j in range(1, window+1)):
                resistance_indices.append(i)
            if all(prices[i] < prices[i-j] for j in range(1, window+1)) and all(prices[i] < prices[i+j] for j in range(1, window+1)):
                support_indices.append(i)
        
        # Get the price levels
        resistance_levels = [prices[i] for i in resistance_indices]
        support_levels = [prices[i] for i in support_indices]
        
        # Cluster similar levels
        resistance_levels = self._cluster_price_levels(resistance_levels)
        support_levels = self._cluster_price_levels(support_levels)
        
        return support_levels, resistance_levels
    
    def _cluster_price_levels(self, levels: List[float], threshold: float = 0.02) -> List[float]:
        """
        Cluster similar price levels.
        
        Args:
            levels: List of price levels
            threshold: Threshold for clustering (as percentage)
            
        Returns:
            List of clustered price levels
        """
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Cluster similar levels
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # If level is within threshold of the average of current cluster
            if abs(level - sum(current_cluster) / len(current_cluster)) / level < threshold:
                current_cluster.append(level)
            else:
                # Add average of current cluster to clusters and start a new cluster
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters
    
    def detect_swing_points(self, prices: List[float], timestamps: List[datetime], window: int = 5, prominence: float = 0.005) -> Tuple[List[int], List[int]]:
        """
        Detect swing high and low points in price data.
        
        Args:
            prices: List of price values
            timestamps: List of corresponding timestamps
            window: Window size for peak detection (automatically adjusted based on data length)
            prominence: Minimum prominence for a peak to be considered
            
        Returns:
            Tuple of (swing_high_indices, swing_low_indices)
        """
        # Automatically adjust window size based on data length
        data_length = len(prices)
        if data_length < 50:
            # Very short timeframes
            adjusted_window = 2
            adjusted_prominence = prominence / 2
        elif data_length < 100:
            # Short timeframes
            adjusted_window = min(3, window)
            adjusted_prominence = prominence / 1.5
        elif data_length < 200:
            # Medium timeframes
            adjusted_window = min(4, window)
            adjusted_prominence = prominence
        else:
            # Longer timeframes
            adjusted_window = window
            adjusted_prominence = prominence
            
        logger.info(f"Adjusted window size to {adjusted_window} and prominence to {adjusted_prominence:.6f} for {data_length} data points")
        
        if len(prices) < (2 * adjusted_window + 1):
            logger.warning(f"Not enough data points for swing detection. Need at least {2 * adjusted_window + 1}, got {len(prices)}")
            return [], []
        
        swing_high_indices = []
        swing_low_indices = []
        
        # Find local maxima and minima
        for i in range(adjusted_window, len(prices) - adjusted_window):
            # Check if this is a local maximum
            if all(prices[i] > prices[i-j] for j in range(1, adjusted_window+1)) and \
               all(prices[i] > prices[i+j] for j in range(1, adjusted_window+1)):
                # Calculate prominence
                left_min = min(prices[i-adjusted_window:i])
                right_min = min(prices[i+1:i+adjusted_window+1])
                base = max(left_min, right_min)
                peak_prominence = (prices[i] - base) / prices[i]
                
                if peak_prominence >= adjusted_prominence:
                    swing_high_indices.append(i)
            
            # Check if this is a local minimum
            if all(prices[i] < prices[i-j] for j in range(1, adjusted_window+1)) and \
               all(prices[i] < prices[i+j] for j in range(1, adjusted_window+1)):
                # Calculate prominence
                left_max = max(prices[i-adjusted_window:i])
                right_max = max(prices[i+1:i+adjusted_window+1])
                base = min(left_max, right_max)
                peak_prominence = (base - prices[i]) / prices[i]
                
                if peak_prominence >= adjusted_prominence:
                    swing_low_indices.append(i)
        
        # If we still don't have enough swing points, try with even lower prominence
        if len(swing_high_indices) < 2 or len(swing_low_indices) < 2:
            logger.info(f"Not enough swing points detected, trying with lower prominence")
            return self.detect_swing_points(prices, timestamps, window, prominence / 2)
        
        return swing_high_indices, swing_low_indices
    
    def generate_trendlines(self, prices: List[float], timestamps: List[datetime], window_size: int = 20, num_lines: int = 3) -> List[Dict[str, Any]]:
        """
        Generate trendlines based on swing high and low points.
        
        Args:
            prices: List of price values
            timestamps: List of corresponding timestamps
            window_size: Window size for swing point detection
            num_lines: Maximum number of trendlines to generate
            
        Returns:
            List of trendline dictionaries with 'type', 'points', 'slope', 'intercept'
        """
        # Detect swing points
        swing_high_indices, swing_low_indices = self.detect_swing_points(prices, timestamps, window=window_size)
        
        if not swing_high_indices or not swing_low_indices:
            logger.warning("No swing points detected for trendline generation")
            return []
        
        trendlines = []
        
        # Generate resistance (upper) trendlines from swing highs
        if len(swing_high_indices) >= 2:
            # Sort by recency (most recent first)
            recent_highs = sorted(swing_high_indices, reverse=True)
            
            # Try to form trendlines from different combinations of points
            for i in range(min(len(recent_highs)-1, num_lines)):
                for j in range(i+1, min(len(recent_highs), i+5)):  # Look at a few combinations
                    idx1, idx2 = recent_highs[i], recent_highs[j]
                    
                    # Convert timestamps to numeric values for calculation
                    t1 = timestamps[idx1].timestamp()
                    t2 = timestamps[idx2].timestamp()
                    
                    # Skip if points are too close in time
                    if abs(t1 - t2) < 3600:  # 1 hour minimum separation
                        continue
                    
                    # Calculate slope and intercept
                    if t2 != t1:  # Avoid division by zero
                        slope = (prices[idx2] - prices[idx1]) / (t2 - t1)
                        intercept = prices[idx1] - slope * t1
                        
                        # Create trendline
                        trendline = {
                            'type': 'resistance',
                            'points': [(timestamps[idx1], prices[idx1]), (timestamps[idx2], prices[idx2])],
                            'slope': slope,
                            'intercept': intercept
                        }
                        
                        # Check if this is a valid trendline (not too steep)
                        if abs(slope) < 0.01:  # Arbitrary threshold
                            trendlines.append(trendline)
                            
                            # Break after finding a good trendline for this starting point
                            break
        
        # Generate support (lower) trendlines from swing lows
        if len(swing_low_indices) >= 2:
            # Sort by recency (most recent first)
            recent_lows = sorted(swing_low_indices, reverse=True)
            
            # Try to form trendlines from different combinations of points
            for i in range(min(len(recent_lows)-1, num_lines)):
                for j in range(i+1, min(len(recent_lows), i+5)):  # Look at a few combinations
                    idx1, idx2 = recent_lows[i], recent_lows[j]
                    
                    # Convert timestamps to numeric values for calculation
                    t1 = timestamps[idx1].timestamp()
                    t2 = timestamps[idx2].timestamp()
                    
                    # Skip if points are too close in time
                    if abs(t1 - t2) < 3600:  # 1 hour minimum separation
                        continue
                    
                    # Calculate slope and intercept
                    if t2 != t1:  # Avoid division by zero
                        slope = (prices[idx2] - prices[idx1]) / (t2 - t1)
                        intercept = prices[idx1] - slope * t1
                        
                        # Create trendline
                        trendline = {
                            'type': 'support',
                            'points': [(timestamps[idx1], prices[idx1]), (timestamps[idx2], prices[idx2])],
                            'slope': slope,
                            'intercept': intercept
                        }
                        
                        # Check if this is a valid trendline (not too steep)
                        if abs(slope) < 0.01:  # Arbitrary threshold
                            trendlines.append(trendline)
                            
                            # Break after finding a good trendline for this starting point
                            break
        
        # Sort trendlines by recency of the first point
        trendlines.sort(key=lambda x: x['points'][0][0], reverse=True)
        
        # Limit the number of trendlines
        return trendlines[:num_lines*2]  # Return at most num_lines*2 trendlines (support and resistance)
    
    def detect_abcd_patterns(self, prices: List[float], timestamps: List[datetime], tolerance: float = 0.15) -> List[Dict[str, Any]]:
        """
        Detect ABCD harmonic patterns in price data.
        
        Args:
            prices: List of price values
            timestamps: List of corresponding timestamps
            tolerance: Tolerance for pattern ratio matching (automatically adjusted based on data length)
            
        Returns:
            List of ABCD pattern dictionaries
        """
        # Adjust tolerance based on timeframe
        data_length = len(prices)
        if data_length < 50:
            # Very short timeframes need more flexibility
            adjusted_tolerance = tolerance * 1.5
        elif data_length < 100:
            # Short timeframes
            adjusted_tolerance = tolerance * 1.25
        else:
            # Longer timeframes can be more strict
            adjusted_tolerance = tolerance
            
        logger.info(f"Using adjusted tolerance of {adjusted_tolerance:.4f} for ABCD pattern detection with {data_length} data points")
        
        # Detect swing points for pattern identification
        swing_high_indices, swing_low_indices = self.detect_swing_points(prices, timestamps)
        
        if not swing_high_indices or not swing_low_indices:
            logger.warning("No swing points detected for ABCD pattern detection")
            return []
        
        logger.info(f"Found {len(swing_high_indices)} swing highs and {len(swing_low_indices)} swing lows for pattern detection")
        
        # Combine and sort all swing points by index
        all_swings = [(idx, 'high') for idx in swing_high_indices] + [(idx, 'low') for idx in swing_low_indices]
        all_swings.sort(key=lambda x: x[0])
        
        patterns = []
        
        # Need at least 4 points for an ABCD pattern
        if len(all_swings) < 4:
            logger.warning("Not enough swing points for ABCD pattern detection")
            return patterns
        
        # Look for patterns in the swing points
        # Don't just look at consecutive points, but try different combinations
        for i in range(len(all_swings) - 3):
            # Try different combinations of points to find valid patterns
            for j in range(i+1, min(i+5, len(all_swings)-2)):
                for k in range(j+1, min(j+5, len(all_swings)-1)):
                    for l in range(k+1, min(k+5, len(all_swings))):
                        # Get four swing points
                        a_idx, a_type = all_swings[i]
                        b_idx, b_type = all_swings[j]
                        c_idx, c_type = all_swings[k]
                        d_idx, d_type = all_swings[l]
                        
                        # ABCD pattern requires alternating highs and lows
                        if a_type == b_type or b_type == c_type or c_type == d_type:
                            continue
                        
                        # Check if points are in chronological order
                        if not (a_idx < b_idx < c_idx < d_idx):
                            continue
                        
                        # Get price values
                        a_price = prices[a_idx]
                        b_price = prices[b_idx]
                        c_price = prices[c_idx]
                        d_price = prices[d_idx]
                        
                        # Calculate price movements
                        ab_move = abs(b_price - a_price)
                        bc_move = abs(c_price - b_price)
                        cd_move = abs(d_price - c_price)
                        
                        # Calculate ratios for pattern validation
                        # In an ideal ABCD pattern: AB/CD ≈ 1.0 and BC/AB ≈ 0.618 (golden ratio)
                        ab_cd_ratio = ab_move / cd_move if cd_move != 0 else 0
                        bc_ab_ratio = bc_move / ab_move if ab_move != 0 else 0
                        
                        # Check if the ratios match ABCD pattern criteria within tolerance
                        is_valid_abcd = (
                            abs(ab_cd_ratio - 1.0) <= adjusted_tolerance and 
                            abs(bc_ab_ratio - 0.618) <= adjusted_tolerance * 2  # Wider tolerance for the Fibonacci ratio
                        )
                        
                        # Also check alternative Fibonacci ratios (0.5 and 0.786)
                        is_valid_alt_abcd = (
                            abs(ab_cd_ratio - 1.0) <= adjusted_tolerance and 
                            (abs(bc_ab_ratio - 0.5) <= adjusted_tolerance * 2 or 
                             abs(bc_ab_ratio - 0.786) <= adjusted_tolerance * 2)
                        )
                        
                        if is_valid_abcd or is_valid_alt_abcd:
                            # Determine if this is a bullish or bearish pattern
                            pattern_type = 'bullish' if a_type == 'low' else 'bearish'
                            
                            # Determine the specific pattern variant
                            pattern_variant = 'classic'
                            if is_valid_alt_abcd and not is_valid_abcd:
                                if abs(bc_ab_ratio - 0.5) <= adjusted_tolerance * 2:
                                    pattern_variant = 'shallow'
                                else:  # 0.786
                                    pattern_variant = 'deep'
                            
                            # Calculate pattern strength (1.0 is perfect)
                            ratio_accuracy = 1.0 - (abs(ab_cd_ratio - 1.0) + abs(bc_ab_ratio - 0.618)) / 2
                            strength = max(0.1, min(1.0, ratio_accuracy))
                            
                            # Create pattern object
                            pattern = {
                                'type': f'{pattern_type}_abcd_{pattern_variant}',
                                'points': {
                                    'A': (timestamps[a_idx], a_price),
                                    'B': (timestamps[b_idx], b_price),
                                    'C': (timestamps[c_idx], c_price),
                                    'D': (timestamps[d_idx], d_price)
                                },
                                'ratios': {
                                    'AB/CD': ab_cd_ratio,
                                    'BC/AB': bc_ab_ratio
                                },
                                'strength': strength,
                                'completion_time': timestamps[d_idx]
                            }
                            
                            patterns.append(pattern)
        
        # Sort patterns by recency (most recent first) and strength
        patterns.sort(key=lambda x: (x['completion_time'], x['strength']), reverse=True)
        
        # Limit to top patterns to avoid overcrowding the chart
        max_patterns = 3
        if len(patterns) > max_patterns:
            logger.info(f"Found {len(patterns)} ABCD patterns, limiting to top {max_patterns}")
            patterns = patterns[:max_patterns]
        else:
            logger.info(f"Found {len(patterns)} ABCD patterns")
        
        return patterns
    
    def generate_trading_chart(self, 
                             crypto_name: str, 
                             data: Dict[str, Any],
                             interval: str,
                             time_range: str) -> go.Figure:
        """
        Generate an interactive trading analysis chart.
        
        Args:
            crypto_name: Name of the cryptocurrency
            data: Dictionary with timestamps, prices, volumes
            interval: Time interval
            time_range: Time range
            
        Returns:
            Plotly figure with trading analysis
        """
        # Convert data to DataFrame
        df = pd.DataFrame({
            'timestamp': data['timestamps'],
            'price': data['prices'],
            'volume': data['volumes']
        })
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Generate trading signals
        df = self.generate_trading_signals(df)
        
        # Identify support and resistance levels
        support_levels, resistance_levels = self.identify_support_resistance(df)
        
        # Generate trendlines
        trendlines = self.generate_trendlines(
            prices=data['prices'],
            timestamps=data['timestamps'],
            window_size=15,
            num_lines=3
        )
        
        # Detect ABCD patterns
        abcd_patterns = self.detect_abcd_patterns(
            prices=data['prices'],
            timestamps=data['timestamps'],
            tolerance=0.15  # Slightly relaxed tolerance for more pattern detection
        )
        
        # Focus on recent data for current trading analysis
        recent_df = df.tail(50)  # Last 50 data points
        
        # Create figure with subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(
                f"{crypto_name.title()} Price Analysis",
                "RSI",
                "MACD",
                "Volume"
            )
        )
        
        # Add price candlestick chart
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['price'],
                mode='lines',
                name=f"{crypto_name.title()} Price",
                line=dict(color='rgb(49, 130, 189)', width=2),
                hovertemplate='%{x}<br>Price: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['sma_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='rgba(255, 165, 0, 0.7)', width=1.5),
                hovertemplate='%{x}<br>SMA 20: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['sma_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='rgba(255, 0, 0, 0.7)', width=1.5),
                hovertemplate='%{x}<br>SMA 50: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['bb_upper'],
                mode='lines',
                name='Bollinger Upper',
                line=dict(color='rgba(0, 128, 0, 0.3)', width=1),
                hovertemplate='%{x}<br>Upper Band: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['bb_lower'],
                mode='lines',
                name='Bollinger Lower',
                line=dict(color='rgba(0, 128, 0, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(0, 128, 0, 0.1)',
                hovertemplate='%{x}<br>Lower Band: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add support and resistance levels
        for level in support_levels:
            fig.add_shape(
                type="line",
                x0=df['timestamp'].iloc[0],
                y0=level,
                x1=df['timestamp'].iloc[-1],
                y1=level,
                line=dict(color="green", width=1, dash="dash"),
                row=1, col=1
            )
            
            # Add annotation
            fig.add_annotation(
                x=df['timestamp'].iloc[-1],
                y=level,
                text=f"Support: ${level:.2f}",
                showarrow=False,
                xshift=10,
                font=dict(color="green"),
                row=1, col=1
            )
        
        for level in resistance_levels:
            fig.add_shape(
                type="line",
                x0=df['timestamp'].iloc[0],
                y0=level,
                x1=df['timestamp'].iloc[-1],
                y1=level,
                line=dict(color="red", width=1, dash="dash"),
                row=1, col=1
            )
            
            # Add annotation
            fig.add_annotation(
                x=df['timestamp'].iloc[-1],
                y=level,
                text=f"Resistance: ${level:.2f}",
                showarrow=False,
                xshift=10,
                font=dict(color="red"),
                row=1, col=1
            )
        
        # Add RSI
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['rsi'],
                mode='lines',
                name='RSI (14)',
                line=dict(color='purple', width=1.5),
                hovertemplate='%{x}<br>RSI: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add RSI reference lines
        fig.add_shape(
            type="line",
            x0=df['timestamp'].iloc[0],
            y0=30,
            x1=df['timestamp'].iloc[-1],
            y1=30,
            line=dict(color="green", width=1, dash="dash"),
            row=2, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=df['timestamp'].iloc[0],
            y0=70,
            x1=df['timestamp'].iloc[-1],
            y1=70,
            line=dict(color="red", width=1, dash="dash"),
            row=2, col=1
        )
        
        # Add MACD
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=1.5),
                hovertemplate='%{x}<br>MACD: %{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['macd_signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=1.5),
                hovertemplate='%{x}<br>Signal: %{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add MACD histogram
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['macd_hist'],
                name='MACD Histogram',
                marker=dict(
                    color=np.where(df['macd_hist'] >= 0, 'rgba(0, 255, 0, 0.7)', 'rgba(255, 0, 0, 0.7)')
                ),
                hovertemplate='%{x}<br>Histogram: %{y:.2f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Add volume
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                name='Volume',
                marker=dict(
                    color='rgba(58, 71, 80, 0.6)',
                    line=dict(color='rgba(58, 71, 80, 1.0)', width=1)
                ),
                hovertemplate='%{x}<br>Volume: $%{y:,.2f}<extra></extra>'
            ),
            row=4, col=1
        )
        
        # Add volume moving average
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['volume_sma'],
                mode='lines',
                name='Volume SMA (20)',
                line=dict(color='rgba(255, 165, 0, 0.7)', width=1.5),
                hovertemplate='%{x}<br>Vol SMA: $%{y:,.2f}<extra></extra>'
            ),
            row=4, col=1
        )
        
        # Add buy/sell signals
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        # Add buy signals
        fig.add_trace(
            go.Scatter(
                x=buy_signals['timestamp'],
                y=buy_signals['price'],
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='green',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='%{x}<br>Buy at: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add sell signals
        fig.add_trace(
            go.Scatter(
                x=sell_signals['timestamp'],
                y=sell_signals['price'],
                mode='markers',
                name='Sell Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='red',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='%{x}<br>Sell at: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Highlight current trading zone (last few data points)
        current_price = df['price'].iloc[-1]
        
        # Add current price marker
        fig.add_trace(
            go.Scatter(
                x=[df['timestamp'].iloc[-1]],
                y=[current_price],
                mode='markers+text',
                name='Current Price',
                marker=dict(
                    symbol='circle',
                    size=12,
                    color='blue',
                    line=dict(color='white', width=2)
                ),
                text=f"${current_price:.2f}",
                textposition="top right",
                hovertemplate='Current Price: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add trendlines to the chart
        for i, trendline in enumerate(trendlines):
            # Extract points and properties
            start_point = trendline['points'][0]
            end_point = trendline['points'][1]
            trendline_type = trendline['type']
            
            # Determine color based on trendline type
            color = 'rgba(0, 128, 0, 0.8)' if trendline_type == 'support' else 'rgba(255, 0, 0, 0.8)'
            
            # Add trendline
            fig.add_shape(
                type="line",
                x0=start_point[0],
                y0=start_point[1],
                x1=end_point[0],
                y1=end_point[1],
                line=dict(color=color, width=2, dash="solid"),
                row=1, col=1
            )
            
            # Extend trendline to the right edge of the chart
            if trendline['slope'] != 0:
                # Calculate extended point
                last_timestamp = df['timestamp'].iloc[-1]
                time_diff = (last_timestamp.timestamp() - start_point[0].timestamp())
                extended_price = start_point[1] + trendline['slope'] * time_diff
                
                # Add extended trendline
                fig.add_shape(
                    type="line",
                    x0=end_point[0],
                    y0=end_point[1],
                    x1=last_timestamp,
                    y1=extended_price,
                    line=dict(color=color, width=2, dash="dash"),
                    row=1, col=1
                )
                
                # Add annotation
                fig.add_annotation(
                    x=last_timestamp,
                    y=extended_price,
                    text=f"{trendline_type.title()} Trendline",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    font=dict(size=10, color=color),
                    row=1, col=1
                )
        
        # Add ABCD patterns to the chart
        for i, pattern in enumerate(abcd_patterns):
            # Extract points
            a_point = pattern['points']['A']
            b_point = pattern['points']['B']
            c_point = pattern['points']['C']
            d_point = pattern['points']['D']
            pattern_type = pattern['type']
            
            # Determine color based on pattern type
            color = 'rgba(0, 128, 0, 0.8)' if 'bullish' in pattern_type else 'rgba(255, 0, 0, 0.8)'
            
            # Create a shape group for the pattern
            pattern_points = [a_point, b_point, c_point, d_point]
            x_values = [point[0] for point in pattern_points]
            y_values = [point[1] for point in pattern_points]
            
            # Add lines connecting the points
            for j in range(len(pattern_points) - 1):
                fig.add_shape(
                    type="line",
                    x0=pattern_points[j][0],
                    y0=pattern_points[j][1],
                    x1=pattern_points[j+1][0],
                    y1=pattern_points[j+1][1],
                    line=dict(color=color, width=2),
                    row=1, col=1
                )
            
            # Add markers for each point
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers+text',
                    name=f"{pattern_type.replace('_', ' ').title()} Pattern",
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color=color,
                        line=dict(color='white', width=1)
                    ),
                    text=['A', 'B', 'C', 'D'],
                    textposition="top center",
                    hovertemplate='Point %{text}<br>Price: $%{y:,.2f}<br>Time: %{x}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add pattern annotation with variant and strength information
            completion_time = pattern['completion_time']
            pattern_name = pattern_type.replace('_', ' ').title()
            pattern_variant = pattern['type'].split('_')[-1].title()
            pattern_strength = pattern.get('strength', 0.5) * 100
            
            # Create a more informative annotation
            fig.add_annotation(
                x=completion_time,
                y=d_point[1],
                text=f"{pattern_name} ABCD ({pattern_variant})<br>Strength: {pattern_strength:.0f}%<br>AB/CD: {pattern['ratios']['AB/CD']:.2f}<br>BC/AB: {pattern['ratios']['BC/AB']:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                font=dict(size=10, color=color),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=color,
                borderwidth=1,
                borderpad=4,
                row=1, col=1
            )
            
            # Add a potential target price projection if it's a recent pattern
            # For bullish patterns, project upward; for bearish patterns, project downward
            if pattern['completion_time'] >= data['timestamps'][-20]:  # Only for recent patterns
                # Calculate price movement for projection
                a_price = pattern['points']['A'][1]
                b_price = pattern['points']['B'][1]
                ab_move = abs(b_price - a_price)
                
                # Calculate projected price movement based on pattern type
                if 'bullish' in pattern_type:
                    # Project upward movement: typically AB distance from D point
                    target_price = d_point[1] + ab_move
                else:
                    # Project downward movement: typically AB distance from D point
                    target_price = d_point[1] - ab_move
                
                # Add projection line
                fig.add_shape(
                    type="line",
                    x0=d_point[0],
                    y0=d_point[1],
                    x1=data['timestamps'][-1],  # Project to the end of the chart
                    y1=target_price,
                    line=dict(color=color, width=1, dash="dot"),
                    row=1, col=1
                )
                
                # Add target annotation
                fig.add_annotation(
                    x=data['timestamps'][-1],
                    y=target_price,
                    text=f"Target: ${target_price:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor=color,
                    font=dict(size=9, color=color),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=2,
                    row=1, col=1
                )
        
        # Add trading recommendation based on signals
        current_signal = df['signal'].iloc[-1]
        current_trend = df['trend'].iloc[-1]
        
        recommendation = "HOLD"
        if current_signal > 0 and current_trend == 'bullish':
            recommendation = "STRONG BUY"
        elif current_signal > 0:
            recommendation = "BUY"
        elif current_signal < 0 and current_trend == 'bearish':
            recommendation = "STRONG SELL"
        elif current_signal < 0:
            recommendation = "SELL"
        
        # Add recommendation annotation
        fig.add_annotation(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"Current Recommendation: {recommendation}",
            showarrow=False,
            font=dict(
                size=16,
                color="white" if recommendation in ["HOLD", "BUY", "SELL"] else "red" if "SELL" in recommendation else "green"
            ),
            bgcolor="rgba(0, 0, 0, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.5)",
            borderwidth=1,
            borderpad=10,
            align="center"
        )
        
        # Update layout with white/gray theme
        fig.update_layout(
            title={
                'text': f"<b>{crypto_name.title()} Trading Analysis</b> ({interval}, {time_range})",
                'font': {'size': 24, 'color': '#444444'}
            },
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            paper_bgcolor='rgba(248, 249, 250, 1)',
            plot_bgcolor='rgba(248, 249, 250, 1)',
            height=1000,
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
            margin=dict(l=50, r=50, t=120, b=50),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.95)",
                font_size=12,
                font_family="Arial",
                font=dict(color='#444444')
            )
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
        
        # Style the RSI chart
        fig.update_yaxes(
            title_text="RSI", 
            title_font=dict(color="#444444"),
            row=2, col=1, 
            gridcolor="rgba(200, 200, 200, 0.3)",
            zerolinecolor="rgba(150, 150, 150, 0.5)",
            tickfont=dict(color="#444444"),
            range=[0, 100]
        )
        
        # Style the MACD chart
        fig.update_yaxes(
            title_text="MACD", 
            title_font=dict(color="#444444"),
            row=3, col=1, 
            gridcolor="rgba(200, 200, 200, 0.3)",
            zerolinecolor="rgba(150, 150, 150, 0.5)",
            tickfont=dict(color="#444444")
        )
        
        # Style the volume chart
        fig.update_yaxes(
            title_text="Volume (USD)", 
            title_font=dict(color="#444444"),
            row=4, col=1, 
            gridcolor="rgba(200, 200, 200, 0.3)",
            zerolinecolor="rgba(150, 150, 150, 0.5)",
            tickfont=dict(color="#444444")
        )
        
        return fig
