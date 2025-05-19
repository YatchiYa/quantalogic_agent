# Financial Market Analysis Flow

A comprehensive market analysis flow that provides detailed technical and structural analysis of financial instruments.

## Features

1. **Price Analysis**
   - Trend identification
   - Support and resistance levels
   - Price patterns
   - Chart formations

2. **Technical Indicators**
   - Moving averages (SMA20, SMA50, SMA200)
   - Momentum indicators (RSI)
   - Volatility measures (Bollinger Bands)
   - Volume analysis

3. **Pattern Recognition**
   - Chart patterns
   - Candlestick patterns
   - Breakout points
   - Reversal signals

4. **Trading Signals**
   - Entry points
   - Exit points
   - Stop levels
   - Position sizing

5. **Market Structure**
   - Smart Money Concepts (SMC)
   - Institutional Composite Theory (ICT)
   - Order flow analysis
   - Market phases

## Usage

```python
from flows.financial_analysis.flow import analyze_market

# Run analysis
analysis = analyze_market(
    symbol="BTC-USD",
    interval="5m",
    days=30
)

# Access results
print(f"Market Direction: {analysis.price_analysis.direction}")
print(f"Risk Level: {analysis.risk_level}/10")
print(f"Recommendation: {analysis.recommendation}")

# Access plots
for i, plot in enumerate(analysis.plots):
    with open(f"plot_{i}.html", "wb") as f:
        f.write(base64.b64decode(plot))
```

## Components

1. **models.py**: Data models for all analysis components
2. **flow.py**: Main workflow implementation
3. **templates/**: Analysis templates and prompts

## Dependencies

- pandas
- numpy
- plotly
- ta (Technical Analysis library)
- yfinance (via quantalogic.tools.finance)
