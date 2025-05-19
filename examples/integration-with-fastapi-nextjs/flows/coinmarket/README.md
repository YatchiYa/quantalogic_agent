# CoinMarketCap Data Visualization Tool

A Python tool for fetching cryptocurrency data from CoinMarketCap's API, generating interactive TradingView-like charts, and saving data to CSV files and news to markdown files.

## Features

- Fetch cryptocurrency price, volume, and market cap data from CoinMarketCap
- Generate interactive Plotly charts similar to TradingView
- Export time series data to CSV files
- Save news and annotations to markdown files
- Support for multiple cryptocurrencies, time intervals, and ranges

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from coinmarketcap_charts import CryptoChartGenerator

# Initialize chart generator
generator = CryptoChartGenerator()

# Process Bitcoin data with daily interval for the past year
csv_path, news_path, chart_path = generator.process_crypto("bitcoin", "1d", "1Y")

# Open the interactive chart in your browser
import webbrowser
webbrowser.open(f"file://{chart_path}")
```

### Command Line

Run the script directly to generate charts for Bitcoin and Ethereum:

```bash
python coinmarketcap_charts.py
```

## Output

The tool creates three directories:

- `data/csv/` - Contains CSV files with time series data
- `data/news/` - Contains markdown files with news and annotations
- `data/charts/` - Contains interactive HTML charts

## Available Cryptocurrencies

- Bitcoin (BTC)
- Ethereum (ETH)
- Tether (USDT)
- Binance Coin (BNB)
- Solana (SOL)
- XRP
- USD Coin (USDC)
- Cardano (ADA)
- Dogecoin (DOGE)
- Avalanche (AVAX)

## Available Time Intervals

- 5m, 15m, 30m (minutes)
- 1h, 2h, 4h (hours)
- 1d, 2d, 3d (days)
- 7d, 30d, 90d (days)

## Available Time Ranges

- 1D (1 day)
- 7D (1 week)
- 1M (1 month)
- 3M (3 months)
- 1Y (1 year)
- YTD (Year to date)
- All (All time)
