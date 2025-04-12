"""Financial data tool using Finnhub API for retrieving market data.

This tool provides functionality to retrieve financial data from Finnhub
with support for different timeframes including minutes, hours, days, etc.
"""

from datetime import datetime, timedelta, date
from typing import Optional, Literal, Union, ClassVar
from pydantic import Field
import pandas as pd
import pytz
import requests
import json
import time

from quantalogic.tools.tool import Tool, ToolArgument


class FinnhubTool(Tool):
    """Tool for retrieving financial data from Finnhub."""

    BASE_URL: ClassVar[str] = "https://finnhub.io/api/v1"
    API_KEY: ClassVar[str] = "YOUR_FINNHUB_API_KEY"  # Replace with your API key

    def __init__(self):
        super().__init__(
            name="finnhub_finance",
            description="Retrieve financial data from Finnhub with different timeframes",
            arguments=[
                ToolArgument(
                    name="symbol",
                    arg_type="string",
                    description="Stock/Crypto symbol (e.g., BINANCE:BTCUSDT, COINBASE:BTC-USD)",    
                    required=True,
                    example="BINANCE:BTCUSDT"
                ),
                ToolArgument(
                    name="interval",
                    arg_type="string",
                    description="Data interval (1,5,15,30,60,D,W,M)",
                    required=False,
                    default="5",
                    example="5"
                ),
                ToolArgument(
                    name="range_type",
                    arg_type="string",
                    description="Type of date range (today, date, week, month, ytd)",
                    required=False,
                    default="today",
                    example="today"
                ),
                ToolArgument(
                    name="start_date",
                    arg_type="string",
                    description="Start date in YYYY-MM-DD format (only for range_type='date')",
                    required=False,
                    default=None,
                    example="2025-04-01"
                ),
                ToolArgument(
                    name="end_date",
                    arg_type="string",
                    description="End date in YYYY-MM-DD format (only for range_type='date')",
                    required=False,
                    default=None,
                    example="2025-04-05"
                ),
            ]
        )

    def _get_date_range(self, range_type: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> tuple[datetime, datetime]:
        """Get start and end dates based on range type."""
        ny_tz = pytz.timezone('America/New_York')
        now = datetime.now(ny_tz)
        
        if range_type == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif range_type == "date":
            if not start_date:
                raise ValueError("start_date is required for range_type='date'")
            start = datetime.strptime(start_date, "%Y-%m-%d")
            if end_date:
                end = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end = start + timedelta(days=1)
            start = ny_tz.localize(start)
            end = ny_tz.localize(end)
        elif range_type == "week":
            start = now - timedelta(days=7)
            end = now
        elif range_type == "month":
            start = now - timedelta(days=30)
            end = now
        elif range_type == "ytd":
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = now
        else:
            raise ValueError(f"Invalid range_type: {range_type}")
            
        return start, end

    def _fetch_finnhub_data(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch data from Finnhub."""
        try:
            # Convert interval to resolution
            resolution = interval.upper() if interval in ['D', 'W', 'M'] else interval
            
            # Convert timestamps to Unix timestamps
            start_ts = int(start.timestamp())
            end_ts = int(end.timestamp())
            
            # Prepare the URL
            url = f"{self.BASE_URL}/crypto/candle"
            params = {
                "symbol": symbol,
                "resolution": resolution,
                "from": start_ts,
                "to": end_ts,
                "token": self.API_KEY
            }
            
            # Make request
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get('s') == 'no_data':
                raise ValueError(f"No data available for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['t'], unit='s'),
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            })
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error fetching data from Finnhub: {str(e)}")

    async def execute(self, symbol: str, interval: str = "5", range_type: str = "today", 
                     start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict:
        """Execute the Finnhub tool to retrieve financial data.

        Args:
            symbol: Stock/Crypto symbol (e.g., BINANCE:BTCUSDT)
            interval: Data interval (1,5,15,30,60,D,W,M)
            range_type: Type of date range (today, date, week, month, ytd)
            start_date: Start date in YYYY-MM-DD format (only for range_type='date')
            end_date: End date in YYYY-MM-DD format (only for range_type='date')

        Returns:
            Dictionary containing the financial data and DataFrame
        """
        try:
            # Get date range
            start, end = self._get_date_range(range_type, start_date, end_date)
            
            # Get historical data
            hist = self._fetch_finnhub_data(symbol, interval, start, end)
            
            # Format the data
            hist.index = hist.index.tz_localize('UTC').tz_convert('America/New_York')
            
            # Convert DataFrame to records for JSON
            formatted_data = []
            for date, row in hist.iterrows():
                formatted_data.append({
                    "timestamp": date.strftime("%Y-%m-%d %H:%M:%S%z"),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"])
                })
            
            # Get market hours (crypto markets are 24/7)
            market_hours = {
                "open": "00:00:00",
                "close": "23:59:59",
                "timezone": "America/New_York"
            }
            
            # Get basic info
            info = {
                "symbol": symbol,
                "type": "Cryptocurrency",
                "currency": "USD",
                "exchange": symbol.split(":")[0] if ":" in symbol else "Unknown"
            }
            
            data = {
                "symbol": symbol,
                "interval": interval,
                "range_type": range_type,
                "start_date": start.strftime("%Y-%m-%d %H:%M:%S%z"),
                "end_date": end.strftime("%Y-%m-%d %H:%M:%S%z"),
                "market_hours": market_hours,
                "data_points": len(formatted_data),
                "data": formatted_data,
                "dataframe": hist,
                "info": info
            }
            
            return data
        except Exception as e:
            return {"error": str(e)}


def main():
    """Test the Finnhub tool with different timeframes."""
    import asyncio
    from pprint import pprint
    
    async def test_finnhub():
        tool = FinnhubTool()
        
        # Test cases with different date ranges
        test_cases = [
            {
                "symbol": "BINANCE:BTCUSDT",
                "interval": "5",
                "range_type": "today",
            },
            {
                "symbol": "BINANCE:BTCUSDT",
                "interval": "60",
                "range_type": "week",
            },
            {
                "symbol": "BINANCE:BTCUSDT",
                "interval": "D",
                "range_type": "month",
            },
            {
                "symbol": "BINANCE:BTCUSDT",
                "interval": "5",
                "range_type": "date",
                "start_date": "2025-04-01",
                "end_date": "2025-04-05",
            }
        ]
        
        for test_case in test_cases:
            print(f"\nTesting {test_case['symbol']} with {test_case['interval']} interval for {test_case['range_type']}:")
            result = await tool.execute(**test_case)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                continue
                
            print(f"Successfully retrieved {result['data_points']} data points")
            print(f"Date Range: {result['start_date']} to {result['end_date']}")
            
            print("\nMarket Hours:")
            print(f"Open: {result['market_hours']['open']} {result['market_hours']['timezone']}")
            print(f"Close: {result['market_hours']['close']} {result['market_hours']['timezone']}")
            
            print(f"\nCrypto Info:")
            pprint(result['info'])
            
            if result['data_points'] > 0:
                print("\nDataFrame Head:")
                print(result['dataframe'].head())
                print("\nDataFrame Tail:")
                print(result['dataframe'].tail())

    # Run the test
    asyncio.run(test_finnhub())


if __name__ == "__main__":
    main()
