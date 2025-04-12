"""Yahoo Finance tool for retrieving financial data with different timeframes.

This tool provides functionality to retrieve financial data from Yahoo Finance
with support for different timeframes including minutes, hours, days, etc.
"""

import yfinance as yf
from datetime import datetime, timedelta, date
from typing import Optional, Literal, Union
from pydantic import Field
import pandas as pd
import pytz

from quantalogic.tools.tool import Tool, ToolArgument


class YahooFinanceTool(Tool):
    """Tool for retrieving financial data from Yahoo Finance."""

    def __init__(self):
        super().__init__(
            name="yahoo_finance",
            description="Retrieve financial data from Yahoo Finance with different timeframes",
            arguments=[
                ToolArgument(
                    name="symbol",
                    arg_type="string",
                    description="Stock symbol (e.g., BTC-USD, ETH-USD)",    
                    required=True,
                    example="BTC-USD"
                ),
                ToolArgument(
                    name="interval",
                    arg_type="string",
                    description="Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d)",
                    required=False,
                    default="5m",
                    example="5m"
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

    async def execute(self, symbol: str, interval: str = "5m", range_type: str = "today", 
                     start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict:
        """Execute the Yahoo Finance tool to retrieve financial data.

        Args:
            symbol: Stock symbol (e.g., AAPL, GOOGL)
            interval: Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d)
            range_type: Type of date range (today, date, week, month, ytd)
            start_date: Start date in YYYY-MM-DD format (only for range_type='date')
            end_date: End date in YYYY-MM-DD format (only for range_type='date')

        Returns:
            Dictionary containing the financial data and DataFrame
        """
        try:
            # Create Ticker object
            ticker = yf.Ticker(symbol)
            
            # Get date range
            start, end = self._get_date_range(range_type, start_date, end_date)
            
            # Get historical data
            hist = ticker.history(start=start, end=end, interval=interval)
            
            # Format the data
            hist.index = hist.index.tz_convert('America/New_York')
            
            # Convert DataFrame to records for JSON
            formatted_data = []
            for date, row in hist.iterrows():
                formatted_data.append({
                    "timestamp": date.strftime("%Y-%m-%d %H:%M:%S%z"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                    "dividends": float(row["Dividends"]),
                    "stock_splits": float(row["Stock Splits"])
                })
            
            # Get market hours
            market_hours = {
                "open": "09:30:00",
                "close": "16:00:00",
                "timezone": "America/New_York"
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
                "dataframe": hist,  # Include the DataFrame directly
                "info": {
                    "currency": ticker.info.get("currency"),
                    "exchange": ticker.info.get("exchange"),
                    "shortName": ticker.info.get("shortName"),
                    "longName": ticker.info.get("longName"),
                    "sector": ticker.info.get("sector"),
                    "industry": ticker.info.get("industry")
                }
            }
            
            return data
        except Exception as e:
            return {"error": str(e)}


def main():
    """Test the Yahoo Finance tool with different timeframes."""
    import asyncio
    from pprint import pprint
    
    async def test_yahoo_finance():
        tool = YahooFinanceTool()
        
        # Test cases with different date ranges
        test_cases = [
            {
                "symbol": "BTC-USD",
                "interval": "5m",
                "range_type": "today",
            },
            {
                "symbol": "BTC-USD",
                "interval": "1h",
                "range_type": "week",
            },
            {
                "symbol": "BTC-USD",
                "interval": "1d",
                "range_type": "month",
            },
            {
                "symbol": "BTC-USD",
                "interval": "5m",
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
            
            print(f"\nStock Info:")
            pprint(result['info'])
            
            if result['data_points'] > 0:
                print("\nDataFrame Head:")
                print(result['dataframe'].head())
                print("\nDataFrame Tail:")
                print(result['dataframe'].tail())

    # Run the test
    asyncio.run(test_yahoo_finance())


if __name__ == "__main__":
    main()