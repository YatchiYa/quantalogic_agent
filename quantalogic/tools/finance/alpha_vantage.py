"""Alpha Vantage tool for retrieving financial data with different timeframes.

This tool provides functionality to retrieve financial data from Alpha Vantage
with support for different timeframes including minutes, hours, days, etc.
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Optional, Literal, Union, Dict, Any, List
import pandas as pd
import pytz
from loguru import logger

from quantalogic.tools.tool import Tool, ToolArgument


class AlphaVantageTool(Tool):
    """Tool for retrieving financial data from Alpha Vantage."""

    def __init__(self):
        super().__init__(
            name="alpha_vantage",
            description="Retrieve financial data from Alpha Vantage with different timeframes",
            arguments=[
                ToolArgument(
                    name="symbol",
                    arg_type="string",
                    description="Symbol (e.g., XAUUSD for Gold, XAGUSD for Silver, AAPL for stocks)",    
                    required=True,
                    example="XAUUSD"
                ),
                ToolArgument(
                    name="function",
                    arg_type="string",
                    description="Data function (CURRENCY_EXCHANGE_RATE for real-time forex/gold, TIME_SERIES_INTRADAY, TIME_SERIES_DAILY for stocks)",
                    required=False,
                    default="CURRENCY_EXCHANGE_RATE",
                    example="CURRENCY_EXCHANGE_RATE"
                ),
                ToolArgument(
                    name="interval",
                    arg_type="string",
                    description="Data interval for intraday data (1min, 5min, 15min, 30min, 60min)",
                    required=False,
                    default="5min",
                    example="5min"
                ),
                ToolArgument(
                    name="outputsize",
                    arg_type="string",
                    description="Amount of data to retrieve (compact or full)",
                    required=False,
                    default="compact",
                    example="compact"
                ),
            ]
        )
        # Get API key from environment variables
        self.api_key = os.environ.get("ALPHA_VANTAGE_KEY")
        if not self.api_key:
            logger.warning("ALPHA_VANTAGE_KEY not found in environment variables")

    def _parse_time_series_data(self, data: Dict[str, Any], function: str) -> List[Dict[str, Any]]:
        """Parse time series data from Alpha Vantage response."""
        formatted_data = []
        
        # Log the keys in the response for debugging
        logger.debug(f"Response keys: {list(data.keys())}")
        
        # Handle CURRENCY_EXCHANGE_RATE differently as it's not a time series
        if function == "CURRENCY_EXCHANGE_RATE":
            if "Realtime Currency Exchange Rate" in data:
                exchange_data = data["Realtime Currency Exchange Rate"]
                # Create a single data point with the current exchange rate
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                formatted_data.append({
                    "timestamp": current_time,
                    "from_currency": exchange_data.get("1. From_Currency Code", ""),
                    "to_currency": exchange_data.get("3. To_Currency Code", ""),
                    "exchange_rate": float(exchange_data.get("5. Exchange Rate", 0)),
                    "last_refreshed": exchange_data.get("6. Last Refreshed", ""),
                    "time_zone": exchange_data.get("7. Time Zone", ""),
                    "bid_price": float(exchange_data.get("8. Bid Price", 0)) if "8. Bid Price" in exchange_data else None,
                    "ask_price": float(exchange_data.get("9. Ask Price", 0)) if "9. Ask Price" in exchange_data else None
                })
            return formatted_data
        
        # For time series data
        time_series_key = None
        if function == "TIME_SERIES_INTRADAY":
            # The key will be something like "Time Series (5min)"
            for key in data.keys():
                if key.startswith("Time Series"):
                    time_series_key = key
                    break
        elif function == "TIME_SERIES_DAILY":
            time_series_key = "Time Series (Daily)"
        elif function == "TIME_SERIES_WEEKLY":
            time_series_key = "Weekly Time Series"
        elif function == "TIME_SERIES_MONTHLY":
            time_series_key = "Monthly Time Series"
        elif function == "FX_INTRADAY":
            # The key will be something like "Time Series FX (5min)"
            for key in data.keys():
                if key.startswith("Time Series FX"):
                    time_series_key = key
                    break
        elif function == "FX_DAILY":
            time_series_key = "Time Series FX (Daily)"
        
        if not time_series_key or time_series_key not in data:
            return formatted_data
        
        time_series = data[time_series_key]
        
        for date_str, values in time_series.items():
            formatted_data.append({
                "timestamp": date_str,
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "volume": int(values.get("5. volume", 0))
            })
        
        # Sort by timestamp in descending order (newest first)
        formatted_data.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return formatted_data

    def _create_dataframe(self, formatted_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert formatted data to pandas DataFrame."""
        if not formatted_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(formatted_data)
        
        # Handle different data formats
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        
        return df

    async def execute(self, symbol: str, function: str = "CURRENCY_EXCHANGE_RATE", 
                     interval: str = "5min", outputsize: str = "compact") -> dict:
        """Execute the Alpha Vantage tool to retrieve financial data.

        Args:
            symbol: Stock symbol (e.g., AAPL, MSFT, GOOGL)
            function: Data function (TIME_SERIES_INTRADAY, TIME_SERIES_DAILY, TIME_SERIES_WEEKLY, TIME_SERIES_MONTHLY)
            interval: Data interval for intraday data (1min, 5min, 15min, 30min, 60min)
            outputsize: Amount of data to retrieve (compact or full)

        Returns:
            Dictionary containing the financial data and DataFrame
        """
        try:
            if not self.api_key:
                return {"error": "ALPHA_VANTAGE_KEY not found in environment variables"}
            
            # Base URL for Alpha Vantage API
            base_url = "https://www.alphavantage.co/query"
            
            # Parameters for the API request
            params = {
                "function": function,
                "apikey": self.api_key,
                "outputsize": outputsize,
                "datatype": "json"
            }
            
            # Handle different function types
            if function == "CURRENCY_EXCHANGE_RATE":
                # For currency exchange rate (including gold)
                if len(symbol) == 6:
                    from_currency = symbol[:3]
                    to_currency = symbol[3:]
                else:
                    # Default to gold if symbol format is not as expected
                    from_currency = "XAU"
                    to_currency = "USD"
                
                params["from_currency"] = from_currency
                params["to_currency"] = to_currency
            elif function.startswith("FX_"):
                # For forex/commodities time series (premium)
                if len(symbol) == 6:
                    from_currency = symbol[:3]
                    to_currency = symbol[3:]
                else:
                    # Default to gold if symbol format is not as expected
                    from_currency = "XAU"
                    to_currency = "USD"
                
                params["from_symbol"] = from_currency
                params["to_symbol"] = to_currency
            else:
                # For regular stocks
                params["symbol"] = symbol
            
            # Add interval parameter for intraday data
            if function in ["TIME_SERIES_INTRADAY", "FX_INTRADAY"]:
                params["interval"] = interval
                
            # Log the request parameters for debugging
            logger.debug(f"Alpha Vantage API request parameters: {params}")
            
            # Make the API request
            response = requests.get(base_url, params=params)
            
            # Check if the request was successful
            if response.status_code != 200:
                return {"error": f"API request failed with status code {response.status_code}"}
            
            # Parse the response
            data = response.json()
            
            # Log the response for debugging
            logger.debug(f"Alpha Vantage API response: {data}")
            
            # Check for error messages in the response
            if "Error Message" in data:
                return {"error": data["Error Message"]}
            
            if "Information" in data:
                return {"error": f"API limit reached: {data['Information']}"}
                
            if "Note" in data:
                logger.warning(f"Alpha Vantage API note: {data['Note']}")
                # If we got a note about API call frequency, it's likely we didn't get data
                if "call frequency" in data["Note"]:
                    return {"error": data["Note"]}
            
            # Parse the time series data
            formatted_data = self._parse_time_series_data(data, function)
            
            # Create DataFrame
            df = self._create_dataframe(formatted_data)
            
            # Get market hours (US market by default)
            market_hours = {
                "open": "09:30:00",
                "close": "16:00:00",
                "timezone": "America/New_York"
            }
            
            # Prepare result based on function type
            if function == "CURRENCY_EXCHANGE_RATE":
                # For currency exchange rate, we have different metadata
                exchange_data = data.get("Realtime Currency Exchange Rate", {})
                result = {
                    "symbol": symbol,
                    "function": function,
                    "from_currency": exchange_data.get("1. From_Currency Code", ""),
                    "to_currency": exchange_data.get("3. To_Currency Code", ""),
                    "exchange_rate": float(exchange_data.get("5. Exchange Rate", 0)),
                    "last_refreshed": exchange_data.get("6. Last Refreshed", "N/A"),
                    "timezone": exchange_data.get("7. Time Zone", "US/Eastern"),
                    "data_points": len(formatted_data),
                    "data": formatted_data,
                    "dataframe": df,  # Include the DataFrame directly
                    "raw_data": exchange_data  # Include the raw data for reference
                }
            else:
                # For time series data
                metadata = data.get("Meta Data", {})
                result = {
                    "symbol": symbol,
                    "function": function,
                    "interval": interval if function in ["TIME_SERIES_INTRADAY", "FX_INTRADAY"] else "N/A",
                    "outputsize": outputsize,
                    "last_refreshed": metadata.get("3. Last Refreshed", "N/A"),
                    "timezone": metadata.get("6. Time Zone", "US/Eastern"),
                    "market_hours": market_hours,
                    "data_points": len(formatted_data),
                    "data": formatted_data,
                    "dataframe": df,  # Include the DataFrame directly
                    "metadata": metadata
                }
            
            return result
        except Exception as e:
            logger.error(f"Error in AlphaVantageTool: {str(e)}")
            return {"error": str(e)}


def main():
    """Test the Alpha Vantage tool with different timeframes."""
    import asyncio
    from pprint import pprint
    import logging
    from loguru import logger
    
    # Set up more verbose logging for debugging
    logger.remove()
    logger.add(logging.StreamHandler(), level="DEBUG")
    
    async def test_alpha_vantage():
        tool = AlphaVantageTool()
        
        # Print API key status (without revealing the key)
        if tool.api_key:
            print(f"API key found with length: {len(tool.api_key)}")
        else:
            print("WARNING: No API key found in environment variables")
        
        # Test cases with different functions
        test_cases = [
            {
                "symbol": "XAUUSD",  # Gold in forex format
                "function": "CURRENCY_EXCHANGE_RATE",
                "interval": "5min",  # Not used for CURRENCY_EXCHANGE_RATE but kept for consistency
                "outputsize": "compact"  # Not used for CURRENCY_EXCHANGE_RATE but kept for consistency
            },
        ]
        
        for test_case in test_cases:
            print(f"\nTesting {test_case['symbol']} with {test_case.get('function')}:")
            result = await tool.execute(**test_case)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                print("\nTroubleshooting tips:")
                print("1. Check if your Alpha Vantage API key is valid")
                print("2. Alpha Vantage free tier has a limit of 25 API calls per day")
                print("3. Make sure you're using the correct function and symbol format")
                print("4. Try using CURRENCY_EXCHANGE_RATE for real-time gold prices")
                continue
                
            print(f"Successfully retrieved {result['data_points']} data points")
            print(f"Last Refreshed: {result['last_refreshed']}")
            
            # Print the first data point as a sample
            if result['data']:
                print("\nSample data point:")
                pprint(result['data'][0])
            
            # Print DataFrame info
            if not result['dataframe'].empty:
                print("\nDataFrame info:")
                print(result['dataframe'].info())
    
    # Run the test
    asyncio.run(test_alpha_vantage())


if __name__ == "__main__":
    main()
