"""Tool for accessing Yahoo Finance data through yfinance."""

import json
from typing import Optional

from loguru import logger
from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.tools.finance.finance_functions import (
    get_current_stock_price,
    get_company_info,
    get_historical_stock_prices,
    get_stock_fundamentals,
    get_income_statements,
    get_key_financial_ratios,
    get_analyst_recommendations,
    get_company_news,
    get_technical_indicators,
)


class YahooFinanceTool(Tool):
    """Tool for accessing financial data from Yahoo Finance."""

    name: str = "yahoo_finance_tool"
    description: str = "Retrieves financial data for stocks and other assets from Yahoo Finance."
    need_validation: bool = False
    arguments: list = [
        ToolArgument(
            name="symbol",
            arg_type="string",
            description="The stock/asset symbol to query (e.g., MSFT, BTC-USD, GC=F).",
            required=True,
            example="AAPL",
        ),
        ToolArgument(
            name="data_type",
            arg_type="string",
            description="""
            The type of financial data to retrieve. Options:
            - price: Current stock price
            - info: Company information and overview
            - historical: Historical stock prices
            - fundamentals: Stock fundamentals
            - income: Income statements
            - ratios: Key financial ratios
            - recommendations: Analyst recommendations
            - news: Company news
            - technical: Technical indicators
            """,
            required=True,
            example="price",
        ),
        ToolArgument(
            name="period",
            arg_type="string",
            description="""
            The period for historical data or technical indicators.
            Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
            Only used for historical and technical data types.
            """,
            required=False,
            default="1mo",
            example="3mo",
        ),
        ToolArgument(
            name="interval",
            arg_type="string",
            description="""
            The interval between data points for historical data.
            Valid intervals: 1d, 5d, 1wk, 1mo, 3mo.
            Only used for historical data type.
            """,
            required=False,
            default="1d",
            example="1d",
        ),
        ToolArgument(
            name="num_stories",
            arg_type="string",
            description="""
            The number of news stories to retrieve.
            Only used for news data type.
            """,
            required=False,
            default="3",
            example="5",
        ),
    ]

    def execute(
        self, 
        symbol: str, 
        data_type: str, 
        period: Optional[str] = "1mo", 
        interval: Optional[str] = "1d", 
        num_stories: Optional[str] = "3"
    ) -> str:
        """Retrieves financial data from Yahoo Finance.

        Args:
            symbol (str): The stock/asset symbol to query.
            data_type (str): The type of financial data to retrieve.
            period (str, optional): The period for historical data. Defaults to "1mo".
            interval (str, optional): The interval between data points. Defaults to "1d".
            num_stories (str, optional): The number of news stories to retrieve. Defaults to "3".

        Returns:
            str: The requested financial data in JSON format or an error message.

        Raises:
            ValueError: If an invalid data_type is provided.
        """
        try:
            # Convert num_stories to int
            num_stories_int = int(num_stories)
            
            # Call the appropriate function based on data_type
            if data_type == "price":
                result = get_current_stock_price(symbol)
            elif data_type == "info":
                result = get_company_info(symbol)
            elif data_type == "historical":
                result = get_historical_stock_prices(symbol, period, interval)
            elif data_type == "fundamentals":
                result = get_stock_fundamentals(symbol)
            elif data_type == "income":
                result = get_income_statements(symbol)
            elif data_type == "ratios":
                result = get_key_financial_ratios(symbol)
            elif data_type == "recommendations":
                result = get_analyst_recommendations(symbol)
            elif data_type == "news":
                result = get_company_news(symbol, num_stories_int)
            elif data_type == "technical":
                result = get_technical_indicators(symbol, period)
            else:
                valid_types = [
                    "price", "info", "historical", "fundamentals", 
                    "income", "ratios", "recommendations", "news", "technical"
                ]
                raise ValueError(f"Invalid data_type: {data_type}. Valid types are: {', '.join(valid_types)}")
            
            # Check if result is already a JSON string
            try:
                # If it's already JSON, parse and re-stringify for consistent formatting
                json_obj = json.loads(result)
                return json.dumps(json_obj, indent=2)
            except (json.JSONDecodeError, TypeError):
                # If it's not JSON, return as is
                return result
                
        except Exception as e:
            logger.error(f"Yahoo Finance tool error: {str(e)}")
            return f"Error retrieving {data_type} data for {symbol}: {str(e)}"


if __name__ == "__main__":
    tool = YahooFinanceTool()
    print(tool.to_markdown())
    
    # Example usage
    print("\nExample: Getting Apple's current stock price")
    result = tool.execute(symbol="AAPL", data_type="price")
    print(result)
