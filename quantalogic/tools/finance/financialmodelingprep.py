"""Financial Modeling Prep (FMP) API integration tool.

This module provides a comprehensive tool for interacting with the Financial Modeling Prep API,
offering access to financial data, economic indicators, and market analysis.
"""

import os
import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import requests
from loguru import logger
from pydantic import BaseModel, Field

from ..tool import Tool, ToolArgument


class FMPEndpoint(str, Enum):
    """Available FMP API endpoints."""
    STOCK_PRICE = "quote"
    COMPANY_PROFILE = "profile"
    FINANCIAL_RATIOS = "ratios"
    BALANCE_SHEET = "balance-sheet-statement"
    INCOME_STATEMENT = "income-statement"
    CASH_FLOW = "cash-flow-statement"
    MARKET_CAP = "market-capitalization"


class FMPResponse(BaseModel):
    """Base model for FMP API responses."""
    status: str = Field(default="success")
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class FMPTool(Tool):
    """Financial Modeling Prep API tool for accessing financial data and market information."""

    name: str = "fmp_tool"
    description: str = "Access financial data and market information through Financial Modeling Prep API"
    base_url: str = "https://financialmodelingprep.com/api/v3"
    api_key: str = Field(default="", description="FMP API key")
    need_validation: bool = True  # Require validation due to API usage and rate limits

    def __init__(self, **data):
        """Initialize FMP tool with API key from environment."""
        super().__init__(**data)
        self.api_key = os.getenv("FMP_API_KEY", "")
        if not self.api_key:
            logger.warning("FMP_API_KEY environment variable not set")

        # Define tool arguments
        self.arguments = [
            ToolArgument(
                name="endpoint",
                arg_type="string",
                description="FMP API endpoint to query (quote, profile, income-statement, balance-sheet-statement, cash-flow-statement)",
                required=True,
                example="quote"
            ),
            ToolArgument(
                name="symbol",
                arg_type="string",
                description="Stock symbol (e.g., AAPL, TSLA, AMZN)",
                required=True,
                example="AAPL"
            ),
            ToolArgument(
                name="limit",
                arg_type="int",
                description="Number of results to return",
                required=False,
                default="1"
            ),
            ToolArgument(
                name="period",
                arg_type="string",
                description="Period for financial statements (annual, quarter)",
                required=False,
                default="annual"
            )
        ]

    async def execute(
        self,
        endpoint: str,
        symbol: str,
        limit: int = 1,
        period: str = "annual",
        **kwargs
    ) -> FMPResponse:
        """Execute FMP API request with provided parameters."""
        try:
            # Get the endpoint value if it's an enum member
            endpoint_value = endpoint.value if isinstance(endpoint, FMPEndpoint) else endpoint
            
            # Validate endpoint
            try:
                endpoint_enum = FMPEndpoint(endpoint_value)
            except ValueError:
                raise ValueError(f"Invalid endpoint: {endpoint_value}")

            # Build request parameters
            params = {"apikey": self.api_key}
            if limit and endpoint_value not in ["profile"]:
                params["limit"] = limit
            if period and endpoint_value in ["income-statement", "balance-sheet-statement", "cash-flow-statement"]:
                params["period"] = period

            # Add additional parameters
            params.update({k: v for k, v in kwargs.items() if v is not None})

            # Build URL
            url = f"{self.base_url}/{endpoint_value}"
            if symbol:
                url = f"{url}/{symbol}"

            # Make request
            logger.info(f"Making FMP API request to {url}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Process response based on endpoint
            processed_data = self._process_response(endpoint_enum, data)
            
            return FMPResponse(
                status="success",
                data=processed_data
            )

        except requests.RequestException as e:
            error_msg = str(e)
            if "403" in error_msg:
                error_msg = f"Access denied to endpoint '{endpoint_value}'. This endpoint may require a premium subscription."
            logger.error(f"FMP API request failed: {error_msg}")
            return FMPResponse(
                status="error",
                error=error_msg
            )
        except Exception as e:
            logger.error(f"Error processing FMP request: {str(e)}")
            return FMPResponse(
                status="error",
                error=str(e)
            )

    def _process_response(self, endpoint: FMPEndpoint, data: Any) -> Dict[str, Any]:
        """Process API response based on endpoint type."""
        try:
            if not isinstance(data, (list, dict)):
                return {"data": data}

            if endpoint == FMPEndpoint.STOCK_PRICE:
                return {
                    "quote": {
                        "symbol": data[0].get("symbol"),
                        "price": data[0].get("price"),
                        "change": data[0].get("change"),
                        "change_percent": data[0].get("changesPercentage"),
                        "volume": data[0].get("volume"),
                        "market_cap": data[0].get("marketCap"),
                        "day_range": {
                            "low": data[0].get("dayLow"),
                            "high": data[0].get("dayHigh")
                        }
                    }
                } if isinstance(data, list) and len(data) > 0 else {"quote": {}}

            elif endpoint == FMPEndpoint.COMPANY_PROFILE:
                return {
                    "profile": {
                        "name": data[0].get("companyName"),
                        "symbol": data[0].get("symbol"),
                        "industry": data[0].get("industry"),
                        "sector": data[0].get("sector"),
                        "description": data[0].get("description"),
                        "ceo": data[0].get("ceo"),
                        "website": data[0].get("website"),
                        "employees": data[0].get("fullTimeEmployees"),
                        "location": {
                            "address": data[0].get("address"),
                            "city": data[0].get("city"),
                            "state": data[0].get("state"),
                            "country": data[0].get("country")
                        }
                    }
                } if isinstance(data, list) and len(data) > 0 else {"profile": {}}

            return {"data": data}

        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return {"data": data}


async def main():
    """Test the FMP tool with various endpoints."""
    # Initialize the tool
    fmp_tool = FMPTool()

    # Test cases for available endpoints
    test_cases = [
        # Tesla Profile
        {
            "endpoint": "profile",
            "symbol": "TSLA",
            "description": "Tesla Company Profile"
        },
        # Amazon Stock Price
        {
            "endpoint": "quote",
            "symbol": "AMZN",
            "description": "Amazon Stock Price"
        },
        # Apple Financial Statements
        {
            "endpoint": "income-statement",
            "symbol": "AAPL",
            "limit": 1,
            "period": "annual",
            "description": "Apple Income Statement"
        }
    ]

    # Run tests
    for test in test_cases:
        logger.info(f"\nTesting: {test['description']}")
        try:
            response = await fmp_tool.execute(**{k: v for k, v in test.items() if k != 'description'})
            if response.status == "success":
                logger.success(f"Success! Data received:")
                logger.info(response.data)
            else:
                logger.error(f"Error: {response.error}")
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
        
        # Add small delay between requests
        await asyncio.sleep(1)


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("FMP_API_KEY"):
        logger.warning("FMP_API_KEY not set. Please set it before running tests.")
        logger.info("Example: export FMP_API_KEY='your_api_key_here'")
        exit(1)

    # Run the tests
    asyncio.run(main())
