import json

try:
    import yfinance as yf
except ImportError:
    raise ImportError("`yfinance` not installed. Please install using `pip install yfinance`.")


def get_current_stock_price(symbol: str) -> str:
    """
    Get the current stock price for a given symbol.

    Args:
        symbol (str): The stock symbol.

    Returns:
        str: The current stock price or error message.
    """
    try:
        stock = yf.Ticker(symbol)
        # Use "regularMarketPrice" for regular market hours, or "currentPrice" for pre/post market
        current_price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
        return f"{current_price:.4f}" if current_price else f"Could not fetch current price for {symbol}"
    except Exception as e:
        return f"Error fetching current price for {symbol}: {e}"


def get_company_info(symbol: str) -> str:
    """Get company information and overview for a given stock symbol.

    Args:
        symbol (str): The stock symbol.

    Returns:
        str: JSON containing company profile and overview.
    """
    try:
        company_info_full = yf.Ticker(symbol).info
        if company_info_full is None:
            return f"Could not fetch company info for {symbol}"

        company_info_cleaned = {
            "Name": company_info_full.get("shortName"),
            "Symbol": company_info_full.get("symbol"),
            "Current Stock Price": f"{company_info_full.get('regularMarketPrice', company_info_full.get('currentPrice'))} {company_info_full.get('currency', 'USD')}",
            "Market Cap": f"{company_info_full.get('marketCap', company_info_full.get('enterpriseValue'))} {company_info_full.get('currency', 'USD')}",
            "Sector": company_info_full.get("sector"),
            "Industry": company_info_full.get("industry"),
            "Address": company_info_full.get("address1"),
            "City": company_info_full.get("city"),
            "State": company_info_full.get("state"),
            "Zip": company_info_full.get("zip"),
            "Country": company_info_full.get("country"),
            "EPS": company_info_full.get("trailingEps"),
            "P/E Ratio": company_info_full.get("trailingPE"),
            "52 Week Low": company_info_full.get("fiftyTwoWeekLow"),
            "52 Week High": company_info_full.get("fiftyTwoWeekHigh"),
            "50 Day Average": company_info_full.get("fiftyDayAverage"),
            "200 Day Average": company_info_full.get("twoHundredDayAverage"),
            "Website": company_info_full.get("website"),
            "Summary": company_info_full.get("longBusinessSummary"),
            "Analyst Recommendation": company_info_full.get("recommendationKey"),
            "Number Of Analyst Opinions": company_info_full.get("numberOfAnalystOpinions"),
            "Employees": company_info_full.get("fullTimeEmployees"),
            "Total Cash": company_info_full.get("totalCash"),
            "Free Cash flow": company_info_full.get("freeCashflow"),
            "Operating Cash flow": company_info_full.get("operatingCashflow"),
            "EBITDA": company_info_full.get("ebitda"),
            "Revenue Growth": company_info_full.get("revenueGrowth"),
            "Gross Margins": company_info_full.get("grossMargins"),
            "Ebitda Margins": company_info_full.get("ebitdaMargins"),
        }
        return json.dumps(company_info_cleaned, indent=2)
    except Exception as e:
        return f"Error fetching company profile for {symbol}: {e}"


def get_historical_stock_prices(symbol: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Get the historical stock price for a given symbol.

    Args:
        symbol (str): The stock symbol.
        period (str): The period for which to retrieve historical prices. Defaults to "1mo".
                    Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval (str): The interval between data points. Defaults to "1d".
                    Valid intervals: 1d,5d,1wk,1mo,3mo

    Returns:
      str: The current stock price or error message.
    """
    try:
        stock = yf.Ticker(symbol)
        historical_price = stock.history(period=period, interval=interval)
        return historical_price.to_json(orient="index")
    except Exception as e:
        return f"Error fetching historical prices for {symbol}: {e}"


def get_stock_fundamentals(symbol: str) -> str:
    """Get fundamental data for a given stock symbol yfinance API.

    Args:
        symbol (str): The stock symbol.

    Returns:
        str: A JSON string containing fundamental data or an error message.
            Keys:
                - 'symbol': The stock symbol.
                - 'company_name': The long name of the company.
                - 'sector': The sector to which the company belongs.
                - 'industry': The industry to which the company belongs.
                - 'market_cap': The market capitalization of the company.
                - 'pe_ratio': The forward price-to-earnings ratio.
                - 'pb_ratio': The price-to-book ratio.
                - 'dividend_yield': The dividend yield.
                - 'eps': The trailing earnings per share.
                - 'beta': The beta value of the stock.
                - '52_week_high': The 52-week high price of the stock.
                - '52_week_low': The 52-week low price of the stock.
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        fundamentals = {
            "symbol": symbol,
            "company_name": info.get("longName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("forwardPE", "N/A"),
            "pb_ratio": info.get("priceToBook", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "eps": info.get("trailingEps", "N/A"),
            "beta": info.get("beta", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
        }
        return json.dumps(fundamentals, indent=2)
    except Exception as e:
        return f"Error getting fundamentals for {symbol}: {e}"


def get_income_statements(symbol: str) -> str:
    """Get income statements for a given stock symbol.

    Args:
        symbol (str): The stock symbol.

    Returns:
        dict: JSON containing income statements or an empty dictionary.
    """
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        return financials.to_json(orient="index")
    except Exception as e:
        return f"Error fetching income statements for {symbol}: {e}"


def get_key_financial_ratios(symbol: str) -> str:
    """Get key financial ratios for a given stock symbol.

    Args:
        symbol (str): The stock symbol.

    Returns:
        dict: JSON containing key financial ratios.
    """
    try:
        stock = yf.Ticker(symbol)
        key_ratios = stock.info
        return json.dumps(key_ratios, indent=2)
    except Exception as e:
        return f"Error fetching key financial ratios for {symbol}: {e}"


def get_analyst_recommendations(symbol: str) -> str:
    """Get analyst recommendations for a given stock symbol.

    Args:
        symbol (str): The stock symbol.

    Returns:
        str: JSON containing analyst recommendations.
    """
    try:
        stock = yf.Ticker(symbol)
        recommendations = stock.recommendations
        return recommendations.to_json(orient="index")
    except Exception as e:
        return f"Error fetching analyst recommendations for {symbol}: {e}"


def get_company_news(symbol: str, num_stories: int = 3) -> str:
    """Get company news and press releases for a given stock symbol.

    Args:
        symbol (str): The stock symbol.
        num_stories (int): The number of news stories to return. Defaults to 3.

    Returns:
        str: JSON containing company news and press releases.
    """
    try:
        news = yf.Ticker(symbol).news
        return json.dumps(news[:num_stories], indent=2)
    except Exception as e:
        return f"Error fetching company news for {symbol}: {e}"


def get_technical_indicators(symbol: str, period: str = "3mo") -> str:
    """Get technical indicators for a given stock symbol.

    Args:
        symbol (str): The stock symbol.
        period (str): The time period for which to retrieve technical indicators.
            Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. Defaults to 3mo.

    Returns:
        str: JSON containing technical indicators.
    """
    try:
        indicators = yf.Ticker(symbol).history(period=period)
        return indicators.to_json(orient="index")
    except Exception as e:
        return f"Error fetching technical indicators for {symbol}: {e}"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Financial data functions using yfinance")
    parser.add_argument("--symbol", type=str, required=True, help="Stock/asset symbol to query (e.g., MSFT, BTC-USD, GC=F)")
    parser.add_argument("--function", type=str, default="price", choices=[
        "price", "info", "fundamentals", "income", "ratios", 
        "recommendations", "news", "technical", "historical"
    ], help="Function to test")
    parser.add_argument("--period", type=str, default="1mo", help="Period for historical data (e.g., 1d, 5d, 1mo, 3mo, 6mo, 1y)")
    parser.add_argument("--interval", type=str, default="1d", help="Interval for historical data (e.g., 1d, 1wk, 1mo)")
    parser.add_argument("--stories", type=int, default=3, help="Number of news stories to fetch")
    
    args = parser.parse_args()
    
    print(f"Querying data for {args.symbol}...")
    
    if args.function == "price":
        result = get_current_stock_price(args.symbol)
        print(f"Current price for {args.symbol}: {result}")
    
    elif args.function == "info":
        result = get_company_info(args.symbol)
        print(f"Company info for {args.symbol}:\n{result}")
    
    elif args.function == "fundamentals":
        result = get_stock_fundamentals(args.symbol)
        print(f"Fundamentals for {args.symbol}:\n{result}")
    
    elif args.function == "income":
        result = get_income_statements(args.symbol)
        print(f"Income statements for {args.symbol}:\n{result}")
    
    elif args.function == "ratios":
        result = get_key_financial_ratios(args.symbol)
        print(f"Financial ratios for {args.symbol}:\n{result}")
    
    elif args.function == "recommendations":
        result = get_analyst_recommendations(args.symbol)
        print(f"Analyst recommendations for {args.symbol}:\n{result}")
    
    elif args.function == "news":
        result = get_company_news(args.symbol, args.stories)
        print(f"News for {args.symbol}:\n{result}")
    
    elif args.function == "technical":
        result = get_technical_indicators(args.symbol, args.period)
        print(f"Technical indicators for {args.symbol}:\n{result}")
    
    elif args.function == "historical":
        result = get_historical_stock_prices(args.symbol, args.period, args.interval)
        print(f"Historical prices for {args.symbol}:\n{result}")
    
    print("\nExample usage:")
    print("1. Get Microsoft stock price:")
    print("   python finance_functions.py --symbol MSFT --function price")
    print("\n2. Get Bitcoin information:")
    print("   python finance_functions.py --symbol BTC-USD --function info")
    print("\n3. Get Gold historical prices for the past year:")
    print("   python finance_functions.py --symbol GC=F --function historical --period 1y")
    print("\n4. Get Microsoft news (5 stories):")
    print("   python finance_functions.py --symbol MSFT --function news --stories 5")
