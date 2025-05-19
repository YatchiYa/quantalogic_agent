#!/usr/bin/env python3
"""
CoinMarketCap API Client

A simple Python client for fetching cryptocurrency chart data from CoinMarketCap's API.
"""

import requests
import json
from typing import Dict, Any, Optional
from loguru import logger
import time
from datetime import datetime

API_KEY = "0ba8131f-7578-4e84-8106-c0b3ddc36418"

class CoinMarketCapClient:
    """Client for interacting with CoinMarketCap's API to fetch cryptocurrency data."""
    
    BASE_URL = "https://api.coinmarketcap.com/data-api/v3.3"
    
    # Common cryptocurrency IDs
    CRYPTO_IDS = {
        "bitcoin": 1,
        "ethereum": 1027,
        "tether": 825,
        "binancecoin": 1839,
        "solana": 5426,
        "xrp": 52,
        "usdc": 3408,
        "cardano": 2010,
        "dogecoin": 74,
        "avalanche": 5805
    }
    
    # Available time intervals
    INTERVALS = ["5m", "15m", "30m", "1h", "2h", "4h", "1d", "2d", "3d", "7d", "30d", "90d"]
    
    # Available time ranges
    RANGES = ["1D", "7D", "1M", "3M", "1Y", "YTD", "All"]
    
    def __init__(self):
        """Initialize the CoinMarketCap API client."""
        self.session = requests.Session()
        self.session.headers.update({
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
            'cache-control': 'no-cache',
            'origin': 'https://coinmarketcap.com',
            'platform': 'web',
            'priority': 'u=1, i',
            'referer': 'https://coinmarketcap.com/',
            'sec-ch-ua': '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Linux"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36'
        })
    
    def get_crypto_id(self, crypto_name: str) -> int:
        """
        Get the CoinMarketCap ID for a cryptocurrency.
        
        Args:
            crypto_name: The name of the cryptocurrency (e.g., 'bitcoin', 'ethereum')
            
        Returns:
            The CoinMarketCap ID for the cryptocurrency
            
        Raises:
            ValueError: If the cryptocurrency name is not recognized
        """
        crypto_name = crypto_name.lower()
        if crypto_name not in self.CRYPTO_IDS:
            raise ValueError(f"Unknown cryptocurrency: {crypto_name}. Available options: {list(self.CRYPTO_IDS.keys())}")
        return self.CRYPTO_IDS[crypto_name]
    
    def get_chart_data(self, 
                      crypto_name: str, 
                      interval: str = "1d", 
                      time_range: str = "1M") -> Dict[str, Any]:
        """
        Fetch chart data for a cryptocurrency.
        
        Args:
            crypto_name: The name of the cryptocurrency (e.g., 'bitcoin', 'ethereum')
            interval: Time interval for data points (e.g., '5m', '1h', '1d', '7d')
            time_range: Time range for the chart (e.g., '1D', '7D', '1M', '1Y', 'All')
            
        Returns:
            A dictionary containing the chart data
            
        Raises:
            ValueError: If the interval or time_range is invalid
            requests.RequestException: If the API request fails
        """
        # Validate parameters
        interval = interval.lower()
        
        # Case-insensitive check for time_range
        valid_range = False
        for valid_range_option in self.RANGES:
            if time_range.upper() == valid_range_option.upper():
                time_range = valid_range_option  # Use the correctly cased version
                valid_range = True
                break
        
        if interval not in self.INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Available options: {self.INTERVALS}")
        
        if not valid_range:
            raise ValueError(f"Invalid time range: {time_range}. Available options: {self.RANGES}")
        
        # Get crypto ID
        crypto_id = self.get_crypto_id(crypto_name)
        
        # Generate request ID (similar to what the browser would do)
        request_id = f"{int(time.time() * 1000)}-{crypto_id}-{interval}-{time_range}"
        
        # Set up headers with request ID
        self.session.headers.update({
            'x-request-id': request_id
        })
        
        # Build URL
        url = f"{self.BASE_URL}/cryptocurrency/detail/chart"
        params = {
            "id": crypto_id,
            "interval": interval,
            "range": time_range
        }
        
        try:
            logger.info(f"Fetching chart data for {crypto_name} with interval={interval}, range={time_range}")
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data
        except requests.RequestException as e:
            logger.error(f"Error fetching chart data: {e}")
            raise
    
    def parse_chart_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the chart data into a more usable format.
        
        Args:
            data: The raw chart data from the API
            
        Returns:
            A dictionary with parsed chart data
        """
        result = {
            "timestamps": [],
            "prices": [],
            "volumes": [],
            "market_caps": [],
            "annotations": []
        }
        
        if "data" not in data or "points" not in data["data"]:
            logger.warning("Invalid data format received from API")
            return result
        
        for point in data["data"]["points"]:
            # Convert timestamp to datetime
            timestamp = int(point["s"])
            dt = datetime.fromtimestamp(timestamp)
            
            # Extract values
            if "v" in point and len(point["v"]) >= 3:
                price = point["v"][0]
                volume = point["v"][1]
                market_cap = point["v"][2]
                
                result["timestamps"].append(dt)
                result["prices"].append(price)
                result["volumes"].append(volume)
                result["market_caps"].append(market_cap)
            
            # Extract annotations if present
            if "annotations" in point:
                for annotation in point["annotations"]:
                    result["annotations"].append({
                        "timestamp": datetime.fromtimestamp(int(annotation.get("eventTime", 0)) / 1000),
                        "type": annotation.get("typeName", ""),
                        "title": annotation.get("title", ""),
                        "description": annotation.get("description", ""),
                        "url": annotation.get("readMoreUrl", "")
                    })
        
        return result


def main():
    """Example usage of the CoinMarketCap client."""
    client = CoinMarketCapClient()
    
    # Example 1: Get Bitcoin data for the last day with 5-minute intervals
    btc_data = client.get_chart_data("bitcoin", interval="5m", time_range="1D")
    parsed_btc_data = client.parse_chart_data(btc_data)
    
    # Print some basic info
    print(f"Bitcoin data points: {len(parsed_btc_data['prices'])}")
    if parsed_btc_data['prices'] and parsed_btc_data['timestamps']:
        latest_time = parsed_btc_data['timestamps'][-1]
        print(f"Latest data time: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Latest price: ${parsed_btc_data['prices'][-1]:,.2f}")
        print(f"Latest volume: ${parsed_btc_data['volumes'][-1]:,.2f}")
    
    # Example 2: Get Ethereum historical data with weekly intervals
    eth_data = client.get_chart_data("ethereum", interval="7d", time_range="1Y")
    parsed_eth_data = client.parse_chart_data(eth_data)
    
    print(f"\nEthereum data points: {len(parsed_eth_data['prices'])}")
    if parsed_eth_data['prices'] and parsed_eth_data['timestamps']:
        latest_time = parsed_eth_data['timestamps'][-1]
        print(f"Latest data time: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Latest price: ${parsed_eth_data['prices'][-1]:,.2f}")
    
    # Example 3: Get all-time Bitcoin data
    btc_all_data = client.get_chart_data("bitcoin", interval="7d", time_range="All")
    parsed_btc_all = client.parse_chart_data(btc_all_data)
    
    print(f"\nBitcoin all-time data points: {len(parsed_btc_all['prices'])}")
    if parsed_btc_all['prices'] and len(parsed_btc_all['prices']) > 1 and parsed_btc_all['timestamps']:
        first_time = parsed_btc_all['timestamps'][0]
        latest_time = parsed_btc_all['timestamps'][-1]
        first_price = parsed_btc_all['prices'][0]
        last_price = parsed_btc_all['prices'][-1]
        roi = (last_price - first_price) / first_price * 100
        print(f"First recorded time: {first_time.strftime('%Y-%m-%d')}")
        print(f"First recorded price: ${first_price:,.2f}")
        print(f"Latest data time: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Latest price: ${last_price:,.2f}")
        print(f"All-time ROI: {roi:,.2f}%")
    
    # Print any annotations/news
    if parsed_btc_data['annotations']:
        print("\nRecent Bitcoin news:")
        for i, annotation in enumerate(parsed_btc_data['annotations'][:3], 1):
            print(f"{i}. [{annotation['timestamp']}] {annotation['title'] or annotation['description'][:50] + '...'}")



if __name__ == "__main__":
    main()
