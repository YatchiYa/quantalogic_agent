#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru>=0.7.2",
#     "rich>=13.0.0"
# ]
# ///

import asyncio
from competitive_flow import analyze_competition
from rich import print_json

async def main():
    # Example company information
    company_name = "Tesla"
    industry = "Automotive & Clean Energy"
    products = [
        "Electric Vehicles",
        "Energy Storage Systems",
        "Solar Products",
        "Autonomous Driving Technology"
    ]
    markets = [
        "North America",
        "Europe",
        "China",
        "Global Automotive Market"
    ]
    strengths = [
        "Strong brand recognition",
        "Advanced technology in EVs",
        "Vertical integration",
        "Manufacturing innovation",
        "Global charging network"
    ]
    strategies = [
        "Direct-to-consumer sales model",
        "Continuous innovation in battery technology",
        "Expansion into energy sector",
        "Global manufacturing presence",
        "Focus on autonomous driving technology"
    ]
    
    try:
        # Run the competitive analysis
        report = await analyze_competition(
            company_name=company_name,
            industry=industry,
            products=products,
            markets=markets,
            strengths=strengths,
            strategies=strategies,
            llm_model="gemini/gemini-2.0-flash"
        )
        
        # Print the results
        print_json(data=report.model_dump())
        
    except Exception as e:
        print(f"Error running competitive analysis: {e}")

if __name__ == "__main__":
    asyncio.run(main())
