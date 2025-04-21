#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru>=0.7.2",
#     "rich>=13.0.0"
# ]
# ///

import asyncio
from topic_flow import monitor_topic
from rich import print_json

async def main():
    # Example topic and context
    topic = "Artificial Intelligence in Healthcare"
    context = {
        "industry_focus": "Healthcare",
        "geographic_scope": "Global",
        "time_range": "Last 3 months",
        "specific_interests": [
            "AI diagnostic tools",
            "Medical imaging analysis",
            "Patient data privacy",
            "Regulatory compliance"
        ],
        "stakeholders": [
            "Healthcare providers",
            "AI technology companies",
            "Medical researchers",
            "Regulatory bodies"
        ]
    }
    
    try:
        # Run the monitoring workflow
        report = await monitor_topic(
            topic=topic,
            context=context,
            llm_model="gemini/gemini-2.0-flash"
        )
        
        # Print the results
        print_json(data=report.model_dump())
        
    except Exception as e:
        print(f"Error running monitoring workflow: {e}")

if __name__ == "__main__":
    asyncio.run(main())
