"""ForexFactory scraper tool for economic calendar data.

This module provides a tool for scraping economic calendar data from ForexFactory.com
using Selenium WebDriver with proper error handling and screenshot capabilities.
"""

import os
import time
import json
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pydantic import BaseModel, Field
import asyncio
from loguru import logger

from ..tool import Tool, ToolArgument
from .utils import reformat_scraped_data
from .config import ALLOWED_ELEMENT_TYPES, ICON_COLOR_MAP

class ForexFactoryScraperConfig(BaseModel):
    """Configuration for ForexFactory scraper.
    
    Attributes:
        screenshots_dir: Directory to save screenshots
        base_url: Base URL for ForexFactory calendar
        scroll_step: Number of pixels to scroll each step
        scroll_delay: Delay between scrolls in seconds
    """
    screenshots_dir: str = Field(default="screenshots", description="Directory to save screenshots")
    base_url: str = Field(
        default="https://www.forexfactory.com/calendar?week=this", 
        description="ForexFactory calendar URL"
    )
    scroll_step: int = Field(default=500, description="Pixels to scroll each step")
    scroll_delay: float = Field(default=2.0, description="Delay between scrolls in seconds")

class ForexFactoryScraperTool(Tool):
    """Tool for scraping economic calendar data from ForexFactory.com"""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self):
        super().__init__(
            name="forex_factory_scraper",
            description="Scrapes economic calendar data from ForexFactory.com with screenshots",
            arguments=[
                ToolArgument(
                    name="save_screenshots",
                    arg_type="boolean",
                    description="Whether to save screenshots during scraping",
                    required=False,
                    default="True"
                ),
                ToolArgument(
                    name="screenshots_dir",
                    arg_type="string",
                    description="Directory to save screenshots",
                    required=False,
                    default="screenshots"
                )
            ],
            need_validation=True
        )
        self.config = ForexFactoryScraperConfig()
        self._driver = None

    def _setup_driver(self) -> None:
        """Set up Chrome WebDriver with proper error handling."""
        try:
            service = Service(ChromeDriverManager().install())
            self._driver = webdriver.Chrome(service=service)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chrome WebDriver: {e}")

    def _save_screenshot(self, counter: int) -> str:
        """Save a screenshot with timestamp.
        
        Args:
            counter: Screenshot counter number
            
        Returns:
            Path to saved screenshot
        """
        if not os.path.exists(self.config.screenshots_dir):
            os.makedirs(self.config.screenshots_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(
            self.config.screenshots_dir, 
            f"forex_calendar_{timestamp}_part{counter}.png"
        )
        self._driver.save_screenshot(screenshot_path)
        return screenshot_path

    def _scroll_and_capture(self, save_screenshots: bool = True) -> None:
        """Scroll through the page and capture screenshots."""
        screenshot_counter = 1
        
        if save_screenshots:
            self._save_screenshot(screenshot_counter)
            
        while True:
            before_scroll = self._driver.execute_script("return window.pageYOffset;")
            self._driver.execute_script(
                f"window.scrollTo(0, window.pageYOffset + {self.config.scroll_step});"
            )
            
            time.sleep(self.config.scroll_delay)
            
            if save_screenshots:
                screenshot_counter += 1
                path = self._save_screenshot(screenshot_counter)
                print(f"Screenshot {screenshot_counter} saved to: {path}")
            
            after_scroll = self._driver.execute_script("return window.pageYOffset;")
            if before_scroll == after_scroll:
                break

    def _extract_row_data(self, row: Any) -> List[str]:
        """Extract data from a table row.
        
        Args:
            row: Selenium WebElement representing a table row
            
        Returns:
            List of extracted cell values
        """
        row_data = []
        cells = row.find_elements(By.TAG_NAME, "td")
        
        for element in cells:
            class_name = element.get_attribute('class')
            
            if class_name in ALLOWED_ELEMENT_TYPES:
                if element.text:
                    row_data.append(element.text)
                elif "calendar__impact" in class_name:
                    impact_elements = element.find_elements(By.TAG_NAME, "span")
                    for impact in impact_elements:
                        impact_class = impact.get_attribute("class")
                        color = ICON_COLOR_MAP[impact_class]
                    if color:
                        row_data.append(color)
                    else:
                        row_data.append("impact")
                elif any(field in class_name for field in ["calendar__actual", "calendar__forecast", "calendar__previous"]):
                    row_data.append("")
                    
            # Extract specific field values
            if "calendar__actual" in class_name:
                actual_value = element.text.strip() if element.text else ""
                if actual_value not in row_data:
                    row_data.append(actual_value)
                    
            if "calendar__forecast" in class_name:
                forecast_value = element.text.strip() if element.text else ""
                if forecast_value not in row_data:
                    row_data.append(forecast_value)
                    
            if "calendar__previous" in class_name:
                previous_text = element.text.strip()
                if "Revised from" in previous_text:
                    previous_value = previous_text.split('\n')[0]
                else:
                    previous_value = previous_text
                if previous_value and previous_value not in row_data:
                    row_data.append(previous_value)
                    
        return row_data

    async def execute(
        self, 
        save_screenshots: bool = True,
        screenshots_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the ForexFactory scraper tool.
        
        Args:
            save_screenshots: Whether to save screenshots during scraping
            screenshots_dir: Directory to save screenshots (optional)
            
        Returns:
            Dictionary containing scraped data and metadata
        """
        if screenshots_dir:
            self.config.screenshots_dir = screenshots_dir
            
        try:
            self._setup_driver()
            self._driver.get(self.config.base_url)
            
            self._scroll_and_capture(save_screenshots)
            
            table = self._driver.find_element(By.CLASS_NAME, "calendar__table")
            data = []
            
            for row in table.find_elements(By.TAG_NAME, "tr"):
                row_data = self._extract_row_data(row)
                if row_data:
                    data.append(row_data)
                    
            month = datetime.now().strftime("%B")
            processed_data = reformat_scraped_data(data, month)
            
            return {
                "status": "success",
                "data": processed_data,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "screenshots_dir": self.config.screenshots_dir if save_screenshots else None,
                    "url": self.config.base_url
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "metadata": {
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        finally:
            if self._driver:
                self._driver.quit()

async def main():
    """Test the ForexFactory scraper tool."""
    logger.info("Starting ForexFactory scraper test")
    
    try:
        # Initialize the scraper tool
        scraper = ForexFactoryScraperTool()
        logger.info("Scraper tool initialized")
        
        # Execute the scraper with default settings
        logger.info("Starting scraping process...")
        result = await scraper.execute(save_screenshots=True)
        
        if result["status"] == "success":
            logger.success("Scraping completed successfully!")
            
            # Save the data to a JSON file
            output_file = "forex_calendar_data.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Data saved to {output_file}")
            
            # Print some statistics
            data = result["data"]
            logger.info(f"Total events scraped: {len(data)}")
            logger.info(f"Screenshot directory: {result['metadata']['screenshots_dir']}")
            logger.info(f"Timestamp: {result['metadata']['timestamp']}")
            
        else:
            logger.error(f"Scraping failed: {result['error']}")
            
    except Exception as e:
        logger.exception(f"Error during scraping: {e}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
