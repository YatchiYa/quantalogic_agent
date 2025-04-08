try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    driver = webdriver.Chrome()
except:
    print ("AF: No Chrome webdriver installed")
    driver = webdriver.Chrome(ChromeDriverManager().install())

import time
import json
import pandas as pd
from datetime import datetime
from config import ALLOWED_ELEMENT_TYPES,ICON_COLOR_MAP
from utils import reformat_scraped_data
from webdriver_manager.chrome import ChromeDriverManager
import os

# Create screenshots directory if it doesn't exist
screenshots_dir = "screenshots"
if not os.path.exists(screenshots_dir):
    os.makedirs(screenshots_dir)

driver.get("https://www.forexfactory.com/calendar?week=this")

# Take initial screenshot
screenshot_counter = 1
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
screenshot_path = os.path.join(screenshots_dir, f"forex_calendar_{timestamp}_part{screenshot_counter}.png")
driver.save_screenshot(screenshot_path)
print(f"Screenshot {screenshot_counter} saved to: {screenshot_path}")

month = datetime.now().strftime("%B")

table = driver.find_element(By.CLASS_NAME, "calendar__table")

data = []
previous_row_count = 0
# Scroll down to the end of the page
while True:
    # Record the current scroll position
    before_scroll = driver.execute_script("return window.pageYOffset;")
    
    # Scroll down a fixed amount
    driver.execute_script("window.scrollTo(0, window.pageYOffset + 500);")
    
    # Wait for a short moment to allow content to load
    time.sleep(2)
    
    # Take screenshot after each scroll
    screenshot_counter += 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join(screenshots_dir, f"forex_calendar_{timestamp}_part{screenshot_counter}.png")
    driver.save_screenshot(screenshot_path)
    print(f"Screenshot {screenshot_counter} saved to: {screenshot_path}")
    
    # Record the new scroll position
    after_scroll = driver.execute_script("return window.pageYOffset;")
    
    # If the scroll position hasn't changed, we've reached the end of the page
    if before_scroll == after_scroll:
        break

# Take final screenshot
screenshot_counter += 1
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
screenshot_path = os.path.join(screenshots_dir, f"forex_calendar_{timestamp}_part{screenshot_counter}.png")
driver.save_screenshot(screenshot_path)
print(f"Screenshot {screenshot_counter} saved to: {screenshot_path}")

# Now that we've scrolled to the end, collect the data
for row in table.find_elements(By.TAG_NAME, "tr"):
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
            # Handle empty actual/forecast/previous cells
            elif any(field in class_name for field in ["calendar__actual", "calendar__forecast", "calendar__previous"]):
                row_data.append("") # Add empty string for missing values
        
        # Additional data extraction for specific fields
        if "calendar__actual" in class_name:
            actual_value = element.text.strip() if element.text else ""
            if actual_value not in row_data:  # Avoid duplicates
                row_data.append(actual_value)
        
        if "calendar__forecast" in class_name:
            forecast_value = element.text.strip() if element.text else ""
            if forecast_value not in row_data:  # Avoid duplicates
                row_data.append(forecast_value)
        
        if "calendar__previous" in class_name:
            previous_text = element.text.strip()
            # Handle revised values
            if "Revised from" in previous_text:
                previous_value = previous_text.split('\n')[0]  # Get the first line which contains the value
            else:
                previous_value = previous_text
            if previous_value and previous_value not in row_data:  # Avoid duplicates
                row_data.append(previous_value)

    if len(row_data):
        data.append(row_data)

reformat_scraped_data(data,month)