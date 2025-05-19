#!/usr/bin/env python3
"""Test script demonstrating the full e-commerce flow using all tools."""

import json
import sys
from pathlib import Path

from loguru import logger

# Add parent directory to path to ensure imports work
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from quantalogic.tools.ecommerce.recommend_popular_products_tool import RecommendPopularProductsTool
from quantalogic.tools.ecommerce.product_identifier_tool import ProductIdentifierTool
from quantalogic.tools.ecommerce.product_memory_tool import ProductMemoryTool
from quantalogic.tools.ecommerce.command_validator_tool import ProductValidatorTool


def format_section(title):
    """Format a section title for better readability."""
    line = "=" * 80
    return f"\n{line}\n{title}\n{line}\n"


def main():
    """Run the full e-commerce flow test."""
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Initialize all tools
    recommend_tool = RecommendPopularProductsTool()
    identifier_tool = ProductIdentifierTool()
    memory_tool = ProductMemoryTool()
    validator_tool = ProductValidatorTool()
    
    # Step 1: Get product recommendations based on user preferences
    print(format_section("STEP 1: PRODUCT RECOMMENDATIONS"))
    user_query = "I'm looking for high-quality noise cancelling headphones"
    print(f"User query: '{user_query}'")
    
    recommendations = recommend_tool.execute(
        preferences=user_query,
        num_recommendations="3"
    )
    print("\nRecommendations:")
    print(recommendations)
    
    # Step 2: User selects a product (simulated by providing a description)
    print(format_section("STEP 2: PRODUCT IDENTIFICATION"))
    user_selection = "I want to buy the Sony noise cancelling headphones in black"
    print(f"User selection: '{user_selection}'")
    
    # Initial parameters (quantity only)
    initial_params = {
        "quantity": 1
    }
    
    # Identify the product
    identification_result = identifier_tool.execute(
        user_input=user_selection,
        current_params=json.dumps(initial_params)
    )
    
    # Parse the identification result
    try:
        identified_data = json.loads(identification_result)
        print("\nIdentified Product:")
        if "identified_product" in identified_data:
            product = identified_data["identified_product"]
            print(f"Product: {product.get('name', 'Unknown')} (ID: {product.get('id', 'Unknown')})")
            print(f"Category: {product.get('category', 'Unknown')}")
            print(f"Price: ${product.get('price', 'Unknown')}")
            
            print("\nRequired Parameters:")
            for param in identified_data.get("required_params", []):
                print(f"- {param}")
                
            print("\nMissing Parameters:")
            for param in identified_data.get("missing_params", []):
                print(f"- {param}")
                
            print("\nAlready Configured Parameters:")
            for param, value in identified_data.get("configured_params", {}).items():
                print(f"- {param}: {value}")
        else:
            print("No product identified.")
            return
    except json.JSONDecodeError:
        print("Error parsing identification result.")
        print(identification_result)
        return
    
    # Step 3: User provides missing parameters (simulated)
    print(format_section("STEP 3: PARAMETER COLLECTION"))
    print("User provides missing parameters:")
    
    # Get the product from identification
    product = identified_data.get("identified_product", {})
    product_id = product.get("id")
    
    # Get missing parameters
    missing_params = identified_data.get("missing_params", [])
    
    # Simulate user providing the missing parameters
    updated_params = initial_params.copy()
    
    if "color" in missing_params:
        color = "black"
        print(f"- color: {color}")
        updated_params["color"] = color
        
    if "warranty" in missing_params:
        warranty = "2 years"
        print(f"- warranty: {warranty}")
        updated_params["warranty"] = warranty
    
    # Add other missing parameters with default values
    for param in missing_params:
        if param not in updated_params:
            updated_params[param] = f"default_{param}"
            print(f"- {param}: {updated_params[param]}")
    
    # Step 4: Store the product in memory
    print(format_section("STEP 4: PRODUCT STORAGE"))
    
    # Create a complete product entry
    product_entry = {
        "id": product_id,
        "name": product.get("name"),
        "category": product.get("category"),
        "price": product.get("price"),
        "quantity": updated_params.get("quantity", 1),
        "description": product.get("description", ""),
        "options": []
    }
    
    # Add options in the correct format (list of dictionaries)
    option_values = {}
    for param, value in updated_params.items():
        if param not in ["id", "name", "category", "price", "quantity", "description"]:
            option_values[param] = value
    
    # Convert to the expected format
    for option_name, option_value in option_values.items():
        product_entry["options"].append({
            "name": option_name,
            "values": [option_value]  # Values should be a list
        })
    
    # Store in memory
    memory_result = memory_tool.execute(
        action="create",
        product_data=json.dumps(product_entry)
    )
    print(memory_result)
    
    # List products in memory
    print("\nProducts in memory:")
    print(memory_tool.execute(action="list"))
    
    # Step 5: Validate the purchase command
    print(format_section("STEP 5: COMMAND VALIDATION"))
    
    command = "purchase"
    print(f"Command to validate: '{command}'")
    
    # Prepare product data for validation
    validation_data = {
        "id": product_id,
        "name": product.get("name"),
        "quantity": updated_params.get("quantity", 1),
    }
    
    # Add options
    for param, value in updated_params.items():
        if param not in ["id", "name", "category", "price", "quantity"]:
            validation_data[param] = value
    
    # Required parameters for validation
    required_params = ["id", "name", "quantity"] + list(set(missing_params) - set(["quantity"]))
    
    # Validate the command
    validation_result = validator_tool.execute(
        command=command,
        product_data=json.dumps(validation_data),
        required_params=",".join(required_params)
    )
    
    print("\nValidation Result:")
    print(validation_result)
    
    # Step 6: Simulate command execution (purchase)
    print(format_section("STEP 6: COMMAND EXECUTION"))
    print(f"Executing command: {command}")
    print(f"Product: {product.get('name')} (ID: {product_id})")
    print(f"Quantity: {updated_params.get('quantity', 1)}")
    
    options_str = ", ".join([f"{k}: {v}" for k, v in updated_params.items() 
                           if k not in ["id", "name", "category", "price", "quantity"]])
    print(f"Options: {options_str}")
    print(f"Price: ${product.get('price', '0.00')}")
    
    total_price = float(product.get('price', '0.00')) * updated_params.get('quantity', 1)
    print(f"Total Price: ${total_price:.2f}")
    
    print("\nOrder successfully placed!")
    
    # Step 7: Clean up (optional)
    print(format_section("STEP 7: CLEANUP"))
    print("Clearing product memory...")
    print(memory_tool.execute(action="clear"))


if __name__ == "__main__":
    main()
