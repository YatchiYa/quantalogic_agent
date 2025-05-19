"""Tool for building and maintaining product information with memory."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from loguru import logger
from pydantic import Field, BaseModel

from quantalogic.tools.tool import Tool, ToolArgument


class ProductSchema(BaseModel):
    """Schema for a product."""
    
    id: str
    name: str
    category: str
    price: str
    description: Optional[str] = None
    options: Optional[List[Dict[str, Union[str, List[str]]]]] = None
    image_url: Optional[str] = None


class ProductMemoryTool(Tool):
    """Tool to build and maintain product information with memory."""

    name: str = Field(default="product_memory")
    description: str = Field(
        default=(
            "Builds and maintains product information with memory. "
            "Can create new products, update existing ones, and retrieve product information. "
            "Products are stored in memory and can be persisted to a file."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="action",
                arg_type="string",
                description="Action to perform: create, update, get, list, or save.",
                required=True,
                example="create",
            ),
            ToolArgument(
                name="product_data",
                arg_type="string",
                description="JSON string with product data for create/update actions.",
                required=False,
                default="",
                example='{"id": "PROD-001", "name": "Premium Headphones", "category": "electronics", "price": "129.99"}',
            ),
            ToolArgument(
                name="product_id",
                arg_type="string",
                description="Product ID for get/update actions.",
                required=False,
                default="",
                example="PROD-001",
            ),
            ToolArgument(
                name="file_path",
                arg_type="string",
                description="Optional file path for saving products (defaults to memory_products.json in sample_data).",
                required=False,
                default="",
                example="/path/to/products.json",
            ),
        ]
    )

    # Memory to store products
    products: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Default path for saving products
    default_save_path: str = Field(
        default=str(Path(__file__).parent / "sample_data" / "memory_products.json"),
        description="Default path for saving products"
    )
    
    def __init__(
        self,
        name: str = "product_memory",
        initial_products: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize the ProductMemoryTool.
        
        Args:
            name: Name of the tool instance.
            initial_products: Optional dictionary of initial products.
        """
        # Set up parameters for super().__init__
        init_params = {
            "name": name,
            "default_save_path": str(Path(__file__).parent / "sample_data" / "memory_products.json")
        }
        
        # Initialize with parent class first
        super().__init__(**init_params)
        
        # Initialize products memory after parent initialization
        if initial_products:
            self.products = initial_products
    
    def _validate_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate product data against schema.
        
        Args:
            product_data: Product data to validate.
            
        Returns:
            Validated product data.
            
        Raises:
            ValueError: If product data is invalid.
        """
        try:
            # Validate using Pydantic model
            validated = ProductSchema(**product_data).model_dump()
            return validated
        except Exception as e:
            logger.error(f"Invalid product data: {e}")
            raise ValueError(f"Invalid product data: {e}")
    
    def _load_from_file(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """Load products from a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Dictionary of products.
            
        Raises:
            FileNotFoundError: If file does not exist.
            Exception: If there's an error loading the file.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # If data has a 'products' key, use that
            if isinstance(data, dict) and 'products' in data:
                products_list = data['products']
            # Otherwise assume it's a list of products
            elif isinstance(data, list):
                products_list = data
            else:
                raise ValueError(f"Invalid file format: {file_path}")
                
            # Convert list to dictionary with ID as key
            return {product['id']: product for product in products_list}
                
        except Exception as e:
            logger.error(f"Error loading products from file: {e}")
            raise Exception(f"Error loading products from file: {e}")
    
    def _save_to_file(self, file_path: str) -> None:
        """Save products to a file.
        
        Args:
            file_path: Path to the file.
            
        Raises:
            Exception: If there's an error saving to the file.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert dictionary to list
            products_list = list(self.products.values())
            
            # Save as JSON with products key
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump({'products': products_list}, file, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving products to file: {e}")
            raise Exception(f"Error saving products to file: {e}")
    
    def execute(
        self, 
        action: str, 
        product_data: str = "", 
        product_id: str = "",
        file_path: str = ""
    ) -> str:
        """Execute the tool to manage product information.
        
        Args:
            action: Action to perform: create, update, get, list, or save.
            product_data: JSON string with product data for create/update actions.
            product_id: Product ID for get/update actions.
            file_path: Optional file path for saving products.
            
        Returns:
            Result of the action.
        """
        try:
            # Normalize action
            action = action.lower().strip()
            
            # Create a new product
            if action == "create":
                if not product_data:
                    return "Error: product_data is required for create action."
                
                try:
                    # Parse JSON string to dictionary
                    product_dict = json.loads(product_data)
                    
                    # Validate product data
                    validated_product = self._validate_product(product_dict)
                    
                    # Check if product ID already exists
                    if validated_product['id'] in self.products:
                        return f"Error: Product with ID {validated_product['id']} already exists."
                    
                    # Add to memory
                    self.products[validated_product['id']] = validated_product
                    
                    return f"Product {validated_product['id']} created successfully."
                    
                except json.JSONDecodeError:
                    return "Error: Invalid JSON in product_data."
                except Exception as e:
                    return f"Error creating product: {str(e)}"
            
            # Update an existing product
            elif action == "update":
                if not product_id and not product_data:
                    return "Error: product_id and product_data are required for update action."
                
                try:
                    # If product_data is provided, parse it
                    if product_data:
                        product_dict = json.loads(product_data)
                        
                        # If product_id is not provided in arguments but is in the data, use that
                        if not product_id and 'id' in product_dict:
                            product_id = product_dict['id']
                    
                    # Check if product exists
                    if product_id not in self.products:
                        return f"Error: Product with ID {product_id} not found."
                    
                    # If product_data is provided, update the product
                    if product_data:
                        # Get existing product
                        existing_product = self.products[product_id]
                        
                        # Parse JSON string to dictionary
                        update_dict = json.loads(product_data)
                        
                        # Update existing product with new values
                        for key, value in update_dict.items():
                            if key != 'id':  # Don't allow changing the ID
                                existing_product[key] = value
                        
                        # Validate updated product
                        validated_product = self._validate_product(existing_product)
                        
                        # Update in memory
                        self.products[product_id] = validated_product
                    
                    return f"Product {product_id} updated successfully."
                    
                except json.JSONDecodeError:
                    return "Error: Invalid JSON in product_data."
                except Exception as e:
                    return f"Error updating product: {str(e)}"
            
            # Get a product by ID
            elif action == "get":
                if not product_id:
                    return "Error: product_id is required for get action."
                
                if product_id not in self.products:
                    return f"Error: Product with ID {product_id} not found."
                
                # Return product as formatted JSON string
                return json.dumps(self.products[product_id], indent=2)
            
            # List all products
            elif action == "list":
                if not self.products:
                    return "No products in memory."
                
                # Return summary of all products
                result = "Products in memory:\n\n"
                for product_id, product in self.products.items():
                    result += f"- {product_id}: {product['name']} (${product['price']}, Category: {product['category']})\n"
                
                return result
            
            # Save products to file
            elif action == "save":
                # Use provided file path or default
                save_path = file_path if file_path else self.default_save_path
                
                try:
                    self._save_to_file(save_path)
                    return f"Products saved successfully to {save_path}."
                except Exception as e:
                    return f"Error saving products: {str(e)}"
            
            # Load products from file
            elif action == "load":
                if not file_path:
                    return "Error: file_path is required for load action."
                
                try:
                    self.products = self._load_from_file(file_path)
                    return f"Loaded {len(self.products)} products from {file_path}."
                except Exception as e:
                    return f"Error loading products: {str(e)}"
            
            # Clear all products from memory
            elif action == "clear":
                count = len(self.products)
                self.products = {}
                return f"Cleared {count} products from memory."
            
            else:
                return f"Error: Unknown action '{action}'. Valid actions are: create, update, get, list, save, load, clear."
                
        except Exception as e:
            logger.error(f"Error in ProductMemoryTool: {e}")
            return f"Error: {str(e)}"


if __name__ == "__main__":
    # Example usage
    tool = ProductMemoryTool()
    
    # Create a product
    print(tool.execute(
        action="create",
        product_data=json.dumps({
            "id": "TEST-001",
            "name": "Test Product",
            "category": "test",
            "price": "99.99",
            "description": "A test product"
        })
    ))
    
    # List products
    print(tool.execute(action="list"))
    
    # Get a product
    print(tool.execute(action="get", product_id="TEST-001"))
    
    # Update a product
    print(tool.execute(
        action="update",
        product_id="TEST-001",
        product_data=json.dumps({
            "price": "89.99",
            "description": "Updated test product"
        })
    ))
    
    # Get updated product
    print(tool.execute(action="get", product_id="TEST-001"))
    
    # Save products
    print(tool.execute(action="save"))
