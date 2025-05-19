"""Tool for identifying products and required parameters using LLM."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from loguru import logger
from pydantic import Field, BaseModel

from quantalogic.tools.llm_tool import LLMTool
from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.event_emitter import EventEmitter

class ProductIdentifierTool(Tool):
    """Tool to identify products and determine required parameters using LLM."""

    name: str = Field(default="identify_product")
    description: str = Field(
        default=(
            "Identifies which product is being discussed based on user input and catalog data. "
            "Determines required parameters for the product and which ones are already configured."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="user_input",
                arg_type="string",
                description="The user's input describing the product or request.",
                required=True,
                example="I want to buy the Sony noise cancelling headphones in black",
            ),
            ToolArgument(
                name="catalog_path",
                arg_type="string",
                description="Path to the product catalog JSON file.",
                required=False,
                default="",
                example="/path/to/catalogue.json",
            ),
            ToolArgument(
                name="catalog_data",
                arg_type="string",
                description="JSON string with catalog data (alternative to catalog_path).",
                required=False,
                default="",
                example='{"products": [...]}',
            ),
            ToolArgument(
                name="current_params",
                arg_type="string",
                description="JSON string with current parameter values (if any).",
                required=False,
                default="",
                example='{"color": "black", "quantity": 1}',
            ),
        ]
    )

    # Default catalog path
    default_catalog_path: str = Field(
        default=str(Path(__file__).parent / "sample_data" / "catalogue.json"),
        description="Default path to the product catalog JSON file"
    )
    
    # LLM tool for product identification
    llm_tool: LLMTool = Field(default=None, exclude=True)
    
    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        name: str = "identify_product",
        on_token: Callable = None,
        event_emitter: EventEmitter = None,
        catalog_path: str = None,
    ):
        """Initialize the ProductIdentifierTool.
        
        Args:
            model_name: Name of the LLM model to use for product identification.
            name: Name of the tool instance.
            on_token: Optional callback function for streaming tokens.
            event_emitter: Optional event emitter for streaming events.
            catalog_path: Optional path to the catalog file.
        """
        # Set up parameters for super().__init__
        init_params = {
            "name": name,
        }
        
        # Set default catalog path if provided
        if catalog_path:
            init_params["default_catalog_path"] = catalog_path
        else:
            init_params["default_catalog_path"] = str(Path(__file__).parent / "sample_data" / "catalogue.json")
        
        # Initialize with parent class first
        super().__init__(**init_params)
        
        # Initialize the LLM tool for product identification after parent initialization
        self.llm_tool = LLMTool(
            model_name=model_name,
            on_token=on_token,
            event_emitter=event_emitter
        )
    
    def _load_catalog(self, catalog_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load the product catalog from a file.
        
        Args:
            catalog_path: Path to the catalog JSON file. If None, uses the default path.
            
        Returns:
            List of product dictionaries.
            
        Raises:
            FileNotFoundError: If the catalog file doesn't exist.
            ValueError: If the catalog format is invalid.
        """
        path = catalog_path or self.default_catalog_path
        
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Check if the catalog has a 'products' key
            if 'products' in data and isinstance(data['products'], list):
                return data['products']
            elif isinstance(data, list):
                return data
            else:
                raise ValueError(f"Invalid catalog format in {path}")
                
        except FileNotFoundError:
            logger.error(f"Catalog file not found: {path}")
            raise FileNotFoundError(f"Catalog file not found: {path}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in catalog file: {path}")
            raise ValueError(f"Invalid JSON in catalog file: {path}")
        except Exception as e:
            logger.error(f"Error loading catalog: {e}")
            raise Exception(f"Error loading catalog: {e}")
    
    def _extract_required_params(self, product: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """Extract required and optional parameters for a product.
        
        Args:
            product: Product dictionary from the catalog.
            
        Returns:
            Tuple of (basic_params, required_params, optional_params) lists.
        """
        # Basic required parameters (not included in missing_params)
        basic_params = ["id", "name"]
        
        # Required parameters that should be in missing_params if not provided
        required_params = ["quantity"]
        
        # Add options as optional parameters if present
        optional_params = ["description"]
        
        # Extract options from the product
        if "options" in product and isinstance(product["options"], list):
            for option in product["options"]:
                if "name" in option:
                    # Consider options as required parameters
                    required_params.append(option["name"])
        
        return basic_params, required_params, optional_params
    
    def execute(
        self, 
        user_input: str,
        catalog_path: str = "",
        catalog_data: str = "",
        current_params: str = "",
    ) -> str:
        """Execute the tool to identify products and determine required parameters.
        
        Args:
            user_input: The user's input describing the product or request.
            catalog_path: Path to the product catalog JSON file.
            catalog_data: JSON string with catalog data (alternative to catalog_path).
            current_params: JSON string with current parameter values (if any).
            
        Returns:
            JSON string with identified product and parameter information.
        """
        try:
            if not user_input:
                return json.dumps({
                    "success": False,
                    "error": "No user input provided"
                })
            
            # Load catalog data
            products = []
            try:
                if catalog_data:
                    # Parse catalog from string
                    data = json.loads(catalog_data)
                    if 'products' in data and isinstance(data['products'], list):
                        products = data['products']
                    elif isinstance(data, list):
                        products = data
                    else:
                        raise ValueError("Invalid catalog data format")
                else:
                    # Load from file
                    products = self._load_catalog(catalog_path if catalog_path else None)
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "error": f"Error loading catalog: {str(e)}"
                })
            
            # Parse current parameters if provided
            current_param_dict = {}
            if current_params:
                try:
                    current_param_dict = json.loads(current_params)
                except json.JSONDecodeError:
                    return json.dumps({
                        "success": False,
                        "error": "Invalid JSON in current_params"
                    })
            
            # Prepare product data for LLM
            product_data = "\n".join([
                f"ID: {p.get('id', 'unknown')}, Name: {p.get('name', 'unknown')}, "
                f"Category: {p.get('category', 'unknown')}, Price: ${p.get('price', 'unknown')}"
                for p in products[:min(20, len(products))]  # Limit to 20 products to avoid token limits
            ])
            
            # Create system prompt for LLM
            system_prompt = """
            You are a product identification assistant. Your task is to:
            
            1. Identify which product the user is referring to based on their input and the catalog
            2. Determine the required parameters for that product
            3. Check which parameters are already configured and which still need values
            
            Respond with a JSON object containing:
            - identified_product: The product details
            - required_params: List of required parameter names
            - configured_params: Dictionary of parameters that already have values
            - missing_params: List of required parameters that still need values
            - confidence: Your confidence level (0-100) in the product identification
            
            Be precise in your identification. If multiple products match, choose the best fit.
            If no products match well, indicate a low confidence level.
            """
            
            # Create user prompt with input and product data
            user_prompt = f"""
            User input: {user_input}
            
            Available products:
            {product_data}
            
            """
            
            # Add current parameters if available
            if current_param_dict:
                user_prompt += f"""
                Current parameters:
                {json.dumps(current_param_dict, indent=2)}
                """
            
            user_prompt += """
            Please identify the product and determine the required parameters.
            Respond in valid JSON format only.
            """
            
            # Get identification from LLM
            llm_response = self.llm_tool.execute(
                system_prompt=system_prompt,
                prompt=user_prompt,
                temperature="0.2"  # Lower temperature for more consistent identification
            )
            
            # Clean the response - remove markdown code blocks if present
            cleaned_response = llm_response
            if "```json" in cleaned_response:
                # Extract content between ```json and ``` markers
                import re
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", cleaned_response)
                if json_match:
                    cleaned_response = json_match.group(1)
            
            # Parse LLM response as JSON
            try:
                result = json.loads(cleaned_response)
                
                # If the LLM identified a product, enhance the response with actual product data
                if "identified_product" in result and isinstance(result["identified_product"], dict):
                    # Normalize field names (handle case sensitivity)
                    normalized_product = {}
                    for key, value in result["identified_product"].items():
                        normalized_key = key.lower()
                        if normalized_key == "id":
                            normalized_product["id"] = value
                        elif normalized_key == "name":
                            normalized_product["name"] = value
                        else:
                            normalized_product[key] = value
                    
                    # Use normalized product for ID lookup
                    product_id = normalized_product.get("id")
                    
                    # Find the full product in the catalog
                    full_product = None
                    for p in products:
                        if p.get("id") == product_id:
                            full_product = p
                            break
                    
                    if full_product:
                        # Extract required and optional parameters
                        basic_params, required_params, optional_params = self._extract_required_params(full_product)
                        
                        # Update the result with actual product data and parameter information
                        result["identified_product"] = full_product
                        result["required_params"] = basic_params + required_params
                        result["optional_params"] = optional_params
                        
                        # Determine configured and missing parameters
                        configured_params = {}
                        missing_params = []
                        
                        # Only check required_params (not basic_params) for missing parameters
                        for param in required_params:
                            if param in current_param_dict and current_param_dict[param]:
                                configured_params[param] = current_param_dict[param]
                            else:
                                missing_params.append(param)
                        
                        # Also include basic_params in configured_params if they exist
                        for param in basic_params:
                            if param in current_param_dict and current_param_dict[param]:
                                configured_params[param] = current_param_dict[param]
                        
                        result["configured_params"] = configured_params
                        result["missing_params"] = missing_params
                
                return json.dumps(result, indent=2)
                
            except json.JSONDecodeError:
                # If LLM response is not valid JSON, return a structured error
                logger.error(f"Invalid JSON response from LLM: {llm_response}")
                
                return json.dumps({
                    "success": False,
                    "error": "Failed to parse LLM response as JSON",
                    "raw_response": llm_response
                })
            
        except Exception as e:
            logger.error(f"Error in ProductIdentifierTool: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })


if __name__ == "__main__":
    # Example usage
    tool = ProductIdentifierTool()
    
    # Identify product from user input
    result = tool.execute(
        user_input="I want to buy the Sony noise cancelling headphones in black",
        current_params=json.dumps({
            "quantity": 1
        })
    )
    
    print("Product Identification Result:")
    print(result)
