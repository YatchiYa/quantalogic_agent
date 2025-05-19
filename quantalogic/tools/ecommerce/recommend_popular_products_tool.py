"""Tool for recommending popular products using LLM."""

import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from loguru import logger
from pydantic import Field

from quantalogic.tools.llm_tool import LLMTool
from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.event_emitter import EventEmitter


class RecommendPopularProductsTool(Tool):
    """Tool to recommend popular products based on user preferences."""

    name: str = Field(default="recommend_popular_products")
    description: str = Field(
        default=(
            "Recommends popular products based on optional user preferences. "
            "Uses a CSV database of popular products and leverages an LLM to make personalized recommendations."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="preferences",
                arg_type="string",
                description="Optional user preferences for product recommendations (e.g., category, price range, features).",
                required=False,
                default="",
                example="I'm looking for electronics under $100 with good ratings",
            ),
            ToolArgument(
                name="num_recommendations",
                arg_type="int",
                description="Number of products to recommend.",
                required=False,
                default="3",
                example="5",
            ),
        ]
    )

    # Path to the CSV file containing popular products
    csv_path: str = Field(
        default=str(Path(__file__).parent / "sample_data" / "popular_products.csv"),
        description="Path to the CSV file containing popular products data"
    )
    
    # LLM tool for generating personalized recommendations
    llm_tool: LLMTool = Field(default=None, exclude=True)
    
    def __init__(
        self,
        csv_path: Optional[str] = None,
        model_name: str = "openai/gpt-4o-mini",
        name: str = "recommend_popular_products",
        on_token: Callable = None,
        event_emitter: EventEmitter = None,
    ):
        """Initialize the RecommendPopularProductsTool.
        
        Args:
            csv_path: Optional path to the CSV file containing popular products.
                     If not provided, uses the default path in the package.
            model_name: Name of the LLM model to use for recommendations.
            name: Name of the tool instance.
            on_token: Optional callback function for streaming tokens.
            event_emitter: Optional event emitter for streaming events.
        """
        # Set up parameters for super().__init__
        init_params = {
            "name": name,
        }
        
        # Use the provided CSV path or the default one
        if csv_path:
            init_params["csv_path"] = csv_path
        else:
            init_params["csv_path"] = str(Path(__file__).parent / "sample_data" / "popular_products.csv")
            
        # Initialize with parent class first
        super().__init__(**init_params)
        
        # Initialize the LLM tool for recommendations after parent initialization
        self.llm_tool = LLMTool(
            model_name=model_name,
            on_token=on_token,
            event_emitter=event_emitter
        )
        
        # Validate that the CSV file exists
        if not os.path.exists(self.csv_path):
            logger.error(f"CSV file not found at {self.csv_path}")
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")
    
    def _load_products(self) -> List[Dict[str, Any]]:
        """Load products from the CSV file.
        
        Returns:
            List of product dictionaries.
        """
        products = []
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    products.append(dict(row))
            return products
        except Exception as e:
            logger.error(f"Error loading products from CSV: {e}")
            raise Exception(f"Error loading products from CSV: {e}")
    
    def execute(self, preferences: str = "", num_recommendations: str = "3") -> str:
        """Execute the tool to recommend popular products.
        
        Args:
            preferences: Optional user preferences for product recommendations.
            num_recommendations: Number of products to recommend.
            
        Returns:
            Formatted string containing product recommendations.
        """
        try:
            # Convert num_recommendations to integer
            try:
                num_recs = int(num_recommendations)
                if num_recs < 1:
                    num_recs = 3  # Default to 3 if invalid
            except ValueError:
                num_recs = 3  # Default to 3 if conversion fails
            
            # Load products from CSV
            products = self._load_products()
            
            # If no products found, return error message
            if not products:
                return "No products found in the database."
            
            # Sort products by popularity score (descending)
            products.sort(key=lambda x: float(x.get('popularity_score', 0)), reverse=True)
            
            # If no preferences provided, return top N products
            if not preferences:
                top_products = products[:num_recs]
                result = "Top recommended products:\n\n"
                for i, product in enumerate(top_products, 1):
                    # Include tags and sustainability score if available
                    tags_info = f", Tags: {product.get('tags', '')}" if 'tags' in product else ""
                    sustainability = f", Sustainability: {product.get('sustainability_score', 'N/A')}" if 'sustainability_score' in product else ""
                    
                    result += (f"{i}. {product['name']} - ${product['price']} "
                              f"(Category: {product['category']}, Rating: {product['rating']}"
                              f"{tags_info}{sustainability})\n")
                return result
            
            # If preferences provided, use LLM to filter and recommend products
            # Prepare product data for LLM
            product_data = "\n".join([
                f"ID: {p['id']}, Name: {p['name']}, Category: {p['category']}, "
                f"Price: ${p['price']}, Rating: {p['rating']}, Stock: {p['stock']}, "
                f"Popularity: {p['popularity_score']}"
                + (f", Tags: {p.get('tags', '')}" if 'tags' in p else "")
                + (f", Release Date: {p.get('release_date', '')}" if 'release_date' in p else "")
                + (f", Sustainability Score: {p.get('sustainability_score', '')}" if 'sustainability_score' in p else "")
                for p in products[:min(20, len(products))]  # Limit to top 20 products to avoid token limits
            ])
            
            # Create system prompt for LLM
            system_prompt = """
            You are a product recommendation assistant. Your task is to recommend products 
            based on user preferences from the provided product catalog. 
            Consider all available product attributes including category, price, rating, 
            popularity, tags, release date, and sustainability score when making recommendations.
            
            Format your response as a numbered list with comprehensive product details.
            For each product, explain why it matches the user's preferences, highlighting
            specific features that align with their needs.
            """
            
            # Create user prompt with preferences and product data
            user_prompt = f"""
            User preferences: {preferences}
            
            Available products:
            {product_data}
            
            Please recommend {num_recs} products that best match the user's preferences.
            For each product, include:
            1. Product name
            2. Price
            3. Category
            4. Rating
            5. Key tags or features
            6. Sustainability score (if relevant to preferences)
            7. A detailed explanation of why this product matches the user's preferences
            
            Format as a numbered list with clear sections for each product.
            """
            
            # Get recommendations from LLM
            recommendations = self.llm_tool.execute(
                system_prompt=system_prompt,
                prompt=user_prompt,
                temperature="0.3"  # Lower temperature for more consistent recommendations
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error recommending products: {e}")
            return f"Error recommending products: {str(e)}"


if __name__ == "__main__":
    # Example usage
    tool = RecommendPopularProductsTool()
    
    # Recommend products without preferences
    print("Recommendations without preferences:")
    print(tool.execute())
    
    print("\n" + "-" * 50 + "\n")
    
    # Recommend products with preferences
    print("Recommendations with preferences:")
    print(tool.execute(
        preferences="I'm looking for electronics under $100 with good ratings",
        num_recommendations="2"
    ))
