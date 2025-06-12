"""
Model pricing information for token cost calculation.
"""
from typing import Dict, Tuple, Optional
from loguru import logger

# Pricing structure: (input_price_per_1k_tokens, output_price_per_1k_tokens)
# Prices in USD
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI models
    "gpt-4": (0.03, 0.06),
    "gpt-4-32k": (0.06, 0.12),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4-turbo-preview": (0.01, 0.03),
    "gpt-4o": (0.01, 0.03),
    "gpt-4o-mini": (0.0015, 0.006),  # $1.50/$6.00 per million tokens
    "gpt-3.5-turbo": (0.001, 0.002),
    "gpt-3.5-turbo-16k": (0.001, 0.002),
    "gpt-3.5-turbo-instruct": (0.0015, 0.002),
    
    # Anthropic models
    "claude-instant-1": (0.0008, 0.0024),
    "claude-2": (0.008, 0.024),
    "claude-2.1": (0.008, 0.024),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    
    # Bedrock models
    "anthropic.claude-v2": (0.008, 0.024),
    "anthropic.claude-v2:1": (0.008, 0.024),
    "anthropic.claude-3-sonnet-20240229-v1:0": (0.003, 0.015),
    "anthropic.claude-3-haiku-20240307-v1:0": (0.00025, 0.00125),
    "anthropic.claude-3-opus-20240229-v1:0": (0.015, 0.075),
    "amazon.titan-text-express-v1": (0.0002, 0.0002),
    "amazon.titan-text-lite-v1": (0.0003, 0.0004),
    "cohere.command-text-v14": (0.0005, 0.0015),
    "cohere.command-light-text-v14": (0.0003, 0.0006),
    "meta.llama2-13b-chat-v1": (0.00075, 0.00095),
    "meta.llama2-70b-chat-v1": (0.00195, 0.00256),
    
    # Google models
    "gemini-pro": (0.00025, 0.0005),
    "gemini-pro-vision": (0.00025, 0.0005),
    "gemini-ultra": (0.00125, 0.00375),
    
    # Default fallback
    "default": (0.001, 0.002)  # Default to GPT-3.5 pricing
}


def get_model_pricing(model_name: str) -> Tuple[float, float]:
    """
    Get the pricing for a specific model.
    
    Args:
        model_name: The name of the model
        
    Returns:
        Tuple of (input_price_per_1k_tokens, output_price_per_1k_tokens)
    """
    # Try exact match first
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    
    # Try partial match for model families
    for key in MODEL_PRICING:
        if key in model_name:
            return MODEL_PRICING[key]
    
    # Return default pricing if no match found
    return MODEL_PRICING["default"]


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate the cost for a specific model and token usage.
    
    Args:
        model_name: The name of the model
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        
    Returns:
        Total cost in USD
    """
    logger.debug(f" == = = = == == = = =Model name for pricing: {model_name}")
    logger.info(f" == = = = == == = = =Model name for pricing: {model_name}")
    input_price, output_price = get_model_pricing(model_name)
    logger.debug(f" == = = = == == = = =Input price: {input_price}, Output price: {output_price}")
    logger.info(f" == = = = == == = = =Input price: {input_price}, Output price: {output_price}")
    # Convert from price per 1k tokens to price per token
    input_cost = (prompt_tokens / 1000) * input_price
    output_cost = (completion_tokens / 1000) * output_price
    logger.debug(f" == = = = == == = = =Input cost: {input_cost}, Output cost: {output_cost}")
    logger.info(f" == = = = == == = = =Input cost: {input_cost}, Output cost: {output_cost}")
    
    return input_cost + output_cost
