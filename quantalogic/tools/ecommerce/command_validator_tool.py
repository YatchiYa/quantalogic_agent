"""Tool for validating commands and product parameters using LLM."""

import json
from typing import Optional, List, Dict, Any, Union

from loguru import logger
from pydantic import Field, BaseModel

from quantalogic.tools.llm_tool import LLMTool
from quantalogic.tools.tool import Tool, ToolArgument
from quantalogic.event_emitter import EventEmitter


class ProductParameter(BaseModel):
    """Model for a product parameter."""
    name: str
    value: Optional[str] = None
    required: bool = False


class ProductValidatorTool(Tool):
    """Tool to validate product parameters and commands using LLM."""

    name: str = Field(default="validate_product_command")
    description: str = Field(
        default=(
            "Validates product parameters and commands. "
            "Checks if all required parameters are filled and asks the client "
            "if they want to change parameters or proceed with validation."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="command",
                arg_type="string",
                description="The command to validate (e.g., 'purchase', 'add_to_cart', 'checkout').",
                required=True,
                example="purchase",
            ),
            ToolArgument(
                name="product_data",
                arg_type="string",
                description="JSON string with product data to validate.",
                required=True,
                example='{"id": "PROD-001", "name": "Premium Headphones", "quantity": 1, "color": "black"}',
            ),
            ToolArgument(
                name="required_params",
                arg_type="string",
                description="Comma-separated list of required parameters.",
                required=False,
                default="id,name",
                example="id,name,quantity,color",
            ),
            ToolArgument(
                name="optional_params",
                arg_type="string",
                description="Comma-separated list of optional parameters.",
                required=False,
                default="",
                example="warranty,gift_wrap,message",
            ),
        ]
    )

    # LLM tool for validation
    llm_tool: LLMTool = Field(default=None, exclude=True)

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        name: str = "validate_product_command",
        on_token: callable = None,
        event_emitter: EventEmitter = None,
    ):
        """Initialize the ProductValidatorTool.

        Args:
            model_name: Name of the LLM model to use for validation.
            name: Name of the tool instance.
            on_token: Optional callback function for streaming tokens.
            event_emitter: Optional event emitter for streaming events.
        """
        # Set up parameters for super().__init__
        init_params = {
            "name": name,
        }

        # Initialize with parent class first
        super().__init__(**init_params)

        # Initialize the LLM tool for validation after parent initialization
        self.llm_tool = LLMTool(
            model_name=model_name,
            on_token=on_token,
            event_emitter=event_emitter
        )

    def _validate_product_params(
        self,
        product_data: Dict[str, Any],
        required_params: List[str],
        optional_params: List[str]
    ) -> Dict[str, Any]:
        """Validate product parameters.

        Args:
            product_data: Dictionary of product data.
            required_params: List of required parameter names.
            optional_params: List of optional parameter names.

        Returns:
            Dictionary with validation results.
        """
        validation_result = {
            "valid": True,
            "missing_params": [],
            "empty_params": [],
            "all_params": {},
        }

        # Check required parameters
        for param in required_params:
            if param not in product_data:
                validation_result["valid"] = False
                validation_result["missing_params"].append(param)
            elif not product_data.get(param):
                validation_result["valid"] = False
                validation_result["empty_params"].append(param)

        # Collect all parameters (required and optional) with their values
        for param in required_params + optional_params:
            if param in product_data:
                validation_result["all_params"][param] = product_data.get(param)
            else:
                validation_result["all_params"][param] = None

        return validation_result

    def execute(
        self,
        command: str,
        product_data: str,
        required_params: str = "id,name",
        optional_params: str = "",
    ) -> str:
        """Execute the tool to validate product parameters and commands.

        Args:
            command: The command to validate.
            product_data: JSON string with product data to validate.
            required_params: Comma-separated list of required parameters.
            optional_params: Comma-separated list of optional parameters.

        Returns:
            Validation result with explanation and interactive options.
        """
        try:
            if not command:
                return "Error: No command provided for validation."

            if not product_data:
                return "Error: No product data provided for validation."

            # Parse product data
            try:
                product_dict = json.loads(product_data)
            except json.JSONDecodeError:
                return "Error: Invalid JSON in product_data."

            # Parse required and optional parameters
            req_params = [p.strip() for p in required_params.split(',') if p.strip()]
            opt_params = [p.strip() for p in optional_params.split(',') if p.strip()]

            # Validate product parameters
            validation = self._validate_product_params(
                product_dict, req_params, opt_params
            )

            # If all required parameters are valid, use LLM to validate the command
            if validation["valid"]:
                # Create system prompt for LLM
                system_prompt = """
                You are a product command validation assistant. Your task is to analyze a command and product data to determine if:

                1. The command is valid for the given product
                2. All parameters are appropriate for the command
                3. The command can be safely executed

                Provide a clear assessment with the following structure:
                - Command: [command name]
                - Valid: Yes/No
                - Product: [product name/id]
                - Parameters: [list key parameters]
                - Explanation: [brief explanation of your assessment]
                - Confirmation: [ask if the user wants to proceed with the command]

                Be thorough in your analysis and present options clearly.
                """

                # Create user prompt with command and product data
                user_prompt = f"""
                Command to validate: {command}

                Product data:
                {json.dumps(product_dict, indent=2)}

                Please validate this command with the product data and ask the user if they want to proceed.
                """

                # Get validation from LLM
                validation_result = self.llm_tool.execute(
                    system_prompt=system_prompt,
                    prompt=user_prompt,
                    temperature="0.2"  # Lower temperature for more consistent validation
                )

                return validation_result
            else:
                # If validation failed, create an interactive response asking for missing parameters
                missing = validation["missing_params"]
                empty = validation["empty_params"]
                all_params = validation["all_params"]

                response = f"Command '{command}' requires additional information before proceeding.\n\n"

                if missing:
                    response += f"Missing parameters: {', '.join(missing)}\n"

                if empty:
                    response += f"Empty parameters: {', '.join(empty)}\n"

                response += "\nCurrent product parameters:\n"
                for param, value in all_params.items():
                    status = "✓" if value else "✗"
                    required = "(required)" if param in req_params else "(optional)"
                    response += f"- {param}: {value if value else 'Not set'} {status} {required}\n"

                # Ask the client to provide missing parameters or modify existing ones
                response += "\nPlease provide the missing information:\n"

                # For each missing or empty parameter, ask for input
                for param in missing + empty:
                    response += f"- {param}: [Please enter a value]\n"

                response += "\nYou can also modify any existing parameter if needed.\n"
                response += "When you're done, please choose one of the following options:\n"
                response += "1. Update parameters\n"
                response += "2. Validate with current parameters\n"
                response += "3. Cancel the command\n"

                # Add all parameters to the template, highlighting missing ones
                for i, (param, value) in enumerate(all_params.items()):
                    comma = "," if i < len(all_params) - 1 else ""
                    if param in missing or param in empty:
                        response += f'  "{param}": "PLEASE FILL THIS"{comma} <!-- Required -->\n'
                    else:
                        response += f'  "{param}": "{value}"{comma}\n'

                response += "}\n```\n"

                return response

        except Exception as e:
            logger.error(f"Error validating product command: {e}")
            return f"Error validating product command: {str(e)}"


if __name__ == "__main__":
    # Example usage
    tool = ProductValidatorTool()

    # Validate a complete product command
    print("Validating complete product command:")
    print(tool.execute(
        command="purchase",
        product_data=json.dumps({
            "id": "PROD-001",
            "name": "Premium Headphones",
            "quantity": 1,
            "color": "black"
        }),
        required_params="id,name,quantity,color"
    ))

    print("\n" + "-" * 50 + "\n")

    # Validate an incomplete product command
    print("Validating incomplete product command:")
    print(tool.execute(
        command="purchase",
        product_data=json.dumps({
            "id": "PROD-001",
            "name": "Premium Headphones",
            "quantity": ""
        }),
        required_params="id,name,quantity,color"
    ))
