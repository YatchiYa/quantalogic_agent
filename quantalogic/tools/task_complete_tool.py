"""Tool for reading a file and returning its content."""

from loguru import logger
from quantalogic.tools.tool import Tool, ToolArgument


class TaskCompleteTool(Tool):
    """Tool to reply answer to the user."""

    name: str = "task_complete"
    description: str = "Replies to the user when the task is completed."
    arguments: list = [
        ToolArgument(
            name="answer",
            arg_type="string",
            description="The answer to the user. Use interpolation if possible example $var1$.",
            required=True,
            example="The answer to the meaning of life",
        ),
    ]

    def execute(self, answer: str) -> str:
        """Attempts to reply to the user.

        Args:
            answer (str): The answer to the user.

        Returns:
            str: The answer to the user.

        Raises:
            ValueError: If the answer contains uninterpolated variables.
        """
        import re

        # Check for uninterpolated variables (those with $ at both start and end)
        var_pattern = r'\$[a-zA-Z_][a-zA-Z0-9_]*\$'
        matches = re.findall(var_pattern, answer)
        
        # Filter out false positives like SCSS variables
        filtered_matches = []
        if matches:
            # Check for code blocks that might contain SCSS/CSS
            code_blocks = re.findall(r'```(?:scss|css|sass)([\s\S]*?)```', answer, re.IGNORECASE)
            
            # Create a list of ranges to exclude (SCSS/CSS code blocks)
            exclude_ranges = []
            for block in code_blocks:
                # Find all occurrences of this block in the answer
                for match in re.finditer(re.escape(block), answer):
                    exclude_ranges.append((match.start(), match.end()))
            
            # Only include matches that aren't in code blocks
            for match in matches:
                match_positions = [(m.start(), m.end()) for m in re.finditer(re.escape(match), answer)]
                
                for start, end in match_positions:
                    in_code_block = False
                    for block_start, block_end in exclude_ranges:
                        if start >= block_start and end <= block_end:
                            in_code_block = True
                            break
                    
                    if not in_code_block:
                        filtered_matches.append(match)
        
        if filtered_matches:
            var_list = ", ".join(set(filtered_matches))  # Use set to remove duplicates
            error_msg = (
                f"Error: Task complete answer contains uninterpolated variables: {var_list}. "
                "Variables should be interpolated before completing the task."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        return answer


if __name__ == "__main__":
    tool = TaskCompleteTool()
    print(tool.to_markdown())
