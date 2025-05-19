"""Contract Comparison Tool for comparing contracts against standards or other contracts."""

import asyncio
from typing import Callable, Dict, Any, List, Optional, Union

from loguru import logger
from pydantic import ConfigDict, Field, BaseModel

from quantalogic.console_print_token import console_print_token
from quantalogic.event_emitter import EventEmitter
from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.tools.tool import Tool, ToolArgument


class ClauseComparison(BaseModel):
    """Model for comparing individual clauses."""
    
    clause_name: str = Field(description="Name or title of the clause")
    primary_text: Optional[str] = Field(default=None, description="Text of the clause in the primary contract")
    comparison_text: Optional[str] = Field(default=None, description="Text of the clause in the comparison contract")
    differences: List[str] = Field(default_factory=list, description="Key differences between the clauses")
    favorability: str = Field(description="Assessment of favorability: 'More favorable', 'Less favorable', 'Neutral', or 'N/A'")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")


class ContractComparison(BaseModel):
    """Model for structured contract comparison results."""
    
    # Basic information
    primary_contract_type: str = Field(description="Type of the primary contract")
    comparison_type: str = Field(description="Type of comparison: 'Contract-to-Contract' or 'Contract-to-Standard'")
    comparison_contract_type: Optional[str] = Field(default=None, description="Type of the comparison contract (if applicable)")
    industry_standard: Optional[str] = Field(default=None, description="Industry standard used for comparison (if applicable)")
    
    # Comparison results
    matching_clauses: List[ClauseComparison] = Field(default_factory=list, description="Clauses found in both contracts with comparison")
    primary_only_clauses: List[Dict[str, Any]] = Field(default_factory=list, description="Clauses found only in the primary contract")
    comparison_only_clauses: List[Dict[str, Any]] = Field(default_factory=list, description="Clauses found only in the comparison contract")
    key_differences: List[Dict[str, Any]] = Field(default_factory=list, description="Summary of key differences between contracts")
    
    # Analysis
    overall_assessment: str = Field(description="Overall assessment of the primary contract compared to the standard/comparison")
    missing_standard_clauses: List[str] = Field(default_factory=list, description="Standard clauses missing from the primary contract")
    improvement_recommendations: List[str] = Field(default_factory=list, description="Recommendations for improving the primary contract")
    
    # Executive summary
    comparison_summary: str = Field(description="Brief executive summary of the comparison")


class ContractComparisonTool(Tool):
    """Tool to compare contracts against standards or other contracts."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="contract_comparison_tool")
    description: str = Field(
        default=(
            "Compares contracts against industry standards or between multiple contracts, "
            "identifying differences, similarities, and potential improvements. The tool "
            "provides clause-by-clause comparison, gap analysis, term favorability assessment, "
            "and specific recommendations for improving contract terms."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="primary_contract_text",
                arg_type="string",
                description=(
                    "The full text of the primary contract to analyze. This is the main contract "
                    "that will be compared against a standard or another contract."
                ),
                required=True,
                example="[Full text of primary contract...]",
            ),
            ToolArgument(
                name="comparison_contract_text",
                arg_type="string",
                description=(
                    "The full text of the secondary contract to compare against the primary contract. "
                    "Leave empty if comparing against an industry standard instead of a specific contract."
                ),
                required=False,
                example="[Full text of comparison contract...]",
            ),
            ToolArgument(
                name="industry_standard",
                arg_type="string",
                description=(
                    "The type of industry standard to compare against if not comparing against a specific contract. "
                    "Examples: 'employment', 'lease', 'sale', 'service', 'nda', etc. "
                    "Leave empty if providing a comparison_contract_text."
                ),
                required=False,
                example="employment",
            ),
            ToolArgument(
                name="comparison_focus",
                arg_type="string",
                description=(
                    "Comma-separated list of specific areas to focus on in the comparison. "
                    "Options include: 'payment_terms', 'termination', 'liability', 'confidentiality', "
                    "'intellectual_property', 'all'. Default is 'all'."
                ),
                required=False,
                default="all",
                example="termination,liability",
            ),
            ToolArgument(
                name="output_format",
                arg_type="string",
                description=(
                    "The format for the comparison output. Options: 'structured' (returns a structured "
                    "object with all comparison details), 'summary' (returns a concise text summary), "
                    "or 'detailed' (returns a comprehensive text analysis). Default is 'structured'."
                ),
                required=False,
                default="structured",
                example="summary",
            ),
            ToolArgument(
                name="language",
                arg_type="string",
                description=(
                    "The language of the contracts. This helps the analyzer understand language-specific "
                    "legal terminology and formatting. Defaults to auto-detection."
                ),
                required=False,
                default="auto",
                example="French",
            ),
            ToolArgument(
                name="temperature",
                arg_type="string",
                description=(
                    'Sampling temperature between "0.0" and "1.0": "0.0" for more factual comparison, '
                    '"1.0" for more interpretive analysis. For contract comparison, '
                    'lower values (0.1-0.3) are generally recommended.'
                ),
                required=False,
                default="0.2",
                example="0.2",
            ),
        ]
    )

    model_name: str = Field(..., description="The name of the language model to use")
    system_prompt: str | None = Field(default=None)
    on_token: Callable | None = Field(default=None, exclude=True)
    generative_model: GenerativeModel | None = Field(default=None, exclude=True)
    event_emitter: EventEmitter | None = Field(default=None, exclude=True)

    def __init__(
        self,
        model_name: str,
        system_prompt: str | None = None,
        on_token: Callable | None = None,
        name: str = "contract_comparison_tool",
        generative_model: GenerativeModel | None = None,
        event_emitter: EventEmitter | None = None,
    ):
        """Initialize the ContractComparisonTool with model configuration and optional callback.

        Args:
            model_name (str): The name of the language model to use.
            system_prompt (str, optional): Default system prompt for the model.
            on_token (Callable, optional): Callback function for streaming tokens.
            name (str): Name of the tool instance. Defaults to "contract_comparison_tool".
            generative_model (GenerativeModel, optional): Pre-initialized generative model.
            event_emitter (EventEmitter, optional): Event emitter for streaming tokens.
        """
        # Default system prompt for contract comparison if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert contract analyst specializing in contract comparison and analysis. "
                "Your expertise includes identifying differences between contracts, assessing the relative "
                "favorability of terms, comparing contracts against industry standards, and providing "
                "recommendations for improvement. You excel at clause-by-clause comparison, gap analysis, "
                "and term favorability assessment. You understand contract terminology across different "
                "industries and can adapt your analysis to specific contract types. Your comparisons are "
                "thorough, precise, and focused on practical business implications."
            )

        # Use dict to pass validated data to parent constructor
        super().__init__(
            **{
                "model_name": model_name,
                "system_prompt": system_prompt,
                "on_token": on_token,
                "name": name,
                "generative_model": generative_model,
                "event_emitter": event_emitter,
            }
        )
        
        # Initialize the generative model
        self.model_post_init(None)

    def model_post_init(self, __context):
        """Initialize the generative model after model initialization."""
        if self.generative_model is None:
            self.generative_model = GenerativeModel(
                model=self.model_name,
                event_emitter=self.event_emitter
            )
            logger.debug(f"Initialized ContractComparisonTool with model: {self.model_name}")

        # Only set up event listener if on_token is provided
        if self.on_token is not None:
            logger.debug(f"Setting up event listener for ContractComparisonTool with model: {self.model_name}")
            self.generative_model.event_emitter.on("stream_chunk", self.on_token)

    def _build_prompt(
        self,
        primary_contract_text: str,
        comparison_contract_text: Optional[str] = None,
        industry_standard: Optional[str] = None,
        comparison_focus: str = "all",
        output_format: str = "structured",
        language: str = "auto",
    ) -> str:
        """Build a comprehensive prompt for contract comparison.

        Args:
            primary_contract_text: The full text of the primary contract to analyze.
            comparison_contract_text: The full text of the secondary contract to compare against.
            industry_standard: The type of industry standard to compare against.
            comparison_focus: Specific areas to focus on in the comparison.
            output_format: Format for the comparison output.
            language: Language of the contracts.

        Returns:
            A structured prompt for the LLM.
        """
        # Validate comparison type
        if comparison_contract_text and industry_standard:
            comparison_type = "contract_to_contract"
            logger.warning("Both comparison_contract_text and industry_standard provided. Using comparison_contract_text.")
        elif comparison_contract_text:
            comparison_type = "contract_to_contract"
        elif industry_standard:
            comparison_type = "contract_to_standard"
        else:
            comparison_type = "contract_to_standard"
            industry_standard = "general"
            logger.warning("Neither comparison_contract_text nor industry_standard provided. Using general standards.")
        
        # Validate output format
        valid_output_formats = ["structured", "summary", "detailed"]
        if output_format.lower() not in valid_output_formats:
            output_format = "structured"
        
        # Process focus areas
        focus_areas_list = [area.strip().lower() for area in comparison_focus.split(",")]
        if "all" in focus_areas_list:
            focus_areas_list = ["all"]
        
        # Language instructions
        language_instruction = (
            f"The contracts are in {language}. Analyze accordingly." 
            if language.lower() != "auto" 
            else "Auto-detect the language of the contracts and analyze accordingly."
        )

        # Define output format instructions
        output_format_instructions = {
            "structured": (
                "Return your comparison as a structured JSON object following this schema:\n"
                "```json\n"
                "{\n"
                '  "primary_contract_type": "string",\n'
                '  "comparison_type": "string",\n'
                '  "comparison_contract_type": "string" or null,\n'
                '  "industry_standard": "string" or null,\n'
                '  "matching_clauses": [\n'
                '    {\n'
                '      "clause_name": "string",\n'
                '      "primary_text": "string",\n'
                '      "comparison_text": "string",\n'
                '      "differences": ["string", ...],\n'
                '      "favorability": "string",\n'
                '      "recommendations": ["string", ...]\n'
                '    },\n'
                '    ...\n'
                '  ],\n'
                '  "primary_only_clauses": [\n'
                '    {"clause_name": "string", "summary": "string", "significance": "string"},\n'
                '    ...\n'
                '  ],\n'
                '  "comparison_only_clauses": [\n'
                '    {"clause_name": "string", "summary": "string", "significance": "string"},\n'
                '    ...\n'
                '  ],\n'
                '  "key_differences": [\n'
                '    {"area": "string", "description": "string", "impact": "string"},\n'
                '    ...\n'
                '  ],\n'
                '  "overall_assessment": "string",\n'
                '  "missing_standard_clauses": ["string", ...],\n'
                '  "improvement_recommendations": ["string", ...],\n'
                '  "comparison_summary": "string"\n'
                "}\n"
                "```\n"
                "Ensure all fields are populated with appropriate values. For lists, include at least one item "
                "if relevant information is available. For fields where no relevant information can be determined, "
                "use empty lists [] for list fields, empty strings \"\" for string fields, and null for optional fields. "
                "Ensure the JSON is valid and properly formatted."
            ),
            "summary": (
                "Return a concise text summary of the contract comparison, focusing on the "
                "most important differences, missing clauses, and key recommendations. The summary "
                "should be clear, informative, and highlight the most significant aspects of the comparison. "
                "Format the summary with clear headings and bullet points for readability."
            ),
            "detailed": (
                "Return a comprehensive text analysis organized into sections with headings. Include sections for: "
                "1) Comparison Overview, 2) Clause-by-Clause Comparison, 3) Missing Clauses Analysis, "
                "4) Key Differences, 5) Term Favorability Assessment, 6) Recommendations for Improvement, "
                "and 7) Executive Summary. Provide detailed explanations for each section and include "
                "specific references to relevant contract clauses."
            )
        }

        # Build the base prompt
        prompt = f"""
Compare the following {"contracts" if comparison_type == "contract_to_contract" else "contract against industry standards"} to identify differences, similarities, and potential improvements.

# PRIMARY CONTRACT
```
{primary_contract_text}
```

"""

        # Add comparison contract if provided
        if comparison_type == "contract_to_contract":
            prompt += f"""
# COMPARISON CONTRACT
```
{comparison_contract_text}
```
"""
        else:
            prompt += f"""
# INDUSTRY STANDARD
Compare against standard practices and typical clauses for {industry_standard} contracts.
"""

        # Add analysis parameters
        prompt += f"""
# ANALYSIS PARAMETERS
- Comparison Type: {"Contract-to-Contract" if comparison_type == "contract_to_contract" else "Contract-to-Standard"}
- Industry Standard: {industry_standard if comparison_type == "contract_to_standard" else "N/A"}
- Focus Areas: {", ".join(focus_areas_list)}
- Output Format: {output_format}
- Language: {language}

# COMPARISON REQUIREMENTS
Perform a comprehensive comparison addressing the following aspects:

1. **Clause-by-Clause Comparison**
   - Identify corresponding clauses between {"the contracts" if comparison_type == "contract_to_contract" else "the contract and industry standards"}
   - For each matching clause:
     * Highlight key differences in language and terms
     * Assess which version is more favorable and to which party
     * Provide specific recommendations for improvement

2. **Gap Analysis**
   - Identify clauses present in one {"contract" if comparison_type == "contract_to_contract" else "source"} but missing in the other
   - Assess the significance of missing clauses
   - Recommend additional clauses to consider

3. **Term Favorability Analysis**
   - Evaluate whether terms are favorable to one party or balanced
   - Highlight potentially one-sided clauses
   - Suggest more balanced alternatives

4. **Key Differences Analysis**
   - Identify the most significant differences between {"the contracts" if comparison_type == "contract_to_contract" else "the contract and industry standards"}
   - Assess the potential impact of these differences
   - Provide recommendations for addressing critical differences

5. **Overall Assessment**
   - Provide an overall assessment of the primary contract
   - Identify major strengths and weaknesses
   - Offer general recommendations for improvement

# FOCUS AREAS
{
    "Analyze all aspects of the contract with balanced attention." if "all" in focus_areas_list
    else f"Pay special attention to the following areas: {', '.join(focus_areas_list)}."
}

# LANGUAGE INSTRUCTIONS
{language_instruction}

# OUTPUT FORMAT
{output_format_instructions[output_format.lower()]}

# APPROACH
- Be thorough in identifying all relevant differences and similarities
- Be precise in describing differences and their implications
- Focus on substantive differences rather than minor wording variations
- Provide specific, actionable recommendations for improvement
- Consider both legal and business implications in your analysis
- When comparing to industry standards, reference typical practices in the industry
- Distinguish between critical issues and minor concerns

Please provide a complete comparison according to these specifications.
"""
        return prompt

    async def async_execute(
        self,
        primary_contract_text: str,
        comparison_contract_text: str = None,
        industry_standard: str = None,
        comparison_focus: str = "all",
        output_format: str = "structured",
        language: str = "auto",
        temperature: str = "0.2",
    ) -> Union[Dict[str, Any], str]:
        """Execute the tool to compare contracts asynchronously.

        Args:
            primary_contract_text: The full text of the primary contract to analyze.
            comparison_contract_text: The full text of the secondary contract to compare against.
            industry_standard: The type of industry standard to compare against.
            comparison_focus: Specific areas to focus on in the comparison.
            output_format: Format for the comparison output.
            language: Language of the contracts.
            temperature: Sampling temperature for the model.

        Returns:
            Union[Dict[str, Any], str]: The comparison results, either as a dictionary (for structured output)
            or as a string (for summary or detailed output).

        Raises:
            ValueError: If temperature is not a valid float between 0 and 1.
            Exception: If there's an error during response generation.
        """
        try:
            temp = float(temperature)
            if not (0.0 <= temp <= 1.0):
                raise ValueError("Temperature must be between 0 and 1.")
        except ValueError as ve:
            logger.error(f"Invalid temperature value: {temperature}")
            raise ValueError(f"Invalid temperature value: {temperature}") from ve

        # Build the prompt for contract comparison
        prompt = self._build_prompt(
            primary_contract_text=primary_contract_text,
            comparison_contract_text=comparison_contract_text,
            industry_standard=industry_standard,
            comparison_focus=comparison_focus,
            output_format=output_format,
            language=language,
        )

        # Prepare the messages history
        messages_history = [
            Message(role="system", content=self.system_prompt),
        ]

        is_streaming = self.on_token is not None

        # Set the model's temperature
        if self.generative_model:
            self.generative_model.temperature = temp

            # Generate the response asynchronously using the generative model
            try:
                result = await self.generative_model.async_generate_with_history(
                    messages_history=messages_history, prompt=prompt, streaming=is_streaming
                )

                if is_streaming:
                    response = ""
                    async for chunk in result:
                        response += chunk
                        # Note: on_token is handled via the event emitter set in model_post_init
                else:
                    response = result.response

                logger.debug(f"Generated contract comparison (first 100 chars): {response[:100]}...")
                
                # Process the response based on output format
                if output_format.lower() == "structured":
                    try:
                        # Try to parse the response as JSON
                        import json
                        import re
                        
                        # Extract JSON from the response if it's wrapped in markdown code blocks
                        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = response
                            
                        # Parse the JSON
                        comparison_dict = json.loads(json_str)
                        
                        # Create a ContractComparison object
                        comparison = ContractComparison(**comparison_dict)
                        
                        # Return as dictionary
                        return comparison.model_dump()
                    except Exception as e:
                        logger.error(f"Error parsing structured output: {e}")
                        # Fall back to returning the raw response
                        return response
                else:
                    # For summary or detailed output, return the raw response
                    return response
            except Exception as e:
                logger.error(f"Error generating contract comparison: {e}")
                raise Exception(f"Error generating contract comparison: {e}") from e
        else:
            raise ValueError("Generative model not initialized")

    def execute(
        self,
        primary_contract_text: str,
        comparison_contract_text: str = None,
        industry_standard: str = None,
        comparison_focus: str = "all",
        output_format: str = "structured",
        language: str = "auto",
        temperature: str = "0.2",
    ) -> Union[Dict[str, Any], str]:
        """Execute the tool to compare contracts synchronously.

        This method provides a synchronous wrapper around the asynchronous implementation.

        Args:
            primary_contract_text: The full text of the primary contract to analyze.
            comparison_contract_text: The full text of the secondary contract to compare against.
            industry_standard: The type of industry standard to compare against.
            comparison_focus: Specific areas to focus on in the comparison.
            output_format: Format for the comparison output.
            language: Language of the contracts.
            temperature: Sampling temperature for the model.

        Returns:
            Union[Dict[str, Any], str]: The comparison results, either as a dictionary (for structured output)
            or as a string (for summary or detailed output).
        """
        return asyncio.run(
            self.async_execute(
                primary_contract_text=primary_contract_text,
                comparison_contract_text=comparison_contract_text,
                industry_standard=industry_standard,
                comparison_focus=comparison_focus,
                output_format=output_format,
                language=language,
                temperature=temperature,
            )
        )


if __name__ == "__main__":
    # Example usage of ContractComparisonTool
    tool = ContractComparisonTool(model_name="openrouter/openai/gpt-4o-mini")
    
    # Example contracts to compare
    primary_contract = """
    EMPLOYMENT AGREEMENT
    
    This Employment Agreement (the "Agreement") is made and entered into as of January 15, 2023 (the "Effective Date"), by and between ABC Corporation, a Delaware corporation with its principal place of business at 123 Business Ave., New York, NY 10001 ("Employer"), and John Smith, an individual residing at 456 Residence St., New York, NY 10002 ("Employee").
    
    1. EMPLOYMENT
    Employer hereby employs Employee, and Employee hereby accepts employment with Employer, upon the terms and conditions set forth in this Agreement.
    
    2. POSITION AND DUTIES
    Employee shall serve as Senior Software Developer and shall perform such duties as are customarily associated with such position and as may be assigned to Employee from time to time by Employer. Employee shall report directly to the Chief Technology Officer.
    
    3. TERM
    The term of this Agreement shall commence on the Effective Date and shall continue for a period of two (2) years, unless earlier terminated as provided herein (the "Initial Term"). This Agreement shall automatically renew for successive one-year periods (each, a "Renewal Term") unless either party provides written notice of non-renewal at least sixty (60) days prior to the end of the Initial Term or any Renewal Term.
    
    4. COMPENSATION
    4.1 Base Salary. Employer shall pay Employee a base salary of $120,000 per year, payable in accordance with Employer's standard payroll practices.
    4.2 Annual Bonus. Employee shall be eligible for an annual bonus of up to 20% of Employee's base salary, based on criteria to be determined by Employer in its sole discretion.
    4.3 Stock Options. Employee shall be granted options to purchase 10,000 shares of Employer's common stock, subject to the terms and conditions of Employer's stock option plan.
    
    5. BENEFITS
    Employee shall be entitled to participate in all benefit programs provided by Employer to its employees, subject to the terms and conditions of such programs.
    
    6. TERMINATION
    6.1 Termination for Cause. Employer may terminate Employee's employment for "Cause" immediately upon written notice to Employee. For purposes of this Agreement, "Cause" shall mean: (i) Employee's material breach of this Agreement; (ii) Employee's conviction of a felony; (iii) Employee's gross negligence or willful misconduct in the performance of Employee's duties; or (iv) Employee's failure to follow the lawful directives of Employer.
    6.2 Termination without Cause. Employer may terminate Employee's employment without Cause upon thirty (30) days' written notice to Employee.
    6.3 Resignation. Employee may resign upon thirty (30) days' written notice to Employer.
    
    7. CONFIDENTIALITY
    Employee agrees to maintain the confidentiality of all proprietary and confidential information of Employer during and after Employee's employment with Employer.
    
    8. NON-COMPETE
    During Employee's employment and for a period of one (1) year thereafter, Employee shall not, directly or indirectly, engage in any business that competes with Employer.
    
    9. GOVERNING LAW
    This Agreement shall be governed by and construed in accordance with the laws of the State of New York, without regard to conflicts of law principles.
    
    10. ENTIRE AGREEMENT
    This Agreement constitutes the entire agreement between the parties with respect to the subject matter hereof and supersedes all prior agreements and understandings, whether written or oral.
    
    IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.
    
    EMPLOYER:                          EMPLOYEE:
    ABC Corporation
    
    By: ___________________           ___________________
    Name: Jane Doe                    John Smith
    Title: CEO
    """
    
    comparison_contract = """
    EMPLOYMENT AGREEMENT
    
    This Employment Agreement (the "Agreement") is made and entered into as of February 1, 2023, by and between XYZ Inc., a California corporation ("Employer"), and Jane Doe, an individual ("Employee").
    
    1. EMPLOYMENT
    Employer agrees to employ Employee, and Employee agrees to accept employment with Employer, subject to the terms and conditions of this Agreement.
    
    2. POSITION AND DUTIES
    Employee shall serve as Senior Software Engineer and shall report to the VP of Engineering. Employee shall perform such duties as are customarily associated with such position and as may be reasonably assigned by Employer.
    
    3. TERM
    The term of this Agreement shall begin on February 15, 2023, and shall continue indefinitely until terminated in accordance with Section 6.
    
    4. COMPENSATION
    4.1 Base Salary. Employer shall pay Employee a base salary of $135,000 per year, payable semi-monthly.
    4.2 Performance Bonus. Employee shall be eligible for a quarterly performance bonus of up to 10% of Employee's base salary, based on achievement of mutually agreed upon objectives.
    4.3 Equity. Employee shall be granted 15,000 restricted stock units, vesting over four years with a one-year cliff.
    
    5. BENEFITS
    5.1 Standard Benefits. Employee shall be entitled to participate in all benefit programs offered to employees of Employer.
    5.2 Vacation. Employee shall be entitled to four (4) weeks of paid vacation per year.
    5.3 Remote Work. Employee may work remotely up to two (2) days per week.
    
    6. TERMINATION
    6.1 Termination for Cause. Employer may terminate Employee's employment for cause immediately.
    6.2 Termination without Cause. Employer may terminate Employee's employment without cause with sixty (60) days' written notice. In such event, Employee shall be entitled to severance pay equal to three (3) months of Employee's base salary.
    6.3 Resignation. Employee may resign with thirty (30) days' written notice.
    
    7. CONFIDENTIALITY AND INTELLECTUAL PROPERTY
    7.1 Confidentiality. Employee shall maintain the confidentiality of all proprietary information of Employer.
    7.2 Intellectual Property. All work product created by Employee during employment shall be the sole and exclusive property of Employer.
    
    8. NON-SOLICITATION
    For twelve (12) months following termination of employment, Employee shall not solicit any employees or customers of Employer.
    
    9. DISPUTE RESOLUTION
    Any disputes arising under this Agreement shall be resolved through binding arbitration in accordance with the rules of the American Arbitration Association.
    
    10. GOVERNING LAW
    This Agreement shall be governed by the laws of the State of California.
    
    11. ENTIRE AGREEMENT
    This Agreement contains the entire understanding between the parties and supersedes all prior agreements.
    
    EMPLOYER:                          EMPLOYEE:
    XYZ Inc.
    
    By: ___________________           ___________________
    Name: John Johnson                Jane Doe
    Title: CEO
    """
    
    # Generate comparison with structured output
    structured_comparison = tool.execute(
        primary_contract_text=primary_contract,
        comparison_contract_text=comparison_contract,
        comparison_focus="termination,compensation",
        output_format="structured",
        language="English",
    )
    
    print("Structured Comparison Results:")
    import json
    print(json.dumps(structured_comparison, indent=2))
    
    # Generate comparison with industry standard
    standard_comparison = tool.execute(
        primary_contract_text=primary_contract,
        industry_standard="employment",
        output_format="summary",
        comparison_focus="all",
    )
    
    print("\nIndustry Standard Comparison:")
    print(standard_comparison)
    
    # Display tool configuration in Markdown
    print("\nTool Documentation:")
    print(tool.to_markdown())
