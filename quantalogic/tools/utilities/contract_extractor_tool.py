"""Contract Extractor Tool for analyzing contracts and extracting key information."""

import asyncio
from typing import Callable, Dict, Any, List, Optional, Union

from loguru import logger
from pydantic import ConfigDict, Field, BaseModel

from quantalogic.console_print_token import console_print_token
from quantalogic.event_emitter import EventEmitter
from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.tools.tool import Tool, ToolArgument


class ContractSection(BaseModel):
    """Model for a contract section with extracted information."""
    
    title: str = Field(description="Title or name of the section")
    content: str = Field(description="Full text content of the section")
    summary: str = Field(description="Brief summary of the section's purpose and key points")
    risk_level: str = Field(description="Risk assessment: 'Low', 'Medium', 'High', or 'N/A'")
    risk_explanation: Optional[str] = Field(default=None, description="Explanation of identified risks")
    key_terms: List[str] = Field(default_factory=list, description="Important terms or concepts in this section")
    obligations: List[str] = Field(default_factory=list, description="Obligations created by this section")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")


class ContractParty(BaseModel):
    """Model for a party to the contract."""
    
    name: str = Field(description="Name of the party")
    role: str = Field(description="Role in the contract (e.g., 'Buyer', 'Seller', 'Lessor')")
    contact_info: Optional[str] = Field(default=None, description="Contact information if available")
    obligations: List[str] = Field(default_factory=list, description="Key obligations of this party")
    rights: List[str] = Field(default_factory=list, description="Key rights of this party")


class ContractExtraction(BaseModel):
    """Model for structured contract extraction results."""
    
    # Basic contract information
    contract_type: str = Field(description="Type of contract (e.g., 'Employment', 'Lease', 'Sale')")
    title: str = Field(description="Title or name of the contract")
    execution_date: Optional[str] = Field(default=None, description="Date the contract was executed")
    effective_date: Optional[str] = Field(default=None, description="Date the contract becomes effective")
    termination_date: Optional[str] = Field(default=None, description="Date the contract terminates")
    
    # Parties involved
    parties: List[ContractParty] = Field(default_factory=list, description="Parties to the contract")
    
    # Key financial terms
    payment_terms: Optional[str] = Field(default=None, description="Summary of payment terms")
    payment_amount: Optional[str] = Field(default=None, description="Payment amount(s)")
    currency: Optional[str] = Field(default=None, description="Currency used")
    
    # Contract sections
    sections: List[ContractSection] = Field(default_factory=list, description="Extracted contract sections")
    
    # Key dates and deadlines
    key_dates: Dict[str, str] = Field(default_factory=dict, description="Important dates and deadlines")
    
    # Overall analysis
    overall_risk_assessment: str = Field(description="Overall risk level: 'Low', 'Medium', or 'High'")
    risk_factors: List[str] = Field(default_factory=list, description="Key risk factors identified")
    opportunities: List[str] = Field(default_factory=list, description="Opportunities for improvement")
    missing_elements: List[str] = Field(default_factory=list, description="Important elements missing from the contract")
    
    # Executive summary
    executive_summary: str = Field(description="Brief executive summary of the contract analysis")


class ContractExtractorTool(Tool):
    """Tool to analyze contracts and extract key information with risk assessment."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="contract_extractor_tool")
    description: str = Field(
        default=(
            "Analyzes contracts to extract key information, identify risks, and provide recommendations. "
            "The tool extracts parties, obligations, payment terms, deadlines, and other critical elements. "
            "It also performs risk assessment on clauses and provides an executive summary of the contract."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="contract_text",
                arg_type="string",
                description=(
                    "The full text of the contract to analyze. Include the complete contract "
                    "for the most accurate analysis."
                ),
                required=True,
                example="[Full text of contract...]",
            ),
            ToolArgument(
                name="output_format",
                arg_type="string",
                description=(
                    "The format for the extraction output. Options: 'structured' (returns a structured "
                    "object with all extracted information), 'summary' (returns a concise text summary), "
                    "or 'detailed' (returns a comprehensive text analysis). Default is 'structured'."
                ),
                required=False,
                default="structured",
                example="summary",
            ),
            ToolArgument(
                name="contract_type_hint",
                arg_type="string",
                description=(
                    "Optional hint about the type of contract to help with extraction. "
                    "Example: 'employment', 'lease', 'sale', 'service', etc."
                ),
                required=False,
                example="employment",
            ),
            ToolArgument(
                name="focus_areas",
                arg_type="string",
                description=(
                    "Comma-separated list of specific areas to focus on in the analysis. "
                    "Options include: 'payment_terms', 'termination', 'liability', 'confidentiality', "
                    "'intellectual_property', 'all'. Default is 'all'."
                ),
                required=False,
                default="all",
                example="termination,liability",
            ),
            ToolArgument(
                name="risk_assessment",
                arg_type="string",
                description=(
                    "Whether to include risk assessment in the analysis. Options: 'true' or 'false'. "
                    "Default is 'true'."
                ),
                required=False,
                default="true",
                example="true",
            ),
            ToolArgument(
                name="language",
                arg_type="string",
                description=(
                    "The language of the contract. This helps the analyzer understand language-specific "
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
                    'Sampling temperature between "0.0" and "1.0": "0.0" for more factual extraction, '
                    '"1.0" for more interpretive analysis. For contract analysis, '
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
        name: str = "contract_extractor_tool",
        generative_model: GenerativeModel | None = None,
        event_emitter: EventEmitter | None = None,
    ):
        """Initialize the ContractExtractorTool with model configuration and optional callback.

        Args:
            model_name (str): The name of the language model to use.
            system_prompt (str, optional): Default system prompt for the model.
            on_token (Callable, optional): Callback function for streaming tokens.
            name (str): Name of the tool instance. Defaults to "contract_extractor_tool".
            generative_model (GenerativeModel, optional): Pre-initialized generative model.
            event_emitter (EventEmitter, optional): Event emitter for streaming tokens.
        """
        # Default system prompt for contract extraction if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert contract analyst with extensive experience in reviewing and extracting "
                "You always answer in french unless it's specified by the user"
                "information from various types of contracts. Your expertise includes identifying parties, "
                "obligations, payment terms, key dates, and risk factors. You excel at summarizing complex "
                "legal language into clear, actionable insights. You can recognize problematic clauses and "
                "provide recommendations for improvement. Your analysis is thorough, accurate, and focused "
                "on practical business implications rather than theoretical legal discussions. You understand "
                "contract terminology across different industries and can adapt your analysis to specific "
                "contract types."
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
            logger.debug(f"Initialized ContractExtractorTool with model: {self.model_name}")

        # Only set up event listener if on_token is provided
        if self.on_token is not None:
            logger.debug(f"Setting up event listener for ContractExtractorTool with model: {self.model_name}")
            self.generative_model.event_emitter.on("stream_chunk", self.on_token)

    def _build_prompt(
        self,
        contract_text: str,
        output_format: str = "structured",
        contract_type_hint: Optional[str] = None,
        focus_areas: str = "all",
        risk_assessment: str = "true",
        language: str = "auto",
    ) -> str:
        """Build a comprehensive prompt for contract extraction and analysis.

        Args:
            contract_text: The full text of the contract to analyze.
            output_format: Format for the extraction output.
            contract_type_hint: Optional hint about the type of contract.
            focus_areas: Specific areas to focus on in the analysis.
            risk_assessment: Whether to include risk assessment.
            language: Language of the contract.

        Returns:
            A structured prompt for the LLM.
        """
        # Validate output format
        valid_output_formats = ["structured", "summary", "detailed"]
        if output_format.lower() not in valid_output_formats:
            output_format = "structured"
        
        # Process focus areas
        focus_areas_list = [area.strip().lower() for area in focus_areas.split(",")]
        if "all" in focus_areas_list:
            focus_areas_list = ["all"]
        
        # Process risk assessment flag
        include_risk = risk_assessment.lower() == "true"
        
        # Build hints section
        hints = []
        if contract_type_hint:
            hints.append(f"Contract Type Hint: {contract_type_hint}")
        hints_section = "\n".join(hints) if hints else "No specific hints provided."

        # Language instructions
        language_instruction = (
            f"The contract is in {language}. Analyze accordingly." 
            if language.lower() != "auto" 
            else "Auto-detect the language of the contract and analyze accordingly."
        )

        # Define output format instructions
        output_format_instructions = {
            "structured": (
                "Return your extraction as a structured JSON object following this schema:\n"
                "```json\n"
                "{\n"
                '  "contract_type": "string",\n'
                '  "title": "string",\n'
                '  "execution_date": "string",\n'
                '  "effective_date": "string",\n'
                '  "termination_date": "string",\n'
                '  "parties": [\n'
                '    {\n'
                '      "name": "string",\n'
                '      "role": "string",\n'
                '      "contact_info": "string",\n'
                '      "obligations": ["string", ...],\n'
                '      "rights": ["string", ...]\n'
                '    },\n'
                '    ...\n'
                '  ],\n'
                '  "payment_terms": "string",\n'
                '  "payment_amount": "string",\n'
                '  "currency": "string",\n'
                '  "sections": [\n'
                '    {\n'
                '      "title": "string",\n'
                '      "content": "string",\n'
                '      "summary": "string",\n'
                '      "risk_level": "string",\n'
                '      "risk_explanation": "string",\n'
                '      "key_terms": ["string", ...],\n'
                '      "obligations": ["string", ...],\n'
                '      "recommendations": ["string", ...]\n'
                '    },\n'
                '    ...\n'
                '  ],\n'
                '  "key_dates": {\n'
                '    "date_description": "date_value",\n'
                '    ...\n'
                '  },\n'
                '  "overall_risk_assessment": "string",\n'
                '  "risk_factors": ["string", ...],\n'
                '  "opportunities": ["string", ...],\n'
                '  "missing_elements": ["string", ...],\n'
                '  "executive_summary": "string"\n'
                "}\n"
                "```\n"
                "Ensure all fields are populated with appropriate values. For lists, include at least one item "
                "if relevant information is available. For fields where no relevant information can be determined, "
                "use empty lists [] for list fields, empty strings \"\" for string fields, and null for optional fields. "
                "Ensure the JSON is valid and properly formatted."
            ),
            "summary": (
                "Return a concise text summary of the contract analysis, focusing on the "
                "most important aspects such as contract type, parties, key terms, important dates, "
                "and major risk factors. The summary should be clear, informative, and highlight "
                "the most critical elements of the contract. Format the summary with clear headings "
                "and bullet points for readability."
            ),
            "detailed": (
                "Return a comprehensive text analysis organized into sections with headings. Include sections for: "
                "1) Contract Overview, 2) Parties and Roles, 3) Key Terms and Conditions, 4) Payment Terms, "
                "5) Important Dates and Deadlines, 6) Obligations and Rights, 7) Risk Assessment, "
                "8) Recommendations for Improvement, and 9) Executive Summary. Provide detailed explanations "
                "for each section and include specific references to relevant contract clauses."
            )
        }

        prompt = f"""
Analyze the following contract to extract key information, identify risks, and provide recommendations.

# CONTRACT TO ANALYZE
```
{contract_text}
```

# ANALYSIS PARAMETERS
- Output Format: {output_format}
- Contract Type Hint: {contract_type_hint if contract_type_hint else "None provided"}
- Focus Areas: {", ".join(focus_areas_list)}
- Include Risk Assessment: {"Yes" if include_risk else "No"}
- Language: {language}

# HINTS
{hints_section}

# EXTRACTION REQUIREMENTS
Extract and analyze the following information from the contract:

1. **Basic Contract Information**
   - Contract type (e.g., employment, lease, sale, service)
   - Title or name of the contract
   - Execution date (when signed)
   - Effective date (when the contract begins)
   - Termination date or contract duration

2. **Parties Involved**
   - Identify all parties to the contract
   - For each party, determine:
     * Full name (individual or entity)
     * Role in the contract (e.g., employer, employee, buyer, seller)
     * Contact information if provided
     * Key obligations of this party
     * Key rights of this party

3. **Key Financial Terms**
   - Payment terms and conditions
   - Payment amounts
   - Currency
   - Payment schedule
   - Late payment penalties
   - Price adjustment mechanisms

4. **Contract Sections**
   - Identify major sections of the contract
   - For each significant section:
     * Title or name
     * Brief summary of purpose and key points
     * Risk assessment (if applicable)
     * Key terms or concepts
     * Obligations created
     * Recommendations for improvement

5. **Key Dates and Deadlines**
   - Identify all important dates mentioned in the contract
   - Include deadlines for deliverables, notifications, renewals, etc.

6. **Risk Assessment** {"(if requested)" if not include_risk else ""}
   - Evaluate overall risk level (Low, Medium, High)
   - Identify specific risk factors
   - Highlight problematic or ambiguous clauses
   - Note any missing elements that should be included
   - Provide recommendations for mitigating identified risks

7. **Executive Summary**
   - Provide a brief (3-5 sentences) executive summary of the contract
   - Highlight the most important aspects and any significant risks

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
- Be thorough in extracting all relevant information
- Be precise in identifying parties, obligations, and rights
- When exact information is not provided, indicate this clearly
- Focus on practical business implications rather than theoretical legal discussions
- Provide actionable recommendations for addressing risks or improving the contract
- Distinguish between clear contractual terms and ambiguous or problematic clauses
- Consider industry standards and best practices in your analysis

Please provide a complete analysis of the contract according to these specifications.
"""
        return prompt

    async def async_execute(
        self,
        contract_text: str,
        output_format: str = "structured",
        contract_type_hint: str = None,
        focus_areas: str = "all",
        risk_assessment: str = "true",
        language: str = "auto",
        temperature: str = "0.2",
    ) -> Union[Dict[str, Any], str]:
        """Execute the tool to analyze a contract asynchronously.

        Args:
            contract_text: The full text of the contract to analyze.
            output_format: Format for the extraction output.
            contract_type_hint: Optional hint about the type of contract.
            focus_areas: Specific areas to focus on in the analysis.
            risk_assessment: Whether to include risk assessment.
            language: Language of the contract.
            temperature: Sampling temperature for the model.

        Returns:
            Union[Dict[str, Any], str]: The extraction results, either as a dictionary (for structured output)
            or as a string (for summary or detailed output).

        Raises:
            ValueError: If temperature is not a valid float between 0 and 1.
            Exception: If there's an error during response generation.
        """
        try:
            # Handle empty string or None by using default temperature
            if temperature is None or temperature.strip() == "":
                temp = 0.2  # Default temperature
                logger.info(f"Using default temperature: {temp}")
            else:
                temp = float(temperature)
                if not (0.0 <= temp <= 1.0):
                    logger.warning(f"Temperature {temp} out of range, using default 0.2")
                    temp = 0.2
        except ValueError as ve:
            logger.warning(f"Invalid temperature value: {temperature}, using default 0.2")
            temp = 0.2  # Default to 0.2 on error instead of raising exception

        # Build the prompt for contract extraction
        prompt = self._build_prompt(
            contract_text=contract_text,
            output_format=output_format,
            contract_type_hint=contract_type_hint,
            focus_areas=focus_areas,
            risk_assessment=risk_assessment,
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

                logger.debug(f"Generated contract analysis (first 100 chars): {response[:100]}...")
                
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
                        extraction_dict = json.loads(json_str)
                        
                        # Create a ContractExtraction object
                        extraction = ContractExtraction(**extraction_dict)
                        
                        # Return as dictionary
                        return extraction.model_dump()
                    except Exception as e:
                        logger.error(f"Error parsing structured output: {e}")
                        # Fall back to returning the raw response
                        return response
                else:
                    # For summary or detailed output, return the raw response
                    return response
            except Exception as e:
                logger.error(f"Error generating contract analysis: {e}")
                raise Exception(f"Error generating contract analysis: {e}") from e
        else:
            raise ValueError("Generative model not initialized")

    def execute(
        self,
        contract_text: str,
        output_format: str = "structured",
        contract_type_hint: str = None,
        focus_areas: str = "all",
        risk_assessment: str = "true",
        language: str = "auto",
        temperature: str = "0.2",
    ) -> Union[Dict[str, Any], str]:
        """Execute the tool to analyze a contract synchronously.

        This method provides a synchronous wrapper around the asynchronous implementation.

        Args:
            contract_text: The full text of the contract to analyze.
            output_format: Format for the extraction output.
            contract_type_hint: Optional hint about the type of contract.
            focus_areas: Specific areas to focus on in the analysis.
            risk_assessment: Whether to include risk assessment.
            language: Language of the contract.
            temperature: Sampling temperature for the model.

        Returns:
            Union[Dict[str, Any], str]: The extraction results, either as a dictionary (for structured output)
            or as a string (for summary or detailed output).
        """
        return asyncio.run(
            self.async_execute(
                contract_text=contract_text,
                output_format=output_format,
                contract_type_hint=contract_type_hint,
                focus_areas=focus_areas,
                risk_assessment=risk_assessment,
                language=language,
                temperature=temperature,
            )
        )


if __name__ == "__main__":
    # Example usage of ContractExtractorTool
    tool = ContractExtractorTool(model_name="openai/gpt-4o-mini")
    
    # Example contract to analyze
    sample_contract = """
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
    
    # Generate extraction with structured output
    structured_extraction = tool.execute(
        contract_text=sample_contract,
        output_format="structured",
        contract_type_hint="employment",
        focus_areas="termination,compensation",
        language="English",
    )
    
    print("Structured Extraction Results:")
    import json
    print(json.dumps(structured_extraction, indent=2))
    
    # Generate extraction with summary output
    summary_extraction = tool.execute(
        contract_text=sample_contract,
        output_format="summary",
        focus_areas="all",
    )
    
    print("\nSummary Extraction:")
    print(summary_extraction)
    
    # Display tool configuration in Markdown
    print("\nTool Documentation:")
    print(tool.to_markdown())
