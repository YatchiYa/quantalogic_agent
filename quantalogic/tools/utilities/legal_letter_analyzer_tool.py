"""Legal Letter Analyzer Tool for extracting key information from legal documents."""

import asyncio
from typing import Callable, Dict, Any, Optional, List

from loguru import logger
from pydantic import ConfigDict, Field

from quantalogic.console_print_token import console_print_token
from quantalogic.event_emitter import EventEmitter
from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.tools.tool import Tool, ToolArgument


class LegalLetterAnalyzerTool(Tool):
    """Tool to analyze legal letters and extract key information."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="legal_letter_analyzer_tool")
    description: str = Field(
        default=(
            "Analyzes legal letters (from defenders, prosecutors, or other legal professionals) "
            "and extracts key information such as parties involved, legal references, case details, "
            "arguments, and requested outcomes. Provides a comprehensive overview of the document "
            "in a structured format."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="letter_text",
                arg_type="string",
                description=(
                    "The full text of the legal letter to analyze. Include the complete letter "
                    "for the most accurate analysis."
                ),
                required=True,
                example="[Full text of legal letter...]",
            ),
            ToolArgument(
                name="output_format",
                arg_type="string",
                description=(
                    "The format for the analysis output. Options: 'markdown' (default), 'json', or 'text'. "
                    "Markdown provides a well-structured, readable format. JSON is useful for programmatic "
                    "processing. Text is a simpler format with less structure."
                ),
                required=False,
                default="markdown",
                example="markdown",
            ),
            ToolArgument(
                name="analysis_depth",
                arg_type="string",
                description=(
                    "The level of detail in the analysis. Options: 'basic', 'standard', or 'detailed'. "
                    "Basic provides a quick overview, standard provides a balanced analysis, and "
                    "detailed provides an in-depth analysis of all aspects of the letter."
                ),
                required=False,
                default="standard",
                example="standard",
            ),
            ToolArgument(
                name="language",
                arg_type="string",
                description=(
                    "The language of the letter. This helps the analyzer understand language-specific "
                    "legal terminology and formatting. Defaults to auto-detection."
                ),
                required=False,
                default="auto",
                example="English",
            ),
            ToolArgument(
                name="focus_areas",
                arg_type="string",
                description=(
                    "Comma-separated list of specific areas to focus on in the analysis. "
                    "Options include: 'parties', 'legal_references', 'arguments', 'evidence', "
                    "'outcomes', 'timeline', 'all'. Default is 'all'."
                ),
                required=False,
                default="all",
                example="legal_references,arguments,outcomes",
            ),
            ToolArgument(
                name="temperature",
                arg_type="string",
                description=(
                    'Sampling temperature between "0.0" and "1.0": "0.0" for more factual extraction, '
                    '"1.0" for more interpretive analysis. For legal document analysis, '
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
        name: str = "legal_letter_analyzer_tool",
        generative_model: GenerativeModel | None = None,
        event_emitter: EventEmitter | None = None,
    ):
        """Initialize the LegalLetterAnalyzerTool with model configuration and optional callback.

        Args:
            model_name (str): The name of the language model to use.
            system_prompt (str, optional): Default system prompt for the model.
            on_token (Callable, optional): Callback function for streaming tokens.
            name (str): Name of the tool instance. Defaults to "legal_letter_analyzer_tool".
            generative_model (GenerativeModel, optional): Pre-initialized generative model.
            event_emitter (EventEmitter, optional): Event emitter for streaming tokens.
        """
        # Default system prompt for legal letter analysis if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert legal document analyst with extensive experience in reviewing and "
                "extracting information from legal letters, court submissions, and other legal documents. "
                "Your expertise includes identifying parties, legal references, case details, arguments, "
                "and requested outcomes. You are skilled at distinguishing between facts, claims, and legal "
                "interpretations. You can recognize legal terminology across different jurisdictions and "
                "practice areas. Your analysis is thorough, accurate, and objective, focusing on extracting "
                "and organizing information rather than making judgments about the merits of arguments."
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
            logger.debug(f"Initialized LegalLetterAnalyzerTool with model: {self.model_name}")

        # Only set up event listener if on_token is provided
        if self.on_token is not None:
            logger.debug(f"Setting up event listener for LegalLetterAnalyzerTool with model: {self.model_name}")
            self.generative_model.event_emitter.on("stream_chunk", self.on_token)

    def _build_prompt(
        self,
        letter_text: str,
        output_format: str = "markdown",
        analysis_depth: str = "standard",
        language: str = "auto",
        focus_areas: str = "all",
    ) -> str:
        """Build a comprehensive prompt for the legal letter analysis.

        Args:
            letter_text: The full text of the legal letter to analyze.
            output_format: Format for the analysis output (markdown, json, text).
            analysis_depth: Level of detail in the analysis (basic, standard, detailed).
            language: Language of the letter.
            focus_areas: Specific areas to focus on in the analysis.

        Returns:
            A structured prompt for the LLM.
        """
        # Validate and process focus areas
        focus_areas_list = [area.strip().lower() for area in focus_areas.split(",")]
        if "all" in focus_areas_list:
            focus_areas_list = ["all"]
        
        # Define depth levels
        depth_descriptions = {
            "basic": "Provide a brief overview focusing only on the most essential information",
            "standard": "Provide a balanced analysis with moderate detail on all key aspects",
            "detailed": "Provide an in-depth analysis with extensive detail on all aspects"
        }
        
        # Get depth description or use standard if invalid
        depth_desc = depth_descriptions.get(analysis_depth.lower(), depth_descriptions["standard"])
        
        # Define output format instructions
        format_instructions = {
            "markdown": (
                "Format the analysis as Markdown with clear headings, lists, and emphasis. "
                "Use level 1 heading for the title, level 2 headings for main sections, "
                "and level 3 headings for subsections. Use bold for key terms and bullet "
                "points for lists."
            ),
            "json": (
                "Format the analysis as a JSON object with the following structure:\n"
                "{\n"
                "  \"executive_summary\": \"string\",\n"
                "  \"document_information\": { \"type\": \"string\", \"date\": \"string\", ... },\n"
                "  \"parties_involved\": [ { \"name\": \"string\", \"role\": \"string\", ... }, ... ],\n"
                "  \"case_details\": { \"type\": \"string\", \"number\": \"string\", ... },\n"
                "  \"legal_references\": [ { \"type\": \"string\", \"citation\": \"string\", ... }, ... ],\n"
                "  \"key_arguments\": [ { \"argument\": \"string\", \"explanation\": \"string\" }, ... ],\n"
                "  \"evidence_referenced\": [ \"string\", ... ],\n"
                "  \"requested_outcomes\": [ \"string\", ... ],\n"
                "  \"timeline\": [ { \"date\": \"string\", \"event\": \"string\" }, ... ],\n"
                "  \"additional_notes\": \"string\"\n"
                "}\n"
                "Ensure all JSON is valid and properly escaped."
            ),
            "text": (
                "Format the analysis as plain text with clear section headers in ALL CAPS, "
                "followed by the section content. Use dashes or asterisks for list items. "
                "Separate sections with blank lines."
            )
        }
        
        # Get format instructions or use markdown if invalid
        format_desc = format_instructions.get(output_format.lower(), format_instructions["markdown"])
        
        # Language instructions
        language_instruction = (
            f"The document is in {language}. Analyze accordingly." 
            if language.lower() != "auto" 
            else "Auto-detect the language of the document and analyze accordingly."
        )

        prompt = f"""
Analyze the following legal letter and extract key information to provide a comprehensive overview.

# LEGAL LETTER TO ANALYZE
```
{letter_text}
```

# ANALYSIS PARAMETERS
- Output Format: {output_format}
- Analysis Depth: {analysis_depth} ({depth_desc})
- Language: {language}
- Focus Areas: {", ".join(focus_areas_list)}

# ANALYSIS REQUIREMENTS
Extract and organize information into the following categories:

1. **Executive Summary**
   - Provide a concise 2-3 sentence overview of the letter's purpose and key points
   - Identify the type of legal document and its primary objective

2. **Document Information**
   - Document type (e.g., legal submission, response, motion)
   - Date of the letter
   - Language and formality assessment
   - Overall structure and organization

3. **Parties Involved**
   - Identify all parties mentioned in the letter
   - For each party, extract:
     * Full name
     * Role (e.g., plaintiff, defendant, attorney, judge)
     * Contact information if provided
     * Relationships between parties

4. **Case Details**
   - Case type (e.g., property dispute, contract breach)
   - Case number or reference (if mentioned)
   - Jurisdiction and court
   - Stage of proceedings
   - Relevant dates (filing date, hearing date, etc.)

5. **Legal References**
   - Statutes, codes, and articles cited (with specific numbers)
   - Case law and precedents referenced
   - Legal principles or doctrines mentioned
   - Regulations or rules cited

6. **Key Arguments**
   - Main claims and positions
   - Supporting arguments for each claim
   - Legal reasoning provided
   - Factual assertions made

7. **Evidence Referenced**
   - Documents mentioned (e.g., surveys, contracts, affidavits)
   - Witness statements or testimonies
   - Expert opinions
   - Physical evidence
   - Exhibits or attachments

8. **Requested Outcomes**
   - Specific remedies or actions requested
   - Legal basis for the requested outcomes
   - Alternative remedies mentioned

9. **Timeline of Events**
   - Chronological list of events mentioned in the letter
   - Include dates where available

10. **Additional Notes**
    - Any other relevant observations
    - Potential strengths or weaknesses in the arguments (if analysis_depth is "detailed")
    - Procedural or formatting issues

# OUTPUT FORMAT INSTRUCTIONS
{format_desc}

# LANGUAGE INSTRUCTIONS
{language_instruction}

# FOCUS AREAS
{
    "Focus on all categories with balanced attention." if "all" in focus_areas_list
    else f"Pay special attention to the following areas: {', '.join(focus_areas_list)}."
}

# ANALYSIS APPROACH
- Be objective and factual in your extraction
- Do not make judgments about the merits of arguments unless explicitly requested
- Be thorough in identifying all relevant information
- When exact information is not provided, indicate this clearly
- For legal references, be precise in citing the exact articles, sections, or cases
- Distinguish between established facts, claimed facts, and legal interpretations
- Maintain the appropriate level of detail based on the analysis_depth parameter

Please provide a complete analysis of the legal letter according to these specifications.
"""
        return prompt

    async def async_execute(
        self,
        letter_text: str,
        output_format: str = "markdown",
        analysis_depth: str = "standard",
        language: str = "auto",
        focus_areas: str = "all",
        temperature: str = "0.2",
    ) -> str:
        """Execute the tool to analyze a legal letter asynchronously.

        Args:
            letter_text: The full text of the legal letter to analyze.
            output_format: Format for the analysis output (markdown, json, text).
            analysis_depth: Level of detail in the analysis (basic, standard, detailed).
            language: Language of the letter.
            focus_areas: Specific areas to focus on in the analysis.
            temperature: Sampling temperature for the model.

        Returns:
            str: The analysis of the legal letter.

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

        # Build the prompt for the legal letter analysis
        prompt = self._build_prompt(
            letter_text=letter_text,
            output_format=output_format,
            analysis_depth=analysis_depth,
            language=language,
            focus_areas=focus_areas,
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

                logger.debug(f"Generated legal letter analysis (first 100 chars): {response[:100]}...")
                return response
            except Exception as e:
                logger.error(f"Error generating legal letter analysis: {e}")
                raise Exception(f"Error generating legal letter analysis: {e}") from e
        else:
            raise ValueError("Generative model not initialized")

    def execute(
        self,
        letter_text: str,
        output_format: str = "markdown",
        analysis_depth: str = "standard",
        language: str = "auto",
        focus_areas: str = "all",
        temperature: str = "0.2",
    ) -> str:
        """Execute the tool to analyze a legal letter synchronously.

        This method provides a synchronous wrapper around the asynchronous implementation.

        Args:
            letter_text: The full text of the legal letter to analyze.
            output_format: Format for the analysis output (markdown, json, text).
            analysis_depth: Level of detail in the analysis (basic, standard, detailed).
            language: Language of the letter.
            focus_areas: Specific areas to focus on in the analysis.
            temperature: Sampling temperature for the model.

        Returns:
            str: The analysis of the legal letter.
        """
        return asyncio.run(
            self.async_execute(
                letter_text=letter_text,
                output_format=output_format,
                analysis_depth=analysis_depth,
                language=language,
                focus_areas=focus_areas,
                temperature=temperature,
            )
        )


if __name__ == "__main__":
    # Example usage of LegalLetterAnalyzerTool
    tool = LegalLetterAnalyzerTool(model_name="openai/gpt-4o-mini")
    
    # Example legal letter to analyze
    sample_letter = """
    [Date]
    
    The Honorable Judge of the District Court of Paris
    Palais de Justice
    10 Boulevard du Palais
    75001 Paris, France
    
    Subject: Property Dispute - Boundary Line Contestation - Case No. 2023-45678
    
    Your Honor,
    
    I, Jean Dupont, residing at 123 Rue de Paris, 75001 Paris, am writing to defend my property rights in the dispute with my neighbor, Marie Dubois, regarding the boundary fence between our properties.
    
    The fence in question was erected based on a professional land survey conducted in December 2022, which clearly established the property boundaries. Contrary to my neighbor's claims, the fence does not encroach on their property but follows the exact boundary line as determined by the surveyor.
    
    The Civil Code Article 691 states that a property owner must exercise their rights without abusing the property of their neighbor. I have fully complied with this provision by ensuring that the fence was placed precisely on the boundary line, not on my neighbor's property.
    
    Furthermore, Article 703 regarding the bornage (boundary marking) has been respected as the fence was installed following a proper survey. The neighbor had ample opportunity to contest the survey results before the fence installation but failed to do so in a timely manner.
    
    I request that the court recognize the validity of my survey and confirm that the fence is correctly positioned on the boundary line. I further request that all claims for removal of the fence be dismissed as they lack legal and factual basis.
    
    Respectfully submitted,
    
    Jean Dupont
    123 Rue de Paris
    75001 Paris, France
    Tel: 01-23-45-67-89
    Email: jean.dupont@email.com
    """
    
    # Generate analysis
    analysis = tool.execute(
        letter_text=sample_letter,
        output_format="markdown",
        analysis_depth="detailed",
        language="English",
        focus_areas="legal_references,parties,arguments",
    )
    
    print("Generated Legal Letter Analysis:")
    print(analysis)
    
    # Display tool configuration in Markdown
    print("\nTool Documentation:")
    print(tool.to_markdown())
