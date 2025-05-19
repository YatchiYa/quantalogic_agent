"""Legal Classifier Tool for categorizing and classifying legal documents and queries."""

import asyncio
from typing import Callable, Dict, Any, List, Optional, Union

from loguru import logger
from pydantic import ConfigDict, Field, BaseModel

from quantalogic.console_print_token import console_print_token
from quantalogic.event_emitter import EventEmitter
from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.tools.tool import Tool, ToolArgument


class LegalClassification(BaseModel):
    """Model for structured legal classification results."""
    
    # Primary classifications
    document_type: str = Field(
        description="Type of legal document (e.g., contract, complaint, motion, brief, letter, opinion)"
    )
    legal_domain: str = Field(
        description="Primary area of law (e.g., property, contract, criminal, family, intellectual property)"
    )
    jurisdiction: str = Field(
        description="Jurisdiction or legal system (e.g., US Federal, French Civil Law, International)"
    )
    
    # Secondary classifications
    sub_domains: List[str] = Field(
        default_factory=list,
        description="Specific sub-areas of law (e.g., real estate, divorce, patent, securities)"
    )
    purpose: str = Field(
        default="",
        description="Primary purpose of the document (e.g., advocacy, informational, decisional, transactional)"
    )
    audience: str = Field(
        default="",
        description="Intended audience (e.g., court, opposing counsel, client, public)"
    )
    
    # Analysis metrics
    complexity_level: int = Field(
        default=0,
        description="Complexity level from 1 (simple) to 5 (highly complex)"
    )
    formality_level: int = Field(
        default=0,
        description="Formality level from 1 (informal) to 5 (highly formal)"
    )
    estimated_reading_time: str = Field(
        default="",
        description="Estimated time required to read and understand the document"
    )
    
    # Content analysis
    key_legal_concepts: List[str] = Field(
        default_factory=list,
        description="Key legal concepts or principles involved"
    )
    potential_precedents: List[str] = Field(
        default_factory=list,
        description="Potential relevant precedents or cases"
    )
    statutes_referenced: List[str] = Field(
        default_factory=list,
        description="Statutes or codes that may be relevant"
    )
    
    # Recommendations
    recommended_resources: List[str] = Field(
        default_factory=list,
        description="Recommended resources for further understanding"
    )
    specialist_areas: List[str] = Field(
        default_factory=list,
        description="Areas where specialist knowledge may be required"
    )
    
    # Confidence
    confidence_score: float = Field(
        default=0.0,
        description="Confidence in the classification (0.0 to 1.0)"
    )
    
    # Summary
    classification_summary: str = Field(
        default="",
        description="Brief summary explaining the classification"
    )


class LegalClassifierTool(Tool):
    """Tool to classify legal documents, queries, and scenarios."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="legal_classifier_tool")
    description: str = Field(
        default=(
            "Classifies legal documents, queries, and scenarios by identifying the type of document, "
            "area of law, jurisdiction, complexity level, and other relevant attributes. "
            "Provides structured classification results to help understand the legal context "
            "and identify appropriate resources or expertise needed."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="text",
                arg_type="string",
                description=(
                    "The text to classify. This can be a legal document, a description of a legal "
                    "scenario, a legal question, or any text with legal content. For best results, "
                    "provide as much context as possible."
                ),
                required=True,
                example="I need to file a motion to dismiss in a contract dispute case in federal court...",
            ),
            ToolArgument(
                name="classification_type",
                arg_type="string",
                description=(
                    "The type of classification to perform. Options: 'document' (classify a legal document), "
                    "'query' (classify a legal question), 'scenario' (classify a legal situation), or "
                    "'comprehensive' (perform all classifications). Default is 'comprehensive'."
                ),
                required=False,
                default="comprehensive",
                example="document",
            ),
            ToolArgument(
                name="output_format",
                arg_type="string",
                description=(
                    "The format for the classification output. Options: 'structured' (returns a structured "
                    "object with all classifications), 'summary' (returns a concise text summary), or "
                    "'detailed' (returns a comprehensive text analysis). Default is 'structured'."
                ),
                required=False,
                default="structured",
                example="summary",
            ),
            ToolArgument(
                name="jurisdiction_hint",
                arg_type="string",
                description=(
                    "Optional hint about the jurisdiction to help with classification. "
                    "Example: 'US', 'EU', 'UK', 'International', etc."
                ),
                required=False,
                example="US",
            ),
            ToolArgument(
                name="domain_hint",
                arg_type="string",
                description=(
                    "Optional hint about the legal domain to help with classification. "
                    "Example: 'property', 'contract', 'criminal', etc."
                ),
                required=False,
                example="intellectual property",
            ),
            ToolArgument(
                name="temperature",
                arg_type="string",
                description=(
                    'Sampling temperature between "0.0" and "1.0": "0.0" for more deterministic classification, '
                    '"1.0" for more creative analysis. For legal classification, '
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
        name: str = "legal_classifier_tool",
        generative_model: GenerativeModel | None = None,
        event_emitter: EventEmitter | None = None,
    ):
        """Initialize the LegalClassifierTool with model configuration and optional callback.

        Args:
            model_name (str): The name of the language model to use.
            system_prompt (str, optional): Default system prompt for the model.
            on_token (Callable, optional): Callback function for streaming tokens.
            name (str): Name of the tool instance. Defaults to "legal_classifier_tool".
            generative_model (GenerativeModel, optional): Pre-initialized generative model.
            event_emitter (EventEmitter, optional): Event emitter for streaming tokens.
        """
        # Default system prompt for legal classification if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert legal classifier with extensive knowledge across multiple legal systems "
                "and domains. Your expertise includes identifying document types, legal domains, jurisdictions, "
                "complexity levels, and other attributes of legal texts. You can recognize patterns in legal "
                "language that indicate specific areas of law, document purposes, and intended audiences. "
                "You are familiar with legal terminology across different jurisdictions including common law "
                "systems (US, UK, etc.), civil law systems (France, Germany, etc.), and international law. "
                "Your classifications are precise, comprehensive, and include confidence levels when uncertainty exists."
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
            logger.debug(f"Initialized LegalClassifierTool with model: {self.model_name}")

        # Only set up event listener if on_token is provided
        if self.on_token is not None:
            logger.debug(f"Setting up event listener for LegalClassifierTool with model: {self.model_name}")
            self.generative_model.event_emitter.on("stream_chunk", self.on_token)

    def _build_prompt(
        self,
        text: str,
        classification_type: str = "comprehensive",
        output_format: str = "structured",
        jurisdiction_hint: Optional[str] = None,
        domain_hint: Optional[str] = None,
    ) -> str:
        """Build a comprehensive prompt for legal classification.

        Args:
            text: The text to classify.
            classification_type: Type of classification to perform.
            output_format: Format for the classification output.
            jurisdiction_hint: Optional hint about the jurisdiction.
            domain_hint: Optional hint about the legal domain.

        Returns:
            A structured prompt for the LLM.
        """
        # Validate classification type
        valid_classification_types = ["document", "query", "scenario", "comprehensive"]
        if classification_type.lower() not in valid_classification_types:
            classification_type = "comprehensive"
        
        # Validate output format
        valid_output_formats = ["structured", "summary", "detailed"]
        if output_format.lower() not in valid_output_formats:
            output_format = "structured"
        
        # Build hints section
        hints = []
        if jurisdiction_hint:
            hints.append(f"Jurisdiction Hint: {jurisdiction_hint}")
        if domain_hint:
            hints.append(f"Legal Domain Hint: {domain_hint}")
        hints_section = "\n".join(hints) if hints else "No specific hints provided."

        # Define classification instructions based on type
        classification_instructions = {
            "document": (
                "Focus on identifying the document type, its purpose, intended audience, and formal characteristics. "
                "Pay attention to formatting, legal language patterns, citations, and structural elements that "
                "indicate specific document types."
            ),
            "query": (
                "Focus on identifying the legal question, the area of law it pertains to, and potential "
                "resources or expertise needed to address it. Consider the complexity of the question and "
                "whether it spans multiple legal domains."
            ),
            "scenario": (
                "Focus on identifying the legal issues present in the scenario, potential causes of action, "
                "applicable laws, and jurisdictional considerations. Assess the complexity of the scenario "
                "and whether it involves multiple areas of law."
            ),
            "comprehensive": (
                "Perform a comprehensive classification covering document characteristics (if applicable), "
                "legal domains, jurisdictional aspects, complexity assessment, and identification of key "
                "legal concepts. Provide a holistic analysis regardless of whether the text is a document, "
                "query, or scenario."
            )
        }

        # Define output format instructions
        output_format_instructions = {
            "structured": (
                "Return your classification as a structured JSON object following this schema:\n"
                "```json\n"
                "{\n"
                '  "document_type": "string",\n'
                '  "legal_domain": "string",\n'
                '  "jurisdiction": "string",\n'
                '  "sub_domains": ["string", "string", ...],\n'
                '  "purpose": "string",\n'
                '  "audience": "string",\n'
                '  "complexity_level": int (1-5),\n'
                '  "formality_level": int (1-5),\n'
                '  "estimated_reading_time": "string",\n'
                '  "key_legal_concepts": ["string", "string", ...],\n'
                '  "potential_precedents": ["string", "string", ...],\n'
                '  "statutes_referenced": ["string", "string", ...],\n'
                '  "recommended_resources": ["string", "string", ...],\n'
                '  "specialist_areas": ["string", "string", ...],\n'
                '  "confidence_score": float (0.0-1.0),\n'
                '  "classification_summary": "string"\n'
                "}\n"
                "```\n"
                "Ensure all fields are populated with appropriate values. For lists, include at least one item "
                "if relevant information is available. For fields where no relevant information can be determined, "
                "use empty lists [] for list fields, empty strings \"\" for string fields, and appropriate default "
                "values for numeric fields. Ensure the JSON is valid and properly formatted."
            ),
            "summary": (
                "Return a concise text summary (3-5 sentences) of the classification results, focusing on the "
                "most important aspects such as document type, legal domain, jurisdiction, and complexity level. "
                "The summary should be clear, informative, and highlight the most distinctive features of the text."
            ),
            "detailed": (
                "Return a comprehensive text analysis organized into sections with headings. Include sections for: "
                "1) Document Classification, 2) Legal Domain Analysis, 3) Jurisdictional Assessment, 4) Complexity "
                "and Formality Analysis, 5) Key Legal Concepts, 6) Recommended Resources and Expertise, and "
                "7) Confidence Assessment. Provide detailed explanations for each classification and include "
                "reasoning for your determinations."
            )
        }

        prompt = f"""
Analyze and classify the following legal text. Determine the document type, legal domain, jurisdiction, complexity level, and other relevant attributes.

# TEXT TO CLASSIFY
```
{text}
```

# CLASSIFICATION TYPE
{classification_type.capitalize()} Classification: {classification_instructions[classification_type.lower()]}

# HINTS
{hints_section}

# CLASSIFICATION REQUIREMENTS
Provide a comprehensive classification of the text addressing the following aspects:

1. **Document Type Classification**
   - Identify the specific type of legal document (e.g., contract, complaint, motion, brief, letter, opinion)
   - If not a document, classify as "query" or "scenario" as appropriate
   - Determine the document's primary purpose (e.g., advocacy, informational, decisional, transactional)
   - Identify the intended audience (e.g., court, opposing counsel, client, public)

2. **Legal Domain Classification**
   - Identify the primary area of law (e.g., property, contract, criminal, family, intellectual property)
   - Identify specific sub-domains (e.g., real estate, divorce, patent, securities)
   - Identify any cross-domain issues where multiple areas of law intersect

3. **Jurisdictional Classification**
   - Identify the legal system (e.g., common law, civil law, religious law, international law)
   - Identify the specific jurisdiction if possible (e.g., US Federal, California State, French Civil Law)
   - Note any jurisdictional complexities or conflicts

4. **Complexity and Formality Assessment**
   - Assess the complexity level on a scale of 1 (simple) to 5 (highly complex)
   - Assess the formality level on a scale of 1 (informal) to 5 (highly formal)
   - Estimate the reading time required for a legal professional to understand the text

5. **Content Analysis**
   - Identify key legal concepts or principles involved
   - Identify potential relevant precedents or cases
   - Identify statutes or codes that may be relevant

6. **Resource Recommendations**
   - Recommend resources for further understanding (e.g., specific treatises, practice guides)
   - Identify areas where specialist knowledge may be required

7. **Confidence Assessment**
   - Provide a confidence score for your classification (0.0 to 1.0)
   - Note any areas of uncertainty in the classification

# OUTPUT FORMAT
{output_format_instructions[output_format.lower()]}

# APPROACH
- Be precise in your classifications, using specific legal terminology
- Base classifications on textual evidence and legal knowledge
- When uncertain, indicate lower confidence rather than making definitive claims
- Consider both explicit information and implicit indicators in the text
- For document classification, pay attention to formatting, citation patterns, and structural elements
- For domain classification, identify legal terminology specific to different areas of law
- For jurisdictional classification, look for references to specific laws, courts, or legal systems

Please provide a complete classification according to these specifications.
"""
        return prompt

    async def async_execute(
        self,
        text: str,
        classification_type: str = "comprehensive",
        output_format: str = "structured",
        jurisdiction_hint: str = None,
        domain_hint: str = None,
        temperature: str = "0.2",
    ) -> Union[Dict[str, Any], str]:
        """Execute the tool to classify legal text asynchronously.

        Args:
            text: The text to classify.
            classification_type: Type of classification to perform.
            output_format: Format for the classification output.
            jurisdiction_hint: Optional hint about the jurisdiction.
            domain_hint: Optional hint about the legal domain.
            temperature: Sampling temperature for the model.

        Returns:
            Union[Dict[str, Any], str]: The classification results, either as a dictionary (for structured output)
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

        # Build the prompt for legal classification
        prompt = self._build_prompt(
            text=text,
            classification_type=classification_type,
            output_format=output_format,
            jurisdiction_hint=jurisdiction_hint,
            domain_hint=domain_hint,
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

                logger.debug(f"Generated legal classification (first 100 chars): {response[:100]}...")
                
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
                        classification_dict = json.loads(json_str)
                        
                        # Create a LegalClassification object
                        classification = LegalClassification(**classification_dict)
                        
                        # Return as dictionary
                        return classification.model_dump()
                    except Exception as e:
                        logger.error(f"Error parsing structured output: {e}")
                        # Fall back to returning the raw response
                        return response
                else:
                    # For summary or detailed output, return the raw response
                    return response
            except Exception as e:
                logger.error(f"Error generating legal classification: {e}")
                raise Exception(f"Error generating legal classification: {e}") from e
        else:
            raise ValueError("Generative model not initialized")

    def execute(
        self,
        text: str,
        classification_type: str = "comprehensive",
        output_format: str = "structured",
        jurisdiction_hint: str = None,
        domain_hint: str = None,
        temperature: str = "0.2",
    ) -> Union[Dict[str, Any], str]:
        """Execute the tool to classify legal text synchronously.

        This method provides a synchronous wrapper around the asynchronous implementation.

        Args:
            text: The text to classify.
            classification_type: Type of classification to perform.
            output_format: Format for the classification output.
            jurisdiction_hint: Optional hint about the jurisdiction.
            domain_hint: Optional hint about the legal domain.
            temperature: Sampling temperature for the model.

        Returns:
            Union[Dict[str, Any], str]: The classification results, either as a dictionary (for structured output)
            or as a string (for summary or detailed output).
        """
        return asyncio.run(
            self.async_execute(
                text=text,
                classification_type=classification_type,
                output_format=output_format,
                jurisdiction_hint=jurisdiction_hint,
                domain_hint=domain_hint,
                temperature=temperature,
            )
        )


if __name__ == "__main__":
    # Example usage of LegalClassifierTool
    tool = LegalClassifierTool(model_name="openai/gpt-4o-mini")
    
    # Example legal text to classify
    sample_text = """
    MOTION TO DISMISS
    
    COMES NOW the Defendant, XYZ Corporation, by and through its undersigned counsel, and pursuant to Federal Rule of Civil Procedure 12(b)(6), hereby moves this Honorable Court to dismiss the Complaint filed by Plaintiff for failure to state a claim upon which relief can be granted, and in support thereof states as follows:
    
    1. Plaintiff alleges breach of contract related to a software licensing agreement executed on January 15, 2023.
    
    2. The agreement contains a mandatory arbitration clause in Section 14, which states: "Any dispute arising out of or relating to this Agreement shall be resolved by binding arbitration in accordance with the Commercial Arbitration Rules of the American Arbitration Association."
    
    3. Plaintiff has failed to pursue arbitration as required by the agreement before filing this action.
    
    4. The Federal Arbitration Act, 9 U.S.C. § 2, provides that arbitration agreements "shall be valid, irrevocable, and enforceable, save upon such grounds as exist at law or in equity for the revocation of any contract."
    
    5. The Supreme Court has consistently upheld the enforceability of arbitration clauses. See AT&T Mobility LLC v. Concepcion, 563 U.S. 333 (2011).
    
    WHEREFORE, Defendant respectfully requests that this Court dismiss Plaintiff's Complaint with prejudice and award Defendant its costs and attorney's fees incurred in defending this action, and for such other relief as the Court deems just and proper.
    
    Respectfully submitted,
    
    Jane Smith
    Smith & Associates
    123 Legal Street
    New York, NY 10001
    Attorney for Defendant
    """
    
    # Generate classification with structured output
    structured_classification = tool.execute(
        text=sample_text,
        classification_type="document",
        output_format="structured",
        jurisdiction_hint="US",
    )
    
    print("Structured Classification Results:")
    import json
    print(json.dumps(structured_classification, indent=2))
    
    # Generate classification with summary output
    summary_classification = tool.execute(
        text=sample_text,
        classification_type="comprehensive",
        output_format="summary",
    )
    
    print("\nSummary Classification:")
    print(summary_classification)
    
    # Display tool configuration in Markdown
    print("\nTool Documentation:")
    print(tool.to_markdown())
