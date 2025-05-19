"""Defender LLM Tool for generating professional legal letters for court submissions."""

import asyncio
from typing import Callable, Dict, Any, List, Optional

from loguru import logger
from pydantic import ConfigDict, Field

from quantalogic.console_print_token import console_print_token
from quantalogic.event_emitter import EventEmitter
from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.tools.tool import Tool, ToolArgument


class DefenderLLMTool(Tool):
    """Tool to generate professional legal letters for court submissions."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="defender_llm_tool")
    description: str = Field(
        default=(
            "Generates professional legal letters for court submissions based on your specific case details. "
            "The tool will create a comprehensive, well-structured legal letter that addresses your situation "
            "with appropriate legal references, arguments, and formal language suitable for court submission. "
            "Provide detailed information about your case for the best results."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="case_type",
                arg_type="string",
                description=(
                    "The type of legal case (e.g., property dispute, contract breach, family law, etc.). "
                    "This helps tailor the letter to the specific legal domain."
                ),
                required=True,
                example="property dispute",
            ),
            ToolArgument(
                name="case_details",
                arg_type="string",
                description=(
                    "Detailed description of your case including relevant facts, dates, parties involved, "
                    "and any previous communications or legal actions. The more details provided, the better "
                    "the resulting letter will be."
                ),
                required=True,
                example="My neighbor has built a fence that encroaches 2 meters onto my property...",
            ),
            ToolArgument(
                name="desired_outcome",
                arg_type="string",
                description=(
                    "What you hope to achieve with this letter (e.g., removal of structure, compensation, "
                    "cease and desist, etc.)."
                ),
                required=True,
                example="I want the court to order removal of the encroaching structure and compensation for damages.",
            ),
            ToolArgument(
                name="legal_references",
                arg_type="string",
                description=(
                    "Any specific laws, codes, or precedents you're aware of that relate to your case. "
                    "Optional but helpful if you have this information."
                ),
                required=False,
                example="Civil Code Articles 691, 703, and 710 regarding property boundaries and servitudes.",
            ),
            ToolArgument(
                name="recipient",
                arg_type="string",
                description=(
                    "Who the letter is addressed to (e.g., 'The Honorable Judge of the District Court', "
                    "'Clerk of the Superior Court', etc.)."
                ),
                required=True,
                example="The Honorable Judge of the District Court of Paris",
            ),
            ToolArgument(
                name="sender_details",
                arg_type="string",
                description=(
                    "Your name, address, and contact information to be included in the letter."
                ),
                required=True,
                example="Jean Dupont, 123 Rue de Paris, 75001 Paris, France, Tel: 01-23-45-67-89, Email: jean.dupont@email.com",
            ),
            ToolArgument(
                name="formality_level",
                arg_type="string",
                description=(
                    "The level of formality for the letter, from 1 (standard formal) to 3 (highly formal with "
                    "extensive legal terminology)."
                ),
                required=False,
                default="2",
                example="2",
            ),
            ToolArgument(
                name="language",
                arg_type="string",
                description=(
                    "The language for the letter (e.g., 'English', 'French', 'German'). "
                    "Defaults to English if not specified."
                ),
                required=False,
                default="English",
                example="French",
            ),
            ToolArgument(
                name="include_attachments_list",
                arg_type="string",
                description=(
                    "Set to 'true' to include a section listing recommended attachments for your submission. "
                    "Set to 'false' to omit this section."
                ),
                required=False,
                default="true",
                example="true",
            ),
            ToolArgument(
                name="temperature",
                arg_type="string",
                description=(
                    'Sampling temperature between "0.0" and "1.0": "0.0" for more conservative, '
                    'factual content, "1.0" for more creative language. For legal documents, '
                    'lower values (0.1-0.3) are generally recommended.'
                ),
                required=False,
                default="0.2",
                example="0.2",
            ),
            ToolArgument(
                name="role",
                arg_type="string",
                description=(
                    "The legal role to adopt when writing the letter (e.g., 'defender', 'attorney', 'judge', 'prosecutor', "
                    "'mediator', 'expert witness', etc.). This helps tailor the tone and perspective of the letter."
                ),
                required=False,
                default="defender",
                example="attorney",
            ),
        ]
    )

    model_name: str = Field(..., description="The name of the language model to use")
    system_prompt: str | None = Field(default=None)
    on_token: Callable | None = Field(default=None, exclude=True)
    generative_model: GenerativeModel | None = Field(default=None, exclude=True)
    event_emitter: EventEmitter | None = Field(default=None, exclude=True)
    role: str = Field(default="defender", description="The legal role to adopt when writing the letter")

    def __init__(
        self,
        model_name: str,
        system_prompt: str | None = None,
        on_token: Callable | None = None,
        name: str = "defender_llm_tool",
        generative_model: GenerativeModel | None = None,
        event_emitter: EventEmitter | None = None,
        role: str = "defender",
    ):
        """Initialize the DefenderLLMTool with model configuration and optional callback.

        Args:
            model_name (str): The name of the language model to use.
            system_prompt (str, optional): Default system prompt for the model.
            on_token (Callable, optional): Callback function for streaming tokens.
            name (str): Name of the tool instance. Defaults to "defender_llm_tool".
            generative_model (GenerativeModel, optional): Pre-initialized generative model.
            event_emitter (EventEmitter, optional): Event emitter for streaming tokens.
            role (str, optional): The legal role to adopt (defender, attorney, judge, etc.). Defaults to "defender".
        """
        # Default system prompt for legal letter generation if none provided
        if system_prompt is None:
            role_descriptions = {
                "defender": "You are an experienced defense attorney specializing in protecting clients' rights and interests",
                "attorney": "You are an experienced attorney with expertise in representing clients in various legal matters",
                "judge": "You are an experienced judge with a deep understanding of legal principles and court procedures",
                "prosecutor": "You are an experienced prosecutor focused on representing the interests of the state or government",
                "mediator": "You are an experienced legal mediator skilled in facilitating dispute resolution between parties",
                "expert witness": "You are an expert witness with specialized knowledge relevant to legal proceedings",
                "legal advisor": "You are a legal advisor providing guidance on legal matters and strategies",
                "paralegal": "You are a paralegal assisting with legal research and document preparation",
            }
            
            # Get role description or use a generic one if role not found
            role_desc = role_descriptions.get(
                role.lower(), 
                f"You are an experienced legal professional acting as a {role}"
            )
            
            system_prompt = (
                f"{role_desc} specializing in drafting formal court submissions "
                "and legal correspondence. Your expertise includes property law, civil disputes, contract law, "
                "family law, and general litigation. You write in a clear, professional, and persuasive manner "
                "appropriate for legal contexts. You understand court protocols and formal legal writing conventions. "
                "When drafting documents, you maintain appropriate formality, cite relevant legal codes and precedents, "
                "organize arguments logically, and focus on factual accuracy. Your goal is to create compelling, "
                "professional legal documents that effectively represent the client's position while adhering to "
                "legal standards and court expectations."
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
                "role": role,
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
            logger.debug(f"Initialized DefenderLLMTool with model: {self.model_name}")

        # Only set up event listener if on_token is provided
        if self.on_token is not None:
            logger.debug(f"Setting up event listener for DefenderLLMTool with model: {self.model_name}")
            self.generative_model.event_emitter.on("stream_chunk", self.on_token)

    def _build_prompt(
        self,
        case_type: str,
        case_details: str,
        desired_outcome: str,
        recipient: str,
        sender_details: str,
        legal_references: Optional[str] = None,
        formality_level: str = "2",
        language: str = "English",
        include_attachments_list: str = "true",
        role: str = "defender",
    ) -> str:
        """Build a comprehensive prompt for the legal letter generation.

        Args:
            case_type: The type of legal case.
            case_details: Detailed description of the case.
            desired_outcome: What the sender hopes to achieve.
            recipient: Who the letter is addressed to.
            sender_details: Sender's name and contact information.
            legal_references: Any specific laws or precedents to reference.
            formality_level: Level of formality (1-3).
            language: The language for the letter.
            include_attachments_list: Whether to include recommended attachments.
            role: The legal role to adopt when writing the letter.

        Returns:
            A structured prompt for the LLM.
        """
        # Convert string boolean to actual boolean
        include_attachments = include_attachments_list.lower() == "true"
        
        # Parse formality level
        try:
            formality = int(formality_level)
            if formality < 1 or formality > 3:
                formality = 2  # Default to medium formality if out of range
        except ValueError:
            formality = 2  # Default to medium formality if invalid input
            
        formality_description = {
            1: "standard formal language appropriate for court",
            2: "highly formal with moderate legal terminology",
            3: "extremely formal with extensive legal terminology and references"
        }[formality]

        prompt = f"""
Create an exceptionally professional, persuasive, and meticulously detailed legal letter addressed to {recipient} regarding a {case_type} case. The letter must be compelling and immediately convincing to a judge or legal authority.

# CASE INFORMATION
- Type of Case: {case_type}
- Case Details: {case_details}
- Desired Outcome: {desired_outcome}
- Sender Information: {sender_details}
- Language: {language}
- Formality Level: {formality_description}
- Your Role: {role}
{f"- Legal References to Include: {legal_references}" if legal_references else ""}

# LETTER STRUCTURE
Create a comprehensive legal letter with the following detailed sections:

1. **Formal Heading**
   - Include current date in proper legal format
   - Court details with full jurisdiction name
   - Case reference/docket number (if applicable)
   - Clear indication of document type (e.g., "Legal Submission," "Formal Request," etc.)

2. **Recipient Details**
   - Full formal title and address of recipient
   - Proper honorifics and professional designations

3. **Subject Line**
   - Concise but specific subject clearly stating the purpose and legal nature of the letter
   - Include any relevant case numbers or identifiers

4. **Introduction (1-2 paragraphs)**
   - Formal introduction of yourself and your capacity/standing in this matter
   - Precise statement of purpose for the letter
   - Brief overview of what will follow in the document
   - Reference to any previous communications or court proceedings

5. **Factual Background (2-3 paragraphs)**
   - Chronological and detailed account of relevant facts
   - Specific dates, locations, and parties involved
   - Clear distinction between undisputed facts and contested claims
   - Reference to any supporting documentation or evidence
   - Objective tone while emphasizing facts favorable to your position

6. **Legal Analysis and Arguments (3-5 paragraphs)**
   - Thorough analysis of applicable laws and precedents
   - Detailed application of law to the specific facts of the case
   - Logical progression of arguments building toward your position
   - Anticipation and preemptive addressing of potential counter-arguments
   - Strategic emphasis on strongest legal points
   - Proper citation of relevant statutes, case law, and legal principles
   - Clear explanation of how the law supports your desired outcome

7. **Evidence and Documentation (1-2 paragraphs)**
   - Summary of key evidence supporting your position
   - Explanation of how this evidence meets legal standards
   - References to specific exhibits, affidavits, or expert opinions
   - Logical connection between evidence and legal arguments

8. **Relief/Remedy Sought (1-2 paragraphs)**
   - Precise and detailed statement of exactly what you are requesting
   - Legal basis for the requested relief
   - Explanation of why the requested remedy is appropriate and just
   - Alternative remedies if applicable

9. **Conclusion (1 paragraph)**
   - Respectful but firm restatement of your position
   - Expression of willingness to provide additional information if needed
   - Request for specific action or response
   - Professional closing appropriate to the recipient's position

10. **Signature Block**
    - Full name with any professional titles
    - Contact information including address, phone, email
    - Bar number or professional credentials if applicable
    - Formal closing phrase appropriate to the jurisdiction

{f"11. **Attachments/Evidence List**\n    - Numbered list of all supporting documents\n    - Brief description of each attachment's relevance\n    - Indication of which documents are originals vs. copies\n    - Certification of authenticity where applicable" if include_attachments else ""}

# PERSUASIVE ELEMENTS TO INCLUDE
- Use precise, authoritative language that demonstrates legal expertise
- Employ strategic repetition of key points for emphasis
- Maintain a respectful but confident tone throughout
- Present a balanced view while subtly emphasizing strengths of your position
- Use rhetorical techniques such as:
  * Logical progression from established principles to specific application
  * Strategic use of rhetorical questions where appropriate
  * Parallel structure for emphasis of key points
  * Concise, impactful topic sentences for each paragraph
  * Varied sentence structure to maintain engagement
- Include transitional phrases between sections for logical flow
- Use appropriate legal maxims or principles to reinforce arguments
- Balance emotional appeal with logical reasoning (pathos and logos)
- Demonstrate thorough understanding of procedural requirements

# FORMATTING REQUIREMENTS
- Use appropriate legal formatting for a court submission in {language}
- Maintain {formality_description}
- Include proper legal citations following the jurisdiction's preferred style
- Use numbered paragraphs where appropriate for easy reference
- Employ strategic use of emphasis (bold, italics) for key points (sparingly)
- Organize arguments logically and persuasively
- Use clear paragraph structure with appropriate headings and subheadings
- Ensure the letter is comprehensive but concise
- Write from the perspective of a {role}
- Use proper legal terminology consistent with the jurisdiction
- Maintain consistent formatting throughout the document

# QUALITY STANDARDS
- Ensure impeccable grammar and punctuation
- Verify all legal citations are accurate and properly formatted
- Confirm all factual assertions are supported by evidence
- Maintain professional tone even when addressing contentious issues
- Ensure all arguments are logically sound and legally defensible
- Verify that the document meets all procedural requirements for submission

Please generate a complete, court-ready letter that would immediately convince a judge of the merits of the position.
"""
        return prompt

    async def async_execute(
        self,
        case_type: str,
        case_details: str,
        desired_outcome: str,
        recipient: str,
        sender_details: str,
        legal_references: str = None,
        formality_level: str = "2",
        language: str = "English",
        include_attachments_list: str = "true",
        temperature: str = "0.2",
        role: str = None,
    ) -> str:
        """Execute the tool to generate a legal letter asynchronously.

        Args:
            case_type: The type of legal case.
            case_details: Detailed description of the case.
            desired_outcome: What the sender hopes to achieve.
            recipient: Who the letter is addressed to.
            sender_details: Sender's name and contact information.
            legal_references: Any specific laws or precedents to reference.
            formality_level: Level of formality (1-3).
            language: The language for the letter.
            include_attachments_list: Whether to include recommended attachments.
            temperature: Sampling temperature for the model.
            role: The legal role to adopt when writing the letter.

        Returns:
            str: The generated legal letter.

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

        # Use the instance role if none provided
        if role is None:
            role = self.role

        # Build the prompt for the legal letter
        prompt = self._build_prompt(
            case_type=case_type,
            case_details=case_details,
            desired_outcome=desired_outcome,
            recipient=recipient,
            sender_details=sender_details,
            legal_references=legal_references,
            formality_level=formality_level,
            language=language,
            include_attachments_list=include_attachments_list,
            role=role,
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

                logger.debug(f"Generated legal letter (first 100 chars): {response[:100]}...")
                return response
            except Exception as e:
                logger.error(f"Error generating legal letter: {e}")
                raise Exception(f"Error generating legal letter: {e}") from e
        else:
            raise ValueError("Generative model not initialized")

    def execute(
        self,
        case_type: str,
        case_details: str,
        desired_outcome: str,
        recipient: str,
        sender_details: str,
        legal_references: str = None,
        formality_level: str = "2",
        language: str = "English",
        include_attachments_list: str = "true",
        temperature: str = "0.2",
        role: str = None,
    ) -> str:
        """Execute the tool to generate a legal letter synchronously.

        This method provides a synchronous wrapper around the asynchronous implementation.

        Args:
            case_type: The type of legal case.
            case_details: Detailed description of the case.
            desired_outcome: What the sender hopes to achieve.
            recipient: Who the letter is addressed to.
            sender_details: Sender's name and contact information.
            legal_references: Any specific laws or precedents to reference.
            formality_level: Level of formality (1-3).
            language: The language for the letter.
            include_attachments_list: Whether to include recommended attachments.
            temperature: Sampling temperature for the model.
            role: The legal role to adopt when writing the letter.

        Returns:
            str: The generated legal letter.
        """
        # Use the instance role if none provided
        if role is None:
            role = self.role
            
        return asyncio.run(
            self.async_execute(
                case_type=case_type,
                case_details=case_details,
                desired_outcome=desired_outcome,
                recipient=recipient,
                sender_details=sender_details,
                legal_references=legal_references,
                formality_level=formality_level,
                language=language,
                include_attachments_list=include_attachments_list,
                temperature=temperature,
                role=role,
            )
        )


if __name__ == "__main__":
    # Example usage of DefenderLLMTool
    tool = DefenderLLMTool(model_name="openai/gpt-4o-mini", role="attorney")
    
    # Example case details
    case_type = "property dispute"
    case_details = """
    My neighbor has built a fence that encroaches 2 meters onto my property according to the land survey.
    Despite multiple friendly requests to remove it since January 2023, they have refused to acknowledge
    the boundary issue. I have documentation from a certified surveyor confirming the property line.
    """
    desired_outcome = "I want the court to order removal of the encroaching fence and compensation for damages to my property."
    recipient = "The Honorable Judge of the District Court of Paris"
    sender_details = "Jean Dupont, 123 Rue de Paris, 75001 Paris, France, Tel: 01-23-45-67-89, Email: jean.dupont@email.com"
    legal_references = "Civil Code Articles 691, 703, and 710 regarding property boundaries and servitudes."
    
    # Synchronous execution
    letter = tool.execute(
        case_type=case_type,
        case_details=case_details,
        desired_outcome=desired_outcome,
        recipient=recipient,
        sender_details=sender_details,
        legal_references=legal_references,
        language="French",
    )
    print("Generated Legal Letter:")
    print(letter)
    
    # 
    # Asynchronous execution with streaming
    # streaming_tool = DefenderLLMTool(
    #     model_name="openai/gpt-4o-mini", 
    #     on_token=console_print_token,
    #     role="judge"
    # )
    
    # print("\nGenerating letter with streaming output:")
    # streaming_letter = asyncio.run(
    #     streaming_tool.async_execute(
    #         case_type=case_type,
    #         case_details=case_details,
    #         desired_outcome=desired_outcome,
    #         recipient=recipient,
    #         sender_details=sender_details,
    #         legal_references=legal_references,
    #         language="arabic",
    #         role="prosecutor",  # Override the instance role for this execution
    #     )
    # )
    
    # # Display tool configuration in Markdown
    # print("\nTool Documentation:")
    # print(tool.to_markdown())
