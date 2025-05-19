"""Prosecutor LLM Tool for generating professional counter-arguments to legal defense letters."""

import asyncio
from typing import Callable, Optional

from loguru import logger
from pydantic import ConfigDict, Field

from quantalogic.console_print_token import console_print_token
from quantalogic.event_emitter import EventEmitter
from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.tools.tool import Tool, ToolArgument


class ProsecutorLLMTool(Tool):
    """Tool to generate professional counter-arguments to legal defense letters."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="prosecutor_llm_tool")
    description: str = Field(
        default=(
            "Generates professional counter-argument letters in response to legal defense submissions. "
            "The tool analyzes a defense letter and creates a comprehensive, well-structured counter-argument "
            "that addresses each point raised by the defense with appropriate legal references, arguments, "
            "and formal language suitable for court submission."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="defense_letter",
                arg_type="string",
                description=(
                    "The full text of the defense letter or submission that you want to counter. "
                    "Include the complete letter for the most effective counter-arguments."
                ),
                required=True,
                example="[Full text of defense letter...]",
            ),
            ToolArgument(
                name="case_type",
                arg_type="string",
                description=(
                    "The type of legal case (e.g., property dispute, contract breach, family law, etc.). "
                    "This helps tailor the counter-argument to the specific legal domain."
                ),
                required=True,
                example="property dispute",
            ),
            ToolArgument(
                name="additional_facts",
                arg_type="string",
                description=(
                    "Any additional facts, evidence, or context not mentioned in the defense letter "
                    "that strengthen your counter-argument. This could include contradictory evidence, "
                    "witness statements, or legal precedents."
                ),
                required=False,
                example="The defendant's survey was conducted by an unlicensed surveyor and contradicts three previous official surveys.",
            ),
            ToolArgument(
                name="desired_outcome",
                arg_type="string",
                description=(
                    "What you hope to achieve with this counter-argument (e.g., dismissal of claims, "
                    "judgment in your favor, rejection of proposed remedy, etc.)."
                ),
                required=True,
                example="Dismissal of the defendant's claims and judgment in favor of the plaintiff with costs.",
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
                example="Marie Dubois, Prosecutor, 456 Avenue de Justice, 75001 Paris, France, Tel: 01-98-76-54-32, Email: marie.dubois@justice.fr",
            ),
            ToolArgument(
                name="legal_references",
                arg_type="string",
                description=(
                    "Any specific laws, codes, or precedents you want to reference in your counter-argument. "
                    "Optional but helpful if you have this information."
                ),
                required=False,
                example="Civil Code Articles 695, 705, and 712 regarding property boundaries and evidence standards.",
            ),
            ToolArgument(
                name="role",
                arg_type="string",
                description=(
                    "The legal role to adopt when writing the counter-argument (e.g., 'prosecutor', 'plaintiff', "
                    "'opposing counsel', 'government representative', etc.)."
                ),
                required=False,
                default="prosecutor",
                example="plaintiff",
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
                example="3",
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
        ]
    )

    model_name: str = Field(..., description="The name of the language model to use")
    system_prompt: str | None = Field(default=None)
    on_token: Callable | None = Field(default=None, exclude=True)
    generative_model: GenerativeModel | None = Field(default=None, exclude=True)
    event_emitter: EventEmitter | None = Field(default=None, exclude=True)
    role: str = Field(default="prosecutor", description="The legal role to adopt when writing the counter-argument")

    def __init__(
        self,
        model_name: str,
        system_prompt: str | None = None,
        on_token: Callable | None = None,
        name: str = "prosecutor_llm_tool",
        generative_model: GenerativeModel | None = None,
        event_emitter: EventEmitter | None = None,
        role: str = "prosecutor",
    ):
        """Initialize the ProsecutorLLMTool with model configuration and optional callback.

        Args:
            model_name (str): The name of the language model to use.
            system_prompt (str, optional): Default system prompt for the model.
            on_token (Callable, optional): Callback function for streaming tokens.
            name (str): Name of the tool instance. Defaults to "prosecutor_llm_tool".
            generative_model (GenerativeModel, optional): Pre-initialized generative model.
            event_emitter (EventEmitter, optional): Event emitter for streaming tokens.
            role (str, optional): The legal role to adopt. Defaults to "prosecutor".
        """
        # Default system prompt for counter-argument generation if none provided
        if system_prompt is None:
            role_descriptions = {
                "prosecutor": "You are an experienced prosecutor specializing in representing the interests of the state or government",
                "plaintiff": "You are an experienced attorney representing plaintiffs in legal disputes",
                "opposing counsel": "You are an experienced attorney representing the party opposing the defense",
                "government representative": "You are a government legal representative with expertise in regulatory matters",
                "corporate counsel": "You are a corporate attorney representing a company's interests",
            }
            
            # Get role description or use a generic one if role not found
            role_desc = role_descriptions.get(
                role.lower(), 
                f"You are an experienced legal professional acting as a {role}"
            )
            
            system_prompt = (
                f"{role_desc} specializing in crafting persuasive counter-arguments to defense submissions. "
                "Your expertise includes identifying logical fallacies, factual misrepresentations, and legal "
                "misinterpretations in defense arguments. You excel at constructing compelling rebuttals that "
                "systematically dismantle opposing arguments while building a strong case for your position. "
                "You write in a clear, authoritative, and persuasive manner appropriate for legal contexts. "
                "You understand court protocols and formal legal writing conventions. When drafting counter-arguments, "
                "you maintain appropriate formality, cite relevant legal codes and precedents, organize arguments "
                "logically, and focus on factual accuracy. Your goal is to create compelling, professional legal "
                "documents that effectively counter defense positions while adhering to legal standards and court expectations."
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
            logger.debug(f"Initialized ProsecutorLLMTool with model: {self.model_name}")

        # Only set up event listener if on_token is provided
        if self.on_token is not None:
            logger.debug(f"Setting up event listener for ProsecutorLLMTool with model: {self.model_name}")
            self.generative_model.event_emitter.on("stream_chunk", self.on_token)

    def _build_prompt(
        self,
        defense_letter: str,
        case_type: str,
        desired_outcome: str,
        recipient: str,
        sender_details: str,
        additional_facts: Optional[str] = None,
        legal_references: Optional[str] = None,
        formality_level: str = "2",
        language: str = "English",
        include_attachments_list: str = "true",
        role: str = "prosecutor",
    ) -> str:
        """Build a comprehensive prompt for the counter-argument letter generation.

        Args:
            defense_letter: The full text of the defense letter to counter.
            case_type: The type of legal case.
            desired_outcome: What the sender hopes to achieve.
            recipient: Who the letter is addressed to.
            sender_details: Sender's name and contact information.
            additional_facts: Any additional facts or evidence to include.
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
Create an exceptionally professional, persuasive, and meticulously detailed counter-argument letter addressed to {recipient} regarding a {case_type} case. The letter must systematically dismantle the defense's arguments while presenting a compelling alternative position.

# DEFENSE LETTER TO COUNTER
```
{defense_letter}
```

# CASE INFORMATION
- Type of Case: {case_type}
- Desired Outcome: {desired_outcome}
- Sender Information: {sender_details}
- Language: {language}
- Formality Level: {formality_description}
- Your Role: {role}
{f"- Additional Facts/Evidence: {additional_facts}" if additional_facts else ""}
{f"- Legal References to Include: {legal_references}" if legal_references else ""}

# COUNTER-ARGUMENT STRUCTURE
Create a comprehensive counter-argument letter with the following detailed sections:

1. **Formal Heading**
   - Include current date in proper legal format
   - Court details with full jurisdiction name
   - Case reference/docket number (if applicable)
   - Clear indication of document type (e.g., "Response to Defense Submission," "Counter-Argument," etc.)

2. **Recipient Details**
   - Full formal title and address of recipient
   - Proper honorifics and professional designations

3. **Subject Line**
   - Concise but specific subject clearly stating this is a response to the defense submission
   - Include any relevant case numbers or identifiers

4. **Introduction (1-2 paragraphs)**
   - Formal introduction of yourself and your capacity/standing in this matter
   - Acknowledge receipt of the defense submission
   - Clear statement that you contest the defense's position
   - Brief overview of your main counter-arguments
   - Reference to any previous communications or court proceedings

5. **Summary of Defense Arguments (1-2 paragraphs)**
   - Accurate and fair summary of the key defense arguments
   - Neutral tone in this section to demonstrate understanding
   - Identification of the core legal and factual claims being made

6. **Systematic Counter-Arguments (3-5 paragraphs)**
   - Point-by-point refutation of each major defense argument
   - Identification of logical fallacies, factual errors, or legal misinterpretations
   - Presentation of contradictory evidence or alternative interpretations
   - Challenges to the defense's legal reasoning or precedent application
   - Exposure of any omissions or misrepresentations in the defense letter

7. **Legal Analysis and Alternative Arguments (2-3 paragraphs)**
   - Thorough analysis of applicable laws and precedents that support your position
   - Detailed application of law to the specific facts of the case
   - Presentation of alternative legal frameworks that better fit the situation
   - Proper citation of relevant statutes, case law, and legal principles
   - Clear explanation of how the law supports your desired outcome

8. **Evidence and Documentation (1-2 paragraphs)**
   - Summary of key evidence contradicting the defense's position
   - Introduction of new evidence not addressed by the defense
   - Explanation of how this evidence undermines the defense's case
   - References to specific exhibits, affidavits, or expert opinions
   - Logical connection between evidence and legal arguments

9. **Relief/Remedy Sought (1-2 paragraphs)**
   - Precise and detailed statement of exactly what you are requesting
   - Legal basis for the requested relief
   - Explanation of why the defense's proposed outcome is inappropriate
   - Alternative remedies if applicable

10. **Conclusion (1 paragraph)**
    - Respectful but firm restatement of your position
    - Summary of why the defense's arguments should be rejected
    - Request for specific action or response
    - Professional closing appropriate to the recipient's position

11. **Signature Block**
    - Full name with any professional titles
    - Contact information including address, phone, email
    - Bar number or professional credentials if applicable
    - Formal closing phrase appropriate to the jurisdiction

{f"12. **Attachments/Evidence List**\n    - Numbered list of all supporting documents\n    - Brief description of each attachment's relevance\n    - Indication of which documents directly contradict defense claims\n    - Certification of authenticity where applicable" if include_attachments else ""}

# COUNTER-ARGUMENT STRATEGIES TO EMPLOY
- Identify and expose any logical fallacies in the defense's arguments
- Challenge unsupported assertions or conclusions
- Highlight any misinterpretations or misapplications of law
- Point out any factual inaccuracies or misrepresentations
- Address any omissions of relevant facts or legal principles
- Question the relevance or applicability of cited precedents
- Demonstrate how the defense's interpretation leads to unreasonable outcomes
- Present alternative interpretations that better align with legal principles
- Use the defense's own logic to reach different conclusions
- Emphasize inconsistencies or contradictions in the defense's position

# PERSUASIVE ELEMENTS TO INCLUDE
- Use precise, authoritative language that demonstrates legal expertise
- Maintain a respectful but confident tone throughout
- Present a balanced view while systematically dismantling the defense's position
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

Please generate a complete, court-ready counter-argument letter that systematically dismantles the defense's position and convincingly presents your alternative arguments.
"""
        return prompt

    async def async_execute(
        self,
        defense_letter: str,
        case_type: str,
        desired_outcome: str,
        recipient: str,
        sender_details: str,
        additional_facts: str = None,
        legal_references: str = None,
        formality_level: str = "2",
        language: str = "English",
        include_attachments_list: str = "true",
        temperature: str = "0.2",
        role: str = None,
    ) -> str:
        """Execute the tool to generate a counter-argument letter asynchronously.

        Args:
            defense_letter: The full text of the defense letter to counter.
            case_type: The type of legal case.
            desired_outcome: What the sender hopes to achieve.
            recipient: Who the letter is addressed to.
            sender_details: Sender's name and contact information.
            additional_facts: Any additional facts or evidence to include.
            legal_references: Any specific laws or precedents to reference.
            formality_level: Level of formality (1-3).
            language: The language for the letter.
            include_attachments_list: Whether to include recommended attachments.
            temperature: Sampling temperature for the model.
            role: The legal role to adopt when writing the letter.

        Returns:
            str: The generated counter-argument letter.

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

        # Build the prompt for the counter-argument letter
        prompt = self._build_prompt(
            defense_letter=defense_letter,
            case_type=case_type,
            desired_outcome=desired_outcome,
            recipient=recipient,
            sender_details=sender_details,
            additional_facts=additional_facts,
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

                logger.debug(f"Generated counter-argument letter (first 100 chars): {response[:100]}...")
                return response
            except Exception as e:
                logger.error(f"Error generating counter-argument letter: {e}")
                raise Exception(f"Error generating counter-argument letter: {e}") from e
        else:
            raise ValueError("Generative model not initialized")

    def execute(
        self,
        defense_letter: str,
        case_type: str,
        desired_outcome: str,
        recipient: str,
        sender_details: str,
        additional_facts: str = None,
        legal_references: str = None,
        formality_level: str = "2",
        language: str = "English",
        include_attachments_list: str = "true",
        temperature: str = "0.2",
        role: str = None,
    ) -> str:
        """Execute the tool to generate a counter-argument letter synchronously.

        This method provides a synchronous wrapper around the asynchronous implementation.

        Args:
            defense_letter: The full text of the defense letter to counter.
            case_type: The type of legal case.
            desired_outcome: What the sender hopes to achieve.
            recipient: Who the letter is addressed to.
            sender_details: Sender's name and contact information.
            additional_facts: Any additional facts or evidence to include.
            legal_references: Any specific laws or precedents to reference.
            formality_level: Level of formality (1-3).
            language: The language for the letter.
            include_attachments_list: Whether to include recommended attachments.
            temperature: Sampling temperature for the model.
            role: The legal role to adopt when writing the letter.

        Returns:
            str: The generated counter-argument letter.
        """
        # Use the instance role if none provided
        if role is None:
            role = self.role
            
        return asyncio.run(
            self.async_execute(
                defense_letter=defense_letter,
                case_type=case_type,
                desired_outcome=desired_outcome,
                recipient=recipient,
                sender_details=sender_details,
                additional_facts=additional_facts,
                legal_references=legal_references,
                formality_level=formality_level,
                language=language,
                include_attachments_list=include_attachments_list,
                temperature=temperature,
                role=role,
            )
        )


if __name__ == "__main__":
    # Example usage of ProsecutorLLMTool
    tool = ProsecutorLLMTool(model_name="openai/gpt-4o-mini", role="prosecutor")
    
    # Example defense letter to counter
    defense_letter = """
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
    
    # Example case details
    case_type = "property dispute"
    additional_facts = """
    1. Three previous official surveys (conducted in 2010, 2015, and 2021) all show that the fence encroaches 2 meters onto the plaintiff's property.
    2. The defendant's surveyor is not licensed in the jurisdiction.
    3. Photographic evidence from 2018 shows the previous boundary marker in a different location than claimed by the defendant.
    4. The defendant was formally notified of the boundary dispute in writing on three occasions before installing the fence.
    """
    desired_outcome = "Court order for immediate removal of the encroaching fence, restoration of the property to its original state, and compensation for legal costs."
    recipient = "The Honorable Judge of the District Court of Paris"
    sender_details = "Marie Dubois, Plaintiff, 456 Avenue de Justice, 75001 Paris, France, Tel: 01-98-76-54-32, Email: marie.dubois@email.fr"
    legal_references = "Civil Code Articles 695, 705, and 712 regarding property boundaries, evidence standards, and remedies for encroachment."
    
    # Generate counter-argument letter
    counter_letter = tool.execute(
        defense_letter=defense_letter,
        case_type=case_type,
        desired_outcome=desired_outcome,
        recipient=recipient,
        sender_details=sender_details,
        additional_facts=additional_facts,
        legal_references=legal_references,
        language="French",
        role="plaintiff",
    )
    
    print("Generated Counter-Argument Letter:")
    print(counter_letter)
    
    # Display tool configuration in Markdown
    print("\nTool Documentation:")
    print(tool.to_markdown())
