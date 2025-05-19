"""Legal Case Triage Tool for preliminary analysis of legal cases and document classification."""

import asyncio
import os
from typing import Callable, Dict, Any, List, Optional, Union

from loguru import logger
from pydantic import ConfigDict, Field, BaseModel

from quantalogic.console_print_token import console_print_token
from quantalogic.event_emitter import EventEmitter
from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.tools.tool import Tool, ToolArgument


class LegalDocument(BaseModel):
    """Model for a classified legal document."""
    
    document_id: str = Field(description="Unique identifier for the document")
    document_type: str = Field(description="Type of document (e.g., 'contract', 'court_filing', 'correspondence')")
    title: str = Field(description="Title or name of the document")
    date: Optional[str] = Field(default=None, description="Date of the document (if available)")
    parties: List[str] = Field(default_factory=list, description="Parties mentioned in the document")
    summary: str = Field(description="Brief summary of the document's content")
    key_points: List[str] = Field(default_factory=list, description="Key points or facts from the document")
    legal_issues: List[str] = Field(default_factory=list, description="Legal issues identified in the document")
    relevance_score: float = Field(description="Relevance score from 0.0 to 1.0")


class LegalIssue(BaseModel):
    """Model for an identified legal issue."""
    
    issue_name: str = Field(description="Name of the legal issue")
    description: str = Field(description="Detailed description of the issue")
    applicable_laws: List[str] = Field(default_factory=list, description="Applicable laws, codes, or regulations")
    relevant_documents: List[str] = Field(default_factory=list, description="Document IDs relevant to this issue")
    importance: str = Field(description="Importance level: 'High', 'Medium', or 'Low'")


class CaseTriage(BaseModel):
    """Model for the complete case triage results."""
    
    # Case overview
    case_title: str = Field(description="Title or name of the case")
    case_type: str = Field(description="Type of case (e.g., 'litigation', 'contract_dispute', 'compliance')")
    jurisdiction: str = Field(description="Legal jurisdiction (e.g., 'France', 'EU')")
    legal_domain: str = Field(description="Area of law (e.g., 'civil', 'commercial', 'labor')")
    parties: Dict[str, str] = Field(default_factory=dict, description="Mapping of roles to party names")
    
    # Document analysis
    documents: List[LegalDocument] = Field(default_factory=list, description="Analyzed documents in the case")
    document_categories: Dict[str, List[str]] = Field(default_factory=dict, description="Categorized document IDs")
    
    # Legal analysis
    key_legal_issues: List[LegalIssue] = Field(default_factory=list, description="Key legal issues identified")
    applicable_laws: List[str] = Field(default_factory=list, description="All applicable laws and regulations")
    case_timeline: List[Dict[str, str]] = Field(default_factory=list, description="Timeline of case events")
    
    # Preliminary assessment
    case_complexity: str = Field(description="Case complexity: 'High', 'Medium', or 'Low'")
    case_urgency: str = Field(description="Case urgency: 'High', 'Medium', or 'Low'")
    estimated_workload: str = Field(description="Estimated workload (e.g., 'X hours', 'X days')")
    
    # Recommendations
    next_steps: List[str] = Field(default_factory=list, description="Recommended next steps")
    required_expertise: List[str] = Field(default_factory=list, description="Areas of expertise required")
    
    # Preliminary memo
    analysis_memo: str = Field(description="Preliminary analysis memo")


class LegalCaseTriageTool(Tool):
    """Tool for preliminary analysis of legal cases and document classification."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="legal_case_triage_tool")
    description: str = Field(
        default=(
            "Performs preliminary analysis of legal cases by classifying documents, "
            "identifying key legal issues, and generating analysis memos. This tool "
            "helps lawyers quickly assess and prioritize cases by organizing documents, "
            "extracting important information, and providing initial recommendations."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="case_documents",
                arg_type="string",
                description=(
                    "List of document texts separated by '---DOCUMENT---' delimiters. "
                    "Each document should start with a title or filename on the first line."
                ),
                required=True,
                example="Contract.pdf\nThis agreement made on...\n---DOCUMENT---\nLetter.pdf\nDear Sir...",
            ),
            ToolArgument(
                name="case_description",
                arg_type="string",
                description=(
                    "Brief description of the case to provide context for the analysis. "
                    "Include the nature of the dispute, key parties, and main issues if known."
                ),
                required=False,
                example="Commercial dispute regarding breach of distribution contract by Company X.",
            ),
            ToolArgument(
                name="jurisdiction",
                arg_type="string",
                description=(
                    "Legal jurisdiction for the case (e.g., 'France', 'EU', 'US-California'). "
                    "This helps identify applicable laws and regulations."
                ),
                required=False,
                default="",
                example="France",
            ),
            ToolArgument(
                name="legal_domain",
                arg_type="string",
                description=(
                    "Primary area of law for the case (e.g., 'civil', 'commercial', 'labor', 'intellectual_property'). "
                    "This helps focus the analysis on relevant legal frameworks."
                ),
                required=False,
                default="",
                example="commercial",
            ),
            ToolArgument(
                name="output_format",
                arg_type="string",
                description=(
                    "The format for the triage output. Options: 'structured' (returns a structured "
                    "object with all analysis details), 'memo' (returns only the analysis memo as text), "
                    "or 'detailed' (returns a comprehensive text analysis). Default is 'structured'."
                ),
                required=False,
                default="structured",
                example="memo",
            ),
            ToolArgument(
                name="language",
                arg_type="string",
                description=(
                    "The language for the analysis output. This should match the language of the input documents. "
                    "Defaults to auto-detection."
                ),
                required=False,
                default="auto",
                example="French",
            ),
            ToolArgument(
                name="temperature",
                arg_type="string",
                description=(
                    'Sampling temperature between "0.0" and "1.0": "0.0" for more factual analysis, '
                    '"1.0" for more interpretive analysis. For legal analysis, '
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
        name: str = "legal_case_triage_tool",
        generative_model: GenerativeModel | None = None,
        event_emitter: EventEmitter | None = None,
    ):
        """Initialize the LegalCaseTriageTool with model configuration and optional callback.

        Args:
            model_name (str): The name of the language model to use.
            system_prompt (str, optional): Default system prompt for the model.
            on_token (Callable, optional): Callback function for streaming tokens.
            name (str): Name of the tool instance. Defaults to "legal_case_triage_tool".
            generative_model (GenerativeModel, optional): Pre-initialized generative model.
            event_emitter (EventEmitter, optional): Event emitter for streaming tokens.
        """
        # Default system prompt for legal case triage if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert legal analyst specializing in preliminary case assessment and document "
                "classification. Your expertise includes identifying document types, extracting key information, "
                "recognizing legal issues, and generating preliminary analysis memos. You have extensive knowledge "
                "of various legal domains and jurisdictions, particularly French and EU law. You excel at organizing "
                "complex legal information into clear, structured analyses that help lawyers quickly understand and "
                "prioritize cases. Your assessments are thorough, objective, and focused on practical legal implications."
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
            logger.debug(f"Initialized LegalCaseTriageTool with model: {self.model_name}")

        # Only set up event listener if on_token is provided
        if self.on_token is not None:
            logger.debug(f"Setting up event listener for LegalCaseTriageTool with model: {self.model_name}")
            self.generative_model.event_emitter.on("stream_chunk", self.on_token)
            
    def _build_prompt(
        self,
        case_documents: str,
        case_description: str = "",
        jurisdiction: str = "",
        legal_domain: str = "",
        output_format: str = "structured",
        language: str = "auto",
        temperature: str = "0.2",
    ) -> str:
        """Build a comprehensive prompt for legal case triage.
        
        Args:
            case_documents: Document texts separated by delimiters
            case_description: Brief description of the case
            jurisdiction: Legal jurisdiction for the case
            legal_domain: Primary area of law for the case
            output_format: Format for the triage output
            language: Language for the analysis output
            temperature: Sampling temperature for model generation
            
        Returns:
            str: Comprehensive prompt for legal case triage
        """
        # Determine language for instructions
        use_french = language.lower() in ["french", "français", "fr"]
        
        # Build the prompt with appropriate language
        if use_french:
            prompt = """# Pré-analyse d'un dossier juridique

## Objectif
Réaliser une pré-analyse complète du dossier juridique fourni en:
1. Classant et organisant les documents
2. Identifiant les points de droit clés
3. Générant une note d'analyse préliminaire
4. Évaluant la complexité et l'urgence du dossier

## Contexte du dossier
"""
        else:
            prompt = """# Legal Case Triage Analysis

## Objective
Perform a comprehensive preliminary analysis of the provided legal case by:
1. Classifying and organizing the documents
2. Identifying key legal issues
3. Generating a preliminary analysis memo
4. Assessing case complexity and urgency

## Case Context
"""

        # Add case description
        if case_description:
            if use_french:
                prompt += f"**Description du dossier**: {case_description}\n\n"
            else:
                prompt += f"**Case Description**: {case_description}\n\n"
        
        # Add jurisdiction and legal domain if provided
        if jurisdiction or legal_domain:
            if use_french:
                if jurisdiction:
                    prompt += f"**Juridiction**: {jurisdiction}\n"
                if legal_domain:
                    prompt += f"**Domaine juridique**: {legal_domain}\n"
            else:
                if jurisdiction:
                    prompt += f"**Jurisdiction**: {jurisdiction}\n"
                if legal_domain:
                    prompt += f"**Legal Domain**: {legal_domain}\n"
            prompt += "\n"
        
        # Add document analysis instructions
        if use_french:
            prompt += """## Analyse des documents
Pour chaque document du dossier:
1. Identifiez le type de document (contrat, correspondance, procédure judiciaire, etc.)
2. Extrayez les informations clés (parties, dates, objets, demandes)
3. Résumez le contenu principal
4. Identifiez les points juridiques soulevés
5. Évaluez la pertinence pour le dossier (score de 0.0 à 1.0)

## Identification des points de droit
1. Identifiez tous les points de droit soulevés dans l'ensemble des documents
2. Déterminez les lois, codes ou règlements applicables
3. Évaluez l'importance de chaque point de droit (Élevée, Moyenne, Faible)
4. Associez chaque point de droit aux documents pertinents

## Évaluation du dossier
1. Évaluez la complexité globale du dossier (Élevée, Moyenne, Faible)
2. Déterminez l'urgence du dossier en fonction des délais identifiés
3. Estimez la charge de travail nécessaire
4. Identifiez les domaines d'expertise requis

## Note d'analyse préliminaire
Rédigez une note d'analyse préliminaire qui:
1. Résume les faits clés du dossier
2. Présente les points de droit principaux
3. Identifie les forces et faiblesses potentielles
4. Recommande les prochaines étapes à suivre
"""
        else:
            prompt += """## Document Analysis
For each document in the case file:
1. Identify the document type (contract, correspondence, court filing, etc.)
2. Extract key information (parties, dates, subjects, requests)
3. Summarize the main content
4. Identify legal issues raised
5. Evaluate relevance to the case (score from 0.0 to 1.0)

## Legal Issue Identification
1. Identify all legal issues raised across the documents
2. Determine applicable laws, codes, or regulations
3. Evaluate the importance of each legal issue (High, Medium, Low)
4. Associate each legal issue with relevant documents

## Case Assessment
1. Evaluate the overall complexity of the case (High, Medium, Low)
2. Determine case urgency based on identified deadlines
3. Estimate the required workload
4. Identify required areas of expertise

## Preliminary Analysis Memo
Draft a preliminary analysis memo that:
1. Summarizes key case facts
2. Presents the main legal issues
3. Identifies potential strengths and weaknesses
4. Recommends next steps
"""

        # Add output format instructions
        if use_french:
            if output_format.lower() == "structured":
                prompt += """
## Format de sortie
Fournissez une analyse structurée au format JSON avec les sections suivantes:
- case_title: titre du dossier
- case_type: type de dossier
- jurisdiction: juridiction applicable
- legal_domain: domaine juridique
- parties: mapping des rôles aux noms des parties
- documents: liste des documents analysés avec leurs métadonnées
- document_categories: documents regroupés par catégories
- key_legal_issues: liste des points de droit identifiés
- applicable_laws: lois et règlements applicables
- case_timeline: chronologie des événements du dossier
- case_complexity: complexité du dossier (Élevée, Moyenne, Faible)
- case_urgency: urgence du dossier (Élevée, Moyenne, Faible)
- estimated_workload: charge de travail estimée
- next_steps: prochaines étapes recommandées
- required_expertise: domaines d'expertise requis
- analysis_memo: note d'analyse préliminaire complète
"""
            elif output_format.lower() == "memo":
                prompt += """
## Format de sortie
Fournissez uniquement la note d'analyse préliminaire au format texte structuré, sans les métadonnées d'analyse.
"""
            else:  # detailed
                prompt += """
## Format de sortie
Fournissez une analyse détaillée au format texte avec les sections suivantes:
1. Résumé du dossier
2. Classification des documents
3. Points de droit identifiés
4. Évaluation du dossier
5. Note d'analyse préliminaire
6. Recommandations
"""
        else:
            if output_format.lower() == "structured":
                prompt += """
## Output Format
Provide a structured analysis in JSON format with the following sections:
- case_title: title of the case
- case_type: type of case
- jurisdiction: applicable jurisdiction
- legal_domain: legal domain
- parties: mapping of roles to party names
- documents: list of analyzed documents with metadata
- document_categories: documents grouped by categories
- key_legal_issues: list of identified legal issues
- applicable_laws: applicable laws and regulations
- case_timeline: timeline of case events
- case_complexity: case complexity (High, Medium, Low)
- case_urgency: case urgency (High, Medium, Low)
- estimated_workload: estimated workload
- next_steps: recommended next steps
- required_expertise: required areas of expertise
- analysis_memo: complete preliminary analysis memo
"""
            elif output_format.lower() == "memo":
                prompt += """
## Output Format
Provide only the preliminary analysis memo as structured text, without the analysis metadata.
"""
            else:  # detailed
                prompt += """
## Output Format
Provide a detailed analysis in text format with the following sections:
1. Case Summary
2. Document Classification
3. Identified Legal Issues
4. Case Assessment
5. Preliminary Analysis Memo
6. Recommendations
"""

        # Add document content
        if use_french:
            prompt += "\n## Documents du dossier\n"
        else:
            prompt += "\n## Case Documents\n"
            
        prompt += case_documents
        
        return prompt

    def _parse_response(self, response: str, output_format: str) -> Union[Dict[str, Any], str]:
        """Parse the model response based on the requested output format.
        
        Args:
            response: Raw response from the model
            output_format: Requested output format
            
        Returns:
            Union[Dict[str, Any], str]: Parsed response in the requested format
        """
        try:
            if output_format.lower() == "structured":
                # For structured output, try to extract JSON
                import json
                import re
                
                # Look for JSON blocks in the response
                json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON without markdown code blocks
                    json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # If no JSON found, return the raw response
                        logger.warning("Could not extract JSON from structured response")
                        return {"error": "Failed to parse structured output", "raw_response": response}
                
                # Parse the JSON
                try:
                    result = json.loads(json_str)
                    # Validate against our model
                    return CaseTriage(**result).model_dump()
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {json_str}")
                    return {"error": "Invalid JSON in response", "raw_response": response}
                except Exception as e:
                    logger.error(f"Error validating response against CaseTriage model: {e}")
                    return {"error": f"Validation error: {str(e)}", "raw_response": response}
            
            elif output_format.lower() == "memo":
                # For memo output, extract just the memo text
                import re
                
                # Try to find a section that looks like a memo
                memo_match = re.search(r'# .*?Memo.*?\n(.*?)(?=^#|\Z)', response, re.DOTALL | re.MULTILINE)
                if memo_match:
                    return memo_match.group(1).strip()
                else:
                    # If no specific memo section found, return the whole response
                    return response
            
            else:  # detailed or any other format
                # Return the raw response for detailed output
                return response
                
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {"error": f"Failed to parse response: {str(e)}", "raw_response": response}

    async def execute(
        self,
        case_documents: str,
        case_description: str = "",
        jurisdiction: str = "",
        legal_domain: str = "",
        output_format: str = "structured",
        language: str = "auto",
        temperature: str = "0.2",
    ) -> Union[Dict[str, Any], str]:
        """Execute the legal case triage analysis.
        
        Args:
            case_documents: Document texts separated by delimiters
            case_description: Brief description of the case
            jurisdiction: Legal jurisdiction for the case
            legal_domain: Primary area of law for the case
            output_format: Format for the triage output
            language: Language for the analysis output
            temperature: Sampling temperature for model generation
            
        Returns:
            Union[Dict[str, Any], str]: Triage results in the requested format
        """
        try:
            # Build the prompt
            prompt = self._build_prompt(
                case_documents=case_documents,
                case_description=case_description,
                jurisdiction=jurisdiction,
                legal_domain=legal_domain,
                output_format=output_format,
                language=language,
                temperature=temperature,
            )
            
            # Log the prompt for debugging
            logger.debug(f"Legal case triage prompt: {prompt[:500]}...")
            
            # Convert temperature string to float
            try:
                temp_value = float(temperature)
                # Ensure temperature is within valid range
                temp_value = max(0.0, min(1.0, temp_value))
            except ValueError:
                logger.warning(f"Invalid temperature value: {temperature}, using default 0.2")
                temp_value = 0.2
            
            # Create messages for the model
            messages = [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=prompt),
            ]
            
            # Generate the response
            response = await self.generative_model.generate_content_async(
                messages=messages,
                temperature=temp_value,
            )
            
            # Extract the text from the response
            response_text = response.text
            
            # Parse the response based on the requested output format
            result = self._parse_response(response_text, output_format)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in legal case triage: {e}")
            return {"error": f"Failed to complete legal case triage: {str(e)}"}
    
    def execute_sync(
        self,
        case_documents: str,
        case_description: str = "",
        jurisdiction: str = "",
        legal_domain: str = "",
        output_format: str = "structured",
        language: str = "auto",
        temperature: str = "0.2",
    ) -> Union[Dict[str, Any], str]:
        """Synchronous wrapper for the execute method.
        
        Args:
            case_documents: Document texts separated by delimiters
            case_description: Brief description of the case
            jurisdiction: Legal jurisdiction for the case
            legal_domain: Primary area of law for the case
            output_format: Format for the triage output
            language: Language for the analysis output
            temperature: Sampling temperature for model generation
            
        Returns:
            Union[Dict[str, Any], str]: Triage results in the requested format
        """
        return asyncio.run(self.execute(
            case_documents=case_documents,
            case_description=case_description,
            jurisdiction=jurisdiction,
            legal_domain=legal_domain,
            output_format=output_format,
            language=language,
            temperature=temperature,
        ))
        
    def to_markdown(self) -> str:
        """Generate markdown documentation for the tool.
        
        Returns:
            str: Markdown documentation
        """
        markdown = f"# {self.name}\n\n"
        markdown += f"{self.description}\n\n"
        
        markdown += "## Arguments\n\n"
        for arg in self.arguments:
            required = "Required" if arg.required else "Optional"
            default = f" (Default: `{arg.default}`)" if hasattr(arg, 'default') and arg.default is not None else ""
            markdown += f"### {arg.name} ({required}{default})\n\n"
            markdown += f"{arg.description}\n\n"
            if hasattr(arg, 'example') and arg.example:
                markdown += f"Example: `{arg.example}`\n\n"
        
        markdown += "## Output\n\n"
        markdown += "The tool provides three output formats:\n\n"
        markdown += "1. **structured**: A comprehensive JSON object with all analysis details\n"
        markdown += "2. **memo**: Only the preliminary analysis memo as text\n"
        markdown += "3. **detailed**: A comprehensive text analysis with all sections\n\n"
        
        markdown += "## Example Usage\n\n"
        markdown += "```python\n"
        markdown += "triage_tool = LegalCaseTriageTool(model_name=\"openrouter/openai/gpt-4o-mini\")\n\n"
        markdown += "result = triage_tool.execute_sync(\n"
        markdown += "    case_documents=\"Contract.pdf\\nThis agreement made on...\\n---DOCUMENT---\\nLetter.pdf\\nDear Sir...\",\n"
        markdown += "    case_description=\"Commercial dispute regarding breach of distribution contract\",\n"
        markdown += "    jurisdiction=\"France\",\n"
        markdown += "    legal_domain=\"commercial\",\n"
        markdown += "    output_format=\"structured\",\n"
        markdown += "    language=\"French\"\n"
        markdown += ")\n"
        markdown += "```\n"
        
        return markdown
