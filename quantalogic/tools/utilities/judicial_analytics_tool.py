"""Judicial Analytics and Prediction Tool for analyzing court decisions and predicting outcomes."""

import asyncio
import json
import os
from datetime import datetime
from typing import Callable, Dict, Any, List, Optional, Union, Tuple

from loguru import logger
from pydantic import ConfigDict, Field, BaseModel

from quantalogic.console_print_token import console_print_token
from quantalogic.event_emitter import EventEmitter
from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.tools.tool import Tool, ToolArgument


class CourtDecision(BaseModel):
    """Model for a court decision with key metadata and analysis."""
    
    case_id: str = Field(description="Unique identifier for the court case")
    court_name: str = Field(description="Name of the court that issued the decision")
    decision_date: str = Field(description="Date of the decision")
    case_type: str = Field(description="Type of case (e.g., 'commercial', 'civil', 'labor')")
    judge_name: Optional[str] = Field(default=None, description="Name of the judge(s) who issued the decision")
    parties: Dict[str, str] = Field(default_factory=dict, description="Mapping of party roles to names")
    outcome: str = Field(description="Outcome of the case (e.g., 'plaintiff_win', 'defendant_win', 'partial')")
    outcome_details: str = Field(description="Detailed description of the outcome")
    damages_awarded: Optional[float] = Field(default=None, description="Amount of damages awarded, if applicable")
    legal_basis: List[str] = Field(default_factory=list, description="Legal provisions used in the decision")
    key_factors: List[str] = Field(default_factory=list, description="Key factors that influenced the decision")
    summary: str = Field(description="Brief summary of the decision")


class CourtStatistics(BaseModel):
    """Model for court decision statistics."""
    
    # General statistics
    total_cases: int = Field(description="Total number of cases analyzed")
    time_period: str = Field(description="Time period covered by the analysis")
    
    # Outcome statistics
    outcome_distribution: Dict[str, float] = Field(
        default_factory=dict, 
        description="Distribution of case outcomes (percentage)"
    )
    average_damages: Optional[float] = Field(
        default=None, 
        description="Average damages awarded (if applicable)"
    )
    
    # Court/Judge statistics
    court_win_rates: Dict[str, float] = Field(
        default_factory=dict, 
        description="Win rates by court (percentage)"
    )
    judge_win_rates: Dict[str, float] = Field(
        default_factory=dict, 
        description="Win rates by judge (percentage)"
    )
    
    # Case type statistics
    case_type_distribution: Dict[str, float] = Field(
        default_factory=dict, 
        description="Distribution of case types (percentage)"
    )
    case_type_outcomes: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, 
        description="Outcome distribution by case type"
    )
    
    # Time-based statistics
    time_trends: Dict[str, List[Tuple[str, float]]] = Field(
        default_factory=dict, 
        description="Trends over time for key metrics"
    )
    
    # Factor analysis
    key_factor_impact: Dict[str, float] = Field(
        default_factory=dict, 
        description="Impact of key factors on case outcomes"
    )


class CasePrediction(BaseModel):
    """Model for case outcome prediction."""
    
    # Case information
    case_type: str = Field(description="Type of case")
    court_name: str = Field(description="Target court for the prediction")
    judge_name: Optional[str] = Field(default=None, description="Target judge for the prediction")
    key_factors: List[str] = Field(default_factory=list, description="Key factors present in the case")
    legal_basis: List[str] = Field(default_factory=list, description="Legal provisions relevant to the case")
    
    # Prediction results
    outcome_probabilities: Dict[str, float] = Field(
        default_factory=dict, 
        description="Probability distribution of possible outcomes"
    )
    most_likely_outcome: str = Field(description="Most likely outcome")
    confidence_level: float = Field(description="Confidence level for the prediction (0.0 to 1.0)")
    
    # Similar cases
    similar_cases: List[str] = Field(default_factory=list, description="IDs of similar cases")
    distinguishing_factors: List[str] = Field(
        default_factory=list, 
        description="Factors that distinguish this case from similar ones"
    )
    
    # Recommendations
    strategic_recommendations: List[str] = Field(
        default_factory=list, 
        description="Strategic recommendations based on the prediction"
    )
    
    # Limitations
    prediction_limitations: List[str] = Field(
        default_factory=list, 
        description="Limitations of the prediction"
    )


class JudicialAnalyticsResult(BaseModel):
    """Model for the complete judicial analytics results."""
    
    # Analysis metadata
    analysis_id: str = Field(description="Unique identifier for this analysis")
    analysis_date: str = Field(description="Date when the analysis was performed")
    query_parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used for the analysis")
    
    # Court decisions
    analyzed_decisions: Optional[List[CourtDecision]] = Field(
        default=None, 
        description="Court decisions that were analyzed"
    )
    
    # Statistics
    statistics: Optional[CourtStatistics] = Field(
        default=None, 
        description="Statistical analysis of court decisions"
    )
    
    # Prediction
    prediction: Optional[CasePrediction] = Field(
        default=None, 
        description="Prediction for a specific case"
    )
    
    # Analysis summary
    summary: str = Field(description="Summary of the analysis results")
    
    # Visualization data
    visualization_data: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Data for visualizing the analysis results"
    )


class JudicialAnalyticsTool(Tool):
    """Tool for analyzing court decisions and predicting case outcomes."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="judicial_analytics_tool")
    description: str = Field(
        default=(
            "Analyzes court decisions to identify trends and patterns, and provides "
            "probabilistic estimates of future judicial decisions. This tool helps legal "
            "professionals develop better litigation strategies and make more informed decisions."
        )
    )
    arguments: list = Field(
        default=[
            ToolArgument(
                name="analysis_type",
                arg_type="string",
                description=(
                    "Type of judicial analysis to perform. Options: 'statistics' (analyze trends and patterns "
                    "in court decisions), 'prediction' (estimate probability of outcomes for a specific case), "
                    "or 'comprehensive' (perform both statistics and prediction)."
                ),
                required=True,
                example="statistics",
            ),
            ToolArgument(
                name="court_decisions",
                arg_type="string",
                description=(
                    "JSON string containing court decisions to analyze. Each decision should include at minimum: "
                    "case_id, court_name, decision_date, case_type, outcome, and legal_basis. "
                    "For small datasets, provide the full text. For large datasets, provide a summary of key information."
                ),
                required=True,
                example='[{"case_id": "2020-1234", "court_name": "Tribunal de Commerce de Paris", "decision_date": "2020-05-15", "case_type": "commercial", "outcome": "plaintiff_win", "legal_basis": ["Article 1134 Code Civil"]}]',
            ),
            ToolArgument(
                name="case_description",
                arg_type="string",
                description=(
                    "Description of the specific case for which to predict outcomes. Required when analysis_type "
                    "is 'prediction' or 'comprehensive'. Should include case facts, legal arguments, and relevant "
                    "circumstances."
                ),
                required=False,
                default="",
                example="Commercial dispute regarding breach of distribution contract with claims of force majeure due to economic crisis.",
            ),
            ToolArgument(
                name="target_court",
                arg_type="string",
                description=(
                    "Name of the court for which to analyze statistics or predict outcomes. "
                    "If not specified, will analyze all courts in the provided decisions."
                ),
                required=False,
                default="",
                example="Tribunal de Commerce de Paris",
            ),
            ToolArgument(
                name="target_judge",
                arg_type="string",
                description=(
                    "Name of the judge for which to analyze statistics or predict outcomes. "
                    "Optional and only used if relevant court decisions with judge information are provided."
                ),
                required=False,
                default="",
                example="Jean Dupont",
            ),
            ToolArgument(
                name="case_type",
                arg_type="string",
                description=(
                    "Type of case to focus on for statistics or prediction. "
                    "If not specified, will analyze all case types in the provided decisions."
                ),
                required=False,
                default="",
                example="commercial",
            ),
            ToolArgument(
                name="time_period",
                arg_type="string",
                description=(
                    "Time period for filtering court decisions (e.g., '2020-2023', 'last_5_years'). "
                    "If not specified, will analyze all provided decisions regardless of date."
                ),
                required=False,
                default="",
                example="2020-2023",
            ),
            ToolArgument(
                name="output_format",
                arg_type="string",
                description=(
                    "Format for the analysis output. Options: 'structured' (returns a structured JSON object), "
                    "'summary' (returns a text summary of key findings), or 'detailed' (returns comprehensive "
                    "analysis with visualizations data). Default is 'structured'."
                ),
                required=False,
                default="structured",
                example="summary",
            ),
            ToolArgument(
                name="language",
                arg_type="string",
                description=(
                    "Language for the analysis output. This should match the language of the input data. "
                    "Defaults to auto-detection."
                ),
                required=False,
                default="auto",
                example="French",
            ),
            ToolArgument(
                name="confidence_threshold",
                arg_type="string",
                description=(
                    "Minimum confidence level (0.0 to 1.0) required for including predictions in the output. "
                    "Higher values result in more conservative predictions. Default is 0.6."
                ),
                required=False,
                default="0.6",
                example="0.7",
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
        name: str = "judicial_analytics_tool",
        generative_model: GenerativeModel | None = None,
        event_emitter: EventEmitter | None = None,
    ):
        """Initialize the JudicialAnalyticsTool with model configuration and optional callback.

        Args:
            model_name (str): The name of the language model to use.
            system_prompt (str, optional): Default system prompt for the model.
            on_token (Callable, optional): Callback function for streaming tokens.
            name (str): Name of the tool instance. Defaults to "judicial_analytics_tool".
            generative_model (GenerativeModel, optional): Pre-initialized generative model.
            event_emitter (EventEmitter, optional): Event emitter for streaming tokens.
        """
        # Default system prompt for judicial analytics if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an expert legal analyst specializing in judicial analytics and predictive modeling "
                "for legal outcomes. Your expertise includes analyzing court decisions to identify patterns, "
                "calculating statistical trends in judicial decision-making, and providing probabilistic "
                "estimates of case outcomes based on historical data. You have extensive knowledge of various "
                "legal systems, particularly French and EU law. You excel at identifying key factors that influence "
                "judicial decisions and providing strategic recommendations based on data-driven insights. "
                "Your analyses are thorough, objective, and always include appropriate caveats about the "
                "limitations of predictive modeling in legal contexts."
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
            logger.debug(f"Initialized JudicialAnalyticsTool with model: {self.model_name}")

        # Only set up event listener if on_token is provided
        if self.on_token is not None:
            logger.debug(f"Setting up event listener for JudicialAnalyticsTool with model: {self.model_name}")
            self.generative_model.event_emitter.on("stream_chunk", self.on_token)
            
    def _build_prompt(
        self,
        analysis_type: str,
        court_decisions: str,
        case_description: str = "",
        target_court: str = "",
        target_judge: str = "",
        case_type: str = "",
        time_period: str = "",
        output_format: str = "structured",
        language: str = "auto",
        confidence_threshold: str = "0.6",
    ) -> str:
        """Build a comprehensive prompt for judicial analytics.
        
        Args:
            analysis_type: Type of judicial analysis to perform
            court_decisions: JSON string containing court decisions to analyze
            case_description: Description of the specific case for prediction
            target_court: Name of the court to focus on
            target_judge: Name of the judge to focus on
            case_type: Type of case to focus on
            time_period: Time period for filtering court decisions
            output_format: Format for the analysis output
            language: Language for the analysis output
            confidence_threshold: Minimum confidence level for predictions
            
        Returns:
            str: Comprehensive prompt for judicial analytics
        """
        # Determine language for instructions
        use_french = language.lower() in ["french", "français", "fr"]
        
        # Build the prompt with appropriate language
        if use_french:
            prompt = """# Analyse de Jurimétrie et Prédiction Judiciaire

## Objectif
"""
            if analysis_type.lower() == "statistics":
                prompt += "Réaliser une analyse statistique complète des décisions de justice fournies pour identifier les tendances et les modèles de décision."
            elif analysis_type.lower() == "prediction":
                prompt += "Fournir une estimation probabiliste des résultats possibles pour le cas spécifique décrit, basée sur l'analyse des décisions de justice similaires."
            else:  # comprehensive
                prompt += "Réaliser une analyse statistique complète des décisions de justice fournies ET fournir une estimation probabiliste des résultats possibles pour le cas spécifique décrit."
                
            prompt += """

## Paramètres d'analyse
"""
        else:
            prompt = """# Judicial Analytics and Prediction Analysis

## Objective
"""
            if analysis_type.lower() == "statistics":
                prompt += "Perform a comprehensive statistical analysis of the provided court decisions to identify trends and decision patterns."
            elif analysis_type.lower() == "prediction":
                prompt += "Provide a probabilistic estimate of possible outcomes for the specific case described, based on analysis of similar court decisions."
            else:  # comprehensive
                prompt += "Perform a comprehensive statistical analysis of the provided court decisions AND provide a probabilistic estimate of possible outcomes for the specific case described."
                
            prompt += """

## Analysis Parameters
"""

        # Add analysis parameters
        if use_french:
            if target_court:
                prompt += f"**Tribunal cible**: {target_court}\n"
            if target_judge:
                prompt += f"**Juge cible**: {target_judge}\n"
            if case_type:
                prompt += f"**Type d'affaire**: {case_type}\n"
            if time_period:
                prompt += f"**Période**: {time_period}\n"
            prompt += f"**Seuil de confiance**: {confidence_threshold}\n\n"
        else:
            if target_court:
                prompt += f"**Target Court**: {target_court}\n"
            if target_judge:
                prompt += f"**Target Judge**: {target_judge}\n"
            if case_type:
                prompt += f"**Case Type**: {case_type}\n"
            if time_period:
                prompt += f"**Time Period**: {time_period}\n"
            prompt += f"**Confidence Threshold**: {confidence_threshold}\n\n"
        
        # Add case description if applicable
        if case_description and (analysis_type.lower() in ["prediction", "comprehensive"]):
            if use_french:
                prompt += f"""## Description du cas à analyser
{case_description}

"""
            else:
                prompt += f"""## Case Description for Prediction
{case_description}

"""
        
        # Add analysis instructions
        if use_french:
            if analysis_type.lower() in ["statistics", "comprehensive"]:
                prompt += """## Analyse statistique
Pour l'ensemble des décisions de justice fournies:
1. Calculez la distribution des résultats (pourcentage de victoires pour le demandeur, le défendeur, etc.)
2. Identifiez les taux de succès par tribunal et par juge
3. Analysez la distribution des types d'affaires et leurs résultats respectifs
4. Identifiez les tendances temporelles dans les décisions
5. Déterminez l'impact des facteurs clés sur les résultats des affaires
6. Calculez les montants moyens des dommages-intérêts accordés, le cas échéant

"""
            
            if analysis_type.lower() in ["prediction", "comprehensive"]:
                prompt += """## Prédiction judiciaire
Pour le cas spécifique décrit:
1. Identifiez les décisions de justice similaires dans l'ensemble de données
2. Calculez la distribution de probabilité des résultats possibles
3. Déterminez le résultat le plus probable et le niveau de confiance associé
4. Identifiez les facteurs distinctifs qui pourraient influencer le résultat
5. Formulez des recommandations stratégiques basées sur l'analyse
6. Expliquez clairement les limites de la prédiction

IMPORTANT: Les prédictions judiciaires ne sont jamais certaines. Incluez toujours des mises en garde appropriées concernant les limites de l'analyse prédictive dans le contexte juridique.

"""
        else:
            if analysis_type.lower() in ["statistics", "comprehensive"]:
                prompt += """## Statistical Analysis
For the provided court decisions:
1. Calculate the distribution of outcomes (percentage of wins for plaintiff, defendant, etc.)
2. Identify success rates by court and by judge
3. Analyze the distribution of case types and their respective outcomes
4. Identify temporal trends in decisions
5. Determine the impact of key factors on case outcomes
6. Calculate average damages awarded, if applicable

"""
            
            if analysis_type.lower() in ["prediction", "comprehensive"]:
                prompt += """## Judicial Prediction
For the specific case described:
1. Identify similar court decisions in the dataset
2. Calculate the probability distribution of possible outcomes
3. Determine the most likely outcome and associated confidence level
4. Identify distinguishing factors that might influence the outcome
5. Formulate strategic recommendations based on the analysis
6. Clearly explain the limitations of the prediction

IMPORTANT: Judicial predictions are never certain. Always include appropriate caveats regarding the limitations of predictive analysis in the legal context.

"""
        
        # Add output format instructions
        if use_french:
            if output_format.lower() == "structured":
                prompt += """## Format de sortie
Fournissez une analyse structurée au format JSON avec les sections suivantes:
- analysis_id: identifiant unique pour cette analyse
- analysis_date: date de l'analyse
- query_parameters: paramètres utilisés pour l'analyse
- analyzed_decisions: décisions de justice analysées (si applicable)
- statistics: analyse statistique des décisions (si applicable)
- prediction: prédiction pour le cas spécifique (si applicable)
- summary: résumé des résultats de l'analyse
- visualization_data: données pour la visualisation des résultats (si applicable)
"""
            elif output_format.lower() == "summary":
                prompt += """## Format de sortie
Fournissez un résumé textuel concis des principales conclusions de l'analyse, sans les détails techniques ou les données brutes.
"""
            else:  # detailed
                prompt += """## Format de sortie
Fournissez une analyse détaillée au format texte avec les sections suivantes:
1. Résumé de l'analyse
2. Méthodologie utilisée
3. Résultats statistiques détaillés (si applicable)
4. Prédiction détaillée avec justification (si applicable)
5. Visualisations recommandées (descriptions et données)
6. Limites et mises en garde
7. Recommandations stratégiques
"""
        else:
            if output_format.lower() == "structured":
                prompt += """## Output Format
Provide a structured analysis in JSON format with the following sections:
- analysis_id: unique identifier for this analysis
- analysis_date: date of the analysis
- query_parameters: parameters used for the analysis
- analyzed_decisions: court decisions that were analyzed (if applicable)
- statistics: statistical analysis of decisions (if applicable)
- prediction: prediction for the specific case (if applicable)
- summary: summary of the analysis results
- visualization_data: data for visualizing the results (if applicable)
"""
            elif output_format.lower() == "summary":
                prompt += """## Output Format
Provide a concise textual summary of the key findings from the analysis, without technical details or raw data.
"""
            else:  # detailed
                prompt += """## Output Format
Provide a detailed analysis in text format with the following sections:
1. Analysis Summary
2. Methodology Used
3. Detailed Statistical Results (if applicable)
4. Detailed Prediction with Justification (if applicable)
5. Recommended Visualizations (descriptions and data)
6. Limitations and Caveats
7. Strategic Recommendations
"""
        
        # Add court decisions data
        if use_french:
            prompt += "\n## Décisions de justice à analyser\n"
        else:
            prompt += "\n## Court Decisions for Analysis\n"
            
        prompt += court_decisions
        
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
                    return JudicialAnalyticsResult(**result).model_dump()
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {json_str}")
                    return {"error": "Invalid JSON in response", "raw_response": response}
                except Exception as e:
                    logger.error(f"Error validating response against JudicialAnalyticsResult model: {e}")
                    return {"error": f"Validation error: {str(e)}", "raw_response": response}
            
            elif output_format.lower() == "summary":
                # For summary output, return the raw text
                return response
            
            else:  # detailed or any other format
                # Return the raw response for detailed output
                return response
                
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {"error": f"Failed to parse response: {str(e)}", "raw_response": response}

    async def execute(
        self,
        analysis_type: str,
        court_decisions: str,
        case_description: str = "",
        target_court: str = "",
        target_judge: str = "",
        case_type: str = "",
        time_period: str = "",
        output_format: str = "structured",
        language: str = "auto",
        confidence_threshold: str = "0.6",
    ) -> Union[Dict[str, Any], str]:
        """Execute the judicial analytics analysis.
        
        Args:
            analysis_type: Type of judicial analysis to perform
            court_decisions: JSON string containing court decisions to analyze
            case_description: Description of the specific case for prediction
            target_court: Name of the court to focus on
            target_judge: Name of the judge to focus on
            case_type: Type of case to focus on
            time_period: Time period for filtering court decisions
            output_format: Format for the analysis output
            language: Language for the analysis output
            confidence_threshold: Minimum confidence level for predictions
            
        Returns:
            Union[Dict[str, Any], str]: Analysis results in the requested format
        """
        try:
            # Validate analysis_type
            if analysis_type.lower() not in ["statistics", "prediction", "comprehensive"]:
                raise ValueError(f"Invalid analysis_type: {analysis_type}. Must be 'statistics', 'prediction', or 'comprehensive'.")
            
            # Validate court_decisions
            try:
                # Check if court_decisions is valid JSON
                json.loads(court_decisions)
            except json.JSONDecodeError:
                raise ValueError("court_decisions must be a valid JSON string.")
            
            # Validate case_description for prediction
            if analysis_type.lower() in ["prediction", "comprehensive"] and not case_description:
                raise ValueError("case_description is required for 'prediction' or 'comprehensive' analysis.")
            
            # Validate confidence_threshold
            try:
                conf_threshold = float(confidence_threshold)
                if not 0.0 <= conf_threshold <= 1.0:
                    raise ValueError("confidence_threshold must be between 0.0 and 1.0.")
            except ValueError:
                raise ValueError(f"Invalid confidence_threshold: {confidence_threshold}. Must be a number between 0.0 and 1.0.")
            
            # Build the prompt
            prompt = self._build_prompt(
                analysis_type=analysis_type,
                court_decisions=court_decisions,
                case_description=case_description,
                target_court=target_court,
                target_judge=target_judge,
                case_type=case_type,
                time_period=time_period,
                output_format=output_format,
                language=language,
                confidence_threshold=confidence_threshold,
            )
            
            # Log the prompt for debugging
            logger.debug(f"Judicial analytics prompt: {prompt[:500]}...")
            
            # Create messages for the model
            messages = [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=prompt),
            ]
            
            # Generate the response
            response = await self.generative_model.generate_content_async(
                messages=messages,
                temperature=0.2,  # Lower temperature for more factual analysis
            )
            
            # Extract the text from the response
            response_text = response.text
            
            # Parse the response based on the requested output format
            result = self._parse_response(response_text, output_format)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in judicial analytics: {e}")
            return {"error": f"Failed to complete judicial analytics: {str(e)}"}
    
    def execute_sync(
        self,
        analysis_type: str,
        court_decisions: str,
        case_description: str = "",
        target_court: str = "",
        target_judge: str = "",
        case_type: str = "",
        time_period: str = "",
        output_format: str = "structured",
        language: str = "auto",
        confidence_threshold: str = "0.6",
    ) -> Union[Dict[str, Any], str]:
        """Synchronous wrapper for the execute method.
        
        Args:
            analysis_type: Type of judicial analysis to perform
            court_decisions: JSON string containing court decisions to analyze
            case_description: Description of the specific case for prediction
            target_court: Name of the court to focus on
            target_judge: Name of the judge to focus on
            case_type: Type of case to focus on
            time_period: Time period for filtering court decisions
            output_format: Format for the analysis output
            language: Language for the analysis output
            confidence_threshold: Minimum confidence level for predictions
            
        Returns:
            Union[Dict[str, Any], str]: Analysis results in the requested format
        """
        return asyncio.run(self.execute(
            analysis_type=analysis_type,
            court_decisions=court_decisions,
            case_description=case_description,
            target_court=target_court,
            target_judge=target_judge,
            case_type=case_type,
            time_period=time_period,
            output_format=output_format,
            language=language,
            confidence_threshold=confidence_threshold,
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
        markdown += "2. **summary**: A concise textual summary of key findings\n"
        markdown += "3. **detailed**: A comprehensive text analysis with all sections\n\n"
        
        markdown += "## Important Notes\n\n"
        markdown += "- Judicial predictions are probabilistic and should be interpreted with caution\n"
        markdown += "- The quality of predictions depends on the quantity and quality of input data\n"
        markdown += "- This tool is designed to support legal decision-making, not replace professional judgment\n\n"
        
        markdown += "## Example Usage\n\n"
        markdown += "```python\n"
        markdown += "analytics_tool = JudicialAnalyticsTool(model_name=\"openrouter/openai/gpt-4o-mini\")\n\n"
        markdown += "# Statistical analysis of court decisions\n"
        markdown += "stats_result = analytics_tool.execute_sync(\n"
        markdown += "    analysis_type=\"statistics\",\n"
        markdown += "    court_decisions=json.dumps(court_decisions_data),\n"
        markdown += "    target_court=\"Tribunal de Commerce de Paris\",\n"
        markdown += "    case_type=\"commercial\",\n"
        markdown += "    output_format=\"structured\",\n"
        markdown += "    language=\"French\"\n"
        markdown += ")\n\n"
        markdown += "# Prediction for a specific case\n"
        markdown += "prediction_result = analytics_tool.execute_sync(\n"
        markdown += "    analysis_type=\"prediction\",\n"
        markdown += "    court_decisions=json.dumps(court_decisions_data),\n"
        markdown += "    case_description=\"Commercial dispute regarding breach of contract with force majeure claims\",\n"
        markdown += "    target_court=\"Tribunal de Commerce de Paris\",\n"
        markdown += "    output_format=\"summary\",\n"
        markdown += "    confidence_threshold=\"0.7\"\n"
        markdown += ")\n"
        markdown += "```\n"
        
        return markdown
