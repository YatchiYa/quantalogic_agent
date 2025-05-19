#!/usr/bin/env python3
"""
Example script demonstrating the use of the Judicial Analytics Tool.
This tool analyzes court decisions to identify trends and provides
probabilistic estimates of future judicial decisions.
"""

import json
from loguru import logger

from quantalogic.tools.utilities.judicial_analytics_tool import JudicialAnalyticsTool


def main():
    """Run a demonstration of the Judicial Analytics Tool."""
    # Configure logging
    logger.info("Starting Judicial Analytics Tool demonstration")
    
    # Initialize the tool with the model of your choice
    analytics_tool = JudicialAnalyticsTool(model_name="openrouter/openai/gpt-4o-mini")
    
    # Sample court decisions data (in a real scenario, these would be loaded from a database)
    sample_decisions = [
        {
            "case_id": "2022-1234",
            "court_name": "Tribunal de Commerce de Paris",
            "decision_date": "2022-03-15",
            "case_type": "commercial",
            "judge_name": "Jean Dupont",
            "parties": {"plaintiff": "Société XYZ", "defendant": "Société ABC"},
            "outcome": "plaintiff_win",
            "outcome_details": "Résiliation du contrat pour faute grave avec dommages-intérêts",
            "damages_awarded": 45000.0,
            "legal_basis": ["Article 1217 Code Civil", "Article 1224 Code Civil"],
            "key_factors": ["non-respect des objectifs de vente", "violation de clause d'exclusivité"],
            "summary": "Résiliation d'un contrat de distribution exclusive pour manquement grave aux obligations contractuelles."
        },
        {
            "case_id": "2022-2345",
            "court_name": "Tribunal de Commerce de Paris",
            "decision_date": "2022-05-20",
            "case_type": "commercial",
            "judge_name": "Jean Dupont",
            "parties": {"plaintiff": "Société DEF", "defendant": "Société GHI"},
            "outcome": "defendant_win",
            "outcome_details": "Rejet de la demande de résiliation pour force majeure économique",
            "damages_awarded": 0.0,
            "legal_basis": ["Article 1218 Code Civil", "Article 1195 Code Civil"],
            "key_factors": ["crise économique reconnue", "tentative de renégociation préalable"],
            "summary": "Rejet d'une demande de résiliation de contrat de distribution, la crise économique étant reconnue comme cas de force majeure."
        },
        {
            "case_id": "2022-3456",
            "court_name": "Tribunal de Commerce de Lyon",
            "decision_date": "2022-07-10",
            "case_type": "commercial",
            "judge_name": "Marie Martin",
            "parties": {"plaintiff": "Société JKL", "defendant": "Société MNO"},
            "outcome": "partial",
            "outcome_details": "Révision judiciaire du contrat pour imprévision",
            "damages_awarded": 15000.0,
            "legal_basis": ["Article 1195 Code Civil"],
            "key_factors": ["changement de circonstances imprévisible", "coût d'exécution excessif"],
            "summary": "Révision judiciaire d'un contrat de distribution pour imprévision suite à une hausse imprévisible des coûts."
        },
        {
            "case_id": "2023-1234",
            "court_name": "Tribunal de Commerce de Paris",
            "decision_date": "2023-02-05",
            "case_type": "commercial",
            "judge_name": "Jean Dupont",
            "parties": {"plaintiff": "Société PQR", "defendant": "Société STU"},
            "outcome": "plaintiff_win",
            "outcome_details": "Résiliation du contrat pour faute grave avec dommages-intérêts",
            "damages_awarded": 60000.0,
            "legal_basis": ["Article 1217 Code Civil", "Article 1224 Code Civil"],
            "key_factors": ["non-respect des objectifs de vente", "absence de force majeure"],
            "summary": "Résiliation d'un contrat de distribution pour non-respect des objectifs de vente, l'argument de force majeure économique ayant été rejeté."
        },
        {
            "case_id": "2023-2345",
            "court_name": "Tribunal de Commerce de Lyon",
            "decision_date": "2023-04-18",
            "case_type": "commercial",
            "judge_name": "Marie Martin",
            "parties": {"plaintiff": "Société VWX", "defendant": "Société YZA"},
            "outcome": "defendant_win",
            "outcome_details": "Rejet de la demande de résiliation, révision des objectifs de vente",
            "damages_awarded": 0.0,
            "legal_basis": ["Article 1195 Code Civil", "Article 1104 Code Civil"],
            "key_factors": ["clause de révision des objectifs", "bonne foi dans l'exécution"],
            "summary": "Rejet d'une demande de résiliation de contrat, le tribunal ayant ordonné une révision des objectifs de vente conformément à la clause contractuelle prévue à cet effet."
        }
    ]
    
    # Convert to JSON string
    court_decisions_json = json.dumps(sample_decisions)
    
    # Run statistical analysis
    stats_result = analytics_tool.execute_sync(
        analysis_type="statistics",
        court_decisions=court_decisions_json,
        target_court="Tribunal de Commerce de Paris",
        case_type="commercial",
        output_format="summary",
        language="French"
    )
    
    # Print the statistical analysis results
    print("\n=== ANALYSE STATISTIQUE DES DÉCISIONS ===")
    print(stats_result)
    
    # Case description for prediction
    case_description = """
    Litige commercial concernant un contrat de distribution exclusive signé en janvier 2022 entre la société Acme (fournisseur) et la société Beta (distributeur).
    
    Le contrat prévoit des objectifs minimaux de vente trimestriels et une clause d'exclusivité interdisant au distributeur de commercialiser des produits concurrents.
    
    Le distributeur n'a pas atteint les objectifs de vente pour les deux derniers trimestres (Q1 et Q2 2025) et invoque une crise économique sectorielle comme cas de force majeure.
    
    Le contrat contient une clause de révision des objectifs en cas de "changement significatif des conditions de marché" (article 5.3) et prévoit que les pénalités ne sont applicables qu'en cas de manquement "délibéré et injustifié" (article 9).
    
    Le distributeur a alerté le fournisseur de la situation économique par courriers en janvier et février 2025, restés sans réponse.
    
    Le fournisseur demande la résiliation du contrat pour faute grave et 45.000 euros de pénalités.
    """
    
    # Run prediction analysis
    prediction_result = analytics_tool.execute_sync(
        analysis_type="prediction",
        court_decisions=court_decisions_json,
        case_description=case_description,
        target_court="Tribunal de Commerce de Paris",
        output_format="detailed",
        language="French",
        confidence_threshold="0.6"
    )
    
    # Print the prediction results
    print("\n=== PRÉDICTION JUDICIAIRE ===")
    print(prediction_result)
    
    # Generate tool documentation
    print("\n=== DOCUMENTATION DE L'OUTIL ===")
    print(analytics_tool.to_markdown())


if __name__ == "__main__":
    main()
