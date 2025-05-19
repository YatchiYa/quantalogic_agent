#!/usr/bin/env python3
"""
Example script demonstrating the use of the Legal Case Triage Tool.
This tool automatically classifies legal documents, identifies key legal issues,
and generates preliminary analysis memos.
"""

import json
from loguru import logger

from quantalogic.tools.utilities.legal_case_triage_tool import LegalCaseTriageTool


def main():
    """Run a demonstration of the Legal Case Triage Tool."""
    # Configure logging
    logger.info("Starting Legal Case Triage Tool demonstration")
    
    # Initialize the tool with the model of your choice
    triage_tool = LegalCaseTriageTool(model_name="openrouter/openai/gpt-4o-mini")
    
    # Example case documents (in a real scenario, these would be loaded from files)
    sample_documents = """Contrat de Distribution.pdf
CONTRAT DE DISTRIBUTION EXCLUSIVE

Entre les soussignés :
La société XYZ, société par actions simplifiée au capital de 100.000 euros, dont le siège social est situé 123 Avenue des Champs-Élysées, 75008 Paris, immatriculée au RCS de Paris sous le numéro 123 456 789, représentée par M. Jean Dupont, en sa qualité de Président, dûment habilité aux fins des présentes,
Ci-après dénommée le "Fournisseur",
D'UNE PART,

ET

La société ABC, société à responsabilité limitée au capital de 50.000 euros, dont le siège social est situé 45 Rue du Commerce, 69002 Lyon, immatriculée au RCS de Lyon sous le numéro 987 654 321, représentée par Mme Marie Martin, en sa qualité de Gérante, dûment habilitée aux fins des présentes,
Ci-après dénommée le "Distributeur",
D'AUTRE PART,

IL A ÉTÉ CONVENU CE QUI SUIT :

Article 1 - Objet du contrat
Le Fournisseur confère au Distributeur, qui l'accepte, le droit exclusif de distribuer les produits décrits en Annexe 1 (ci-après les "Produits") sur le territoire défini à l'article 2 (ci-après le "Territoire").

Article 2 - Territoire
Le Territoire sur lequel le Distributeur bénéficie de l'exclusivité est le suivant : France métropolitaine.

Article 3 - Durée
Le présent contrat est conclu pour une durée initiale de trois (3) ans à compter de sa date de signature. Il se renouvellera ensuite par tacite reconduction pour des périodes successives d'un (1) an, sauf dénonciation par l'une des parties moyennant un préavis de trois (3) mois avant l'échéance.

Article 8 - Résiliation
Le présent contrat pourra être résilié par l'une ou l'autre des parties en cas de manquement grave de l'autre partie à l'une quelconque de ses obligations, non réparé dans un délai de trente (30) jours à compter de la réception d'une lettre recommandée avec accusé de réception notifiant ledit manquement.

Article 12 - Loi applicable et juridiction compétente
Le présent contrat est soumis au droit français. Tout litige relatif à son interprétation ou à son exécution sera de la compétence exclusive du Tribunal de Commerce de Paris.

Fait à Paris, le 15 janvier 2023, en deux exemplaires originaux.

Pour le Fournisseur                           Pour le Distributeur
Jean Dupont                                   Marie Martin
Président                                     Gérante
---DOCUMENT---
Mise en Demeure.pdf
LETTRE RECOMMANDÉE AVEC ACCUSÉ DE RÉCEPTION

ABC SARL
45 Rue du Commerce
69002 Lyon

À l'attention de Mme Marie Martin, Gérante

Paris, le 10 avril 2025

Objet : Mise en demeure - Manquement à vos obligations contractuelles

Madame,

Je vous écris en ma qualité de conseil de la société XYZ SAS, dont le siège social est situé 123 Avenue des Champs-Élysées, 75008 Paris.

Ma cliente m'a chargé de vous adresser la présente mise en demeure concernant vos manquements graves et répétés à vos obligations contractuelles découlant du contrat de distribution exclusive conclu entre nos clients respectifs en date du 15 janvier 2023.

En effet, il apparaît que votre société n'a pas respecté les objectifs minimaux de vente définis à l'article 5 du contrat pour les deux derniers trimestres. De plus, ma cliente a constaté que vous commercialisez des produits concurrents en violation de la clause d'exclusivité prévue à l'article 4 du contrat.

Ces manquements constituent des violations substantielles du contrat qui causent un préjudice important à ma cliente, tant en termes de perte de chiffre d'affaires que d'atteinte à l'image de marque de ses produits.

Par conséquent, je vous mets en demeure, au nom et pour le compte de ma cliente, de :
1. Cesser immédiatement la commercialisation de tout produit concurrent aux Produits ;
2. Mettre en œuvre tous les moyens nécessaires pour atteindre les objectifs minimaux de vente pour le trimestre en cours ;
3. Verser à ma cliente la somme de 45.000 euros au titre des pénalités contractuelles prévues à l'article 9 du contrat.

À défaut de régularisation complète de votre situation dans un délai de trente (30) jours à compter de la réception de la présente, ma cliente se réserve le droit de résilier le contrat conformément à son article 8 et d'engager toute procédure judiciaire à votre encontre en vue d'obtenir réparation de son préjudice.

Je vous prie d'agréer, Madame, l'expression de mes salutations distinguées.

Pierre Avocat
Avocat à la Cour
---DOCUMENT---
Réponse à Mise en Demeure.pdf
LETTRE RECOMMANDÉE AVEC ACCUSÉ DE RÉCEPTION

Maître Pierre Avocat
Cabinet d'Avocats
10 Rue de la Paix
75002 Paris

Lyon, le 25 avril 2025

Objet : Réponse à votre mise en demeure du 10 avril 2025

Maître,

J'accuse réception de votre courrier du 10 avril 2025 adressé à ma cliente, la société ABC SARL, et vous réponds en ma qualité de conseil.

Votre mise en demeure appelle de notre part les observations suivantes :

1. Concernant les objectifs minimaux de vente
Les objectifs fixés à l'article 5 du contrat sont devenus manifestement irréalistes en raison de la crise économique qui affecte actuellement le marché. L'article 5.3 du contrat prévoit expressément une révision des objectifs en cas de "changement significatif des conditions de marché", ce qui est précisément le cas en l'espèce.

Ma cliente a d'ailleurs alerté votre cliente de cette situation par courriers des 15 janvier et 28 février 2025, restés sans réponse.

2. Concernant la prétendue violation de la clause d'exclusivité
Les produits commercialisés par ma cliente ne sont pas concurrents des Produits au sens de l'article 4.2 du contrat, qui définit les produits concurrents comme ceux "ayant la même fonction et visant la même clientèle". Or, les produits distribués par ma cliente visent une clientèle différente et présentent des fonctionnalités distinctes.

3. Concernant les pénalités contractuelles
L'article 9 du contrat prévoit que les pénalités ne sont applicables qu'en cas de manquement "délibéré et injustifié", ce qui n'est manifestement pas le cas en l'espèce.

Pour ces raisons, nous contestons formellement les manquements allégués et rejetons vos demandes.

Ma cliente souhaite néanmoins maintenir des relations commerciales constructives avec votre cliente et propose l'organisation d'une réunion dans les meilleurs délais afin de discuter d'une révision des objectifs de vente adaptée à la situation actuelle du marché.

À défaut d'accord amiable, ma cliente se réserve le droit de saisir le Tribunal de Commerce compétent d'une demande de révision judiciaire du contrat pour imprévision, conformément à l'article 1195 du Code civil.

Je vous prie d'agréer, Maître, l'expression de mes salutations distinguées.

Sophie Legrand
Avocate à la Cour"""

    # Run the tool with structured output
    structured_result = triage_tool.execute_sync(
        case_documents=sample_documents,
        case_description="Litige commercial concernant un contrat de distribution exclusive",
        jurisdiction="France",
        legal_domain="commercial",
        output_format="structured",
        language="French"
    )
    
    # Print the structured results
    print("\n=== STRUCTURED ANALYSIS RESULTS ===")
    print(f"Case Title: {structured_result.get('case_title', 'N/A')}")
    print(f"Case Type: {structured_result.get('case_type', 'N/A')}")
    print(f"Complexity: {structured_result.get('case_complexity', 'N/A')}")
    print(f"Urgency: {structured_result.get('case_urgency', 'N/A')}")
    
    print("\nKey Legal Issues:")
    for issue in structured_result.get('key_legal_issues', []):
        print(f"- {issue.get('issue_name')}: {issue.get('importance')} importance")
        print(f"  {issue.get('description')[:100]}...")
    
    print("\nApplicable Laws:")
    for law in structured_result.get('applicable_laws', [])[:5]:
        print(f"- {law}")
    
    print("\nNext Steps:")
    for step in structured_result.get('next_steps', []):
        print(f"- {step}")
    
    # Run the tool with memo output
    memo_result = triage_tool.execute_sync(
        case_documents=sample_documents,
        case_description="Litige commercial concernant un contrat de distribution exclusive",
        jurisdiction="France",
        legal_domain="commercial",
        output_format="memo",
        language="French"
    )
    
    # Print the memo
    print("\n=== PRELIMINARY ANALYSIS MEMO ===")
    print(memo_result[:500] + "...\n")
    
    # Generate detailed documentation for the tool
    print("\n=== TOOL DOCUMENTATION ===")
    print(triage_tool.to_markdown())


if __name__ == "__main__":
    main()
