#!/usr/bin/env python3

import asyncio
from context_analyser import analyze_context_with_instructions
from rich.console import Console

console = Console()

async def main():
    # Simple test context and instructions
    test_context = """
    Context 1: Facture_infos a extraire.pdf
### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Contrat n° 003503 du 25/10/2017
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
34 RUE DES CAYENNES
78700 CONFLANS STE HONORINE
RICOH MP301SP
Matricule : W907P601853
Compteur NB
Estimation au 31/03/2025 : 49428
Dernier facturé le 31/12/2024 : 48362
.

000005 FA4N Format A4 noir 1066 0.00731 7.79 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
64 RUE DES HAUTES RAYES
78700 CONFLANS STE HONORINE
RICOH MP301SP
Matricule : W907P601861
Compteur NB
Estimation au 31/03/2025 : 61792
Dernier facturé le 31/12/2024 : 59664
.

000005 FA4N Format A4 noir 2128 0.00731 15.56 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 1 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

1 RUE DE L'ANCIENNE MAIRIE
95490 VAUREAL
RICOH MP301SP
Matricule : W907P601862
Compteur NB
Relevé machine au 24/03/2025 : 111260
Dernier facturé le 27/12/2024 : 109754
.

000005 FA4N Format A4 noir 1506 0.00731 11.01 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
64 RUE DES HAUTES RAYES
78700 CONFLANS STE HONORINE
RICOH MP301SP
Matricule : W907P601865
Compteur NB
Estimation au 31/03/2025 : 61760
Dernier facturé le 31/12/2024 : 60669
.

000005 FA4N Format A4 noir 1091 0.00731 7.98 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
34 RUE DES CAYENNES
78700 CONFLANS STE HONORINE
RICOH MP301SP
Matricule : W907P700081

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 2 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Compteur NB
Estimation au 31/03/2025 : 44068
Dernier facturé le 31/12/2024 : 41019
.

000005 FA4N Format A4 noir 3049 0.00658 20.06 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
ASSOCIATION EQUALIS
11 RUE DES MARAICHERS
78260 ACHERES
RICOH MP2501SP
Matricule : E337M421051
Compteur NB
Relevé machine au 24/03/2025 : 129163
Dernier facturé le 27/12/2024 : 128681
.

000005 FA4N Format A4 noir 482 0.00731 3.52 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
7 RUE DESIRE CLEMENT
78700 CONFLANS STE HONORINE
RICOH MP2501SP
Matricule : E337M420949
Compteur NB
Relevé machine au 24/03/2025 : 205294
Dernier facturé le 31/12/2024 : 202221
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 3 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

000005 FA4N Format A4 noir 3073 0.00731 22.46 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
400 CHEMIN DE CRECY
CS 50278 MAREUIL LES MEAUX
77334 MEAUX CEDEX
RICOH MP2501SP
Matricule : E337M421042
Compteur NB
Estimation au 31/03/2025 : 146812
Dernier facturé le 15/01/2025 : 146797
.

000005 FA4N Format A4 noir 15 0.00667 0.10 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
64 RUE DES HAUTES RAYES
78700 CONFLANS STE HONORINE
RICOH MP2501SP
Matricule : E337M420957
Compteur NB
Estimation au 31/03/2025 : 98591
Dernier facturé le 27/12/2024 : 95485
.

000005 FA4N Format A4 noir 3106 0.00731 22.70 N
.

Contrat n° 003504 du 25/10/2017

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 4 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
400 CHEMIN DE CRECY
CS 50278 MAREUIL LES MEAUX
77334 MEAUX CEDEX
RICOH MP2555SP
Matricule : C397P701041
Compteur NB
Estimation au 31/03/2025 : 375756
Dernier facturé le 15/01/2025 : 373620
.

000005 FA4N Format A4 noir 2136 0.00731 15.61 N
.

Contrat n° 003505 du 25/10/2017
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
1 RUE GABRIEL PERI
78700 CONFLANS STE HONORINE
RICOH MPC3504EXSP
Matricule : C727R610047
Compteur NB
Relevé machine au 24/03/2025 : 116101
Dernier facturé le 27/12/2024 : 115014
.

000005 FA4N Format A4 noir 1087 0.00683 7.42 N
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 5 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 243520
Dernier facturé le 27/12/2024 : 242443
.

000008 FA4C Format A4 Couleur 1077 0.06815 73.40 N
.
Contrat n° 003601 du 08/12/2017
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
72 RUE DESIRE CLEMENT
78700 CONFLANS STE HONORINE
RICOH MPC3003SP
Matricule : E153M931250
Compteur NB
Estimation au 31/03/2025 : 245170
Dernier facturé le 15/01/2025 : 244975
.

000005 FA4N Format A4 noir 195 0.00731 1.43 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Estimation au 31/03/2025 : 247642
Dernier facturé le 15/01/2025 : 247092
.

000008 FA4C Format A4 Couleur 550 0.07325 40.29 N
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 6 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Contrat n° 003802 du 23/03/2018
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
1 RUE RONSARD
77100 MAREUIL LES MEAUX
RICOH MPC3003SP
Matricule : E155M530951
Compteur NB
Relevé machine au 24/03/2025 : 120919
Dernier facturé le 27/12/2024 : 119858
.

000005 FA4N Format A4 noir 1061 0.00756 8.02 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 161247
Dernier facturé le 27/12/2024 : 158071
.

000008 FA4C Format A4 Couleur 3176 0.07588 240.99 N
.
Contrat n° 003807 du 26/03/2018
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
17 BOULEVARD DE LA MALIBRAN
PARKING: ENTREE 2511# - SORTIE 7768#
77680 ROISSY EN BRIE

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 7 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

RICOH MPC3004SP
Matricule : G696MA30836
Compteur NB
Relevé machine au 24/03/2025 : 38399
Dernier facturé le 27/12/2024 : 37698
.

000005 FA4N Format A4 noir 701 0.00756 5.30 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 63834
Dernier facturé le 27/12/2024 : 60900
.

000008 FA4C Format A4 Couleur 2934 0.07588 222.63 N
.

Contrat n° 003813 du 21/03/2018
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
21 RUE JACQUARD
77400 LAGNY SUR MARNE
RICOH MPC3003SP
Matricule : E153M931273
Compteur COULEUR
Estimation au 31/03/2025 : 202148
Dernier facturé le 15/01/2025 : 202120
.

000008 FA4C Format A4 Couleur 28 0.07588 2.12 N

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 8 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

.
Contrat n° 003814 du 28/03/2018
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
18 AVENUE DU GENERAL DE GAULLE
77140 NEMOURS
RICOH MPC3004SP
Matricule : G696MA30821
Compteur NB
Relevé machine au 24/03/2025 : 60454
Dernier facturé le 27/12/2024 : 58686
.

000005 FA4N Format A4 noir 1768 0.00756 13.37 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 88955
Dernier facturé le 27/12/2024 : 83097
.

000008 FA4C Format A4 Couleur 5858 0.07588 444.51 N
.
Contrat n° 004192 du 21/01/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR) - I002
9BIS QUAI CONTI
78430 LOUVECIENNES

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 9 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

RICOH MPC2504EXSP
Matricule : C778R820078
Compteur NB
Relevé machine au 24/03/2025 : 119942
Dernier facturé le 27/12/2024 : 116905
.

000005 FA4N Format A4 noir 3037 0.00717 21.78 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 251921
Dernier facturé le 27/12/2024 : 244545
.

000008 FA4C Format A4 Couleur 7376 0.07172 529.01 N
.
Contrat n° 004193 du 21/01/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
17 RUE SAINT VINCENT
78100 ST GERMAIN EN LAYE
RICOH MPC2504EXSP
Matricule : C778R820077
Compteur NB
Relevé machine au 24/03/2025 : 63105
Dernier facturé le 27/12/2024 : 61438
.

000005 FA4N Format A4 noir 1667 0.00717 11.95 N

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 10 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 127288
Dernier facturé le 27/12/2024 : 118854
.

000008 FA4C Format A4 Couleur 8434 0.07172 604.89 N
.
Contrat n° 004194 du 22/01/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
39 BOULEVARD DE LAGNY
77600 BUSSY ST GEORGES
RICOH MPC2504EXSP
Matricule : C778R820069
Compteur NB
Relevé machine au 24/03/2025 : 44955
Dernier facturé le 27/12/2024 : 44110
.

000005 FA4N Format A4 noir 845 0.00717 6.06 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 115006
Dernier facturé le 27/12/2024 : 110769
.

000008 FA4C Format A4 Couleur 4237 0.07172 303.88 N

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 11 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

.
Contrat n° 004203 du 25/01/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
12 AVENUE DE LA SOCIETE DES NATIONS
77144 MONTEVRAIN
RICOH MPC2504EXSP
Matricule : C778R820065
Compteur NB
Relevé machine au 24/03/2025 : 72528
Dernier facturé le 30/12/2024 : 71359
.

000005 FA4N Format A4 noir 1169 0.00717 8.38 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 131046
Dernier facturé le 30/12/2024 : 127046
.

000008 FA4C Format A4 Couleur 4000 0.07172 286.88 N
.
Contrat n° 004213 du 30/01/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
18 AVENUE DU GENERAL DE GAULLE
77140 NEMOURS

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 12 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

RICOH MPC2504EXSP
Matricule : C778R820041
Compteur NB
Relevé machine au 24/03/2025 : 123562
Dernier facturé le 27/12/2024 : 122402
.

000005 FA4N Format A4 noir 1160 0.00717 8.32 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 188159
Dernier facturé le 27/12/2024 : 184452
.

000008 FA4C Format A4 Couleur 3707 0.07172 265.87 N
.

Contrat n° 004221 du 06/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
400 CHEMIN DE CRECY
77100 MAREUIL LES MEAUX
RICOH MPC2504EXSP
Matricule : C778R820022
Compteur NB
Relevé machine au 24/03/2025 : 161705
Dernier facturé le 27/12/2024 : 156743
.

000005 FA4N Format A4 noir 4962 0.00717 35.58 N

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 13 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 246506
Dernier facturé le 27/12/2024 : 240607
.

000008 FA4C Format A4 Couleur 5899 0.07172 423.08 N
.
Contrat n° 004222 du 31/01/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
400 CHEMIN DE CRECY
77100 MAREUIL LES MEAUX
RICOH MPC2504EXSP
Matricule : C778R820027
Compteur NB
Relevé machine au 24/03/2025 : 79770
Dernier facturé le 27/12/2024 : 78985
.

000005 FA4N Format A4 noir 785 0.00717 5.63 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 135051
Dernier facturé le 27/12/2024 : 133773
.

000008 FA4C Format A4 Couleur 1278 0.07172 91.66 N

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 14 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

.

Contrat n° 004223 du 31/01/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
400 CHEMIN DE CRECY
77100 MAREUIL LES MEAUX
RICOH MPC2504EXSP
Matricule : C778R820029
Compteur NB
Relevé PDA au 27/03/2025 : 60015
Dernier facturé le 27/12/2024 : 58417
.

000005 FA4N Format A4 noir 1598 0.00717 11.46 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé PDA au 27/03/2025 : 129430
Dernier facturé le 27/12/2024 : 127726
.

000008 FA4C Format A4 Couleur 1704 0.07172 122.21 N
.

Contrat n° 004224 du 31/01/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
591 AVENUE SAINT JUST
77000 VAUX LE PENIL

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 15 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

RICOH MPC2504EXSP
Matricule : C778R820035
SECTION 5017 PASH
Compteur NB
Relevé machine au 24/03/2025 : 61432
Dernier facturé le 27/12/2024 : 57338
.

000005 FA4N Format A4 noir 4094 0.00717 29.35 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 85116
Dernier facturé le 27/12/2024 : 80027
.

000008 FA4C Format A4 Couleur 5089 0.07172 364.98 N
.

Contrat n° 004225 du 01/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
2 A RUE D'ORGEMONT
77100 MEAUX
RICOH MPC2504EXSP
Matricule : C778R820049
Compteur NB
Relevé machine au 24/03/2025 : 102720
Dernier facturé le 27/12/2024 : 97693
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 16 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

000005 FA4N Format A4 noir 5027 0.00717 36.04 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 252285
Dernier facturé le 27/12/2024 : 237872
.

000008 FA4C Format A4 Couleur 14413 0.07172 1033.70 N
.
Contrat n° 004226 du 01/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
2 A RUE D'ORGEMONT
77100 MEAUX
RICOH MPC307SPF
Matricule : C508PB01844
Compteur NB
Relevé machine au 24/03/2025 : 74428
Dernier facturé le 27/12/2024 : 70164
.

000005 FA4N Format A4 noir 4264 0.00717 30.57 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 79623
Dernier facturé le 27/12/2024 : 76402
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 17 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

000008 FA4C Format A4 Couleur 3221 0.07172 231.01 N
.
Contrat n° 004228 du 04/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
2A RUE D'ORGEMONT
77100 MEAUX
RICOH MPC307SPF
Matricule : C508PB01871
Compteur NB
Relevé machine au 24/03/2025 : 26180
Dernier facturé le 27/12/2024 : 24700
.

000005 FA4N Format A4 noir 1480 0.00717 10.61 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 27994
Dernier facturé le 27/12/2024 : 25562
.

000008 FA4C Format A4 Couleur 2432 0.07172 174.42 N
.
Contrat n° 004229 du 05/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
3 RUE DE LA CRECHE

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 18 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

SEMNA77
77100 MEAUX
RICOH MPC2504EXSP
Matricule : C778J300399
Compteur NB
Relevé machine au 24/03/2025 : 63576
Dernier facturé le 27/12/2024 : 62181
.

000005 FA4N Format A4 noir 1395 0.00717 10.00 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 179607
Dernier facturé le 27/12/2024 : 169384
.

000008 FA4C Format A4 Couleur 10223 0.07172 733.19 N
.

Contrat n° 004233 du 06/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
29 RUE DE LA CRECHE
77100 MEAUX
RICOH MPC2504EXSP
Matricule : C778R820036
Compteur NB
Relevé machine au 24/03/2025 : 88269
Dernier facturé le 27/12/2024 : 85278

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 19 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

.

000005 FA4N Format A4 noir 2991 0.00717 21.45 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 161578
Dernier facturé le 27/12/2024 : 156004
.

000008 FA4C Format A4 Couleur 5574 0.07172 399.77 N
.
Contrat n° 004234 du 06/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
29 RUE DE LA CRECHE
77100 MEAUX
RICOH MPC2504EXSP
Matricule : C778R820039
2 ème étage
Compteur NB
Relevé machine au 24/03/2025 : 121136
Dernier facturé le 27/12/2024 : 116351
.

000005 FA4N Format A4 noir 4785 0.00717 34.31 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 216013

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 20 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Dernier facturé le 27/12/2024 : 204843
.

000008 FA4C Format A4 Couleur 11170 0.07172 801.11 N
.
Contrat n° 004235 du 06/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
400 CHEMIN DE CRECY
77100 MAREUIL LES MEAUX
RICOH MPC2504EXSP
Matricule : C778R820021
Compteur NB
Relevé PDA au 31/03/2025 : 78971
Dernier facturé le 27/12/2024 : 77347
.

000005 FA4N Format A4 noir 1624 0.00717 11.64 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé PDA au 31/03/2025 : 172697
Dernier facturé le 27/12/2024 : 168255
.

000008 FA4C Format A4 Couleur 4442 0.07172 318.58 N
.
Contrat n° 004236 du 06/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 21 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

EQUALIS (LA ROSE DES VENTS)
400 CHEMIN DE CRECY
77100 MAREUIL LES MEAUX
RICOH MPC2504EXSP
Matricule : C778R820032
Compteur NB
Relevé machine au 24/03/2025 : 163732
Dernier facturé le 27/12/2024 : 163032
.

000005 FA4N Format A4 noir 700 0.00717 5.02 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 102954
Dernier facturé le 27/12/2024 : 102751
.

000008 FA4C Format A4 Couleur 203 0.07172 14.56 N
.

Contrat n° 004243 du 08/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
9 RUE DE LA PLAINE DE LA CROIX BESNARD
77000 VAUX LE PENIL
RICOH MPC2504EXSP
Matricule : C778R820020
Compteur NB
Estimation au 31/03/2025 : 102157

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 22 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Dernier facturé le 27/12/2024 : 97865
.

000005 FA4N Format A4 noir 4292 0.00717 30.77 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Estimation au 31/03/2025 : 86655
Dernier facturé le 27/12/2024 : 83485
.

000008 FA4C Format A4 Couleur 3170 0.07172 227.35 N
.
Contrat n° 004244 du 08/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
CENTRE COMMERCIAL DE LA BUTTE
MONTCEAU
77210 AVON
RICOH MPC2504EXSP
Matricule : C778R820014
Compteur NB
Relevé machine au 24/03/2025 : 74656
Dernier facturé le 27/12/2024 : 73374
.

000005 FA4N Format A4 noir 1282 0.00717 9.19 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 23 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Relevé machine au 24/03/2025 : 193416
Dernier facturé le 27/12/2024 : 187522
.

000008 FA4C Format A4 Couleur 5894 0.07172 422.72 N
.
Contrat n° 004246 du 11/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
400 CHEMIN DE CRECY
77100 MAREUIL LES MEAUX
RICOH MPC2504EXSP
Matricule : C778R820083
Compteur NB
Relevé machine au 24/03/2025 : 41497
Dernier facturé le 27/12/2024 : 39944
.

000005 FA4N Format A4 noir 1553 0.00717 11.14 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 94398
Dernier facturé le 27/12/2024 : 85452
.

000008 FA4C Format A4 Couleur 8946 0.07172 641.61 N
.
Contrat n° 004247 du 11/02/2019
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 24 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
4 PLACE DU CALVAIRE
77130 MONTEREAU FAULT YONNE
RICOH MPC2504EXSP
Matricule : C778R820007
Compteur NB
Relevé machine au 24/03/2025 : 24577
Dernier facturé le 27/12/2024 : 23807
.

000005 FA4N Format A4 noir 770 0.00717 5.52 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 74114
Dernier facturé le 27/12/2024 : 71929
.

000008 FA4C Format A4 Couleur 2185 0.07172 156.71 N
.

Contrat n° 004253 du 12/02/2019
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
EQUALIS
1 RUE RONSARD
77100 MAREUIL LES MEAUX
RICOH MPC307SPF
Matricule : C508PB01813
Compteur COULEUR

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 25 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Estimation au 31/03/2025 : 15108
Dernier facturé le 15/01/2025 : 15101
.

000008 FA4C Format A4 Couleur 7 0.05971 0.42 N
.
Contrat n° 004254 du 04/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
21 RUE JACQUARD
77400 LAGNY SUR MARNE
RICOH MPC2504EXSP
Matricule : C778R820052
Compteur NB
Relevé machine au 24/03/2025 : 44025
Dernier facturé le 27/12/2024 : 42654
.

000005 FA4N Format A4 noir 1371 0.00717 9.83 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 98216
Dernier facturé le 27/12/2024 : 94110
.

000008 FA4C Format A4 Couleur 4106 0.07172 294.48 N
.
Contrat n° 004257 du 13/02/2019
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 26 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
7 AVENUE DU GENERAL DE GAULLE
91090 LISSES
RICOH MPC2504EXSP
Matricule : C778R820009
Compteur NB
Relevé machine au 24/03/2025 : 60734
Dernier facturé le 27/12/2024 : 59759
.

000005 FA4N Format A4 noir 975 0.00717 6.99 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 156538
Dernier facturé le 27/12/2024 : 151848
.

000008 FA4C Format A4 Couleur 4690 0.07172 336.37 N
.

Contrat n° 004271 du 20/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
1 BIS RUE DE VENISE
77124 VILLENOY
RICOH MPC2504EXSP
Matricule : C778R820003
Compteur NB

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 27 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Relevé machine au 24/03/2025 : 23024
Dernier facturé le 27/12/2024 : 22279
.

000005 FA4N Format A4 noir 745 0.00717 5.34 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 28648
Dernier facturé le 27/12/2024 : 26778
.

000008 FA4C Format A4 Couleur 1870 0.07172 134.12 N
.
Contrat n° 004272 du 20/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
591 AVENUE SAINT JUST
77000 VAUX LE PENIL
RICOH MPC2504EXSP
Matricule : C778J600156
centre hébergement
Compteur NB
Relevé machine au 24/03/2025 : 93704
Dernier facturé le 27/12/2024 : 91363
.

000005 FA4N Format A4 noir 2341 0.00717 16.78 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 28 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Compteur COULEUR
Relevé machine au 24/03/2025 : 223188
Dernier facturé le 27/12/2024 : 216682
.

000008 FA4C Format A4 Couleur 6506 0.07172 466.61 N
.
Contrat n° 004273 du 20/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
591 AVENUE SAINT JUST
77000 VAUX LE PENIL
RICOH MPC2504EXSP
Matricule : C778R620143
gestion locative
Compteur NB
Relevé machine au 24/03/2025 : 47788
Dernier facturé le 27/12/2024 : 45581
.

000005 FA4N Format A4 noir 2207 0.00717 15.82 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 133095
Dernier facturé le 27/12/2024 : 124793
.

000008 FA4C Format A4 Couleur 8302 0.07172 595.42 N
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 29 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Contrat n° 004280 du 21/02/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
25 RUE DE BAGNEAUX
45140 ST JEAN DE LA RUELLE
RICOH MPC2504EXSP
Matricule : C778R820005
Compteur NB
Relevé machine au 24/03/2025 : 65552
Dernier facturé le 27/12/2024 : 64486
.

000005 FA4N Format A4 noir 1066 0.00717 7.64 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 134872
Dernier facturé le 27/12/2024 : 129575
.

000008 FA4C Format A4 Couleur 5297 0.07172 379.90 N
.

Contrat n° 004285 du 25/02/2019
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
EQUALIS (ACR)
7-9 RUE DENIS PAPIN
78190 TRAPPES
RICOH MPC2504EXSP

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 30 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Matricule : C778J600159
Compteur COULEUR
Estimation au 31/03/2025 : 175753
Dernier facturé le 31/12/2024 : 174489
.

000008 FA4C Format A4 Couleur 1264 0.07172 90.65 N
.

Contrat n° 004299 du 01/03/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
400 CHEMIN DE CRECY
77100 MAREUIL LES MEAUX
RICOH MPC2504EXSP
Matricule : C778J600157
RDC
Compteur NB
Relevé machine au 24/03/2025 : 118074
Dernier facturé le 27/12/2024 : 116616
.

000005 FA4N Format A4 noir 1458 0.00717 10.45 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 210090
Dernier facturé le 27/12/2024 : 209332
.

000008 FA4C Format A4 Couleur 758 0.07172 54.36 N

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 31 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

.
Contrat n° 004328 du 28/12/2018
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
7 AVENUE DU GENERAL DE GAULLE
91090 LISSES
RICOH MPC2504EXSP
Matricule : C778R820079
Compteur NB
Relevé machine au 24/03/2025 : 56323
Dernier facturé le 15/12/2024 : 56318
.

000005 FA4N Format A4 noir 5 0.00699 0.03 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 124915
Dernier facturé le 15/12/2024 : 124869
.

000008 FA4C Format A4 Couleur 46 0.06983 3.21 N
.
Contrat n° 004410 du 03/06/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
34 RUE DES CAYENNES
78700 CONFLANS STE HONORINE

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 32 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

RICOH MPC2003SP
Matricule : E203RC60012
Compteur NB
Relevé machine au 24/03/2025 : 52389
Dernier facturé le 27/12/2024 : 52118
.

000005 FA4N Format A4 noir 271 0.00654 1.77 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
31 BOULEVARD DE LA PAIX
78100 ST GERMAIN EN LAYE
RICOH MPC2003SP
Matricule : E204R962805
Compteur NB
Relevé machine au 24/03/2025 : 97010
Dernier facturé le 27/12/2024 : 94565
.

000005 FA4N Format A4 noir 2445 0.00654 15.99 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
EQUALIS (ACR)
34 RUE DES CAYENNES
78700 CONFLANS STE HONORINE
RICOH MPC2003SP
Matricule : E203RC60012
Compteur COULEUR
Relevé machine au 24/03/2025 : 113982

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 33 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Dernier facturé le 27/12/2024 : 113134
.

000008 FA4C Format A4 Couleur 848 0.06544 55.49 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
EQUALIS
31 BOULEVARD DE LA PAIX
78100 ST GERMAIN EN LAYE
RICOH MPC2003SP
Matricule : E204R962805
Compteur COULEUR
Relevé machine au 24/03/2025 : 155369
Dernier facturé le 27/12/2024 : 150070
.

000008 FA4C Format A4 Couleur 5299 0.06544 346.77 N
.

Contrat n° 004413 du 05/06/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
221 RUE LA FAYETTE
75010 PARIS
RICOH MPC3004EXASP
Matricule : C718RA30357
Compteur NB
Relevé machine au 24/03/2025 : 28063
Dernier facturé le 27/12/2024 : 26939
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 34 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

000005 FA4N Format A4 noir 1124 0.00654 7.35 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 91214
Dernier facturé le 27/12/2024 : 88146
.

000008 FA4C Format A4 Couleur 3068 0.06544 200.77 N
.
Contrat n° 004483 du 18/07/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
CHEMIN DE LA MARE LORIN
77127 PENCHARD
RICOH MPC3003SP
Matricule : E155M132042
Compteur NB
Relevé machine au 24/03/2025 : 76268
Dernier facturé le 27/12/2024 : 74985
.

000005 FA4N Format A4 noir 1283 0.00786 10.08 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 120989
Dernier facturé le 27/12/2024 : 116572
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 35 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

000008 FA4C Format A4 Couleur 4417 0.07855 346.96 N
.
Contrat n° 004514 du 22/08/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
54 ROUTE DE SARTROUVILLE
78230 LE PECQ
RICOH MPC3003SP
Matricule : E154M330147
Compteur NB
Relevé machine au 24/03/2025 : 140138
Dernier facturé le 27/12/2024 : 137348
.

000005 FA4N Format A4 noir 2790 0.00786 21.93 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 191205
Dernier facturé le 27/12/2024 : 185744
.

000008 FA4C Format A4 Couleur 5461 0.07855 428.96 N
.
Contrat n° 004543 du 16/09/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
54 ROUTE DE SARTROUVILLE

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 36 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

78230 LE PECQ
RICOH MPC3003SP
Matricule : E153MA34306
Compteur NB
Estimation au 31/03/2025 : 64426
Dernier facturé le 25/03/2024 : 64138
.

000005 FA4N Format A4 noir 288 0.00786 2.26 N
.

Contrat n° 004600 du 24/10/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
42 BOULEVARD DE L'ALMONT
77000 MELUN
RICOH MPC307SPF
Matricule : C508PB01824
Compteur NB
Relevé machine au 24/03/2025 : 10782
Dernier facturé le 27/12/2024 : 10347
.

000005 FA4N Format A4 noir 435 0.00654 2.84 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 12782
Dernier facturé le 27/12/2024 : 12239
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 37 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

000008 FA4C Format A4 Couleur 543 0.06544 35.53 N
.
Contrat n° 004603 du 25/10/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
CHU DE PALMA
77 AVENUE GAMBETTA
75020 PARIS
RICOH IM C2500
Matricule : 3099R610284
Compteur NB
Relevé machine au 24/03/2025 : 15116
Dernier facturé le 27/12/2024 : 14340
.

000005 FA4N Format A4 noir 776 0.00742 5.76 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 53166
Dernier facturé le 27/12/2024 : 50566
.

000008 FA4C Format A4 Couleur 2600 0.08256 214.66 N
.
Contrat n° 004671 du 04/12/2019
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
64 RUE DES HAUTES RAYES

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 38 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

78700 CONFLANS STE HONORINE
RICOH MPC3003SP
Matricule : E154M831983
Compteur NB
Relevé machine au 24/03/2025 : 77366
Dernier facturé le 27/12/2024 : 77292
.

000005 FA4N Format A4 noir 74 0.00776 0.57 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 109653
Dernier facturé le 27/12/2024 : 109535
.

000008 FA4C Format A4 Couleur 118 0.07763 9.16 N
.

Contrat n° 004773 du 12/02/2020
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR)
39 RUE DESIRE CLEMENT
78700 CONFLANS STE HONORINE
RICOH IM C2500
Matricule : 3099R610299
Compteur NB
Relevé machine au 24/03/2025 : 36227
Dernier facturé le 31/12/2024 : 34630
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 39 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

000005 FA4N Format A4 noir 1597 0.00808 12.90 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 62114
Dernier facturé le 31/12/2024 : 56721
.

000008 FA4C Format A4 Couleur 5393 0.08068 435.11 N
.
Contrat n° 004789 du 20/02/2020
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
50 BIS ROUTE D'ORLEANS
45380 LA CHAPELLE ST MESMIN
RICOH MPC3003SP
Matricule : E154M831959
Compteur NB
Relevé machine au 24/03/2025 : 118532
Dernier facturé le 27/12/2024 : 117094
.

000005 FA4N Format A4 noir 1438 0.00808 11.62 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 186681
Dernier facturé le 27/12/2024 : 181248
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 40 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

000008 FA4C Format A4 Couleur 5433 0.08068 438.33 N
.
Contrat n° 004809 du 04/03/2020
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
39 BOULEVARD DE LAGNY
77600 BUSSY ST GEORGES
RICOH IMC300F
Matricule : 3929PC02778
Compteur NB
Estimation au 31/03/2025 : 25131
Dernier facturé le 15/01/2025 : 24888
.

000005 FA4N Format A4 noir 243 0.00808 1.96 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Estimation au 31/03/2025 : 67926
Dernier facturé le 15/01/2025 : 65832
.

000008 FA4C Format A4 Couleur 2094 0.08068 168.94 N
.
Contrat n° 004870 du 19/06/2020
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
400 CHEMIN DE CRECY

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 41 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

CS 50278 MAREUIL LES MEAUX
77334 MEAUX CEDEX
RICOH MPC3004EXASP
Matricule : C718RA30353
Compteur NB
Estimation au 31/03/2025 : 6542
Dernier facturé le 15/01/2025 : 6261
.

000005 FA4N Format A4 noir 281 0.00737 2.07 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Estimation au 31/03/2025 : 25824
Dernier facturé le 15/01/2025 : 24715
.

000008 FA4C Format A4 Couleur 1109 0.07361 81.63 N
.

Contrat n° 004885 du 26/06/2020
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
400 CHEMIN DE CRECY
77100 MAREUIL LES MEAUX
RICOH MPC3003SP
Matricule : E153M830571
Compteur COULEUR
Estimation au 31/03/2025 : 129565
Dernier facturé le 31/12/2024 : 125979

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 42 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

.

000008 FA4C Format A4 Couleur 3586 0.07275 260.88 N
.

Contrat n° 004972 du 14/09/2020
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
400 CHEMIN DE CRECY
77100 MAREUIL LES MEAUX
RICOH MPC3003SP
Matricule : E154M832018
Compteur NB
Relevé machine au 24/03/2025 : 99198
Dernier facturé le 27/12/2024 : 98248
.

000005 FA4N Format A4 noir 950 0.00737 7.00 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
32 QUAI HIPPOLYTE ROSSIGNOL
HOTEL DE LA MARINE - SECTION 5017 PASH
77000 MELUN
RICOH MPC3003SP
Matricule : E155M132034
Compteur NB
Relevé machine au 24/03/2025 : 25245
Dernier facturé le 27/12/2024 : 25159
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 43 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

000005 FA4N Format A4 noir 86 0.00737 0.63 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
400 CHEMIN DE CRECY
77100 MAREUIL LES MEAUX
RICOH MPC3003SP
Matricule : E154M832018
Compteur COULEUR
Relevé machine au 24/03/2025 : 143697
Dernier facturé le 27/12/2024 : 139712
.

000008 FA4C Format A4 Couleur 3985 0.07361 293.34 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
EQUALIS
32 QUAI HIPPOLYTE ROSSIGNOL
HOTEL DE LA MARINE - SECTION 5017 PASH
77000 MELUN
RICOH MPC3003SP
Matricule : E155M132034
Compteur COULEUR
Relevé machine au 24/03/2025 : 68460
Dernier facturé le 27/12/2024 : 68322
.

000008 FA4C Format A4 Couleur 138 0.07361 10.16 N
.

Contrat n° 005017 du 24/11/2020

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 44 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
34 PLACE DU MARCHE
77120 COULOMMIERS
RICOH MPC2004EXSP
Matricule : C768R720540
Compteur NB
Relevé machine au 24/03/2025 : 31821
Dernier facturé le 27/12/2024 : 31129
.

000005 FA4N Format A4 noir 692 0.00737 5.10 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 67427
Dernier facturé le 27/12/2024 : 64085
.

000008 FA4C Format A4 Couleur 3342 0.07361 246.00 N
.

Contrat n° 005024 du 26/11/2020
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
39 BOULEVARD DE LAGNY
77600 BUSSY ST GEORGES
RICOH MPC2003SP
Matricule : E204R960175

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 45 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Compteur COULEUR
Relevé machine au 24/03/2025 : 54840
Dernier facturé le 27/12/2024 : 53210
.

000008 FA4C Format A4 Couleur 1630 0.07361 119.98 N
.
Contrat n° 005033 du 10/12/2020
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
28 PLACE SAINT JACQUES
78200 MANTES LA JOLIE
RICOH MPC3003SP
Matricule : E156M220098
Compteur NB
Relevé machine au 24/03/2025 : 105913
Dernier facturé le 27/12/2024 : 103146
.

000005 FA4N Format A4 noir 2767 0.00737 20.39 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 117257
Dernier facturé le 27/12/2024 : 113646
.

000008 FA4C Format A4 Couleur 3611 0.07361 265.81 N
.
Contrat n° 005049 du 04/01/2021

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 46 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
7 AVENUE DU GENERAL DE GAULLE
91090 LISSES
RICOH MPC3003SP
Matricule : E155MC21368
Compteur NB
Relevé machine au 24/03/2025 : 117071
Dernier facturé le 27/12/2024 : 113938
.

000005 FA4N Format A4 noir 3133 0.00755 23.65 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 278853
Dernier facturé le 27/12/2024 : 268179
.

000008 FA4C Format A4 Couleur 10674 0.0756 806.95 N
.
Contrat n° 005055 du 11/01/2021
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
23 ALLEE DES IMPRESSIONNISTES
BATIMENT LE SISLEY
93420 VILLEPINTE
RICOH MPC3004SP

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 47 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Matricule : G696M532700
Compteur NB
Relevé machine au 24/03/2025 : 47529
Dernier facturé le 27/12/2024 : 46625
.

000005 FA4N Format A4 noir 904 0.00755 6.83 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 158623
Dernier facturé le 27/12/2024 : 153633
.

000008 FA4C Format A4 Couleur 4990 0.0756 377.24 N
.

Contrat n° 005075 du 01/02/2021
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (ACR) - I002
9BIS QUAI CONTI
78430 LOUVECIENNES
RICOH MPC2003SP
Matricule : E205R160533
Compteur NB
Estimation au 31/03/2025 : 55856
Dernier facturé le 31/12/2024 : 53968
.

000005 FA4N Format A4 noir 1888 0.00755 14.25 N
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 48 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Estimation au 31/03/2025 : 62019
Dernier facturé le 31/12/2024 : 60891
.

000008 FA4C Format A4 Couleur 1128 0.0756 85.28 N
.
Contrat n° 005128 du 05/03/2021
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
5 RUE DE LA BELLE ETOILE
77230 LONGPERRIER
RICOH MPC2003SP
Matricule : E206R162329
Compteur NB
Relevé machine au 24/03/2025 : 48657
Dernier facturé le 27/12/2024 : 46939
.

000005 FA4N Format A4 noir 1718 0.00755 12.97 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 89119
Dernier facturé le 27/12/2024 : 85144
.

000008 FA4C Format A4 Couleur 3975 0.0756 300.51 N
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 49 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Contrat n° 005252 du 20/07/2021
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
6 RUE DE MICY
45380 LA CHAPELLE ST MESMIN
RICOH MPC3503SP
Matricule : E165J400081
Compteur NB
Relevé machine au 24/03/2025 : 52933
Dernier facturé le 27/12/2024 : 52333
.

000005 FA4N Format A4 noir 600 0.00689 4.13 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 137459
Dernier facturé le 27/12/2024 : 133092
.

000008 FA4C Format A4 Couleur 4367 0.06898 301.24 N
.

Contrat n° 005336 du 18/11/2021
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
14 PLACE DUPONT PERROT
77370 NANGIS
RICOH MPC2504EXSP

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 50 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

Matricule : C777RB20140
Compteur NB
Relevé PDA au 26/03/2025 : 52609
Dernier facturé le 27/12/2024 : 51846
.

000005 FA4N Format A4 noir 763 0.00683 5.21 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé PDA au 26/03/2025 : 36065
Dernier facturé le 27/12/2024 : 34616
.

000008 FA4C Format A4 Couleur 1449 0.06818 98.79 N
.
Contrat n° 005395 du 19/01/2022
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
17 BOULEVARD DE LA MALIBRAN
PARKING: ENTREE 2511# - SORTIE 7768#
77680 ROISSY EN BRIE
RICOH MPC2504EXSP
Matricule : C778J300383
Compteur NB
Relevé machine au 24/03/2025 : 38626
Dernier facturé le 30/12/2024 : 38142
.

000005 FA4N Format A4 noir 484 0.00672 3.25 N

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 51 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
HOTEL ADAGIO
48 RUE PIERRE SEMARD
78200 MANTES LA JOLIE
RICOH MPC2004EXSP
Matricule : C768R420231
Compteur NB
Relevé machine au 24/03/2025 : 119267
Dernier facturé le 27/12/2024 : 117887
.

000005 FA4N Format A4 noir 1380 0.00613 8.46 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
EQUALIS
17 BOULEVARD DE LA MALIBRAN
PARKING: ENTREE 2511# - SORTIE 7768#
77680 ROISSY EN BRIE
RICOH MPC2504EXSP
Matricule : C778J300383
Compteur COULEUR
Relevé machine au 24/03/2025 : 61661
Dernier facturé le 30/12/2024 : 60693
.

000008 FA4C Format A4 Couleur 968 0.06715 65.00 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
HOTEL ADAGIO

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 52 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

48 RUE PIERRE SEMARD
78200 MANTES LA JOLIE
RICOH MPC2004EXSP
Matricule : C768R420231
Compteur COULEUR
Relevé machine au 24/03/2025 : 27055
Dernier facturé le 27/12/2024 : 25061
.

000008 FA4C Format A4 Couleur 1994 0.06715 133.90 N
.

Contrat n° 005413 du 28/01/2022
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
21 RUE JACQUARD
77400 LAGNY SUR MARNE
RICOH IM 2702
Matricule : 3291MA21202
2EME ETAGE
Compteur NB

Relevé machine au 24/03/2025 : 10864
Dernier facturé le 27/12/2024 : 10205
.

000005 FA4N Format A4 noir 659 0.00724 4.77 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
22 BIS CHEMIN DE LA TOUFFE

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 53 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

77870 VULAINES SUR SEINE
RICOH IM 2702
Matricule : 3291MA21201
Compteur NB
Relevé machine au 24/03/2025 : 10578
Dernier facturé le 27/12/2024 : 9915
.

000005 FA4N Format A4 noir 663 0.00724 4.80 N
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS
2 A RUE D'ORGEMONT
77100 MEAUX
RICOH IM 2702
Matricule : 3291MA20497
Compteur NB
Relevé machine au 24/03/2025 : 44280
Dernier facturé le 27/12/2024 : 40464
.

000005 FA4N Format A4 noir 3816 0.00724 27.63 N
.

Contrat n° 005414 du 28/01/2022
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
CENTRE COMMERCIAL DE LA BUTTE
MONTCEAU
77210 AVON

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 54 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

RICOH MPC2504EXASP
Matricule : C778J600171
RDC
Compteur NB
Relevé machine au 24/03/2025 : 36306
Dernier facturé le 27/12/2024 : 35538
.

000005 FA4N Format A4 noir 768 0.00672 5.16 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Relevé machine au 24/03/2025 : 74127
Dernier facturé le 27/12/2024 : 69666
.

000008 FA4C Format A4 Couleur 4461 0.06715 299.56 N
.

Contrat n° 005474 du 12/04/2022
.

RELEVE COPIES NOIRES A4 TRIMESTRIEL
EQUALIS (LA ROSE DES VENTS)
400 CHEMIN DE CRECY
77100 MAREUIL LES MEAUX
RICOH IMC530FB
Matricule : 4201XC50254
Compteur NB
Estimation au 31/03/2025 : 218
Dernier facturé le 15/01/2025 : 203
.

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 55 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874

### 8/12 rue de Lisbonne 93110 ROSNY SOUS BOIS Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62

## EQUALIS

#### 400 CHEMIN DE CRECY CS 50278 MAREUIL LES MEAUX 77334 MEAUX CEDEX

# FACTURE

### DATE CLIENT FACTURE

#### 31-03-2025 002219 100870 Référence Désignation Qté Liv. PU H.T. Net Montant H.T. TVA

000005 FA4N Format A4 noir 15 0.00613 0.09 N
.

RELEVE COPIES COULEUR A4 TRIMESTRIEL
Compteur COULEUR
Estimation au 31/03/2025 : 357
Dernier facturé le 15/01/2025 : 333
.

000008 FA4C Format A4 Couleur 24 0.06127 1.47 N

FTC
0.00

Taux TVA Base H.T. Montant T.V.A.
20.00 (N) 18796.49 3759.30

#### Total H.T. 18796.49 Total TVA 3759.30 Total T.T.C. 22555.79 Acompte 0.00

Mode de Règlement Echéance Montant
Prélèvement automatique - PRELE VEMENT LE
10 DU MOIS SUIVANT 10-04-2025 22555.79

### Net à Payer 22555.79 €

IBAN : FR76 3008 7338 3100 0204 0710 134 - BIC : CMCIFRPPXXX / Domiciliation : CIC ENTREPRISES MARNE LA VALLEE
Dans le cas où le paiement intégral n'interviendrait pas à la date prévue par les parties, le vendeur se réserve le droit de reprendre la livraison et de
dissoudre le contrat. En cas de retard de paiement, une indemnité forfaitaire de 40 € pour frais de recouvrement sera due de plein droit dès le premier
jour de retard et les pénalités seront calculées sur la base de 1 fois et demi le taux d'intérêt légal, par jour de retard. En cas de rejet de prélèvement, une
indemnité forfaitaire de 15 € HT vous sera facturée.

#### Page 56 / 56

8/12 rue de Lisbonne - 93110 ROSNY SOUS BOIS
Tél. : 01 43 32 17 56 - Fax : 01 43 32 20 62
SAS au Capital de 300 000 Euros - RCS Nanterre B 511 917 874 000 30 - Siret 51191787400030 - APE 4666Z - TVA Intra. : FR67511917874


    """
    
    test_instructions = """
    En tant que spécialiste de l'extraction de données précises, analyse le document fourni qui présente une liste de factures et effectue les tâches suivantes :

    EXIGENCES D'EXTRACTION : 
    Traite toutes les pages du document :
    Pour chaque Contrat du document, extraie toutes les quantités livrées (colonne Qté Liv.) avec les informations suivantes : 
    - Extraie le Numéro de contrat 
    - Extraie le matricule
    - Identifier la spécification de couleur pour le format A4 (e.g. noir ou Couleur) 
    - Collecte toutes les quantités du format de ligne A4 (Information colonne Qté Liv.) et les Montants HT associés (colonne Montant H.T.)
    - Fournis le Numéro de page du document pdf en lecture (e.g. page 1 pour 1 / 56, page 2 pour 2 / 56))
    Attention, Les informations peuvent etre réparties sur 2 pages successives

    FORMATAGE DE LA SORTIE : 
    Ecris un tableau avec six colonnes "N° Contrat", "Matricule", "Couleur", "Quantity", "Montant HT", "Numéro de page" 
    | N° Contrat | Matricule | Couleur  | Quantity | Montant HT | Numéro de Page |

    Chaque entrée sur une nouvelle ligne 
    Maintenir l'ordre original des données par ordre croissant des pages 
    Fournir le tableau complet pour la lecture de toutes les pages du document

    La dernière ligne du tableau propose le cumul du Montant H.T. de toutes les lignes du tableau
    """
    
    try:
        # Test with CSV output
        console.print("\n[bold blue]Testing with CSV output...[/]")
        analysis_csv = await analyze_context_with_instructions(
            context=test_context,
            instructions=test_instructions,
            output_format="csv",
            save_to_file=True
        )
        console.print("[bold green]✓ CSV analysis completed[/]")
        console.print(f"Summary: {analysis_csv.summary}")
        console.print(f"Output format: {analysis_csv.format_type}")
        console.print(f"Content:\n{analysis_csv.content}")
        
        if analysis_csv.file_path:
            console.print(f"File saved to: {analysis_csv.file_path}")
        
    except Exception as e:
        console.print(f"[bold red]Error during testing:[/] {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
