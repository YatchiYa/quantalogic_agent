from typing import List, Optional
from pydantic import BaseModel

class KeyCompetitor(BaseModel):
    """Model for key competitor information."""
    name: str
    strengths: List[str]
    weaknesses: List[str]
    threat_level: str  # High/Medium/Low

class MarketDynamics(BaseModel):
    """Model for market dynamics information."""
    growth_drivers: List[str]
    challenges: List[str]
    opportunities: List[str]

class MarketPosition(BaseModel):
    """Model for market position information."""
    company_position: str
    key_competitors: List[KeyCompetitor]

class CompetitiveLandscape(BaseModel):
    """Model for competitive landscape information."""
    market_position: MarketPosition
    market_dynamics: MarketDynamics

class ExecutiveSummary(BaseModel):
    """Model for executive summary information."""
    key_findings: List[str]
    strategic_implications: str
    recommended_actions: List[str]

class StrategicAction(BaseModel):
    """Model for strategic action information."""
    action: str
    priority: str  # High/Medium/Low
    expected_impact: str
    timeline: str

class StrategicRecommendations(BaseModel):
    """Model for strategic recommendations."""
    short_term: List[StrategicAction]
    long_term: List[StrategicAction]

class Risk(BaseModel):
    """Model for risk information."""
    risk: str
    severity: str  # High/Medium/Low
    mitigation: str

class RiskAssessment(BaseModel):
    """Model for risk assessment information."""
    immediate_risks: List[Risk]
    emerging_risks: List[Risk]

class CompetitiveReport(BaseModel):
    """Model for the complete competitive intelligence report."""
    executive_summary: ExecutiveSummary
    competitive_landscape: CompetitiveLandscape
    strategic_recommendations: StrategicRecommendations
    risk_assessment: RiskAssessment
