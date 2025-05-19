#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru>=0.7.2",
#     "litellm>=1.0.0",
#     "pydantic>=2.0.0",
#     "asyncio",
#     "jinja2>=3.1.0",
#     "quantalogic",
#     "instructor>=0.5.2",
#     "typer>=0.9.0",
#     "rich>=13.0.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "plotly>=5.0.0",
#     "yfinance>=0.2.0"
# ]
# ///

import asyncio
from collections.abc import Callable
import datetime
from decimal import Decimal
import os
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict

from loguru import logger
from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from quantalogic.tools.finance.yahoo_finance import YahooFinanceTool

import asyncio
from collections.abc import Callable
import datetime
from decimal import Decimal
import os
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict

from loguru import logger
from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def get_template_path(template_name: str) -> str:
    """Get the full path to a template file."""
    return os.path.join(TEMPLATES_DIR, template_name)

# Data Models
class FinancialMetrics(BaseModel):
    """Key financial metrics and ratios"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    profitability_ratios: Dict[str, Decimal] = Field(description="ROE, ROA, Profit margins, etc.")
    liquidity_ratios: Dict[str, Decimal] = Field(description="Current ratio, Quick ratio, etc.")
    solvency_ratios: Dict[str, Decimal] = Field(description="Debt/Equity, Interest coverage, etc.")
    efficiency_ratios: Dict[str, Decimal] = Field(description="Asset turnover, Inventory turnover, etc.")
    market_ratios: Dict[str, Decimal] = Field(description="P/E, P/B, Dividend yield, etc.")
    trend_analysis: Dict[str, List[Decimal]] = Field(description="Historical trends of key metrics")

class MarketAnalysis(BaseModel):
    """Market and industry analysis"""
    market_size: Dict[str, Union[int, Decimal]] = Field(description="Market size and growth")
    market_share: Dict[str, Decimal] = Field(description="Market share analysis")
    competition: List[Dict[str, Any]] = Field(description="Competitor analysis")
    industry_trends: List[Dict[str, Any]] = Field(description="Key industry trends")
    market_drivers: List[Dict[str, Any]] = Field(description="Market growth drivers")
    risks: List[Dict[str, Any]] = Field(description="Market risks and challenges")

class TechnicalAnalysis(BaseModel):
    """Technical analysis indicators and patterns"""
    price_trends: Dict[str, Any] = Field(description="Price trend analysis")
    support_resistance: Dict[str, List[Decimal]] = Field(description="Support and resistance levels")
    technical_indicators: Dict[str, Any] = Field(description="Moving averages, RSI, MACD, etc.")
    chart_patterns: List[Dict[str, Any]] = Field(description="Identified chart patterns")
    volume_analysis: Dict[str, Any] = Field(description="Volume trends and analysis")
    momentum_indicators: Dict[str, Any] = Field(description="Momentum-based indicators")

class RiskAssessment(BaseModel):
    """Risk analysis and assessment"""
    market_risks: List[Dict[str, Any]] = Field(description="Market-related risks")
    financial_risks: List[Dict[str, Any]] = Field(description="Financial risks")
    operational_risks: List[Dict[str, Any]] = Field(description="Operational risks")
    regulatory_risks: List[Dict[str, Any]] = Field(description="Regulatory risks")
    risk_metrics: Dict[str, Any] = Field(description="Risk metrics and measurements")
    mitigation_strategies: List[Dict[str, Any]] = Field(description="Risk mitigation strategies")

class InvestmentStrategy(BaseModel):
    """Investment strategy recommendations"""
    asset_allocation: Dict[str, Decimal] = Field(description="Recommended asset allocation")
    sector_weights: Dict[str, Decimal] = Field(description="Sector weightings")
    investment_vehicles: List[Dict[str, Any]] = Field(description="Recommended investment vehicles")
    entry_points: Dict[str, Any] = Field(description="Entry point recommendations")
    exit_strategies: Dict[str, Any] = Field(description="Exit strategy recommendations")
    rebalancing_strategy: Dict[str, Any] = Field(description="Portfolio rebalancing strategy")

class ScenarioAnalysis(BaseModel):
    """Scenario and sensitivity analysis"""
    scenarios: List[Dict[str, Any]] = Field(description="Different scenario analyses")
    sensitivity_factors: List[Dict[str, Any]] = Field(description="Key sensitivity factors")
    impact_analysis: Dict[str, Any] = Field(description="Impact on key metrics")
    probability_assessment: Dict[str, Decimal] = Field(description="Scenario probabilities")
    recommended_actions: Dict[str, List[str]] = Field(description="Actions for each scenario")

class FinancialAnalysisReport(BaseModel):
    """Complete financial analysis report"""
    analysis_id: str = Field(description="Unique analysis identifier")
    target_name: str = Field(description="Name of analyzed entity")
    analysis_date: datetime.datetime = Field(description="Date of analysis")
    financial_metrics: FinancialMetrics = Field(description="Financial metrics analysis")
    market_analysis: MarketAnalysis = Field(description="Market analysis")
    technical_analysis: TechnicalAnalysis = Field(description="Technical analysis")
    risk_assessment: RiskAssessment = Field(description="Risk assessment")
    investment_strategy: InvestmentStrategy = Field(description="Investment strategy")
    scenario_analysis: ScenarioAnalysis = Field(description="Scenario analysis")
    executive_summary: str = Field(description="Executive summary")
    recommendations: List[Dict[str, Any]] = Field(description="Key recommendations")
    action_items: List[Dict[str, Any]] = Field(description="Specific action items")

class MarketData(BaseModel):
    """Market data from Yahoo Finance"""
    symbol: str = Field(description="Stock/Asset symbol")
    interval: str = Field(description="Data interval")
    range_type: str = Field(description="Date range type")
    start_date: str = Field(description="Start date")
    end_date: str = Field(description="End date")
    market_hours: Dict[str, str] = Field(description="Market hours")
    data_points: int = Field(description="Number of data points")
    data: List[Dict[str, Any]] = Field(description="Time series data")
    info: Dict[str, Any] = Field(description="Asset information")

# Workflow Nodes
@Nodes.transform_node(
    transformer=lambda x: x,  # Identity transformer as the actual transformation is done in the function
    output="market_data"
)
async def fetch_market_data(
    symbols: List[str],
    interval: str = "1d",
    range_type: str = "month"
) -> Dict[str, MarketData]:
    """Fetch market data from Yahoo Finance."""
    logger.info(f"Fetching market data for {len(symbols)} symbols")
    
    tool = YahooFinanceTool()
    results = {}
    
    for symbol in symbols:
        try:
            data = await tool.execute(
                symbol=symbol,
                interval=interval,
                range_type=range_type
            )
            results[symbol] = MarketData(**data)
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            continue
    
    return results

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_financial_metrics.j2"),
    output="financial_metrics",
    response_model=FinancialMetrics,
    prompt_file=get_template_path("prompt_financial_metrics.j2")
)
async def analyze_financial_metrics(
    financial_data: Dict[str, Any],
    historical_data: Dict[str, Any],
    model: str
) -> FinancialMetrics:
    """Analyze financial metrics and ratios."""
    logger.info("Analyzing financial metrics")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_market_analysis.j2"),
    output="market_analysis",
    response_model=MarketAnalysis,
    prompt_file=get_template_path("prompt_market_analysis.j2")
)
async def analyze_market(
    market_data: Dict[str, Any],
    industry_data: Dict[str, Any],
    competitor_data: Dict[str, Any],
    model: str
) -> MarketAnalysis:
    """Perform market and industry analysis."""
    logger.info("Analyzing market conditions")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_technical_analysis.j2"),
    output="technical_analysis",
    response_model=TechnicalAnalysis,
    prompt_file=get_template_path("prompt_technical_analysis.j2")
)
async def perform_technical_analysis(
    price_data: Dict[str, Any],
    volume_data: Dict[str, Any],
    model: str
) -> TechnicalAnalysis:
    """Perform technical analysis."""
    logger.info("Performing technical analysis")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_risk_assessment.j2"),
    output="risk_assessment",
    response_model=RiskAssessment,
    prompt_file=get_template_path("prompt_risk_assessment.j2")
)
async def assess_risks(
    financial_metrics: FinancialMetrics,
    market_analysis: MarketAnalysis,
    technical_analysis: TechnicalAnalysis,
    model: str
) -> RiskAssessment:
    """Assess various risks and develop mitigation strategies."""
    logger.info("Assessing risks")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_investment_strategy.j2"),
    output="investment_strategy",
    response_model=InvestmentStrategy,
    prompt_file=get_template_path("prompt_investment_strategy.j2")
)
async def develop_investment_strategy(
    financial_metrics: FinancialMetrics,
    market_analysis: MarketAnalysis,
    technical_analysis: TechnicalAnalysis,
    risk_assessment: RiskAssessment,
    investment_goals: Dict[str, Any],
    model: str
) -> InvestmentStrategy:
    """Develop investment strategy recommendations."""
    logger.info("Developing investment strategy")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_scenario_analysis.j2"),
    output="scenario_analysis",
    response_model=ScenarioAnalysis,
    prompt_file=get_template_path("prompt_scenario_analysis.j2")
)
async def perform_scenario_analysis(
    financial_metrics: FinancialMetrics,
    market_analysis: MarketAnalysis,
    risk_assessment: RiskAssessment,
    investment_strategy: InvestmentStrategy,
    scenario_params: Dict[str, Any],
    model: str
) -> ScenarioAnalysis:
    """Perform scenario and sensitivity analysis."""
    logger.info("Performing scenario analysis")
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_final_report.j2"),
    output="final_report",
    response_model=FinancialAnalysisReport,
    prompt_file=get_template_path("prompt_final_report.j2")
)
async def compile_final_report(
    analysis_id: str,
    target_name: str,
    financial_metrics: FinancialMetrics,
    market_analysis: MarketAnalysis,
    technical_analysis: TechnicalAnalysis,
    risk_assessment: RiskAssessment,
    investment_strategy: InvestmentStrategy,
    scenario_analysis: ScenarioAnalysis,
    model: str
) -> FinancialAnalysisReport:
    """Compile the final analysis report."""
    logger.info("Compiling final report")
    pass

def create_financial_analysis_workflow() -> Workflow:
    """Create the workflow for financial analysis."""
    workflow = (
        Workflow("fetch_market_data")
        .then("analyze_financial_metrics")
        .then("analyze_market")
        .then("perform_technical_analysis")
        .then("assess_risks")
        .then("develop_investment_strategy")
        .then("perform_scenario_analysis")
        .then("compile_final_report")
    )
    
    workflow.node_input_mappings = {
        "fetch_market_data": {
            "symbols": "target_symbols",
            "interval": "data_interval",
            "range_type": "data_range"
        },
        "analyze_financial_metrics": {
            "model": "llm_model",
            "financial_data": "financial_data",
            "historical_data": "historical_data"
        },
        "analyze_market": {
            "model": "llm_model",
            "market_data": "market_data",
            "industry_data": "industry_data",
            "competitor_data": "competitor_data"
        },
        "perform_technical_analysis": {
            "model": "llm_model",
            "price_data": "price_data",
            "volume_data": "volume_data"
        },
        "assess_risks": {
            "model": "llm_model"
        },
        "develop_investment_strategy": {
            "model": "llm_model",
            "investment_goals": "investment_goals"
        },
        "perform_scenario_analysis": {
            "model": "llm_model",
            "scenario_params": "scenario_params"
        },
        "compile_final_report": {
            "model": "llm_model",
            "analysis_id": "analysis_id",
            "target_name": "target_name"
        }
    }
    
    return workflow

async def analyze_financials(
    analysis_id: str,
    target_name: str,
    target_symbols: List[str],
    financial_data: Dict[str, Any],
    historical_data: Dict[str, Any],
    market_data: Dict[str, Any],
    industry_data: Dict[str, Any],
    competitor_data: Dict[str, Any],
    investment_goals: Dict[str, Any],
    scenario_params: Dict[str, Any],
    data_interval: str = "1d",
    data_range: str = "month",
    llm_model: str = "gemini/gemini-2.0-flash",
    _handle_event: Optional[Callable[[str, dict], None]] = None
) -> FinancialAnalysisReport:
    """Run the complete financial analysis workflow."""
    
    initial_context = {
        "analysis_id": analysis_id,
        "target_name": target_name,
        "target_symbols": target_symbols,
        "data_interval": data_interval,
        "data_range": data_range,
        "financial_data": financial_data,
        "historical_data": historical_data,
        "market_data": market_data,
        "industry_data": industry_data,
        "competitor_data": competitor_data,
        "investment_goals": investment_goals,
        "scenario_params": scenario_params,
        "llm_model": llm_model
    }
    
    workflow = create_financial_analysis_workflow()
    engine = workflow.build()
    
    result = await engine.run(initial_context)
    
    logger.info(f"Financial analysis completed for {target_name}")
    return result["final_report"]

def cli_analyze_financials(
    target_name: str,
    target_symbols: List[str],
    financial_data_file: str,
    historical_data_file: str,
    market_data_file: str,
    industry_data_file: str,
    competitor_data_file: str,
    investment_goals_file: str,
    scenario_params_file: str,
    data_interval: str = "1d",
    data_range: str = "month",
    output_file: Optional[str] = None,
    model: str = "gemini/gemini-2.0-flash"
):
    """CLI wrapper for financial analysis."""
    # Generate analysis ID
    analysis_id = f"FIN-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Read input files
    def read_json_file(file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return eval(f.read())  # In production, use proper JSON parsing
    
    result = asyncio.run(analyze_financials(
        analysis_id=analysis_id,
        target_name=target_name,
        target_symbols=target_symbols,
        data_interval=data_interval,
        data_range=data_range,
        financial_data=read_json_file(financial_data_file),
        historical_data=read_json_file(historical_data_file),
        market_data=read_json_file(market_data_file),
        industry_data=read_json_file(industry_data_file),
        competitor_data=read_json_file(competitor_data_file),
        investment_goals=read_json_file(investment_goals_file),
        scenario_params=read_json_file(scenario_params_file),
        model=model
    ))
    
    # Output results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.model_dump_json(indent=2))
    else:
        print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    import typer
    typer.run(cli_analyze_financials)
