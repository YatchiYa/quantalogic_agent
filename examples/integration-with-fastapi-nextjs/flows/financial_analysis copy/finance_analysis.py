## analyse finance & stocks

#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru",
#     "litellm",
#     "pydantic>=2.0",
#     "anyio",
#     "quantalogic>=0.35",
#     "jinja2",
#     "typer>=0.9.0",
#     "yfinance",
#     "pandas",
#     "pytz"
# ]
# ///

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Optional, Union

import typer
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from quantalogic.flow.flow import Nodes, Workflow, WorkflowEvent, WorkflowEventType
from quantalogic.tools.finance.yahoo_finance import YahooFinanceTool
# from ..service import event_observer

# Get the templates directory path
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# Helper function to get template paths
def get_template_path(template_name):
    return os.path.join(TEMPLATES_DIR, template_name)

# Initialize Typer app
app = typer.Typer(help="Financial data analysis and trading insights")

# Pydantic Models
class FinancialSymbol(BaseModel):
    symbol: str
    name: str = ""
    type: str = ""  # stock, crypto, forex, etc.
    
class FinancialDataRequest(BaseModel):
    symbols: List[FinancialSymbol]
    interval: str = "1d"
    range_type: str = "month"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
class FinancialDataPoint(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    
class FinancialDataSeries(BaseModel):
    symbol: str
    name: str
    interval: str
    range_type: str
    start_date: str
    end_date: str
    data_points: int
    data: List[FinancialDataPoint]
    currency: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    
class FinancialDataCollection(BaseModel):
    series: List[FinancialDataSeries]
    timestamp: str
    
class TechnicalIndicators(BaseModel):
    symbol: str
    moving_averages: Dict[str, float]  # e.g., "MA50": 150.25
    rsi: Optional[float] = None
    macd: Optional[Dict[str, float]] = None
    bollinger_bands: Optional[Dict[str, float]] = None
    
class TechnicalAnalysis(BaseModel):
    indicators: List[TechnicalIndicators]
    
class MarketTrend(BaseModel):
    symbol: str
    trend: str  # bullish, bearish, neutral
    strength: int  # 1-10
    key_levels: Dict[str, float]  # support, resistance levels
    
    model_config = {
        "json_schema_extra": {"examples": [{"key_levels": {"support": 100.0, "resistance": 200.0}}]}
    }
    
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        # Pre-process the key_levels field if it exists and contains string values
        if isinstance(obj, dict) and "key_levels" in obj and isinstance(obj["key_levels"], dict):
            obj["key_levels"] = {k: float(v) if isinstance(v, str) else v for k, v in obj["key_levels"].items()}
        return super().model_validate(obj, *args, **kwargs)
    
class FinancialAnalysis(BaseModel):
    timestamp: str
    overview: str
    key_insights: List[str]
    price_action_summary: str
    volume_analysis: str
    technical_outlook: str
    market_trends: List[MarketTrend]
    
class TradingRecommendation(BaseModel):
    symbol: str
    action: str  # buy, sell, hold
    confidence: int  # 1-10
    entry_points: List[float]
    stop_loss: float
    take_profit: List[float]
    time_horizon: str  # short-term, medium-term, long-term
    rationale: str
    risk_level: int  # 1-10
    
class TradingAnalysis(BaseModel):
    timestamp: str
    market_overview: str
    key_observations: List[str]
    recommendations: List[TradingRecommendation]
    risk_assessment: str
    market_sentiment: str
    correlation_insights: Optional[str] = None
    
# Report section models
class ReportExecutiveSummary(BaseModel):
    timestamp: str
    content: str

class ReportMarketContext(BaseModel):
    timestamp: str
    content: str

class ReportTechnicalAnalysisOverview(BaseModel):
    timestamp: str
    content: str

class ReportMarketStructureAnalysis(BaseModel):
    timestamp: str
    content: str

class ReportIctSmcAnalysis(BaseModel):
    timestamp: str
    content: str

class ReportFibonacciAnalysis(BaseModel):
    timestamp: str
    content: str

class ReportVolumeProfileAnalysis(BaseModel):
    timestamp: str
    content: str

class ReportWyckoffAnalysis(BaseModel):
    timestamp: str
    content: str

class ReportFundamentalFactors(BaseModel):
    timestamp: str
    content: str

class ReportTradingStrategy(BaseModel):
    timestamp: str
    content: str

class ReportEntryStrategy(BaseModel):
    timestamp: str
    content: str

class ReportExitStrategy(BaseModel):
    timestamp: str
    content: str

class ReportPositionSizing(BaseModel):
    timestamp: str
    content: str

class ReportRiskManagement(BaseModel):
    timestamp: str
    content: str

class ReportForwardOutlook(BaseModel):
    timestamp: str
    content: str

class ReportScenarioAnalysis(BaseModel):
    timestamp: str
    content: str

class ReportConclusion(BaseModel):
    timestamp: str
    content: str

# Combined final report model
class FinancialReport(BaseModel):
    timestamp: str
    executive_summary: str
    market_context: str
    technical_analysis_overview: str
    market_structure_analysis: str
    ict_smc_analysis: str
    fibonacci_analysis: str
    volume_profile_analysis: str
    wyckoff_analysis: str
    fundamental_factors: str
    trading_strategy: str
    entry_strategy: str
    exit_strategy: str
    position_sizing: str
    risk_management: str
    forward_outlook: str
    scenario_analysis: str
    conclusion: str

# Nodes
@Nodes.define(output="financial_data_request")
async def prepare_financial_request(symbols_list: List[str], interval: str = "1d", 
                                   range_type: str = "month") -> FinancialDataRequest:
    """Prepare the financial data request with the provided symbols."""
    symbols = []
    for symbol in symbols_list:
        symbols.append(FinancialSymbol(symbol=symbol))
    
    logger.info(f"Preparing financial request for {len(symbols)} symbols with {interval} interval")
    return FinancialDataRequest(
        symbols=symbols,
        interval=interval,
        range_type=range_type
    )

@Nodes.define(output="financial_data_collection")
async def fetch_financial_data(request: FinancialDataRequest) -> FinancialDataCollection:
    """Fetch financial data for all symbols in the request."""
    yahoo_tool = YahooFinanceTool()
    series_list = []
    
    for symbol_obj in request.symbols:
        symbol = symbol_obj.symbol
        logger.info(f"Fetching data for {symbol} with {request.interval} interval")
        
        try:
            # Fetch data using the Yahoo Finance tool
            data = await yahoo_tool.execute(
                symbol=symbol,
                interval=request.interval,
                range_type=request.range_type,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            # Process the data
            data_points = []
            for point in data.get("data", []):
                data_points.append(FinancialDataPoint(
                    timestamp=point["timestamp"],
                    open=point["open"],
                    high=point["high"],
                    low=point["low"],
                    close=point["close"],
                    volume=point["volume"]
                ))
            
            # Create a data series for this symbol
            series = FinancialDataSeries(
                symbol=symbol,
                name=data.get("info", {}).get("shortName", symbol),
                interval=request.interval,
                range_type=request.range_type,
                start_date=data.get("start_date", ""),
                end_date=data.get("end_date", ""),
                data_points=len(data_points),
                data=data_points,
                currency=data.get("info", {}).get("currency"),
                exchange=data.get("info", {}).get("exchange"),
                sector=data.get("info", {}).get("sector"),
                industry=data.get("info", {}).get("industry")
            )
            
            series_list.append(series)
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
    
    # Create the collection with all series
    collection = FinancialDataCollection(
        series=series_list,
        timestamp=datetime.now().isoformat()
    )
    
    logger.info(f"Fetched data for {len(series_list)} symbols")
    return collection

# Helper functions for templates
def get_price_range(data_points):
    """Calculate the min and max values from a list of data points."""
    if not data_points:
        return (0, 0)
    low_values = [point.low for point in data_points]
    high_values = [point.high for point in data_points]
    return (min(low_values), max(high_values))

@Nodes.define(output="technical_analysis")
async def calculate_technical_indicators(data_collection: FinancialDataCollection) -> TechnicalAnalysis:
    """Calculate technical indicators for each symbol in the collection."""
    indicators_list = []
    
    for series in data_collection.series:
        # Simple moving averages calculation
        closes = [point.close for point in series.data]
        
        # Calculate moving averages if we have enough data points
        moving_averages = {}
        if len(closes) >= 20:
            ma20 = sum(closes[-20:]) / 20
            moving_averages["MA20"] = round(ma20, 2)
        
        if len(closes) >= 50:
            ma50 = sum(closes[-50:]) / 50
            moving_averages["MA50"] = round(ma50, 2)
        
        if len(closes) >= 200:
            ma200 = sum(closes[-200:]) / 200
            moving_averages["MA200"] = round(ma200, 2)
        
        # Simple RSI calculation (14-period)
        rsi = None
        if len(closes) >= 15:  # Need at least 15 data points for a 14-period RSI
            gains = []
            losses = []
            for i in range(1, 15):
                change = closes[-i] - closes[-i-1]
                if change >= 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi = round(rsi, 2)
        
        # Create technical indicators object
        indicators = TechnicalIndicators(
            symbol=series.symbol,
            moving_averages=moving_averages,
            rsi=rsi
        )
        
        indicators_list.append(indicators)
    
    logger.info(f"Calculated technical indicators for {len(indicators_list)} symbols")
    return TechnicalAnalysis(indicators=indicators_list)

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_analyze_financial_data.j2"),
    output="financial_analysis",
    response_model=FinancialAnalysis,
    prompt_file=get_template_path("prompt_analyze_financial_data.j2"),
    max_tokens=7000
)
async def analyze_financial_data(financial_data_collection: FinancialDataCollection, 
                               technical_analysis: TechnicalAnalysis) -> FinancialAnalysis:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_generate_trading_recommendations.j2"),
    output="trading_analysis",
    response_model=TradingAnalysis,
    prompt_file=get_template_path("prompt_generate_trading_recommendations.j2"),
    max_tokens=7000
)
async def generate_trading_recommendations(financial_data_collection: FinancialDataCollection,
                                          technical_analysis: TechnicalAnalysis,
                                          financial_analysis: FinancialAnalysis,
                                          get_price_range) -> TradingAnalysis:
    pass

# Report section nodes
@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_executive_summary.j2"),
    output="report_executive_summary",
    response_model=ReportExecutiveSummary,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_executive_summary(financial_data_collection: FinancialDataCollection,
                                         technical_analysis: TechnicalAnalysis,
                                         financial_analysis: FinancialAnalysis,
                                         trading_analysis: TradingAnalysis,
                                         get_price_range) -> ReportExecutiveSummary:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_market_context.j2"),
    output="report_market_context",
    response_model=ReportMarketContext,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_market_context(financial_data_collection: FinancialDataCollection,
                                       technical_analysis: TechnicalAnalysis,
                                       financial_analysis: FinancialAnalysis,
                                       trading_analysis: TradingAnalysis,
                                       get_price_range) -> ReportMarketContext:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_technical_overview.j2"),
    output="report_technical_overview",
    response_model=ReportTechnicalAnalysisOverview,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_technical_overview(financial_data_collection: FinancialDataCollection,
                                           technical_analysis: TechnicalAnalysis,
                                           financial_analysis: FinancialAnalysis,
                                           trading_analysis: TradingAnalysis,
                                           get_price_range) -> ReportTechnicalAnalysisOverview:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_market_structure.j2"),
    output="report_market_structure",
    response_model=ReportMarketStructureAnalysis,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_market_structure(financial_data_collection: FinancialDataCollection,
                                         technical_analysis: TechnicalAnalysis,
                                         financial_analysis: FinancialAnalysis,
                                         trading_analysis: TradingAnalysis,
                                         get_price_range) -> ReportMarketStructureAnalysis:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_ict_smc.j2"),
    output="report_ict_smc",
    response_model=ReportIctSmcAnalysis,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_ict_smc(financial_data_collection: FinancialDataCollection,
                                technical_analysis: TechnicalAnalysis,
                                financial_analysis: FinancialAnalysis,
                                trading_analysis: TradingAnalysis,
                                get_price_range) -> ReportIctSmcAnalysis:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_fibonacci.j2"),
    output="report_fibonacci",
    response_model=ReportFibonacciAnalysis,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_fibonacci(financial_data_collection: FinancialDataCollection,
                                  technical_analysis: TechnicalAnalysis,
                                  financial_analysis: FinancialAnalysis,
                                  trading_analysis: TradingAnalysis,
                                  get_price_range) -> ReportFibonacciAnalysis:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_volume_profile.j2"),
    output="report_volume_profile",
    response_model=ReportVolumeProfileAnalysis,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_volume_profile(financial_data_collection: FinancialDataCollection,
                                       technical_analysis: TechnicalAnalysis,
                                       financial_analysis: FinancialAnalysis,
                                       trading_analysis: TradingAnalysis,
                                       get_price_range) -> ReportVolumeProfileAnalysis:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_wyckoff.j2"),
    output="report_wyckoff",
    response_model=ReportWyckoffAnalysis,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_wyckoff(financial_data_collection: FinancialDataCollection,
                                technical_analysis: TechnicalAnalysis,
                                financial_analysis: FinancialAnalysis,
                                trading_analysis: TradingAnalysis,
                                get_price_range) -> ReportWyckoffAnalysis:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_fundamental.j2"),
    output="report_fundamental",
    response_model=ReportFundamentalFactors,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_fundamental(financial_data_collection: FinancialDataCollection,
                                    technical_analysis: TechnicalAnalysis,
                                    financial_analysis: FinancialAnalysis,
                                    trading_analysis: TradingAnalysis,
                                    get_price_range) -> ReportFundamentalFactors:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_trading_strategy.j2"),
    output="report_trading_strategy",
    response_model=ReportTradingStrategy,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_trading_strategy(financial_data_collection: FinancialDataCollection,
                                         technical_analysis: TechnicalAnalysis,
                                         financial_analysis: FinancialAnalysis,
                                         trading_analysis: TradingAnalysis,
                                         get_price_range) -> ReportTradingStrategy:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_entry_strategy.j2"),
    output="report_entry_strategy",
    response_model=ReportEntryStrategy,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_entry_strategy(financial_data_collection: FinancialDataCollection,
                                       technical_analysis: TechnicalAnalysis,
                                       financial_analysis: FinancialAnalysis,
                                       trading_analysis: TradingAnalysis,
                                       get_price_range) -> ReportEntryStrategy:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_exit_strategy.j2"),
    output="report_exit_strategy",
    response_model=ReportExitStrategy,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_exit_strategy(financial_data_collection: FinancialDataCollection,
                                      technical_analysis: TechnicalAnalysis,
                                      financial_analysis: FinancialAnalysis,
                                      trading_analysis: TradingAnalysis,
                                      get_price_range) -> ReportExitStrategy:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_position_sizing.j2"),
    output="report_position_sizing",
    response_model=ReportPositionSizing,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_position_sizing(financial_data_collection: FinancialDataCollection,
                                        technical_analysis: TechnicalAnalysis,
                                        financial_analysis: FinancialAnalysis,
                                        trading_analysis: TradingAnalysis,
                                        get_price_range) -> ReportPositionSizing:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_risk_management.j2"),
    output="report_risk_management",
    response_model=ReportRiskManagement,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_risk_management(financial_data_collection: FinancialDataCollection,
                                        technical_analysis: TechnicalAnalysis,
                                        financial_analysis: FinancialAnalysis,
                                        trading_analysis: TradingAnalysis,
                                        get_price_range) -> ReportRiskManagement:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_forward_outlook.j2"),
    output="report_forward_outlook",
    response_model=ReportForwardOutlook,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_forward_outlook(financial_data_collection: FinancialDataCollection,
                                        technical_analysis: TechnicalAnalysis,
                                        financial_analysis: FinancialAnalysis,
                                        trading_analysis: TradingAnalysis,
                                        get_price_range) -> ReportForwardOutlook:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_scenario_analysis.j2"),
    output="report_scenario_analysis",
    response_model=ReportScenarioAnalysis,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_scenario_analysis(financial_data_collection: FinancialDataCollection,
                                          technical_analysis: TechnicalAnalysis,
                                          financial_analysis: FinancialAnalysis,
                                          trading_analysis: TradingAnalysis,
                                          get_price_range) -> ReportScenarioAnalysis:
    pass

@Nodes.structured_llm_node(
    system_prompt_file=get_template_path("system_report_conclusion.j2"),
    output="report_conclusion",
    response_model=ReportConclusion,
    prompt_file=get_template_path("prompt_report_section.j2"),
    max_tokens=3000,
    temperature=0.7,
    timeout=120  # Increased timeout to 2 minutes
)
async def generate_report_conclusion(financial_data_collection: FinancialDataCollection,
                                   technical_analysis: TechnicalAnalysis,
                                   financial_analysis: FinancialAnalysis,
                                   trading_analysis: TradingAnalysis,
                                   get_price_range) -> ReportConclusion:
    pass

# Final report assembly node
@Nodes.define(output="financial_report")
async def assemble_financial_report(
    report_executive_summary: Optional[ReportExecutiveSummary] = None,
    report_market_context: Optional[ReportMarketContext] = None,
    report_technical_overview: Optional[ReportTechnicalAnalysisOverview] = None,
    report_market_structure: Optional[ReportMarketStructureAnalysis] = None,
    report_ict_smc: Optional[ReportIctSmcAnalysis] = None,
    report_fibonacci: Optional[ReportFibonacciAnalysis] = None,
    report_volume_profile: Optional[ReportVolumeProfileAnalysis] = None,
    report_wyckoff: Optional[ReportWyckoffAnalysis] = None,
    report_fundamental: Optional[ReportFundamentalFactors] = None,
    report_trading_strategy: Optional[ReportTradingStrategy] = None,
    report_entry_strategy: Optional[ReportEntryStrategy] = None,
    report_exit_strategy: Optional[ReportExitStrategy] = None,
    report_position_sizing: Optional[ReportPositionSizing] = None,
    report_risk_management: Optional[ReportRiskManagement] = None,
    report_forward_outlook: Optional[ReportForwardOutlook] = None,
    report_scenario_analysis: Optional[ReportScenarioAnalysis] = None,
    report_conclusion: Optional[ReportConclusion] = None
) -> FinancialReport:
    """Assemble all report sections into a final comprehensive report."""
    
    # Get current timestamp if no sections are available
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S%z")
    
    # Use the timestamp from any available section, prioritizing executive_summary
    if report_executive_summary and hasattr(report_executive_summary, 'timestamp'):
        timestamp = report_executive_summary.timestamp
    
    # Helper function to safely get content from a section
    def get_content(section):
        if section and hasattr(section, 'content'):
            return section.content
        logger.warning(f"Content not available for a report section: {section}")
        return "Content not available at this time. This section will be completed in a future update."
    
    # Create the final report by combining all sections
    final_report = FinancialReport(
        timestamp=timestamp,
        executive_summary=get_content(report_executive_summary),
        market_context=get_content(report_market_context),
        technical_analysis_overview=get_content(report_technical_overview),
        market_structure_analysis=get_content(report_market_structure),
        ict_smc_analysis=get_content(report_ict_smc),
        fibonacci_analysis=get_content(report_fibonacci),
        volume_profile_analysis=get_content(report_volume_profile),
        wyckoff_analysis=get_content(report_wyckoff),
        fundamental_factors=get_content(report_fundamental),
        trading_strategy=get_content(report_trading_strategy),
        entry_strategy=get_content(report_entry_strategy),
        exit_strategy=get_content(report_exit_strategy),
        position_sizing=get_content(report_position_sizing),
        risk_management=get_content(report_risk_management),
        forward_outlook=get_content(report_forward_outlook),
        scenario_analysis=get_content(report_scenario_analysis),
        conclusion=get_content(report_conclusion)
    )
    
    # Log which sections were successfully included
    available_sections = []
    if report_executive_summary and hasattr(report_executive_summary, 'content'):
        available_sections.append("executive_summary")
    if report_market_context and hasattr(report_market_context, 'content'):
        available_sections.append("market_context")
    if report_technical_overview and hasattr(report_technical_overview, 'content'):
        available_sections.append("technical_overview")
    if report_market_structure and hasattr(report_market_structure, 'content'):
        available_sections.append("market_structure")
    if report_ict_smc and hasattr(report_ict_smc, 'content'):
        available_sections.append("ict_smc")
    if report_fibonacci and hasattr(report_fibonacci, 'content'):
        available_sections.append("fibonacci")
    if report_volume_profile and hasattr(report_volume_profile, 'content'):
        available_sections.append("volume_profile")
    if report_wyckoff and hasattr(report_wyckoff, 'content'):
        available_sections.append("wyckoff")
    if report_fundamental and hasattr(report_fundamental, 'content'):
        available_sections.append("fundamental")
    if report_trading_strategy and hasattr(report_trading_strategy, 'content'):
        available_sections.append("trading_strategy")
    if report_entry_strategy and hasattr(report_entry_strategy, 'content'):
        available_sections.append("entry_strategy")
    if report_exit_strategy and hasattr(report_exit_strategy, 'content'):
        available_sections.append("exit_strategy")
    if report_position_sizing and hasattr(report_position_sizing, 'content'):
        available_sections.append("position_sizing")
    if report_risk_management and hasattr(report_risk_management, 'content'):
        available_sections.append("risk_management")
    if report_forward_outlook and hasattr(report_forward_outlook, 'content'):
        available_sections.append("forward_outlook")
    if report_scenario_analysis and hasattr(report_scenario_analysis, 'content'):
        available_sections.append("scenario_analysis")
    if report_conclusion and hasattr(report_conclusion, 'content'):
        available_sections.append("conclusion")
        
    logger.info(f"Assembled financial report with {len(available_sections)}/{17} sections: {', '.join(available_sections)}")
    return final_report

@Nodes.define(output="result")
async def format_results(financial_data_collection: FinancialDataCollection,
                        technical_analysis: TechnicalAnalysis,
                        financial_analysis: FinancialAnalysis,
                        trading_analysis: TradingAnalysis,
                        financial_report: FinancialReport) -> Dict[str, Any]:
    """Format all results into a single dictionary."""
    result = {
        "timestamp": datetime.now().isoformat(),
        "data": financial_data_collection.model_dump(),
        "technical_indicators": technical_analysis.model_dump(),
        "financial_analysis": financial_analysis.model_dump(),
        "trading_recommendations": trading_analysis.model_dump(),
        "financial_report": financial_report.model_dump()
    }
    
    logger.info("Financial analysis workflow completed successfully")
    return result

# Workflow
def create_financial_analysis_workflow() -> Workflow:
    """Create a workflow for financial data analysis and trading recommendations."""
    wf = Workflow("prepare_financial_request")
    
    # Define the flow
    wf.node("prepare_financial_request").then("fetch_financial_data")
    wf.node("fetch_financial_data").then("calculate_technical_indicators")
    wf.node("calculate_technical_indicators").then("analyze_financial_data")
    wf.node("analyze_financial_data").then("generate_trading_recommendations")
    
    # Report section generation - sequential processing
    wf.node("generate_trading_recommendations").then("generate_report_executive_summary")
    wf.node("generate_report_executive_summary").then("generate_report_market_context")
    wf.node("generate_report_market_context").then("generate_report_technical_overview")
    wf.node("generate_report_technical_overview").then("generate_report_market_structure")
    wf.node("generate_report_market_structure").then("generate_report_ict_smc")
    wf.node("generate_report_ict_smc").then("generate_report_fibonacci")
    wf.node("generate_report_fibonacci").then("generate_report_volume_profile")
    wf.node("generate_report_volume_profile").then("generate_report_wyckoff")
    wf.node("generate_report_wyckoff").then("generate_report_fundamental")
    wf.node("generate_report_fundamental").then("generate_report_trading_strategy")
    wf.node("generate_report_trading_strategy").then("generate_report_entry_strategy")
    wf.node("generate_report_entry_strategy").then("generate_report_exit_strategy")
    wf.node("generate_report_exit_strategy").then("generate_report_position_sizing")
    wf.node("generate_report_position_sizing").then("generate_report_risk_management")
    wf.node("generate_report_risk_management").then("generate_report_forward_outlook")
    wf.node("generate_report_forward_outlook").then("generate_report_scenario_analysis")
    wf.node("generate_report_scenario_analysis").then("generate_report_conclusion")
    
    # Assemble final report after all sections are generated
    wf.node("generate_report_conclusion").then("assemble_financial_report")
    
    wf.node("assemble_financial_report").then("format_results")
    
    # Input mappings
    wf.node_input_mappings["fetch_financial_data"] = {
        "request": "financial_data_request"
    }
    
    wf.node_input_mappings["calculate_technical_indicators"] = {
        "data_collection": "financial_data_collection"
    }
    
    wf.node_input_mappings["analyze_financial_data"] = {
        "financial_data_collection": "financial_data_collection",
        "technical_analysis": "technical_analysis"
    }
    
    wf.node_input_mappings["generate_trading_recommendations"] = {
        "financial_data_collection": "financial_data_collection",
        "technical_analysis": "technical_analysis",
        "financial_analysis": "financial_analysis",
        "get_price_range": "get_price_range"
    }
    
    # Report section input mappings
    report_section_inputs = {
        "financial_data_collection": "financial_data_collection",
        "technical_analysis": "technical_analysis",
        "financial_analysis": "financial_analysis",
        "trading_analysis": "trading_analysis",
        "get_price_range": "get_price_range"
    }
    
    wf.node_input_mappings["generate_report_executive_summary"] = report_section_inputs
    wf.node_input_mappings["generate_report_market_context"] = report_section_inputs
    wf.node_input_mappings["generate_report_technical_overview"] = report_section_inputs
    wf.node_input_mappings["generate_report_market_structure"] = report_section_inputs
    wf.node_input_mappings["generate_report_ict_smc"] = report_section_inputs
    wf.node_input_mappings["generate_report_fibonacci"] = report_section_inputs
    wf.node_input_mappings["generate_report_volume_profile"] = report_section_inputs
    wf.node_input_mappings["generate_report_wyckoff"] = report_section_inputs
    wf.node_input_mappings["generate_report_fundamental"] = report_section_inputs
    wf.node_input_mappings["generate_report_trading_strategy"] = report_section_inputs
    wf.node_input_mappings["generate_report_entry_strategy"] = report_section_inputs
    wf.node_input_mappings["generate_report_exit_strategy"] = report_section_inputs
    wf.node_input_mappings["generate_report_position_sizing"] = report_section_inputs
    wf.node_input_mappings["generate_report_risk_management"] = report_section_inputs
    wf.node_input_mappings["generate_report_forward_outlook"] = report_section_inputs
    wf.node_input_mappings["generate_report_scenario_analysis"] = report_section_inputs
    wf.node_input_mappings["generate_report_conclusion"] = report_section_inputs
    
    # Assemble financial report input mappings
    wf.node_input_mappings["assemble_financial_report"] = {
        "report_executive_summary": "report_executive_summary",
        "report_market_context": "report_market_context",
        "report_technical_overview": "report_technical_overview",
        "report_market_structure": "report_market_structure",
        "report_ict_smc": "report_ict_smc",
        "report_fibonacci": "report_fibonacci",
        "report_volume_profile": "report_volume_profile",
        "report_wyckoff": "report_wyckoff",
        "report_fundamental": "report_fundamental",
        "report_trading_strategy": "report_trading_strategy",
        "report_entry_strategy": "report_entry_strategy",
        "report_exit_strategy": "report_exit_strategy",
        "report_position_sizing": "report_position_sizing",
        "report_risk_management": "report_risk_management",
        "report_forward_outlook": "report_forward_outlook",
        "report_scenario_analysis": "report_scenario_analysis",
        "report_conclusion": "report_conclusion"
    }
    
    wf.node_input_mappings["format_results"] = {
        "financial_data_collection": "financial_data_collection",
        "technical_analysis": "technical_analysis",
        "financial_analysis": "financial_analysis",
        "trading_analysis": "trading_analysis",
        "financial_report": "financial_report"
    }
    
    logger.info("Financial analysis workflow created")
    return wf

# Run Workflow
async def analyze_financial_markets(
    symbols: List[str],
    interval: str = "1d",
    range_type: str = "month",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model: str = "openai/gpt-4o-mini",
    task_id: str = "default",
    _handle_event: Optional[callable] = None) -> Dict[str, Any]:
    """Execute the financial analysis workflow with the given symbols and parameters."""
    initial_context = {
        "symbols_list": symbols,
        "interval": interval,
        "range_type": range_type,
        "start_date": start_date,
        "end_date": end_date,
        "model": model,
        "get_price_range": get_price_range  # Add helper function to context
    }
    
    workflow = create_financial_analysis_workflow()
    engine = workflow.build()
    
    # Add the event observer if _handle_event is provided
    # if _handle_event:
    #     bound_observer = lambda event: asyncio.create_task(
    #         event_observer(event, task_id=task_id, _handle_event=_handle_event)
    #     )
    #     engine.add_observer(bound_observer)
    
    result = await engine.run(initial_context)
    
    logger.info(f"Financial analysis completed for {len(symbols)} symbols")
    return result

@app.command()
def analyze(
    symbols: List[str] = typer.Argument(..., help="List of symbols to analyze (e.g., AAPL MSFT BTC-USD)"),
    interval: str = typer.Option("1d", help="Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d)"),
    range_type: str = typer.Option("month", help="Type of date range (today, date, week, month, ytd)"),
    start_date: Optional[str] = typer.Option(None, help="Start date in YYYY-MM-DD format"),
    end_date: Optional[str] = typer.Option(None, help="End date in YYYY-MM-DD format"),
    output_file: Optional[str] = typer.Option(None, help="Output file path for the analysis results")
):
    """Analyze financial data and generate trading recommendations."""
    result = asyncio.run(analyze_financial_markets(
        symbols=symbols,
        interval=interval,
        range_type=range_type,
        start_date=start_date,
        end_date=end_date
    ))
    
    # Print a summary to the console
    print(f"\nFinancial Analysis Summary ({datetime.now().isoformat()})\n")
    print(f"Symbols analyzed: {', '.join(symbols)}")
    print(f"Interval: {interval}, Range: {range_type}")
    
    # Print financial analysis overview
    print("\nFinancial Analysis Overview:")
    print(result["financial_analysis"]["overview"])
    
    # Print key insights
    print("\nKey Insights:")
    for insight in result["financial_analysis"]["key_insights"]:
        print(f"- {insight}")
    
    # Print trading recommendations
    print("\nTrading Recommendations:")
    for rec in result["trading_recommendations"]["recommendations"]:
        print(f"\n{rec['symbol']} - {rec['action'].upper()} (Confidence: {rec['confidence']}/10, Risk: {rec['risk_level']}/10)")
        print(f"Entry points: {', '.join(map(str, rec['entry_points']))}")
        print(f"Stop loss: {rec['stop_loss']}")
        print(f"Take profit: {', '.join(map(str, rec['take_profit']))}")
        print(f"Time horizon: {rec['time_horizon']}")
        print(f"Rationale: {rec['rationale']}")
    
    # Save to file if specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull analysis saved to {output_file}")

def main():
    """Test the financial analysis flow with predefined symbols and parameters."""
    # Define test symbols and parameters
    test_symbols = ["GC=F"]
    test_interval = "1h"
    test_range = "week"
    
    print(f"\nRunning financial analysis for: {', '.join(test_symbols)}")
    print(f"Interval: {test_interval}, Range: {test_range}\n")
    
    try:
        # Run the analysis
        result = asyncio.run(analyze_financial_markets(
            symbols=test_symbols,
            interval=test_interval,
            range_type=test_range,
            model="openai/gpt-4o-mini"  # Explicitly use OpenAI model
        ))
        
        # Access the Pydantic model attributes directly
        if "financial_analysis" in result and "trading_analysis" in result:
            financial_analysis = result["financial_analysis"]
            trading_analysis = result["trading_analysis"]
            
            # Print financial analysis overview
            print("\n=== Financial Analysis Overview ===")
            print(financial_analysis.overview)
            
            # Print key insights
            print("\n=== Key Insights ===")
            for insight in financial_analysis.key_insights:
                print(f"- {insight}")
            
            # Print market trends
            print("\n=== Market Trends ===")
            for trend in financial_analysis.market_trends:
                print(f"{trend.symbol}: {trend.trend} (Strength: {trend.strength}/10)")
                print(f"Key Levels: {', '.join([f'{k}: {v}' for k, v in trend.key_levels.items()])}")
            
            # Print trading recommendations
            print("\n=== Trading Recommendations ===")
            for rec in trading_analysis.recommendations:
                print(f"\n{rec.symbol} - {rec.action.upper()} (Confidence: {rec.confidence}/10, Risk: {rec.risk_level}/10)")
                print(f"Entry points: {', '.join(map(str, rec.entry_points))}")
                print(f"Stop loss: {rec.stop_loss}")
                print(f"Take profit: {', '.join(map(str, rec.take_profit))}")
                print(f"Time horizon: {rec.time_horizon}")
                print(f"Rationale: {rec.rationale}")
                
            # Print financial report summary
            financial_report = result["financial_report"]
            print("\n=== Financial Report ===")
            print("\nExecutive Summary:")
            print(financial_report.executive_summary)
            print("\nConclusion:")
            print(financial_report.conclusion)
            
            # Convert to dict for JSON serialization and save to file
            result_dict = {
                "financial_data": result["financial_data_collection"].model_dump(),
                "technical_analysis": result["technical_analysis"].model_dump(),
                "financial_analysis": financial_analysis.model_dump(),
                "trading_analysis": trading_analysis.model_dump(),
                "financial_report": financial_report.model_dump()
            }
            
            # Save the full report to a separate markdown file
            report_file = "financial_report.md"
            with open(report_file, "w") as f:
                f.write(f"# Financial Analysis Report - {datetime.now().strftime('%Y-%m-%d')}\n\n")
                f.write(f"## Executive Summary\n\n{financial_report.executive_summary}\n\n")
                f.write(f"## Market Context\n\n{financial_report.market_context}\n\n")
                f.write(f"## Technical Analysis Overview\n\n{financial_report.technical_analysis_overview}\n\n")
                f.write(f"## Market Structure Analysis\n\n{financial_report.market_structure_analysis}\n\n")
                f.write(f"## ICT & SMC Analysis\n\n{financial_report.ict_smc_analysis}\n\n")
                f.write(f"## Fibonacci Analysis\n\n{financial_report.fibonacci_analysis}\n\n")
                f.write(f"## Volume Profile Analysis\n\n{financial_report.volume_profile_analysis}\n\n")
                f.write(f"## Wyckoff Analysis\n\n{financial_report.wyckoff_analysis}\n\n")
                f.write(f"## Fundamental Factors\n\n{financial_report.fundamental_factors}\n\n")
                f.write(f"## Trading Strategy\n\n{financial_report.trading_strategy}\n\n")
                f.write(f"## Entry Strategy\n\n{financial_report.entry_strategy}\n\n")
                f.write(f"## Exit Strategy\n\n{financial_report.exit_strategy}\n\n")
                f.write(f"## Position Sizing\n\n{financial_report.position_sizing}\n\n")
                f.write(f"## Risk Management\n\n{financial_report.risk_management}\n\n")
                f.write(f"## Forward Outlook\n\n{financial_report.forward_outlook}\n\n")
                f.write(f"## Scenario Analysis\n\n{financial_report.scenario_analysis}\n\n")
                f.write(f"## Conclusion\n\n{financial_report.conclusion}\n\n")
            print(f"\nDetailed financial report saved to {report_file}")
            
            # Save results to a file
            output_file = "financial_analysis_results.json"
            with open(output_file, "w") as f:
                json.dump(result_dict, f, indent=2)
            print(f"\nFull analysis saved to {output_file}")
        else:
            print("\nError: Missing expected data in results.")
            print(f"Available keys: {list(result.keys())}")

    except Exception as e:
        print(f"\nError running financial analysis: {e}")

if __name__ == "__main__":
    # Run the direct test function instead of the CLI app
    main()