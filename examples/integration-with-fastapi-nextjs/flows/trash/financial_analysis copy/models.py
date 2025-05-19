"""Models for enhanced financial market analysis."""
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime

class PriceLevel(BaseModel):
    level: float
    type: str  # "support", "resistance", "dynamic", "orderblock", "liquidity"
    strength: int  # 1-10
    description: str
    confidence: float  # 0-1
    timeframe: str  # e.g., "1h", "4h", "1d"

class OrderBlock(BaseModel):
    price_range: tuple[float, float]
    type: str  # "buy", "sell"
    strength: int  # 1-10
    timeframe: str
    description: str
    volume_profile: dict  # Volume analysis at the order block
    mitigation_status: str  # "active", "partially_mitigated", "mitigated"

class MarketContext(BaseModel):
    market_condition: str  # "trending", "ranging", "volatile"
    sentiment: str  # "bullish", "bearish", "neutral"
    key_events: List[str]  # Major market events/news
    volatility_state: str  # "high", "medium", "low"
    liquidity_state: str  # "high", "medium", "low"
    institutional_bias: str  # "bullish", "bearish", "neutral"
    description: str

class SmartMoneyAnalysis(BaseModel):
    order_blocks: List[OrderBlock]
    liquidity_levels: List[PriceLevel]
    institutional_levels: List[PriceLevel]
    manipulation_scenarios: List[str]
    volume_analysis: Dict[str, float]  # Key metrics from volume analysis
    order_flow_bias: str  # "bullish", "bearish", "neutral"
    description: str

class TechnicalIndicator(BaseModel):
    name: str
    value: float
    signal: str  # "buy", "sell", "neutral"
    description: str

class AdvancedPattern(BaseModel):
    name: str
    type: str  # "reversal", "continuation", "complex"
    reliability: int  # 1-10
    description: str
    entry_points: List[float]
    targets: List[float]
    stop_loss: float
    timeframes: List[str]  # Timeframes where pattern is valid
    confluence_factors: List[str]  # Other supporting factors

class TradingPlan(BaseModel):
    entry_strategy: str
    exit_strategy: str
    position_sizing: Dict[str, float]
    risk_parameters: Dict[str, float]
    trade_management: List[str]
    scenarios: List[str]
    contingency_plans: List[str]
    timeframe_analysis: Dict[str, str]  # Analysis for each timeframe

class EnhancedMarketAnalysis(BaseModel):
    symbol: str
    timestamp: datetime
    market_context: MarketContext
    smart_money_analysis: SmartMoneyAnalysis
    advanced_patterns: List[AdvancedPattern]
    technical_indicators: List[TechnicalIndicator]
    trading_plan: TradingPlan
    risk_level: int  # 1-10
    recommendation: str
    summary: str
    plots: List[str]  # Base64 encoded plot data

class VolumeAnalysis(BaseModel):
    average_volume: float
    volume_trend: str  # "increasing", "decreasing", "stable"
    notable_levels: List[float]
    description: str

class MarketStructure(BaseModel):
    structure_type: str  # "accumulation", "distribution", "markup", "markdown"
    key_levels: List[PriceLevel]
    description: str

class TradingSignal(BaseModel):
    type: str  # "entry" or "exit"
    price: float
    direction: str  # "long" or "short"
    confidence: int  # 1-10
    description: str
    stop_loss: float
    take_profit: List[float]
    risk_reward_ratio: float

class Strategy(BaseModel):
    name: str
    timeframe: str
    signals: List[TradingSignal]
    description: str
    performance_metrics: Dict[str, float]

class TrendInfo(BaseModel):
    direction: str  # "bullish", "bearish", "sideways"
    strength: int  # 1-10
    key_levels: List[PriceLevel]
    description: str
    momentum_score: float  # 0-1
    trend_age: int  # Number of candles in current trend
    trend_health: str  # "healthy", "weakening", "reversal_likely"

class PatternInfo(BaseModel):
    name: str
    type: str  # "reversal", "continuation"
    reliability: int  # 1-10
    description: str
    entry_points: List[float]
    targets: List[float]
    stop_loss: float
    confirmation_signals: List[str]
    invalidation_points: List[float]

class MarketAnalysis(BaseModel):
    symbol: str
    timestamp: datetime
    price_analysis: TrendInfo
    patterns: List[PatternInfo]
    technical_indicators: List[TechnicalIndicator]
    volume_analysis: VolumeAnalysis
    market_structure: MarketStructure
    strategies: List[Strategy]
    kpis: Dict[str, float]  # Added KPIs field
    summary: str
    risk_level: int  # 1-10
    recommendation: str
    plots: List[str]  # Base64 encoded plotly figures

class MarketSentiment(BaseModel):
    overall_sentiment: str  # bullish, bearish, neutral
    confidence_score: float  # 0-1
    key_factors: List[str]
    risks: List[str]
    opportunities: List[str]

class TechnicalAnalysis(BaseModel):
    trend_strength: int  # 1-10
    support_levels: List[float]
    resistance_levels: List[float]
    key_indicators: Dict[str, str]  # indicator name -> signal
    pattern_analysis: str

class TradeRecommendation(BaseModel):
    action: str  # buy, sell, hold
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit_levels: List[float]
    timeframe: str
    confidence_score: float  # 0-1
    rationale: List[str]

class StructuredAnalysis(BaseModel):
    timestamp: datetime
    symbol: str
    market_sentiment: MarketSentiment
    technical_analysis: TechnicalAnalysis
    trade_recommendation: TradeRecommendation
    summary: str
    risk_rating: int  # 1-10
