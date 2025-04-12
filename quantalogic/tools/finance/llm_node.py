"""Specialized LLM Tool for financial trading analysis and strategy generation."""

import asyncio
from typing import Callable, ClassVar, List
from enum import Enum

from loguru import logger
from pydantic import ConfigDict, Field, BaseModel

from quantalogic.console_print_token import console_print_token
from quantalogic.event_emitter import EventEmitter
from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.tools.llm_tool import LLMTool


class TimeFrame(str, Enum):
    """Trading timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "daily"
    W1 = "weekly"


class MarketStructure(BaseModel):
    """Market structure analysis results."""
    trend: str
    key_levels: List[float]
    support_zones: List[float]
    resistance_zones: List[float]
    liquidity_levels: List[float]
    order_blocks: List[dict]
    fair_value_gaps: List[dict]


class FinanceLLMTool(LLMTool):
    """Advanced LLM tool specialized for financial trading analysis and strategy generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="finance_llm_tool")
    description: str = Field(
        default=(
            "Advanced financial analysis tool with multi-timeframe support, ICT concepts, "
            "and comprehensive market structure analysis. Provides deep market analysis "
            "across multiple timeframes and trading methodologies."
        )
    )

    # Default system prompt for financial analysis
    DEFAULT_SYSTEM_PROMPT: ClassVar[str] = """You are an elite financial trading advisor with mastery in:

1. Advanced Technical Analysis
   - ICT Concepts (Internal, Composite, Technical)
   - Order Flow Analysis
   - Order Blocks and Breaker Blocks
   - Fair Value Gaps (FVG)
   - Liquidity Pools and Sweeps
   - Smart Money Concepts (SMC)
   - Market Structure (MS)
   - Wyckoff Method
   - Volume Profile Analysis
   - Market Profile Analysis
   - Footprint Charts
   - Delta Analysis
   - Fibonacci Relationships
   - Harmonic Patterns
   - Elliott Wave Theory

2. Multi-Timeframe Analysis
   - Higher Timeframe (HTF) Structure
   - Lower Timeframe (LTF) Execution
   - Timeframe Relationships
   - Trend Alignment
   - Nested Market Structure
   - Fractal Analysis

3. Advanced Risk Management
   - Position Sizing Optimization
   - Risk:Reward Scenarios
   - Portfolio Heat Management
   - Correlation Risk Analysis
   - Drawdown Management
   - Risk Distribution Models
   - Volatility-Based Sizing

4. Market Psychology & Order Flow
   - Institutional Order Flow
   - Retail Order Flow
   - Market Maker Methods
   - Stop Hunt Patterns
   - Liquidity Engineering
   - Price Action Psychology
   - Volume Analysis

5. Advanced Trading Strategies
   - ICT Concepts Implementation
   - Smart Money Strategy
   - Institutional Order Flow
   - Algorithmic Pattern Recognition
   - Mean Reversion
   - Momentum Trading
   - Volatility Trading
   - Range Trading
   - Breakout Systems
   - Market Profile Trading

6. Comprehensive Market Analysis
   - Intermarket Analysis
   - Cross-Asset Correlations
   - Global Macro Impact
   - Market Microstructure
   - Order Flow Imbalances
   - Liquidity Analysis
   - Volume Analysis
   - Market Depth Analysis

Analysis Guidelines:
1. Start with HTF analysis for context
2. Identify key market structure levels
3. Locate institutional order blocks
4. Map liquidity pools and sweeps
5. Identify FVGs and breaker blocks
6. Analyze order flow and volume
7. Consider multiple timeframe alignment
8. Evaluate risk scenarios

Response Structure:
1. Market Structure Analysis
   - Trend analysis across timeframes
   - Key structural levels
   - Order blocks and breaker blocks
   - Fair value gaps
   - Liquidity levels

2. Order Flow Analysis
   - Volume analysis
   - Delta divergence
   - Institutional footprint
   - Stop order clusters

3. Trade Opportunities
   - Entry scenarios
   - Stop placement
   - Take profit targets
   - Position sizing
   - Risk management rules

4. Risk Assessment
   - Multiple scenario analysis
   - Risk:reward calculations
   - Position sizing recommendations
   - Portfolio impact analysis
"""

    def __init__(
        self,
        model_name: str,
        system_prompt: str | None = None,
        on_token: Callable | None = None,
        name: str = "finance_llm_tool",
        generative_model: GenerativeModel | None = None,
        event_emitter: EventEmitter | None = None,
    ):
        """Initialize the FinanceLLMTool with specialized configuration."""
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            on_token=on_token,
            name=name,
            generative_model=generative_model,
            event_emitter=event_emitter,
        )

    async def analyze_market_structure(
        self,
        symbol: str,
        timeframes: List[str] = ["4h", "1h", "15m"],
        analysis_type: str = "comprehensive",
        temperature: str = "0.7",
    ) -> str:
        """Perform deep market structure analysis across multiple timeframes.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD", "AAPL", "EUR/USD")
            timeframes: List of timeframes to analyze
            analysis_type: Type of analysis ("technical", "orderflow", "comprehensive")
            temperature: Model temperature for response generation

        Returns:
            Detailed multi-timeframe market analysis
        """
        prompt = f"""Conduct a deep market structure analysis for {symbol} across multiple timeframes: {', '.join(timeframes)}.

Please provide:
1. Higher Timeframe Context
   - Market structure analysis
   - Key structural levels
   - Order blocks and breaker blocks
   - Fair value gaps
   - Institutional order flow

2. Multiple Timeframe Analysis
   For each timeframe ({', '.join(timeframes)}):
   - Market structure status
   - Key levels and zones
   - Order blocks and liquidity
   - Fair value gaps
   - Volume analysis
   - Delta divergence

3. Order Flow Analysis
   - Institutional footprint
   - Volume profile analysis
   - Delta analysis
   - Stop order clusters
   - Liquidity pools

4. Trade Opportunities
   - Key entry zones
   - Stop loss placement
   - Take profit targets
   - Position sizing
   - Risk management rules

5. Risk Assessment
   - Multiple scenario analysis
   - Risk:reward calculations
   - Position sizing recommendations
   - Portfolio impact analysis

Analysis type: {analysis_type}"""

        return await self.async_execute(prompt=prompt, temperature=temperature)

    async def generate_advanced_strategy(
        self,
        strategy_type: str,
        market_type: str,
        timeframes: List[str] = ["4h", "1h", "15m"],
        risk_level: str = "moderate",
        include_ict: bool = True,
        temperature: str = "0.7",
    ) -> str:
        """Generate a comprehensive trading strategy with ICT concepts.

        Args:
            strategy_type: Type of strategy (e.g., "ict", "smc", "orderflow")
            market_type: Target market (e.g., "crypto", "forex", "stocks")
            timeframes: List of timeframes to consider
            risk_level: Risk tolerance level
            include_ict: Include ICT concepts in the strategy
            temperature: Model temperature for response generation

        Returns:
            Detailed trading strategy with ICT concepts
        """
        advanced_concepts = """
Market Analysis Components:

1. Smart Money Concepts (SMC)
   - Supply and Demand zones
   - Institutional order blocks
   - Liquidity pools identification
   - Market manipulation tactics
   - Stop hunt zones
   - Fair value gaps (FVG)
   - Breaker blocks
   - Mitigation blocks

2. Order Flow Analysis
   - Volume profile analysis
   - Footprint charts
   - Delta volume analysis
   - Cumulative volume delta
   - Market depth analysis
   - Time and sales analysis
   - Order flow imbalances
   - Absorption analysis

3. Institutional Trading Concepts (ICT)
   - Premium/discount zones
   - Optimal trade entry
   - Institutional order flow
   - Market structure analysis
   - Liquidity engineering
   - Manipulation points
   - Inefficient price points
   - Price delivery points

4. Advanced Price Action
   - Wyckoff method integration
   - Market structure shifts
   - Swing failure patterns
   - Equal highs/lows
   - Deviation analysis
   - Range expansion/contraction
   - Price rejection analysis
   - Volume spread analysis (VSA)

5. Multi-Timeframe Analysis
   - Higher timeframe dominance
   - Lower timeframe confirmation
   - Timeframe relationships
   - Nested market structure
   - Fractal pattern analysis
   - Time projection tools
   - Cycle analysis
   - Momentum transfer"""

        prompt = f"""As a senior trading analyst with over 25 years of experience, provide a comprehensive market analysis for {market_type} markets
focusing on {strategy_type} strategies across timeframes: {', '.join(timeframes)}.

{advanced_concepts}

Analysis Requirements:

1. Market Context Analysis
   - Current market regime identification
   - Macro perspective and influences
   - Intermarket correlations
   - Market sentiment indicators
   - Key structural levels mapping
   - Volume profile assessment
   - Institutional activity signals

2. Technical Structure Analysis
   - Multi-timeframe market structure
   - Key support/resistance zones
   - Supply/demand imbalances
   - Order block identification
   - Liquidity levels mapping
   - Fair value gaps analysis
   - Breaker block locations
   - Mitigation blocks assessment

3. Volume Analysis
   - Volume profile distribution
   - Delta volume patterns
   - Institutional volume signals
   - Volume node identification
   - Absorption/distribution patterns
   - Cumulative delta analysis
   - Time-weighted analysis
   - Price-volume relationships

4. Order Flow Assessment
   - Large order detection
   - Order flow imbalances
   - Market depth analysis
   - Footprint patterns
   - Absorption levels
   - Institutional positioning
   - Stop order clusters
   - Liquidity sweeps

5. Market Inefficiencies
   - Fair value gaps
   - Price inefficiencies
   - Liquidity voids
   - Premium/discount zones
   - Manipulation points
   - Stop hunt zones
   - Deviation levels
   - Price delivery zones

6. Pattern Recognition
   - Wyckoff phases
   - Market structure shifts
   - Swing failure patterns
   - Equal highs/lows analysis
   - Range analysis
   - Distribution patterns
   - Accumulation patterns
   - Reversal signatures

7. Risk Assessment
   - Volatility analysis
   - Market correlation impact
   - Risk exposure levels
   - Position sizing scenarios
   - Portfolio heat analysis
   - Maximum drawdown projections
   - Risk-reward scenarios
   - Leverage considerations

8. Execution Framework
   - Key timeframe alignments
   - Optimal execution zones
   - Risk parameters
   - Position sizing guidelines
   - Entry zone identification
   - Exit zone mapping
   - Stop placement levels
   - Scale-in/out points

9. Market Regime Analysis
   - Trend strength assessment
   - Volatility regime analysis
   - Momentum characteristics
   - Range identification
   - Cycle analysis
   - Seasonal patterns
   - Market efficiency metrics
   - Regime transition signals

10. Institutional Activity
    - Smart money footprints
    - Large player positioning
    - Order flow patterns
    - Block trade analysis
    - Dark pool activity
    - Market maker activity
    - Institutional accumulation/distribution
    - Smart money divergence

Please provide a detailed analysis considering all these aspects, highlighting the most significant factors currently affecting the market."""

        return await self.async_execute(prompt=prompt, temperature=temperature)


if __name__ == "__main__":
    # Initialize the finance tool with streaming capability
    finance_tool = FinanceLLMTool(
        model_name="gemini/gemini-2.0-flash", # "deepseek/deepseek-chat",
        on_token=console_print_token
    )

    async def run_examples():
        # Example 1: Multi-timeframe Market Analysis
        print("\n=== Bitcoin Multi-timeframe Analysis ===")
        btc_analysis = await finance_tool.analyze_market_structure(
            symbol="BTC/USD",
            timeframes=["4h", "1h", "15m"],
            analysis_type="comprehensive"
        )
        print(f"\nAnalysis Result:\n{btc_analysis}")

        # Example 2: Advanced ICT Strategy Generation
        print("\n=== Generate ICT Trading Strategy ===")
        strategy = await finance_tool.generate_advanced_strategy(
            strategy_type="ict",
            market_type="crypto",
            timeframes=["4h", "1h", "15m"],
            risk_level="moderate",
            include_ict=True
        )
        print(f"\nStrategy:\n{strategy}")

    # Run the examples
    asyncio.run(run_examples())