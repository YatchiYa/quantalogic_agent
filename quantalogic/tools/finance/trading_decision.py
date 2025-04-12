"""Trading decision tool for position taking based on comprehensive market analysis."""

import asyncio
from typing import Callable, List, Dict, Optional, ClassVar
from enum import Enum
from dataclasses import dataclass
from decimal import Decimal

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict

from quantalogic.console_print_token import console_print_token
from quantalogic.event_emitter import EventEmitter
from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.tools.llm_tool import LLMTool


class TradingDecision(str, Enum):
    """Possible trading decisions."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    WAIT = "wait"


class EntryType(str, Enum):
    """Types of trade entries."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class PriceLevel:
    """Price level with description."""
    price: Decimal
    description: str
    confidence: float  # 0-1


class TradeSetup(BaseModel):
    """Complete trade setup with entry, exit, and risk parameters."""
    
    symbol: str
    decision: TradingDecision
    entry_type: EntryType
    timeframe: str
    confidence_score: float  # 0-1

    # Entry parameters
    entry_price: Decimal
    stop_loss: Decimal
    take_profit_levels: List[PriceLevel]
    
    # Risk parameters
    position_size: Decimal
    risk_amount: Decimal
    reward_potential: Decimal
    risk_reward_ratio: float

    # Market context
    market_structure: str
    key_levels: List[PriceLevel]
    catalysts: List[str]
    warnings: List[str]

    # Trade management
    entry_triggers: List[str]
    exit_rules: List[str]
    adjustment_rules: List[str]

    class Config:
        arbitrary_types_allowed = True


class TradingDecisionTool(LLMTool):
    """Advanced LLM tool for making trading decisions."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="trading_decision_tool")
    description: str = Field(
        default=(
            "Advanced trading decision tool that analyzes market conditions "
            "across multiple timeframes and provides specific trade setups "
            "with entry, exit, and risk management parameters."
        )
    )

    DEFAULT_SYSTEM_PROMPT: ClassVar[str] = """You are an elite trading decision maker with expertise in:

1. Decision Making Process
   - Multi-timeframe confluence
   - Order flow analysis
   - Risk assessment
   - Market structure analysis
   - Position sizing optimization
   - Entry and exit timing
   - Trade management rules

2. Market Analysis Integration
   - Technical confluence
   - Order block validation
   - Liquidity analysis
   - Volume profile assessment
   - Market structure alignment
   - Institutional order flow
   - Stop order clusters

3. Risk Management
   - Position sizing calculation
   - Risk:reward optimization
   - Portfolio heat management
   - Correlation impact
   - Maximum risk per trade
   - Drawdown management
   - Risk distribution

4. Trade Execution
   - Entry type selection
   - Entry trigger definition
   - Stop loss placement
   - Take profit targeting
   - Position scaling rules
   - Trade management rules
   - Exit strategy

Decision Framework:
1. Market Context
   - Analyze higher timeframe trend
   - Identify key market structure
   - Locate institutional interest
   - Assess order flow

2. Setup Validation
   - Confirm technical confluence
   - Validate order blocks
   - Check liquidity levels
   - Verify volume structure

3. Risk Assessment
   - Calculate position size
   - Determine risk exposure
   - Set stop loss levels
   - Define take profit targets

4. Execution Plan
   - Specify entry conditions
   - Define entry triggers
   - Set trade management rules
   - Plan exit strategy

Response Requirements:
1. Clear Decision
   - Specific trading decision
   - Confidence level
   - Entry type and price
   - Stop loss and take profit

2. Risk Parameters
   - Position size
   - Risk amount
   - Reward potential
   - Risk:reward ratio

3. Trade Management
   - Entry triggers
   - Exit rules
   - Adjustment criteria
   - Warning signals

4. Context
   - Market structure
   - Key levels
   - Potential catalysts
   - Risk warnings
"""

    def __init__(
        self,
        model_name: str,
        system_prompt: str | None = None,
        on_token: Callable | None = None,
        name: str = "trading_decision_tool",
        generative_model: GenerativeModel | None = None,
        event_emitter: EventEmitter | None = None,
    ):
        """Initialize the TradingDecisionTool."""
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            on_token=on_token,
            name=name,
            generative_model=generative_model,
            event_emitter=event_emitter,
        )

    async def get_trade_decision(
        self,
        symbol: str,
        market_context: str,
        risk_profile: str = "moderate",
        max_risk_per_trade: float = 0.02,  # 2% max risk per trade
        temperature: str = "0.7",
    ) -> str:
        """Generate a comprehensive trade decision based on provided market context.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD", "AAPL", "EUR/USD")
            market_context: Comprehensive market analysis from previous analysis
            risk_profile: Risk tolerance level
            max_risk_per_trade: Maximum risk percentage per trade
            temperature: Model temperature for response generation

        Returns:
            Detailed trade decision with setup parameters
        """
        prompt = f"""Based on the provided comprehensive market analysis for {symbol}, generate a detailed trading decision.

MARKET CONTEXT:
{market_context}

RISK PARAMETERS:
• Profile: {risk_profile}
• Maximum Risk Allocation: {max_risk_per_trade*100}% per trade

Please analyze the provided market context and provide a detailed trading decision with the following structure:

1. CONTEXT ANALYSIS
   • Key Technical Levels Identified
   • Market Structure Assessment
   • Volume Profile Analysis
   • Order Flow Analysis
   • Institutional Activity Analysis
   • Risk Points Identification

2. DECISION FRAMEWORK
   A. Primary Assessment
      • Trading Bias: [STRONG_BUY/BUY/NEUTRAL/SELL/STRONG_SELL/WAIT]
      • Confidence Score: [0-1] with reasoning
      • Entry Type: [MARKET/LIMIT/STOP] with justification
      • Setup Quality Score: [0-1] with analysis

   B. Technical Confluence
      • Support/Resistance Alignment
      • Order Block Validation
      • Fair Value Gap Assessment
      • Liquidity Level Analysis
      • Volume Structure Confirmation

3. TRADE PARAMETERS
   A. Entry Strategy
      • Entry Price/Zone: [PRICE]
      • Entry Trigger Conditions
      • Technical Confirmations Required
      • Order Flow Validations

   B. Risk Management
      • Stop Loss Price: [PRICE] with justification
      • Position Size: [UNITS/CONTRACTS]
      • Risk Amount: [CURRENCY VALUE]
      • Risk:Reward Ratio: [RATIO]

   C. Profit Targets
      • TP1: [PRICE] - [REASON] - [PERCENTAGE]
      • TP2: [PRICE] - [REASON] - [PERCENTAGE]
      • TP3: [PRICE] - [REASON] - [PERCENTAGE]

4. EXECUTION PLAN
   A. Position Management
      • Entry Execution Method
      • Position Scaling Rules
      • Stop Loss Management
      • Take Profit Management

   B. Risk Factors
      • Technical Invalidation Points
      • Warning Signs to Monitor
      • Risk Mitigation Strategies
      • Position Heat Management

5. TRADE MONITORING
   • Critical Price Levels
   • Volume Thresholds
   • Order Flow Signals
   • Technical Invalidation Points
   • Risk Management Checkpoints

6. TRADE EXECUTION SUMMARY
   • Entry Price/Zone
   • Stop Loss Level
   • Take Profit Targets
   • Position Size
   • Risk Amount
   • Expected Return
   • Time Horizon
   • Key Invalidation Criteria
   
   """

        return await self.async_execute(prompt=prompt, temperature=temperature)

if __name__ == "__main__":
    # Initialize the trading decision tool
    decision_tool = TradingDecisionTool(
        model_name="gemini/gemini-2.0-flash", #  gemini/gemini-2.0-flash "deepseek/deepseek-chat",
        on_token=console_print_token
    )

    async def run_example():
        # Example market context from previous analysis
        market_context = """
        Market Analysis for BTC/USD:
        
        1. Smart Money Concepts (SMC)
        - Strong institutional order block identified at $58,500
        - Multiple fair value gaps above current price ($62,500-$63,000)
        - Clear breaker block at $64,200
        - Stop hunt zone detected below $57,800
        
        2. Order Flow Analysis
        - Heavy buying pressure at $60,000 support
        - Significant delta volume divergence (positive)
        - Large absorption of selling at current levels
        - Clear institutional footprint at $61,000
        
        3. Technical Structure
        - Bullish market structure on higher timeframes
        - Key support established at $60,000
        - Multiple liquidity pools above $65,000
        - Strong accumulation pattern visible
        
        4. Volume Analysis
        - High volume node concentrated at $61,000
        - Clear volume profile support at $59,800
        - Institutional buying clusters detected
        - Positive delta divergence on volume
        """
        
        # Get trade decision
        print("\n=== Generate Trading Decision for BTC/USD ===")
        decision = await decision_tool.get_trade_decision(
            symbol="BTC/USD",
            market_context=market_context,
            risk_profile="moderate",
            max_risk_per_trade=0.02  # 2% risk per trade
        )
        print(f"\nTrading Decision:\n{decision}")

    # Run the example
    asyncio.run(run_example())
