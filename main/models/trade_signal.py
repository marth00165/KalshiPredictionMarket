"""Trade signal model with Kelly criterion position sizing"""

from dataclasses import dataclass
from .market_data import MarketData


@dataclass
class TradeSignal:
    """
    Complete trading signal with position sizing calculated via Kelly criterion
    
    Represents a ready-to-execute trade including the market, direction, sizing,
    and reasoning.
    """
    
    market: MarketData  # The market to trade
    action: str  # 'buy_yes', 'buy_no', 'sell_yes', 'sell_no'
    fair_value: float  # Claude's estimated fair probability (0-1)
    market_price: float  # Current market price for the outcome we're trading (0-1)
    edge: float  # Estimated edge: fair_value - market_price (can be +/-)
    kelly_fraction: float  # Kelly criterion position sizing (0.0-1.0 of bankroll)
    position_size: float  # Dollar amount to invest in this trade
    expected_value: float  # Expected profit/loss in dollars
    reasoning: str  # Summary of why this trade is recommended
    
    def __post_init__(self):
        """Validate trade signal after initialization"""
        valid_actions = {'buy_yes', 'buy_no', 'sell_yes', 'sell_no'}
        if self.action not in valid_actions:
            raise ValueError(f"action must be one of {valid_actions}, got {self.action}")
        
        if not 0 <= self.fair_value <= 1:
            raise ValueError(f"fair_value must be between 0 and 1, got {self.fair_value}")
        
        if not 0 <= self.market_price <= 1:
            raise ValueError(f"market_price must be between 0 and 1, got {self.market_price}")
        
        if not 0 <= self.kelly_fraction <= 1:
            raise ValueError(f"kelly_fraction must be between 0 and 1, got {self.kelly_fraction}")
        
        if self.position_size < 0:
            raise ValueError(f"position_size must be >= 0, got {self.position_size}")
    
    def __str__(self) -> str:
        """Human-readable representation"""
        return (
            f"{self.action.upper()}: {self.market.title[:40]}... "
            f"| Fair: {self.fair_value:.1%} vs Market: {self.market_price:.1%} "
            f"| ${self.position_size:,.2f} | EV: ${self.expected_value:+,.2f}"
        )
    
    @property
    def is_bullish(self) -> bool:
        """Returns True if signal is to buy YES (bullish outcome)"""
        return self.action in ('buy_yes', 'sell_no')
    
    @property
    def is_bearish(self) -> bool:
        """Returns True if signal is to buy NO (bearish outcome)"""
        return self.action in ('buy_no', 'sell_yes')
    
    @property
    def is_buy(self) -> bool:
        """Returns True if signal involves buying (not selling)"""
        return 'buy' in self.action
    
    @property
    def is_sell(self) -> bool:
        """Returns True if signal involves selling (not buying)"""
        return 'sell' in self.action
    
    def risk_adjusted_ev(self) -> float:
        """
        Calculate risk-adjusted expected value accounting for position size
        
        A larger position size carries proportionally more risk.
        """
        # EV already accounts for position size, but we can normalize to per-dollar basis
        if self.position_size > 0:
            return self.expected_value / self.position_size
        return 0.0
