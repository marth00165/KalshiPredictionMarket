"""Strategy module for filtering markets and generating trade signals"""

import logging
from typing import List, Tuple

from models import MarketData, FairValueEstimate, TradeSignal
from utils import (
    ConfigManager,
    InsufficientCapitalError,
    NoOpportunitiesError,
    PositionLimitError,
)

logger = logging.getLogger(__name__)


class KellyCriterion:
    """Calculate optimal position sizes using Kelly Criterion"""
    
    @staticmethod
    def calculate_kelly_fraction(
        probability: float,
        market_price: float,
        max_fraction: float = 0.25
    ) -> float:
        """
        Calculate Kelly fraction for binary outcome
        
        Kelly = (p * (b + 1) - 1) / b
        where:
        - p = true probability of winning
        - b = odds received on bet (decimal odds - 1)
        
        For prediction markets:
        - If buying YES at price p_market, odds = (1/p_market) - 1
        - Kelly = (p_true * (1/p_market) - 1) * p_market
        """
        
        if market_price <= 0 or market_price >= 1:
            return 0
        
        # Calculate edge
        edge = probability - market_price
        
        if edge <= 0:
            return 0  # No edge, no bet
        
        # Simplified Kelly for binary markets
        # Kelly = edge / (1 - market_price)
        kelly = edge / (1 - market_price)
        
        # Cap at max_fraction to reduce risk
        kelly_fraction = min(kelly, max_fraction)
        
        # Don't bet if edge is too small
        if kelly_fraction < 0.01:  # Less than 1%
            return 0
        
        return kelly_fraction
    
    @staticmethod
    def calculate_position_size(
        kelly_fraction: float,
        bankroll: float,
        max_position: float = 1000
    ) -> float:
        """Calculate dollar amount to bet"""
        
        position = kelly_fraction * bankroll
        
        # Apply position limits
        position = min(position, max_position)
        
        return round(position, 2)


class Strategy:
    """
    Trading strategy: filter markets, find opportunities, generate signals
    
    Encapsulates all strategy logic separate from execution.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize strategy with configuration
        
        Args:
            config: ConfigManager with strategy, risk, and filter settings
        """
        self.config = config
        self.kelly = KellyCriterion()
    
    # ========================================================================
    # MARKET FILTERING
    # ========================================================================
    
    def filter_markets(self, markets: List[MarketData]) -> List[MarketData]:
        """
        Filter markets by volume, liquidity, and price extremes
        
        Args:
            markets: List of all available markets
        
        Returns:
            Filtered list of tradeable markets
        """
        
        min_volume = self.config.filters.min_volume
        min_liquidity = self.config.filters.min_liquidity
        
        filtered = [
            m for m in markets
            if m.volume >= min_volume and m.liquidity >= min_liquidity
        ]
        
        # Remove markets with extreme prices (likely already resolved)
        filtered = [
            m for m in filtered
            if 0.01 < m.yes_price < 0.99
        ]
        
        logger.debug(f"Filtered {len(markets)} → {len(filtered)} markets")
        return filtered
    
    # ========================================================================
    # OPPORTUNITY FINDING
    # ========================================================================
    
    def find_opportunities(
        self,
        estimates: List[FairValueEstimate],
        markets: List[MarketData]
    ) -> List[Tuple[MarketData, FairValueEstimate]]:
        """
        Find markets with significant mispricings
        
        Args:
            estimates: Claude's fair value estimates for markets
            markets: Market data with current prices
        
        Returns:
            List of (market, estimate) tuples for mispriced markets
        """
        
        # Create lookup dict
        market_dict = {m.market_id: m for m in markets}
        
        opportunities = []
        
        min_edge = self.config.strategy.min_edge
        min_confidence = self.config.strategy.min_confidence
        
        for est in estimates:
            if est.has_significant_edge(min_edge, min_confidence):
                market = market_dict.get(est.market_id)
                if market:
                    opportunities.append((market, est))
        
        # Sort by edge * confidence (best opportunities first)
        opportunities.sort(
            key=lambda x: abs(x[1].edge) * x[1].confidence_level,
            reverse=True
        )
        
        logger.debug(f"Found {len(opportunities)} opportunities with >{min_edge*100:.0f}% edge")
        return opportunities
    
    # ========================================================================
    # SIGNAL GENERATION
    # ========================================================================
    
    def generate_trade_signals(
        self,
        opportunities: List[Tuple[MarketData, FairValueEstimate]],
        current_bankroll: float,
        current_exposure: float = 0.0
    ) -> List[TradeSignal]:
        """
        Generate trade signals with Kelly criterion position sizing
        
        Args:
            opportunities: List of (market, estimate) tuples
            current_bankroll: Current available capital
            current_exposure: Total capital already at risk
        
        Returns:
            List of ready-to-execute TradeSignal objects
        
        Raises:
            InsufficientCapitalError: If no capital available for trading
            PositionLimitError: If max positions reached before any signals generated
            NoOpportunitiesError: If opportunities list is empty
        """
        
        if not opportunities:
            raise NoOpportunitiesError("No opportunities provided for signal generation")
        
        if current_bankroll <= 0:
            raise InsufficientCapitalError(
                required=1.0,
                available=current_bankroll
            )
        
        signals = []
        max_positions = self.config.risk.max_positions
        
        for market, estimate in opportunities[:max_positions]:
            # Don't add more positions if we already have the max
            if len(signals) >= max_positions:
                raise PositionLimitError(
                    current_positions=len(signals),
                    max_allowed=max_positions
                )
            
            # Determine action based on edge direction
            if estimate.is_buy_yes_signal():
                action = 'buy_yes'
                market_price = market.yes_price
            else:
                action = 'buy_no'
                market_price = market.no_price
            
            # Calculate Kelly fraction
            kelly_frac = self.kelly.calculate_kelly_fraction(
                probability=estimate.estimated_probability,
                market_price=market_price,
                max_fraction=self.config.risk.max_kelly_fraction
            )
            
            if kelly_frac <= 0:
                logger.debug(f"Skipping {market.market_id}: Kelly fraction too small ({kelly_frac})")
                continue
            
            # Calculate position size
            position_size = self.kelly.calculate_position_size(
                kelly_fraction=kelly_frac,
                bankroll=current_bankroll,
                max_position=self.config.risk.max_position_size
            )
            
            # Skip if position would exceed available capital
            if position_size > current_bankroll:
                logger.warning(
                    f"Skipping {market.market_id}: position size ${position_size:.2f} "
                    f"exceeds bankroll ${current_bankroll:.2f}"
                )
                continue
            
            # Calculate expected value
            if estimate.is_buy_yes_signal():
                # Buying YES at market_price
                ev = (estimate.estimated_probability * (1 - market_price) - 
                      (1 - estimate.estimated_probability) * market_price) * position_size
            else:
                # Buying NO at market_price
                ev = ((1 - estimate.estimated_probability) * (1 - market_price) - 
                      estimate.estimated_probability * market_price) * position_size
            
            # Create signal
            try:
                signal = TradeSignal(
                    market=market,
                    action=action,
                    fair_value=estimate.estimated_probability,
                    market_price=market_price,
                    edge=estimate.edge,
                    kelly_fraction=kelly_frac,
                    position_size=position_size,
                    expected_value=ev,
                    reasoning=estimate.reasoning[:200] if estimate.reasoning else ""
                )
                signals.append(signal)
                logger.debug(f"Generated signal: {signal}")
            
            except ValueError as e:
                logger.error(f"Error creating signal for {market.market_id}: {e}")
                continue
        
        if not signals:
            raise NoOpportunitiesError(
                "No valid signals generated (all opportunities filtered or failed validation)"
            )
        
        logger.info(f"Generated {len(signals)} trade signals")
        return signals
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_strategy_summary(self) -> dict:
        """Get current strategy settings"""
        return {
            'min_edge': self.config.min_edge_percentage,
            'min_confidence': self.config.min_confidence_percentage,
            'max_positions': self.config.risk.max_positions,
            'max_kelly_fraction': self.config.risk.max_kelly_fraction,
            'max_position_size': self.config.risk.max_position_size,
            'min_volume': self.config.filters.min_volume,
            'min_liquidity': self.config.filters.min_liquidity,
        }
