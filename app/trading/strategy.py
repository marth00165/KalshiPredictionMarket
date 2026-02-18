"""Strategy module for filtering markets and generating trade signals"""

import logging
from typing import List, Tuple, Optional, Set

from app.models import MarketData, FairValueEstimate, TradeSignal
from app.utils import (
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
    # DUPLICATE GUARD HELPERS
    # ========================================================================

    @staticmethod
    def get_event_key_for_platform_market(platform: str, market_id: str) -> str:
        """
        Canonical event key for duplicate guards.

        Falls back to market ticker prefix (without final '-' segment) when
        richer event metadata is unavailable.
        """
        platform_norm = str(platform or "").strip().lower()
        market_id_norm = str(market_id or "").strip()
        if "-" in market_id_norm:
            base = market_id_norm.rsplit("-", 1)[0]
        else:
            base = market_id_norm
        return f"{platform_norm}:{base}"

    def get_event_key_for_market(self, market: MarketData) -> str:
        event_ticker = str(getattr(market, "event_ticker", "") or "").strip()
        if event_ticker:
            return f"{str(market.platform).strip().lower()}:{event_ticker}"
        return self.get_event_key_for_platform_market(market.platform, market.market_id)

    @staticmethod
    def get_opportunity_score_tuple(
        market: MarketData,
        estimate: FairValueEstimate,
    ) -> tuple[float, float, str]:
        """
        Deterministic ranking score for dedupe winner selection.
        """
        abs_edge = abs(float(estimate.effective_edge))
        confidence = float(estimate.effective_confidence)
        primary = abs_edge * confidence
        return (primary, abs_edge, str(market.market_id))

    @staticmethod
    def is_better_score(
        candidate: tuple[float, float, str],
        current: tuple[float, float, str],
    ) -> bool:
        cand_primary, cand_abs_edge, cand_market_id = candidate
        cur_primary, cur_abs_edge, cur_market_id = current
        if cand_primary != cur_primary:
            return cand_primary > cur_primary
        if cand_abs_edge != cur_abs_edge:
            return cand_abs_edge > cur_abs_edge
        return cand_market_id < cur_market_id
    
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

    def evaluate_market_filters(self, market: MarketData) -> dict:
        """Return detailed filter pass/fail info for a single market."""
        min_volume = self.config.filters.min_volume
        min_liquidity = self.config.filters.min_liquidity

        volume_ok = market.volume >= min_volume
        liquidity_ok = market.liquidity >= min_liquidity
        price_ok = 0.01 < market.yes_price < 0.99

        passed = bool(volume_ok and liquidity_ok and price_ok)
        reasons = []
        if not volume_ok:
            reasons.append(f"volume<{min_volume}")
        if not liquidity_ok:
            reasons.append(f"liquidity<{min_liquidity}")
        if not price_ok:
            reasons.append("price_out_of_bounds")

        return {
            "passed": passed,
            "checks": {
                "volume_ok": volume_ok,
                "liquidity_ok": liquidity_ok,
                "price_ok": price_ok,
            },
            "reasons": reasons,
        }

    def classify_volume_tier(self, market: MarketData) -> str:
        """
        Classify a market into a volume tier for polling frequency.
        
        Returns:
            Tier name: 'high', 'medium', 'low', or 'skip'
        """
        volume_tiers = getattr(self.config.filters, 'volume_tiers', None)
        
        if not volume_tiers:
            # Default tiers if not configured
            volume_tiers = {
                'high': {'min_volume': 100000},
                'medium': {'min_volume': 10000},
                'low': {'min_volume': 1000},
            }
        
        vol = market.volume
        
        if vol >= volume_tiers.get('high', {}).get('min_volume', 100000):
            return 'high'
        elif vol >= volume_tiers.get('medium', {}).get('min_volume', 10000):
            return 'medium'
        elif vol >= volume_tiers.get('low', {}).get('min_volume', 1000):
            return 'low'
        else:
            return 'skip'

    def get_tier_poll_seconds(self, tier: str) -> int:
        """Get the polling interval for a volume tier."""
        volume_tiers = getattr(self.config.filters, 'volume_tiers', None)
        
        defaults = {
            'high': 30,
            'medium': 120,
            'low': 600,
            'skip': 0,
        }
        
        if not volume_tiers:
            return defaults.get(tier, 120)
        
        tier_config = volume_tiers.get(tier, {})
        return tier_config.get('poll_seconds', defaults.get(tier, 120))

    def categorize_markets_by_tier(self, markets: List[MarketData]) -> dict:
        """
        Group markets by volume tier.
        
        Returns:
            Dict with tier names as keys and lists of markets as values
        """
        result = {'high': [], 'medium': [], 'low': [], 'skip': []}
        
        for market in markets:
            tier = self.classify_volume_tier(market)
            result[tier].append(market)
        
        return result
    
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
            key=lambda x: abs(x[1].effective_edge) * x[1].effective_confidence,
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
        current_exposure: float = 0.0,
        current_open_positions: int = 0,
        current_open_market_keys: Optional[Set[str]] = None,
        current_open_event_keys: Optional[Set[str]] = None,
        max_new_allocation: Optional[float] = None,
    ) -> List[TradeSignal]:
        """
        Generate trade signals with Kelly criterion position sizing
        
        Args:
            opportunities: List of (market, estimate) tuples
            current_bankroll: Current available capital
            current_exposure: Total capital already at risk
            current_open_positions: Number of positions currently open
            current_open_market_keys: Set of 'platform:market_id' for already open positions
            max_new_allocation: Optional hard cap on new capital deployed this cycle
        
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

        # Exposure-aware sizing: only allocate from remaining deployable capital.
        # current_exposure reflects open capital already at risk.
        effective_exposure = max(0.0, float(current_exposure))
        deployable_capital = current_bankroll - effective_exposure
        if max_new_allocation is not None:
            deployable_capital = min(deployable_capital, max(0.0, float(max_new_allocation)))
        if deployable_capital <= 0:
            raise InsufficientCapitalError(
                required=1.0,
                available=deployable_capital
            )
        
        signals = []
        allocated_this_cycle = 0.0
        max_positions = self.config.risk.max_positions
        if current_open_positions >= max_positions:
            raise PositionLimitError(
                current_positions=current_open_positions,
                max_allowed=max_positions
            )
        available_slots = max_positions - current_open_positions
        seen_in_cycle = set()
        seen_event_in_cycle = set()

        # Event-level dedupe before signal generation:
        # keep only the best-ranked opportunity for each canonical event key.
        best_by_event = {}
        for market, estimate in opportunities:
            event_key = self.get_event_key_for_market(market)
            score = self.get_opportunity_score_tuple(market, estimate)
            existing = best_by_event.get(event_key)
            if existing is None:
                best_by_event[event_key] = (market, estimate, score)
            else:
                _, _, existing_score = existing
                if self.is_better_score(score, existing_score):
                    logger.info(
                        f"Skipping signal candidate for {existing[0].market_id} "
                        f"(duplicate_event_guard: replaced_by_higher_score {market.market_id})"
                    )
                    best_by_event[event_key] = (market, estimate, score)
                else:
                    logger.info(
                        f"Skipping signal candidate for {market.market_id} "
                        f"(duplicate_event_guard: lower_score_same_event)"
                    )

        deduped_opportunities = [
            (market, estimate) for market, estimate, _ in best_by_event.values()
        ]
        deduped_opportunities.sort(
            key=lambda item: (
                -self.get_opportunity_score_tuple(item[0], item[1])[0],
                -self.get_opportunity_score_tuple(item[0], item[1])[1],
                self.get_opportunity_score_tuple(item[0], item[1])[2],
            )
        )

        for market, estimate in deduped_opportunities:
            if len(signals) >= available_slots:
                break

            # Remaining capital after accounting for already-open exposure and
            # signals added earlier in this cycle.
            remaining_capital = deployable_capital - allocated_this_cycle
            if remaining_capital <= 0:
                logger.info(
                    "Stopping signal generation: no deployable capital remains "
                    f"(bankroll=${current_bankroll:.2f}, exposure=${effective_exposure:.2f}, "
                    f"allocated_this_cycle=${allocated_this_cycle:.2f})"
                )
                break
            
            # Duplicate market guard
            market_key = f"{market.platform}:{market.market_id}"
            if current_open_market_keys and market_key in current_open_market_keys:
                logger.info(f"Skipping signal for {market_key} (duplicate_market_guard: already open)")
                continue

            if market_key in seen_in_cycle:
                logger.info(f"Skipping signal for {market_key} (duplicate_market_guard: duplicate in cycle)")
                continue

            event_key = self.get_event_key_for_market(market)
            if current_open_event_keys and event_key in current_open_event_keys:
                logger.info(f"Skipping signal for {market_key} (duplicate_event_guard: already open event {event_key})")
                continue
            if event_key in seen_event_in_cycle:
                logger.info(f"Skipping signal for {market_key} (duplicate_event_guard: duplicate in cycle {event_key})")
                continue

            # Determine action based on edge direction
            if estimate.is_buy_yes_signal():
                action = 'buy_yes'
                market_price = market.yes_price
                probability_for_kelly = estimate.effective_probability
            else:
                action = 'buy_no'
                market_price = market.no_price
                probability_for_kelly = 1 - estimate.effective_probability

            # Guardrail: avoid fading heavy favorites with buy_no unless
            # edge/confidence are materially stronger than baseline thresholds.
            if action == 'buy_no':
                heavy_yes_threshold = float(
                    getattr(self.config.strategy, "heavy_favorite_yes_price_threshold", 0.80)
                )
                strict_edge = float(
                    getattr(self.config.strategy, "heavy_favorite_buy_no_min_edge", 0.15)
                )
                strict_confidence = float(
                    getattr(self.config.strategy, "heavy_favorite_buy_no_min_confidence", 0.85)
                )

                if market.yes_price >= heavy_yes_threshold:
                    abs_edge = abs(float(estimate.effective_edge))
                    confidence = float(estimate.effective_confidence)
                    if abs_edge < strict_edge or confidence < strict_confidence:
                        logger.info(
                            "Skipping signal for %s (heavy_favorite_buy_no_guard: "
                            "yes_price=%.3f edge=%.3f confidence=%.3f requires "
                            "edge>=%.3f confidence>=%.3f)",
                            market_key,
                            market.yes_price,
                            abs_edge,
                            confidence,
                            strict_edge,
                            strict_confidence,
                        )
                        continue
            
            # Calculate Kelly fraction
            kelly_frac = self.kelly.calculate_kelly_fraction(
                probability=probability_for_kelly,
                market_price=market_price,
                max_fraction=self.config.risk.max_kelly_fraction
            )
            
            if kelly_frac <= 0:
                logger.debug(f"Skipping {market.market_id}: Kelly fraction too small ({kelly_frac})")
                continue
            
            # Calculate position size
            position_size = self.kelly.calculate_position_size(
                kelly_fraction=kelly_frac,
                bankroll=remaining_capital,
                max_position=self.config.risk.max_position_size
            )
            
            # Skip if position would exceed available capital
            if position_size > remaining_capital:
                logger.warning(
                    f"Skipping {market.market_id}: position size ${position_size:.2f} "
                    f"exceeds remaining deployable capital ${remaining_capital:.2f}"
                )
                continue
            
            # Calculate expected value
            probability = estimate.effective_probability
            no_probability = 1 - probability
            if estimate.is_buy_yes_signal():
                # Buying YES at market_price
                ev = (probability * (1 - market_price) - 
                      (1 - probability) * market_price) * position_size
            else:
                # Buying NO at market_price
                ev = (no_probability * (1 - market_price) - 
                      (1 - no_probability) * market_price) * position_size
            
            # Create signal
            try:
                signal = TradeSignal(
                    market=market,
                    action=action,
                    fair_value=probability,
                    market_price=market_price,
                    edge=estimate.effective_edge,
                    kelly_fraction=kelly_frac,
                    position_size=position_size,
                    expected_value=ev,
                    reasoning=estimate.reasoning[:200] if estimate.reasoning else ""
                )
                signals.append(signal)
                seen_in_cycle.add(market_key)
                seen_event_in_cycle.add(event_key)
                allocated_this_cycle += position_size
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
