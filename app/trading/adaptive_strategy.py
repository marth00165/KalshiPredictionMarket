"""Adaptive strategy logic for dynamic market parameter adjustment

This module provides tools for:
- Market regime detection (trending, mean-reverting, volatile)
- Dynamic parameter adjustment based on performance
- Risk adjustment based on recent P&L
- Confidence-weighted strategy adjustments
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from enum import Enum

from app.models import FairValueEstimate, TradeSignal
from app.utils import ConfigManager

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"      # Prices consistently moving up
    TRENDING_DOWN = "trending_down"  # Prices consistently moving down
    MEAN_REVERTING = "mean_reverting"  # Prices oscillating around mean
    VOLATILE = "volatile"            # High volatility, no clear direction
    STABLE = "stable"                # Low volatility, stable prices


@dataclass
class PerformanceMetrics:
    """Track strategy performance for adaptive adjustments"""
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 1.0  # Total profit / Total loss
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    def update(self, pnl: float):
        """Update metrics with a new trade result"""
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)
        
        self.total_trades += 1
        
        # Recalculate aggregates
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        if self.total_loss > 0:
            self.profit_factor = self.total_profit / self.total_loss
        
        if self.winning_trades > 0:
            self.avg_win = self.total_profit / self.winning_trades
        
        if self.losing_trades > 0:
            self.avg_loss = self.total_loss / self.losing_trades


@dataclass
class RegimeAnalysis:
    """Results of market regime detection"""
    
    regime: MarketRegime
    confidence: float  # 0-1, how confident in this classification
    trend_strength: float  # 0-1, strength of trending behavior
    volatility: float  # Normalized volatility metric
    mean_price: float  # Average price over analysis window
    analysis_time: datetime = field(default_factory=datetime.now)


class AdaptiveParameterAdjuster:
    """Dynamically adjust strategy parameters based on performance and market conditions"""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize adaptive parameter adjuster
        
        Args:
            config: ConfigManager with base strategy settings
        """
        self.config = config
        self.base_kelly_fraction = config.risk.max_kelly_fraction
        self.base_min_edge = config.strategy.min_edge
        self.performance = PerformanceMetrics()
        self.adjustment_history: List[Dict] = []
    
    def adjust_kelly_fraction(
        self,
        current_drawdown: float,
        win_rate: float
    ) -> float:
        """
        Reduce Kelly fraction during drawdowns or low win rates
        
        Args:
            current_drawdown: Current peak-to-trough drawdown (0-1)
            win_rate: Recent win rate (0-1)
        
        Returns:
            Adjusted Kelly fraction (reduced during difficult periods)
        """
        
        multiplier = 1.0
        
        # Reduce by 50% if in drawdown > 15%
        if current_drawdown > 0.15:
            multiplier *= 0.5
        
        # Reduce by 25% if win rate below 45%
        if win_rate < 0.45:
            multiplier *= 0.75
        
        # Increase by 10% if win rate above 60%
        if win_rate > 0.60:
            multiplier *= 1.10
        
        adjusted = self.base_kelly_fraction * multiplier
        
        logger.info(
            f"Kelly adjustment: {self.base_kelly_fraction:.3f} → {adjusted:.3f} "
            f"(drawdown={current_drawdown:.1%}, win_rate={win_rate:.1%})"
        )
        
        return adjusted
    
    def adjust_min_edge(
        self,
        confidence_in_estimates: float,
        regime: MarketRegime
    ) -> float:
        """
        Adjust minimum edge requirement based on conditions
        
        Args:
            confidence_in_estimates: Claude's average confidence (0-1)
            regime: Current market regime
        
        Returns:
            Adjusted minimum edge percentage
        """
        
        adjusted = self.base_min_edge
        
        # Higher edge requirement in trending markets (harder to predict)
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            adjusted *= 1.5
        
        # Higher edge requirement in volatile markets
        elif regime == MarketRegime.VOLATILE:
            adjusted *= 1.3
        
        # Lower edge requirement when we have high confidence estimates
        if confidence_in_estimates > 0.75:
            adjusted *= 0.8
        elif confidence_in_estimates < 0.50:
            adjusted *= 1.2
        
        logger.info(
            f"Min edge adjustment: {self.base_min_edge:.1%} → {adjusted:.1%} "
            f"(confidence={confidence_in_estimates:.1%}, regime={regime.value})"
        )
        
        return adjusted
    
    def adjust_max_positions(
        self,
        profit_factor: float,
        current_positions: int
    ) -> int:
        """
        Dynamically adjust max concurrent positions based on profitability
        
        Args:
            profit_factor: Total profit / Total loss ratio
            current_positions: Current number of open positions
        
        Returns:
            Adjusted maximum position limit
        """
        
        base_max = self.config.risk.max_positions
        
        # If unprofitable, reduce max positions
        if profit_factor < 1.0:
            adjusted = max(1, int(base_max * 0.5))
        # If marginally profitable, keep conservative
        elif profit_factor < 1.5:
            adjusted = base_max
        # If highly profitable, allow more positions
        elif profit_factor > 2.0:
            adjusted = min(int(base_max * 1.5), 20)
        else:
            adjusted = base_max
        
        logger.info(
            f"Max positions adjustment: {base_max} → {adjusted} "
            f"(profit_factor={profit_factor:.2f})"
        )
        
        return adjusted


class MarketRegimeDetector:
    """Detect current market regime from price/volume data"""
    
    @staticmethod
    def detect_regime(
        prices: List[float],
        volumes: List[float],
        window_size: int = 20
    ) -> RegimeAnalysis:
        """
        Classify market regime from recent price action
        
        Args:
            prices: List of recent prices (oldest to newest)
            volumes: List of corresponding volumes
            window_size: Number of periods to analyze
        
        Returns:
            RegimeAnalysis with detected regime and confidence
        """
        
        if len(prices) < window_size:
            # Not enough data, default to stable
            return RegimeAnalysis(
                regime=MarketRegime.STABLE,
                confidence=0.1,
                trend_strength=0.0,
                volatility=0.0,
                mean_price=sum(prices) / len(prices) if prices else 0.5
            )
        
        recent = prices[-window_size:]
        
        # Calculate trend (linear regression approximation)
        trend = (recent[-1] - recent[0]) / recent[0]
        
        # Calculate volatility (std dev of returns)
        returns = [
            (recent[i] - recent[i-1]) / recent[i-1]
            for i in range(1, len(recent))
        ]
        volatility = (
            sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns)
        ) ** 0.5 if returns else 0
        
        # Mean reversion detection (autocorrelation approximation)
        mean_price = sum(recent) / len(recent)
        mean_reversions = sum(
            1 for i in range(1, len(recent))
            if (recent[i] - mean_price) * (recent[i-1] - mean_price) < 0
        )
        reversion_score = mean_reversions / (len(recent) - 1)
        
        # Classify regime
        if volatility > 0.05:
            regime = MarketRegime.VOLATILE
            confidence = 0.7
        elif reversion_score > 0.4:
            regime = MarketRegime.MEAN_REVERTING
            confidence = 0.6
        elif abs(trend) > 0.05:
            regime = MarketRegime.TRENDING_UP if trend > 0 else MarketRegime.TRENDING_DOWN
            confidence = 0.7
        else:
            regime = MarketRegime.STABLE
            confidence = 0.8
        
        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            trend_strength=abs(trend),
            volatility=volatility,
            mean_price=mean_price
        )


class SignalConfidenceWeighter:
    """Weight trade signals by estimate confidence and market conditions"""
    
    @staticmethod
    def weight_signals(
        signals: List[TradeSignal],
        market_regime: MarketRegime,
        estimate_confidences: List[float]
    ) -> List[TradeSignal]:
        """
        Adjust signal position sizes by confidence scores
        
        Args:
            signals: Original trade signals
            market_regime: Current market regime
            estimate_confidences: Confidence for each signal's estimate
        
        Returns:
            Signals with adjusted position sizes
        """
        
        if len(signals) != len(estimate_confidences):
            logger.warning("Signal count != confidence count, returning original signals")
            return signals
        
        weighted_signals = []
        
        for signal, confidence in zip(signals, estimate_confidences):
            # Base multiplier from estimate confidence
            multiplier = confidence
            
            # Reduce sizes in volatile/trending markets (less predictable)
            if market_regime == MarketRegime.VOLATILE:
                multiplier *= 0.75
            elif market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                multiplier *= 0.85
            
            # Create new signal with adjusted position size
            adjusted_signal = TradeSignal(
                market=signal.market,
                action=signal.action,
                fair_value=signal.fair_value,
                market_price=signal.market_price,
                edge=signal.edge,
                kelly_fraction=signal.kelly_fraction,
                position_size=signal.position_size * multiplier,  # Adjusted
                expected_value=signal.expected_value * multiplier,
                reasoning=signal.reasoning,
                confidence_adjustment=multiplier
            )
            
            weighted_signals.append(adjusted_signal)
        
        return weighted_signals


class StrategyPerformanceMonitor:
    """Monitor strategy performance and generate improvement recommendations"""
    
    def __init__(self, lookback_periods: int = 100):
        """
        Initialize performance monitor
        
        Args:
            lookback_periods: Number of recent trades to analyze
        """
        self.lookback_periods = lookback_periods
        self.trade_history: List[Dict] = []
    
    def record_trade_result(
        self,
        market_id: str,
        action: str,
        entry_price: float,
        exit_price: float,
        position_size: float,
        estimated_probability: float
    ):
        """Record a completed trade for analysis"""
        
        pnl = (exit_price - entry_price) * position_size
        
        self.trade_history.append({
            'market_id': market_id,
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'estimated_probability': estimated_probability,
            'pnl': pnl,
            'timestamp': datetime.now()
        })
        
        # Keep only recent trades
        if len(self.trade_history) > self.lookback_periods:
            self.trade_history = self.trade_history[-self.lookback_periods:]
    
    def get_performance_summary(self) -> Dict:
        """Get summary of recent performance"""
        
        if not self.trade_history:
            return {}
        
        recent = self.trade_history[-20:]  # Last 20 trades
        
        pnls = [t['pnl'] for t in recent]
        win_count = sum(1 for p in pnls if p > 0)
        
        return {
            'recent_trades': len(recent),
            'win_rate': win_count / len(recent) if recent else 0,
            'avg_pnl': sum(pnls) / len(pnls) if pnls else 0,
            'total_pnl': sum(pnls),
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(pnls)
        }
    
    @staticmethod
    def _calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio for returns"""
        
        if len(returns) < 2:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
        
        return (mean_return - risk_free_rate) / std_dev
    
    def get_improvement_recommendations(self) -> List[str]:
        """Generate recommendations for strategy improvement"""
        
        summary = self.get_performance_summary()
        if not summary:
            return ["Insufficient trade history for recommendations"]
        
        recommendations = []
        
        # Win rate analysis
        if summary['win_rate'] < 0.40:
            recommendations.append(
                f"Low win rate ({summary['win_rate']:.1%}): Consider increasing min_edge requirement"
            )
        elif summary['win_rate'] > 0.70:
            recommendations.append(
                f"High win rate ({summary['win_rate']:.1%}): Consider increasing position sizes"
            )
        
        # Profitability analysis
        if summary['total_pnl'] < 0:
            recommendations.append(
                "Negative total P&L: Consider reducing position sizes or increasing edge threshold"
            )
        
        # Sharpe ratio analysis
        if summary['sharpe_ratio'] < 0.5:
            recommendations.append(
                "Low Sharpe ratio: Consider adjusting kelly fraction or position sizing"
            )
        
        if not recommendations:
            recommendations.append("Strategy performing well, no immediate adjustments needed")
        
        return recommendations
