"""Position manager for tracking trades, bankroll, and performance"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime

from ..models import TradeSignal
from ..utils import (
    PositionNotFoundError,
    PositionAlreadyClosedError,
    InsufficientBankrollError,
)

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of an executed trade"""
    
    market_id: str
    action: str  # 'buy_yes', 'buy_no', 'sell_yes', 'sell_no'
    position_size: float  # Amount invested
    entry_price: float  # Price at entry
    timestamp: datetime = field(default_factory=datetime.now)
    result: float = 0.0  # Profit/loss in dollars (calculated on close)
    closed: bool = False
    
    def __str__(self) -> str:
        """Human-readable representation"""
        status = "CLOSED" if self.closed else "OPEN"
        result_str = f" | Result: ${self.result:+,.2f}" if self.closed else ""
        return (
            f"{status}: {self.action} {self.market_id} "
            f"${self.position_size:,.2f} @ {self.entry_price:.2%}{result_str}"
        )


class PositionManager:
    """
    Manages open positions, trade history, and bankroll
    
    Tracks:
    - Current open positions
    - Closed trades and results
    - Bankroll changes
    - Performance metrics (win rate, total profit, etc.)
    """
    
    def __init__(self, initial_bankroll: float):
        """
        Initialize position manager
        
        Args:
            initial_bankroll: Starting capital in dollars
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        
        self.open_positions: List[Trade] = []
        self.trade_history: List[Trade] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
    
    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================
    
    def add_position(self, signal: TradeSignal) -> Trade:
        """
        Open a new position from a trade signal
        
        Args:
            signal: TradeSignal with market and sizing information
        
        Returns:
            Trade object representing the opened position
        
        Raises:
            InsufficientBankrollError: If position size exceeds available bankroll
        """
        
        # Validate sufficient bankroll
        if signal.position_size > self.current_bankroll:
            raise InsufficientBankrollError(
                position_size=signal.position_size,
                bankroll=self.current_bankroll
            )
        
        # Determine entry price based on action
        if signal.action in ('buy_yes', 'sell_no'):
            entry_price = signal.market.yes_price
        else:  # buy_no or sell_yes
            entry_price = signal.market.no_price
        
        trade = Trade(
            market_id=signal.market.market_id,
            action=signal.action,
            position_size=signal.position_size,
            entry_price=entry_price,
        )
        
        # Update bankroll
        self.current_bankroll -= signal.position_size
        self.open_positions.append(trade)
        
        logger.info(f"✅ Added position: {trade}")
        return trade
    
    def close_position(self, market_id: str, exit_price: float) -> Trade:
        """
        Close an open position and record the result
        
        Args:
            market_id: ID of market to close
            exit_price: Price at which position is closed
        
        Returns:
            Closed Trade object with result recorded
        
        Raises:
            PositionNotFoundError: If no open position found for market_id
            PositionAlreadyClosedError: If position is already closed
        """
        # Find position
        position = None
        for p in self.open_positions:
            if p.market_id == market_id:
                if p.closed:
                    raise PositionAlreadyClosedError(market_id)
                position = p
                break
        
        if not position:
            raise PositionNotFoundError(market_id)
        
        # Calculate profit/loss
        if position.action in ('buy_yes', 'sell_no'):
            # Long position on YES
            pnl = (exit_price - position.entry_price) * position.position_size
        else:
            # Long position on NO (short YES)
            pnl = (position.entry_price - exit_price) * position.position_size
        
        # Update position
        position.result = pnl
        position.closed = True
        
        # Update bankroll and stats
        self.current_bankroll += position.position_size + pnl
        self.total_profit += max(0, pnl)
        self.total_loss += abs(min(0, pnl))
        
        # Move to history
        self.open_positions.remove(position)
        self.trade_history.append(position)
        
        # Update metrics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        logger.info(f"✅ Closed position: {position}")
        return position
    
    # ========================================================================
    # QUERY METHODS
    # ========================================================================
    
    def get_open_positions(self) -> List[Trade]:
        """Get all currently open positions"""
        return [p for p in self.open_positions if not p.closed]
    
    def get_position(self, market_id: str) -> Trade:
        """
        Get a specific open position
        
        Args:
            market_id: Market ID to look up
        
        Returns:
            Trade object or None if not found
        """
        for p in self.open_positions:
            if p.market_id == market_id and not p.closed:
                return p
        return None
    
    def has_position(self, market_id: str) -> bool:
        """Check if a market has an open position"""
        return self.get_position(market_id) is not None
    
    def get_total_exposure(self) -> float:
        """Calculate total dollars currently at risk in open positions"""
        return sum(p.position_size for p in self.get_open_positions())
    
    def get_available_capital(self) -> float:
        """Get available capital not committed to positions"""
        return self.current_bankroll
    
    def get_position_count(self) -> int:
        """Get count of open positions"""
        return len(self.get_open_positions())
    
    # ========================================================================
    # PERFORMANCE METRICS
    # ========================================================================
    
    def get_win_rate(self) -> float:
        """Calculate win rate as percentage (0-1)"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    def get_avg_profit_per_trade(self) -> float:
        """Calculate average profit per trade"""
        if self.total_trades == 0:
            return 0.0
        return self.total_profit / self.total_trades
    
    def get_return_on_capital(self) -> float:
        """Calculate return on initial capital as percentage"""
        profit = self.current_bankroll - self.initial_bankroll
        if self.initial_bankroll == 0:
            return 0.0
        return profit / self.initial_bankroll
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.current_bankroll,
            'profit': self.current_bankroll - self.initial_bankroll,
            'return_percent': self.get_return_on_capital() * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_percent': self.get_win_rate() * 100,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'avg_profit_per_trade': self.get_avg_profit_per_trade(),
            'open_positions': len(self.get_open_positions()),
            'total_exposure': self.get_total_exposure(),
            'available_capital': self.get_available_capital(),
        }
    
    def log_stats(self) -> None:
        """Log performance statistics"""
        stats = self.get_stats()
        
        logger.info("\n" + "=" * 80)
        logger.info("💼 POSITION MANAGER STATS")
        logger.info("=" * 80)
        logger.info(f"Bankroll: ${stats['initial_bankroll']:,.2f} → ${stats['current_bankroll']:,.2f} "
                    f"({stats['return_percent']:+.2f}%)")
        logger.info(f"Trades: {stats['total_trades']} "
                    f"({stats['winning_trades']}W / {stats['losing_trades']}L) "
                    f"| Win Rate: {stats['win_rate_percent']:.1f}%")
        logger.info(f"Profit/Loss: ${stats['total_profit']:,.2f} profit, "
                    f"${stats['total_loss']:,.2f} loss")
        logger.info(f"Open Positions: {stats['open_positions']} | "
                    f"Exposure: ${stats['total_exposure']:,.2f} | "
                    f"Available: ${stats['available_capital']:,.2f}")
        logger.info("=" * 80)
