"""Position manager for tracking trades, bankroll, and performance"""

import logging
import json
import aiosqlite
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.models import TradeSignal
from app.storage.db import DatabaseManager
from app.utils import (
    PositionNotFoundError,
    PositionAlreadyClosedError,
    InsufficientBankrollError,
)

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of an executed trade"""
    
    market_id: str
    platform: str
    side: str  # 'yes' or 'no'
    action: str  # 'buy_yes', 'buy_no', 'sell_yes', 'sell_no'
    quantity: float
    position_size: float  # Amount invested (cost)
    entry_price: float  # Price at entry
    timestamp: datetime = field(default_factory=datetime.utcnow)
    result: float = 0.0  # Profit/loss in dollars (calculated on close)
    status: str = "open"  # 'open', 'closed', 'recovered'
    external_order_id: Optional[str] = None
    
    @property
    def closed(self) -> bool:
        return self.status == "closed"

    def __str__(self) -> str:
        """Human-readable representation"""
        result_str = f" | Result: ${self.result:+,.2f}" if self.closed else ""
        return (
            f"{self.status.upper()}: {self.action} {self.market_id} "
            f"qty={self.quantity} cost=${self.position_size:,.2f} @ {self.entry_price:.2%}{result_str}"
        )


from .bankroll_manager import BankrollManager

class PositionManager:
    """
    Manages open positions and trade history with SQLite persistence.
    """
    
    def __init__(self, db: DatabaseManager, bankroll_manager: Optional[BankrollManager] = None):
        """
        Initialize position manager
        
        Args:
            db: DatabaseManager instance
            bankroll_manager: Optional BankrollManager for replenishment
        """
        self.db = db
        self.bankroll_manager = bankroll_manager
        self.open_positions: List[Trade] = []
        self.trade_history: List[Trade] = []
        
        # Performance tracking (will be updated when history is loaded)
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
    
    async def load_positions(self):
        """Load open positions and trade history from database."""
        self.open_positions = []
        self.trade_history = []

        # Reset stats before loading
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0

        async with self.db.connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM positions") as cursor:
                async for row in cursor:
                    trade = Trade(
                        market_id=row['market_id'],
                        platform=row['platform'],
                        side=row['side'],
                        action=f"buy_{row['side']}", # Approximation
                        quantity=row['quantity'],
                        position_size=row['cost'],
                        entry_price=row['entry_price'],
                        timestamp=datetime.fromisoformat(row['opened_at_utc']),
                        status=row['status'],
                        external_order_id=row['external_order_id']
                    )

                    if row['closed_at_utc']:
                        trade.result = row['result'] or 0.0
                        self.trade_history.append(trade)

                        # Update stats
                        self.total_trades += 1
                        if trade.result > 0:
                            self.winning_trades += 1
                            self.total_profit += trade.result
                        else:
                            self.losing_trades += 1
                            self.total_loss += abs(trade.result)
                    else:
                        self.open_positions.append(trade)

        logger.info(f"📂 Loaded {len(self.open_positions)} open positions and {len(self.trade_history)} historical trades from database.")

    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================
    
    async def add_position(self, signal: TradeSignal, external_order_id: Optional[str] = None) -> Trade:
        """
        Open a new position from a trade signal and persist to DB
        
        Args:
            signal: TradeSignal with market and sizing information
            external_order_id: ID from the trading platform
        
        Returns:
            Trade object representing the opened position
        """
        
        # Determine entry price based on action
        if signal.action in ('buy_yes', 'sell_no'):
            entry_price = signal.market.yes_price
            side = 'yes'
        else:  # buy_no or sell_yes
            entry_price = signal.market.no_price
            side = 'no'
        
        # Kalshi count is usually position_size / price
        # But signals already have position_size (dollars)
        quantity = signal.position_size / max(0.01, entry_price)

        trade = Trade(
            market_id=signal.market.market_id,
            platform=signal.market.platform,
            side=side,
            action=signal.action,
            quantity=quantity,
            position_size=signal.position_size,
            entry_price=entry_price,
            external_order_id=external_order_id,
            status="open"
        )
        
        self.open_positions.append(trade)
        
        # Persist to DB
        now = datetime.utcnow().isoformat() + "Z"
        # Ensure trade.timestamp is also formatted with Z
        timestamp_str = trade.timestamp.isoformat()
        if not timestamp_str.endswith("Z"):
            timestamp_str += "Z"

        async with self.db.connect() as db:
            await db.execute("""
                INSERT INTO positions
                (market_id, platform, side, entry_price, quantity, cost, status, external_order_id, opened_at_utc, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.market_id, trade.platform, trade.side, trade.entry_price,
                trade.quantity, trade.position_size, trade.status,
                trade.external_order_id, timestamp_str, now
            ))
            await db.commit()

        logger.info(f"✅ Added position: {trade}")
        return trade
    
    async def add_recovered_position(self, pos_data: Dict[str, Any]) -> Trade:
        """
        Add a position discovered during reconciliation.
        """
        trade = Trade(
            market_id=pos_data['market_id'],
            platform=pos_data['platform'],
            side=pos_data['side'],
            action=f"buy_{pos_data['side']}",
            quantity=pos_data['quantity'],
            position_size=pos_data['quantity'] * pos_data['entry_price'],
            entry_price=pos_data['entry_price'],
            status="recovered"
        )

        self.open_positions.append(trade)

        now = datetime.utcnow().isoformat() + "Z"
        timestamp_str = trade.timestamp.isoformat()
        if not timestamp_str.endswith("Z"):
            timestamp_str += "Z"

        async with self.db.connect() as db:
            await db.execute("""
                INSERT INTO positions
                (market_id, platform, side, entry_price, quantity, cost, status, opened_at_utc, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.market_id, trade.platform, trade.side, trade.entry_price,
                trade.quantity, trade.position_size, trade.status,
                timestamp_str, now
            ))
            await db.commit()

        logger.info(f"🔄 Recovered position: {trade}")
        return trade

    async def close_position(self, market_id: str, exit_price: float) -> Trade:
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
                if p.status == "closed":
                    raise PositionAlreadyClosedError(market_id)
                position = p
                break
        
        if not position:
            raise PositionNotFoundError(market_id)
        
        # Calculate profit/loss
        # Assumption: exit_price is the price of the side held (YES or NO)
        pnl = (exit_price - position.entry_price) * position.quantity
        
        # Update position
        position.result = pnl
        position.status = "closed"
        
        # Update stats
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
        
        # Persist to DB
        now = datetime.utcnow().isoformat() + "Z"
        async with self.db.connect() as db:
            await db.execute("""
                UPDATE positions
                SET status = ?, closed_at_utc = ?, result = ?
                WHERE market_id = ? AND status != 'closed'
            """, (position.status, now, position.result, market_id))
            await db.commit()

        # Replenish bankroll: return cost + PnL (which is quantity * exit_price)
        if self.bankroll_manager:
            replenishment = position.position_size + pnl
            await self.bankroll_manager.adjust_balance(
                replenishment,
                reason="position_closed",
                reference_id=market_id
            )
            # Record in pnl_history
            async with self.db.connect() as db:
                await db.execute("""
                    INSERT INTO pnl_history (timestamp_utc, profit, balance, created_at_utc)
                    VALUES (?, ?, ?, ?)
                """, (now, pnl, self.bankroll_manager.get_balance(), now))
                await db.commit()

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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_percent': self.get_win_rate() * 100,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'avg_profit_per_trade': self.get_avg_profit_per_trade(),
            'open_positions': len(self.get_open_positions()),
            'total_exposure': self.get_total_exposure(),
        }
    
    def log_stats(self) -> None:
        """Log performance statistics"""
        stats = self.get_stats()
        
        logger.info("\n" + "=" * 80)
        logger.info("💼 POSITION MANAGER STATS")
        logger.info("=" * 80)
        logger.info(f"Trades: {stats['total_trades']} "
                    f"({stats['winning_trades']}W / {stats['losing_trades']}L) "
                    f"| Win Rate: {stats['win_rate_percent']:.1f}%")
        logger.info(f"Profit/Loss: ${stats['total_profit']:,.2f} profit, "
                    f"${stats['total_loss']:,.2f} loss")
        logger.info(f"Open Positions: {stats['open_positions']} | "
                    f"Exposure: ${stats['total_exposure']:,.2f}")
        logger.info("=" * 80)
