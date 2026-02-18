"""Bankroll manager for persistent bankroll tracking in SQLite"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from app.storage.db import DatabaseManager

logger = logging.getLogger(__name__)

class BankrollManager:
    """
    Manages persistent bankroll tracking in SQLite database.

    Ensures that bankroll state survives restarts and provides
    a history of all balance changes.
    """

    def __init__(self, db: DatabaseManager, initial_bankroll: float):
        """
        Initialize BankrollManager.

        Args:
            db: DatabaseManager instance for persistence
            initial_bankroll: Starting bankroll if no history exists
        """
        self.db = db
        self.initial_bankroll = initial_bankroll
        self._current_balance: Optional[float] = None

    async def initialize(self):
        """
        Load latest bankroll from database or initialize if empty.
        """
        if self._current_balance is not None:
            return

        async with self.db.connect() as db:
            async with db.execute("SELECT balance FROM bankroll_history ORDER BY id DESC LIMIT 1") as cursor:
                row = await cursor.fetchone()
                if row:
                    self._current_balance = row[0]
                    logger.info(f"💰 Loaded existing bankroll: ${self._current_balance:,.2f}")
                else:
                    self._current_balance = self.initial_bankroll
                    logger.info(f"💰 Initializing new bankroll: ${self._current_balance:,.2f}")
                    # We can't call adjust_balance here because it also tries to connect.
                    # Instead, we perform the initial insert directly.
                    now = datetime.utcnow().isoformat() + "Z"
                    await db.execute("""
                        INSERT INTO bankroll_history
                        (timestamp_utc, balance, change, reason, reference_id, created_at_utc)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (now, self._current_balance, 0.0, "initialization", "init", now))
                    await db.commit()

    def get_balance(self) -> float:
        """Get current bankroll balance."""
        if self._current_balance is None:
            return self.initial_bankroll
        return self._current_balance

    async def adjust_balance(self, amount: float, reason: str, reference_id: Optional[str] = None):
        """
        Adjust bankroll balance and persist change to database.

        Args:
            amount: Amount to add (positive) or subtract (negative)
            reason: Reason for change (e.g., 'trade_execution', 'api_cost', 'settlement')
            reference_id: Optional reference ID (e.g., order_id, cycle_id)
        """
        if self._current_balance is None:
            await self.initialize()

        old_balance = self._current_balance
        self._current_balance += amount

        now = datetime.utcnow().isoformat() + "Z"

        async with self.db.connect() as db:
            await db.execute("""
                INSERT INTO bankroll_history
                (timestamp_utc, balance, change, reason, reference_id, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (now, self._current_balance, amount, reason, reference_id, now))
            await db.commit()

        logger.info(f"💰 Bankroll adjusted: ${old_balance:,.2f} -> ${self._current_balance:,.2f} "
                    f"(change: ${amount:+,.2f}, reason: {reason})")

    async def get_stats(self) -> Dict[str, Any]:
        """Get bankroll statistics."""
        return {
            "current_balance": self.get_balance(),
            "initial_bankroll": self.initial_bankroll,
            "total_change": self.get_balance() - self.initial_bankroll
        }
