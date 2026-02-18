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

    async def reset_for_new_dry_run_session(self):
        """
        Reset paper bankroll to initial value for a fresh dry-run app start.

        This should be called once per process startup in dry-run mode.
        """
        now = datetime.utcnow().isoformat() + "Z"
        previous_balance: Optional[float] = None

        async with self.db.connect() as db:
            async with db.execute("SELECT balance FROM bankroll_history ORDER BY id DESC LIMIT 1") as cursor:
                row = await cursor.fetchone()
                if row:
                    previous_balance = float(row[0])

            change = 0.0 if previous_balance is None else (self.initial_bankroll - previous_balance)
            self._current_balance = self.initial_bankroll

            await db.execute("""
                INSERT INTO bankroll_history
                (timestamp_utc, balance, change, reason, reference_id, created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                now,
                self._current_balance,
                change,
                "dry_run_session_reset",
                "dry_run_reset",
                now,
            ))
            await db.commit()

        logger.info(
            f"🧹 Dry-run session reset bankroll: ${previous_balance if previous_balance is not None else self.initial_bankroll:,.2f} "
            f"-> ${self._current_balance:,.2f}"
        )

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

    async def get_daily_starting_balance(self) -> float:
        """Get the bankroll balance at the start of the current UTC day."""
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"

        async with self.db.connect() as db:
            # Try to find the last balance before today began
            async with db.execute(
                "SELECT balance FROM bankroll_history WHERE timestamp_utc < ? ORDER BY id DESC LIMIT 1",
                (today_start,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return float(row[0])

            # If no record before today, use the very first record
            async with db.execute(
                "SELECT balance FROM bankroll_history ORDER BY id ASC LIMIT 1",
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return float(row[0])

            return self.get_balance()

    async def get_stats(self) -> Dict[str, Any]:
        """Get bankroll statistics."""
        return {
            "current_balance": self.get_balance(),
            "initial_bankroll": self.initial_bankroll,
            "total_change": self.get_balance() - self.initial_bankroll
        }
