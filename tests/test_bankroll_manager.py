from datetime import datetime, timedelta

import pytest

from app.storage.db import DatabaseManager
from app.trading.bankroll_manager import BankrollManager


@pytest.mark.asyncio
async def test_daily_starting_balance_anchors_to_latest_dry_run_reset(tmp_path):
    db = DatabaseManager(str(tmp_path / "bankroll.sqlite"))
    await db.initialize()

    manager = BankrollManager(db, initial_bankroll=100.0)
    await manager.initialize()

    # Seed prior-day paper state that would otherwise inflate "daily start".
    yesterday = (datetime.utcnow() - timedelta(days=1)).replace(microsecond=0).isoformat() + "Z"
    async with db.connect() as conn:
        await conn.execute(
            """
            INSERT INTO bankroll_history
            (timestamp_utc, balance, change, reason, reference_id, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (yesterday, 325.17, 225.17, "trade_execution", "legacy-paper", yesterday),
        )
        await conn.commit()

    await manager.reset_for_new_dry_run_session()

    legacy_daily_start = await manager.get_daily_starting_balance()
    anchored_daily_start = await manager.get_daily_starting_balance(dry_run_session_anchored=True)

    assert legacy_daily_start == pytest.approx(325.17)
    assert anchored_daily_start == pytest.approx(100.0)
