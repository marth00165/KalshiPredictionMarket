import importlib.util
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from app.models import MarketData, TradeSignal
from app.storage.db import DatabaseManager
from app.trading.engine import log_model_divergence_warning
from app.trading.executor import TradeExecutor

HAS_PYTEST_ASYNCIO = importlib.util.find_spec("pytest_asyncio") is not None
if not HAS_PYTEST_ASYNCIO:
    pytest.skip("pytest-asyncio plugin not installed in this environment", allow_module_level=True)


def _build_config():
    cfg = MagicMock()
    cfg.is_dry_run = True
    cfg.max_price_drift = 1.0
    cfg.min_edge_at_execution = 0.0

    cfg.execution = MagicMock()
    cfg.execution.max_submit_slippage = 1.0

    cfg.risk = MagicMock()
    cfg.risk.max_total_exposure_fraction = 0.80
    cfg.risk.max_new_exposure_per_day_fraction = 0.20

    cfg.trading = MagicMock()
    cfg.trading.initial_bankroll = 1000.0
    return cfg


def _build_signal(market_id: str, position_size: float) -> TradeSignal:
    market = MarketData(
        platform="kalshi",
        market_id=market_id,
        title="Test Market",
        description="",
        yes_price=0.5,
        no_price=0.5,
        volume=1000,
        liquidity=1000,
        end_date="2026-12-31T00:00:00Z",
        category="sports",
    )
    return TradeSignal(
        market=market,
        action="buy_yes",
        fair_value=0.60,
        market_price=0.50,
        edge=0.10,
        kelly_fraction=0.10,
        position_size=position_size,
        expected_value=10.0,
        reasoning="test",
    )


def test_model_divergence_warning_logs(caplog):
    caplog.set_level("WARNING")
    emitted = log_model_divergence_warning(
        team="DEN",
        opponent="GSW",
        probability_final=0.80,
        market_probability=0.50,
        edge=0.30,
    )
    assert emitted is True
    assert "MODEL_DIVERGENCE_WARNING" in caplog.text

    caplog.clear()
    emitted = log_model_divergence_warning(
        team="DEN",
        opponent="GSW",
        probability_final=0.60,
        market_probability=0.50,
        edge=0.10,
    )
    assert emitted is False
    assert "MODEL_DIVERGENCE_WARNING" not in caplog.text


@pytest.mark.asyncio
async def test_executor_blocks_when_daily_exposure_limit_reached(tmp_path):
    db = DatabaseManager(str(tmp_path / "daily_exposure.sqlite"))
    await db.initialize()

    today = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    async with db.connect() as conn:
        await conn.execute(
            """
            INSERT INTO bankroll_history
            (timestamp_utc, balance, change, reason, reference_id, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (today, 1000.0, 0.0, "init", None, today),
        )
        # Daily cap = 20% of 1000 => 200. Already opened 200 today.
        await conn.execute(
            """
            INSERT INTO executions
            (market_id, platform, action, quantity, price, cost, external_order_id, status, executed_at_utc, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("MKT1", "kalshi", "buy_yes", 400.0, 0.5, 200.0, "ord1", "executed", today, today),
        )
        await conn.commit()

    cfg = _build_config()
    executor = TradeExecutor(cfg, db, kalshi_client=None)

    results = await executor.execute_signals([_build_signal("MKT2", 50.0)])
    assert len(results) == 1
    assert results[0]["success"] is False
    assert results[0]["skipped"] is True
    assert results[0]["status"] == "skipped_daily_exposure"


@pytest.mark.asyncio
async def test_executor_blocks_when_total_exposure_limit_reached(tmp_path):
    db = DatabaseManager(str(tmp_path / "total_exposure.sqlite"))
    await db.initialize()

    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    async with db.connect() as conn:
        await conn.execute(
            """
            INSERT INTO bankroll_history
            (timestamp_utc, balance, change, reason, reference_id, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (now, 100.0, 0.0, "init", None, now),
        )
        # Total cap = 80% of (bankroll + exposure) = 80% of (100 + 400) = 400.
        # Current exposure is already 400, so any additional notional should be blocked.
        await conn.execute(
            """
            INSERT INTO positions
            (market_id, platform, side, entry_price, quantity, cost, status, external_order_id, opened_at_utc, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("MKT_OPEN", "kalshi", "yes", 0.5, 800.0, 400.0, "open", "ord-open", now, now),
        )
        await conn.commit()

    cfg = _build_config()
    executor = TradeExecutor(cfg, db, kalshi_client=None)

    results = await executor.execute_signals([_build_signal("MKT3", 100.0)])
    assert len(results) == 1
    assert results[0]["success"] is False
    assert results[0]["skipped"] is True
    assert results[0]["status"] == "skipped_total_exposure"
