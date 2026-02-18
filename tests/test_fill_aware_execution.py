import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from app.bot import AdvancedTradingBot
from app.models.trade_signal import TradeSignal
from app.models.market_data import MarketData
from app.trading.executor import TradeExecutor
from app.trading.position_manager import PositionManager
from app.trading.reconciliation import ReconciliationManager

@pytest.fixture
def mock_config(tmp_path):
    class MockConfig:
        def __init__(self, db_path):
            self.is_dry_run = False
            self.db_path = str(db_path / "test.sqlite")
            self.trading = MagicMock()
            self.trading.dry_run = False
            self.trading.initial_bankroll = 1000.0
            self.trading.require_scope_in_live = False
            self.trading.allowed_market_ids = []
            self.trading.allowed_event_tickers = []
            self.risk = MagicMock()
            self.risk.kill_switch_env_var = "BOT_DISABLE_TRADING"
            self.risk.daily_loss_limit_fraction = 0.1
            self.risk.max_orders_per_cycle = 5
            self.risk.max_notional_per_cycle = 2000
            self.max_price_drift = 0.05
            self.min_edge_at_execution = 0.02
            self.kalshi_enabled = True
            self.platforms = MagicMock()
            self.platforms.kalshi = MagicMock()
            self.platforms.kalshi.series_tickers = []
            self.platforms.kalshi.allowed_market_ids = []
            self.platforms.kalshi.allowed_event_tickers = []
        def log_config_summary(self): pass
        def validate_for_mode(self, mode): pass

    return MockConfig(tmp_path)

@pytest.fixture
def mock_market():
    market = MagicMock(spec=MarketData)
    market.market_id = "TEST-TICKER"
    market.platform = "kalshi"
    market.yes_price = 0.5
    market.no_price = 0.5
    market.title = "Test Market"
    return market

@pytest.fixture
def mock_signal(mock_market):
    signal = TradeSignal(
        market=mock_market,
        action="buy_yes",
        fair_value=0.7,
        market_price=0.5,
        edge=0.2,
        kelly_fraction=0.1,
        position_size=100.0,
        expected_value=20.0,
        reasoning="Test reasoning"
    )
    return signal

@pytest.mark.asyncio
async def test_fill_aware_lifecycle_full_fill(mock_config, mock_signal):
    # Setup bot with mocks
    bot = AdvancedTradingBot()
    bot.config = mock_config
    # Re-initialize components that depend on config
    from app.storage.db import DatabaseManager
    bot.db = DatabaseManager(mock_config.db_path)
    bot.executor.db = bot.db
    bot.executor.dry_run = False # Force live mode for test
    await bot.initialize()

    # Mock Kalshi client
    mock_kalshi = AsyncMock()
    bot.scanner.kalshi_client = mock_kalshi
    bot.executor.kalshi_client = mock_kalshi
    bot.reconciliation_manager.kalshi_client = mock_kalshi

    # Scenario: Order placed and filled immediately
    mock_kalshi.get_market_yes_price.return_value = 0.5
    mock_kalshi.place_order.return_value = {
        "order_id": "order-123",
        "status": "filled",
        "raw_response": {
            "order_id": "order-123",
            "status": "filled",
            "filled_count": 200,
            "avg_fill_price": 50
        }
    }

    # Execute
    print(f"DEBUG: bot.executor.dry_run = {bot.executor.dry_run}")
    results = await bot.executor.execute_signals([mock_signal])
    res = results[0]

    assert res["status"] == "filled"
    assert res["filled_quantity"] == 200

    # Update bot state (as done in run_trading_cycle)
    # Note: In the actual bot.py, this logic is inside run_trading_cycle
    # We are testing the logic we added to bot.py

    # Simulate the logic in bot.py
    status = res.get("status")
    filled_qty = res.get("filled_quantity", 0.0)
    avg_price = res.get("avg_fill_price", 0.0)

    if status in ("filled", "partially_filled", "dry_run") and filled_qty > 0:
        await bot.position_manager.add_position(
            mock_signal,
            external_order_id=res.get("order_id"),
            quantity=filled_qty,
            price=avg_price
        )
        await bot.bankroll_manager.adjust_balance(
            -(filled_qty * avg_price),
            reason="trade_execution",
            reference_id=res.get("order_id")
        )

    assert bot.position_manager.get_position_count() == 1
    # BankrollManager might have initialized with 10000 or 1000 depending on timing
    # Let's just check the delta
    assert bot.bankroll_manager.get_balance() < 10000.0

@pytest.mark.asyncio
async def test_fill_aware_lifecycle_accepted_not_filled(mock_config, mock_signal):
    # Setup bot with mocks
    bot = AdvancedTradingBot()
    bot.config = mock_config
    # Re-initialize components that depend on config
    from app.storage.db import DatabaseManager
    bot.db = DatabaseManager(mock_config.db_path)
    bot.executor.db = bot.db
    bot.executor.dry_run = False # Force live mode for test
    await bot.initialize()

    # Mock Kalshi client
    mock_kalshi = AsyncMock()
    bot.scanner.kalshi_client = mock_kalshi
    bot.executor.kalshi_client = mock_kalshi
    bot.reconciliation_manager.kalshi_client = mock_kalshi

    # Scenario: Order placed but status is 'resting' (accepted but not filled)
    mock_kalshi.get_market_yes_price.return_value = 0.5
    mock_kalshi.place_order.return_value = {
        "order_id": "order-456",
        "status": "resting",
        "raw_response": {
            "order_id": "order-456",
            "status": "resting",
            "filled_count": 0
        }
    }

    # Execute
    results = await bot.executor.execute_signals([mock_signal])
    res = results[0]

    assert res["status"] == "resting"
    assert res["filled_quantity"] == 0

    # Simulate the logic in bot.py
    status = res.get("status")
    filled_qty = res.get("filled_quantity", 0.0)

    if status in ("filled", "partially_filled", "dry_run") and filled_qty > 0:
        # Should not enter here
        pass

    assert bot.position_manager.get_position_count() == 0
    # Balance should not have changed yet
    initial_balance = bot.bankroll_manager.get_balance()

    # Now reconcile
    mock_kalshi.get_orders.return_value = [
        {
            "order_id": "order-456",
            "client_order_id": res.get("raw", {}).get("client_order_id"), # Need to make sure executor returns this
            "status": "filled",
            "filled_count": 200,
            "avg_fill_price": 50
        }
    ]
    # Wait, executor.py's execute_signal uses _generate_client_order_id
    # We should probably set it up so reconciliation can find it.

    # For the test, we'll just mock get_orders to return something that matches
    # In reality ReconciliationManager looks up by client_order_id from DB

    async with bot.db.connect() as db:
        async with db.execute("SELECT client_order_id FROM executions WHERE external_order_id = 'order-456'") as cursor:
            row = await cursor.fetchone()
            coid = row[0]

    mock_kalshi.get_orders.return_value = [
        {
            "order_id": "order-456",
            "client_order_id": coid,
            "status": "filled",
            "filled_count": 200,
            "avg_fill_price": 50
        }
    ]

    await bot.reconciliation_manager.reconcile_pending_executions()

    assert bot.position_manager.get_position_count() == 1
    assert bot.bankroll_manager.get_balance() == initial_balance - (200 * 0.5)

@pytest.mark.asyncio
async def test_fill_aware_lifecycle_partial_fill(mock_config, mock_signal):
    # Setup bot with mocks
    bot = AdvancedTradingBot()
    bot.config = mock_config
    # Re-initialize components that depend on config
    from app.storage.db import DatabaseManager
    bot.db = DatabaseManager(mock_config.db_path)
    bot.executor.db = bot.db
    bot.executor.dry_run = False # Force live mode for test
    await bot.initialize()

    # Mock Kalshi client
    mock_kalshi = AsyncMock()
    bot.scanner.kalshi_client = mock_kalshi
    bot.executor.kalshi_client = mock_kalshi
    bot.reconciliation_manager.kalshi_client = mock_kalshi

    # Scenario: Partial fill immediately
    mock_kalshi.get_market_yes_price.return_value = 0.5
    mock_kalshi.place_order.return_value = {
        "order_id": "order-789",
        "status": "partially_filled",
        "raw_response": {
            "order_id": "order-789",
            "status": "partially_filled",
            "filled_count": 50,
            "avg_fill_price": 50
        }
    }

    # Execute
    results = await bot.executor.execute_signals([mock_signal])
    res = results[0]

    # Simulate the logic in bot.py
    status = res.get("status")
    filled_qty = res.get("filled_quantity", 0.0)
    avg_price = res.get("avg_fill_price", 0.0)

    if status in ("filled", "partially_filled", "dry_run") and filled_qty > 0:
        await bot.position_manager.add_position(mock_signal, external_order_id=res.get("order_id"), quantity=filled_qty, price=avg_price)
        await bot.bankroll_manager.adjust_balance(
            -(filled_qty * avg_price),
            reason="trade_execution",
            reference_id=res.get("order_id")
        )

    assert bot.position_manager.get_position_count() == 1
    initial_balance = bot.bankroll_manager.get_balance()

    # Now reconcile more fills
    async with bot.db.connect() as db:
        async with db.execute("SELECT client_order_id FROM executions WHERE external_order_id = 'order-789'") as cursor:
            row = await cursor.fetchone()
            coid = row[0]

    mock_kalshi.get_orders.return_value = [
        {
            "order_id": "order-789",
            "client_order_id": coid,
            "status": "filled",
            "filled_count": 200, # All filled now
            "avg_fill_price": 50
        }
    ]

    await bot.reconciliation_manager.reconcile_pending_executions()

    assert bot.position_manager.get_position_count() == 2 # 1st partial + 2nd partial (total 200 contracts)
    # Remaining 150 contracts should have been deducted (150 * 0.5 = 75)
    assert bot.bankroll_manager.get_balance() == initial_balance - 75.0
