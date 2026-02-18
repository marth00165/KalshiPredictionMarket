import pytest
import os
import json
import aiosqlite
from unittest.mock import MagicMock, AsyncMock, patch
from app.bot import AdvancedTradingBot
from app.config import ConfigManager, RiskConfig
from app.models import MarketData, TradeSignal
from app.trading import TradeExecutor

@pytest.fixture
def mock_config():
    config = MagicMock(spec=ConfigManager)
    config.is_dry_run = True
    config.is_autonomous = False
    config.is_non_interactive = False
    config.kalshi_enabled = True
    config.polymarket_enabled = False
    config.db_path = ":memory:"
    config.trading = MagicMock()
    config.trading.dry_run = True
    config.trading.require_scope_in_live = True
    config.trading.allowed_market_ids = []
    config.trading.allowed_event_tickers = []
    config.platforms = MagicMock()
    config.platforms.kalshi = MagicMock()
    config.platforms.kalshi.series_tickers = []
    config.platforms.kalshi.allowed_market_ids = []
    config.platforms.kalshi.allowed_event_tickers = []
    config.risk = MagicMock()
    config.risk.kill_switch_env_var = "BOT_DISABLE_TRADING"
    config.risk.daily_loss_limit_fraction = 0.1
    config.risk.max_orders_per_cycle = 5
    config.risk.max_notional_per_cycle = 2000
    config.risk.critical_webhook_url = None
    config.api = MagicMock()
    config.api.batch_size = 5
    config.api.api_cost_limit_per_cycle = 5.0
    config.strategy = MagicMock()
    config.strategy.min_edge = 0.08
    config.strategy.min_confidence = 0.60
    config.filters = MagicMock()
    config.filters.min_volume = 1000
    config.filters.min_liquidity = 500
    config.max_price_drift = 0.05
    config.min_edge_at_execution = 0.02
    return config

@pytest.mark.asyncio
async def test_scope_guard_series_prefix_match(mock_config):
    mock_config.platforms.kalshi.series_tickers = ["KXSERIES"]
    bot = AdvancedTradingBot.__new__(AdvancedTradingBot)
    bot.config = mock_config

    # 1. Market matching series_ticker field
    m1 = MarketData(
        platform="kalshi", market_id="m1", title="M1", description="",
        yes_price=0.5, no_price=0.5, volume=1000, liquidity=100,
        end_date="2025-01-01", category="test", series_ticker="KXSERIES"
    )
    assert bot._is_market_in_allowed_scope(m1) is True

    # 2. Market matching market_id prefix fallback
    m2 = MarketData(
        platform="kalshi", market_id="KXSERIES-25JAN01", title="M2", description="",
        yes_price=0.5, no_price=0.5, volume=1000, liquidity=100,
        end_date="2025-01-01", category="test"
    )
    assert bot._is_market_in_allowed_scope(m2) is True

    # 3. Unmatched market should fail
    m3 = MarketData(
        platform="kalshi", market_id="OTHER-25JAN01", title="M3", description="",
        yes_price=0.5, no_price=0.5, volume=1000, liquidity=100,
        end_date="2025-01-01", category="test"
    )
    assert bot._is_market_in_allowed_scope(m3) is False

@pytest.mark.asyncio
async def test_live_requires_scope_when_enabled(mock_config):
    mock_config.is_dry_run = False
    mock_config.trading.dry_run = False
    mock_config.trading.require_scope_in_live = True

    bot = AdvancedTradingBot.__new__(AdvancedTradingBot)
    bot.config = mock_config
    bot.db = MagicMock()
    bot._initialized = False

    with pytest.raises(ValueError, match="Live trading requires an explicit market scope"):
        await bot.initialize()

@pytest.mark.asyncio
async def test_kill_switch_blocks_execution(mock_config):
    bot = AdvancedTradingBot.__new__(AdvancedTradingBot)
    bot.config = mock_config
    bot.notifier = AsyncMock()

    with patch.dict(os.environ, {"BOT_DISABLE_TRADING": "1"}):
        signals = [MagicMock()]
        result = await bot._run_risk_guards(signals)
        assert result == []
        bot.notifier.send_notification.assert_called()

@pytest.mark.asyncio
async def test_execution_revalidation_drift_blocks_order(mock_config, tmp_path):
    from app.storage.db import DatabaseManager
    db_file = str(tmp_path / "test1.sqlite")
    db = DatabaseManager(db_file)
    await db.initialize()

    mock_kalshi = AsyncMock()
    mock_kalshi.get_market_yes_price.return_value = 0.60 # Drifted from 0.50

    mock_config.max_price_drift = 0.05
    executor = TradeExecutor(mock_config, db, kalshi_client=mock_kalshi)
    executor.dry_run = False

    m1 = MarketData(
        platform="kalshi", market_id="m1", title="M1", description="",
        yes_price=0.5, no_price=0.5, volume=1000, liquidity=100,
        end_date="2025-01-01", category="test"
    )
    signal = TradeSignal(market=m1, action="buy_yes", fair_value=0.7, market_price=0.5, edge=0.2, kelly_fraction=0.1, position_size=100, expected_value=20, reasoning="test")

    result = await executor.execute_signal(signal)
    assert result['success'] is False
    assert result['status'] == 'skipped_drift'
    mock_kalshi.place_order.assert_not_called()

@pytest.mark.asyncio
async def test_execution_revalidation_edge_blocks_order(mock_config, tmp_path):
    from app.storage.db import DatabaseManager
    db_file = str(tmp_path / "test2.sqlite")
    db = DatabaseManager(db_file)
    await db.initialize()

    mock_kalshi = AsyncMock()
    mock_kalshi.get_market_yes_price.return_value = 0.69 # Edge is now 0.70 - 0.69 = 0.01

    mock_config.max_price_drift = 0.20
    mock_config.min_edge_at_execution = 0.05
    executor = TradeExecutor(mock_config, db, kalshi_client=mock_kalshi)
    executor.dry_run = False

    m1 = MarketData(
        platform="kalshi", market_id="m1", title="M1", description="",
        yes_price=0.5, no_price=0.5, volume=1000, liquidity=100,
        end_date="2025-01-01", category="test"
    )
    signal = TradeSignal(market=m1, action="buy_yes", fair_value=0.7, market_price=0.5, edge=0.2, kelly_fraction=0.1, position_size=100, expected_value=20, reasoning="test")

    result = await executor.execute_signal(signal)
    assert result['success'] is False
    assert result['status'] == 'skipped_edge'
    mock_kalshi.place_order.assert_not_called()

@pytest.mark.asyncio
async def test_idempotency_existing_client_order_id_skips_place_order(mock_config, tmp_path):
    from app.storage.db import DatabaseManager
    db_file = str(tmp_path / "test3.sqlite")
    db = DatabaseManager(db_file)
    await db.initialize()

    mock_kalshi = AsyncMock()
    executor = TradeExecutor(mock_config, db, kalshi_client=mock_kalshi)
    executor.dry_run = False

    m1 = MarketData(
        platform="kalshi", market_id="m1", title="M1", description="",
        yes_price=0.5, no_price=0.5, volume=1000, liquidity=100,
        end_date="2025-01-01", category="test"
    )
    signal = TradeSignal(market=m1, action="buy_yes", fair_value=0.7, market_price=0.5, edge=0.2, kelly_fraction=0.1, position_size=100, expected_value=20, reasoning="test")

    client_order_id = executor._generate_client_order_id(signal)

    async with db.connect() as conn:
        await conn.execute(
            "INSERT INTO executions (market_id, platform, action, quantity, price, cost, external_order_id, status, executed_at_utc, created_at_utc, client_order_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("m1", "kalshi", "buy_yes", 200, 0.5, 100, "ord123", "executed", "now", "now", client_order_id)
        )
        await conn.commit()

    result = await executor.execute_signal(signal)
    assert result['idempotent'] is True
    assert result['success'] is True
    assert result['order_id'] == "ord123"
    mock_kalshi.place_order.assert_not_called()

@pytest.mark.asyncio
async def test_cli_fast_fail_missing_scope_live_non_interactive():
    from app.__main__ import main
    import sys

    config_path = "test_fail_config.json"
    config_data = {
        "trading": {
            "dry_run": False,
            "require_scope_in_live": True,
            "autonomous_mode": True
        },
        "platforms": {
            "kalshi": {
                "enabled": True,
                "api_key": "test",
                "private_key": "test"
            }
        },
        "api": {
            "claude_api_key": "test"
        }
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    test_args = ["app", "--config", config_path, "--mode", "trade", "--non-interactive"]

    with patch.object(sys, 'argv', test_args), \
         patch('app.bot.AdvancedTradingBot.initialize', side_effect=ValueError("Live trading requires an explicit market scope")):
        with pytest.raises(SystemExit) as cm:
            await main()
        assert cm.value.code == 1

    if os.path.exists(config_path):
        os.remove(config_path)

@pytest.mark.asyncio
async def test_persistent_cycle_count(mock_config, tmp_path):
    from app.storage.db import DatabaseManager
    db_file = str(tmp_path / "test4.sqlite")
    db = DatabaseManager(db_file)
    await db.initialize()

    bot = AdvancedTradingBot.__new__(AdvancedTradingBot)
    bot.config = mock_config
    bot.db = db
    bot._initialized = False
    bot.position_manager = MagicMock()
    bot.position_manager.reset_for_new_dry_run_session = AsyncMock()
    bot.position_manager.load_positions = AsyncMock()
    bot.bankroll_manager = MagicMock()
    bot.bankroll_manager.reset_for_new_dry_run_session = AsyncMock()
    bot.bankroll_manager.initialize = AsyncMock()
    bot.scanner = MagicMock()
    bot.scanner.kalshi_client = None
    bot.executor = MagicMock()
    bot.reconciliation_manager = MagicMock()
    bot.runtime_scan_scope_description = "all_markets"
    bot.runtime_scan_series_tickers = None
    bot.error_reporter = MagicMock()
    bot.notifier = AsyncMock()
    bot.strategy = MagicMock()
    bot.signal_fusion_service = MagicMock()
    bot.signal_fusion_service.report_status.return_value = {}

    await db.update_status('cycle_count', 42)
    await bot.initialize()
    assert bot.cycle_count == 42

    with patch.object(bot, 'scanner'), \
         patch.object(bot, 'bankroll_manager'), \
         patch.object(bot, 'position_manager'), \
         patch.object(bot, 'reconciliation_manager'), \
         patch.object(bot, 'executor'), \
         patch.object(bot, '_update_heartbeat'):

        bot.scanner.scan_all_markets = AsyncMock(return_value=[])
        await bot.run_trading_cycle()
        assert bot.cycle_count == 43

        status = await db.get_last_status()
        assert int(status['cycle_count']) == 43

def test_risk_config_validation():
    # Valid
    rc = RiskConfig(max_orders_per_cycle=5, max_notional_per_cycle=1000, daily_loss_limit_fraction=0.1, kill_switch_env_var="TEST")
    rc.validate()

    # Invalid max_orders
    with pytest.raises(ValueError, match="max_orders_per_cycle"):
        RiskConfig(max_orders_per_cycle=0).validate()

    # Invalid max_notional
    with pytest.raises(ValueError, match="max_notional_per_cycle"):
        RiskConfig(max_notional_per_cycle=0).validate()

    # Invalid loss limit
    with pytest.raises(ValueError, match="daily_loss_limit_fraction"):
        RiskConfig(daily_loss_limit_fraction=1.5).validate()

    # Invalid kill switch
    with pytest.raises(ValueError, match="kill_switch_env_var"):
        RiskConfig(kill_switch_env_var="").validate()
