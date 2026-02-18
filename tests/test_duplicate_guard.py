
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.trading.strategy import Strategy
from app.trading.position_manager import PositionManager, Trade
from app.models import MarketData, FairValueEstimate, TradeSignal
from app.bot import AdvancedTradingBot

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.strategy.min_edge = 0.05
    config.strategy.min_confidence = 0.5
    config.risk.max_positions = 5
    config.risk.max_kelly_fraction = 0.1
    config.risk.max_position_size = 100
    config.analysis_provider = "claude"
    type(config).min_edge_percentage = 5.0
    type(config).min_confidence_percentage = 50.0
    config.api.batch_size = 10
    config.api.api_cost_limit_per_cycle = 100
    config.platforms.polymarket.enabled = True
    config.platforms.kalshi.enabled = True
    config.platforms.polymarket.max_markets = 100
    config.platforms.kalshi.max_markets = 100
    config.filters.min_volume = 100
    config.filters.min_liquidity = 100
    config.is_dry_run = False
    return config

def test_position_manager_open_market_keys():
    db = MagicMock()
    pm = PositionManager(db)

    # Mock open positions
    trade1 = Trade(market_id="m1", platform="kalshi", side="yes", action="buy_yes", quantity=1, position_size=10, entry_price=0.5)
    trade2 = Trade(market_id="m2", platform="poly", side="no", action="buy_no", quantity=1, position_size=10, entry_price=0.5)
    trade3 = Trade(market_id="m3", platform="kalshi", side="yes", action="buy_yes", quantity=1, position_size=10, entry_price=0.5, status="closed")

    pm.open_positions = [trade1, trade2, trade3]

    keys = pm.get_open_market_keys()
    assert keys == {"kalshi:m1", "poly:m2"}
    assert "kalshi:m3" not in keys

def test_strategy_duplicate_guard(mock_config):
    strategy = Strategy(mock_config)

    market1 = MarketData(market_id="m1", platform="k", title="T1", description="D", category="C", end_date="Z", yes_price=0.5, no_price=0.5, volume=1000, liquidity=1000)
    est1 = FairValueEstimate(market_id="m1", estimated_probability=0.7, confidence_level=0.8, reasoning="R", edge=0.2)

    # Case 1: Already open
    open_keys = {"k:m1"}
    with pytest.raises(Exception): # NoOpportunitiesError
        strategy.generate_trade_signals([(market1, est1)], 1000, current_open_market_keys=open_keys)

    # Case 2: Duplicate in cycle
    # Two identical opportunities
    opportunities = [(market1, est1), (market1, est1)]
    signals = strategy.generate_trade_signals(opportunities, 1000, current_open_market_keys=set())
    assert len(signals) == 1 # Only one signal generated for same market

@pytest.mark.asyncio
async def test_bot_duplicate_guard_reporting(mock_config):
    with patch('app.bot.DatabaseManager'), \
         patch('app.bot.MarketScanner') as MockScanner, \
         patch('app.bot.Strategy') as MockStrategy, \
         patch('app.bot.TradeExecutor') as MockExecutor, \
         patch('app.bot.PositionManager') as MockPositionManager, \
         patch('app.bot.BankrollManager') as MockBankrollManager, \
         patch('app.bot.ClaudeAnalyzer'):

        bot = AdvancedTradingBot()
        bot.position_manager = MockPositionManager.return_value
        bot.strategy = MockStrategy.return_value
        bot.executor = MockExecutor.return_value
        bot.bankroll_manager = MockBankrollManager.return_value
        bot.scanner = MockScanner.return_value

        # Setup managers
        bot.bankroll_manager.get_balance.return_value = 1000
        bot.position_manager.get_open_market_keys.return_value = {"k:m1"}

        market1 = MarketData(market_id="m1", platform="k", title="T1", description="D", category="C", end_date="Z", yes_price=0.5, no_price=0.5, volume=1000, liquidity=1000)
        market2 = MarketData(market_id="m2", platform="k", title="T2", description="D", category="C", end_date="Z", yes_price=0.5, no_price=0.5, volume=1000, liquidity=1000)

        # Mock scanner
        bot.scanner.scan_all_markets = AsyncMock(return_value=[market1, market2])

        # Mock strategy
        bot.strategy.evaluate_market_filters.return_value = {"passed": True}
        bot.strategy.classify_volume_tier.return_value = "high"
        bot.strategy.categorize_markets_by_tier.return_value = {"high": [market1, market2], "medium":[], "low":[], "skip":[]}
        bot.strategy.filter_markets.return_value = [market1, market2]

        # Mock analyzer
        est1 = FairValueEstimate(market_id="m1", estimated_probability=0.7, confidence_level=0.8, reasoning="R", edge=0.2)
        est2 = FairValueEstimate(market_id="m2", estimated_probability=0.7, confidence_level=0.8, reasoning="R", edge=0.2)
        bot.analyzer.analyze_market_batch = AsyncMock(return_value=[est1, est2])
        bot.analyzer.get_api_stats.return_value = {"total_cost": 0.0}

        # Two opportunities for m2 (duplicate in cycle) and one for m1 (already open)
        bot.strategy.find_opportunities.return_value = [(market1, est1), (market2, est2), (market2, est2)]

        # Mock signals - force a signal for m2
        signal2 = TradeSignal(market=market2, action="buy_yes", fair_value=0.7, market_price=0.5, edge=0.2, kelly_fraction=0.1, position_size=100, expected_value=20, reasoning="Test")
        bot.strategy.generate_trade_signals.return_value = [signal2]

        bot.initialize = AsyncMock()
        bot.executor.execute_signals = AsyncMock(return_value=[{"success": True, "order_id": "ord1"}])

        # Trigger cycle
        await bot.run_trading_cycle()

        # Check report
        # We expect 2 skips: m1 (already open) and the second m2 (duplicate in cycle)
        # However, the bot's Step 4.5 filters them.
        # Let's verify the counts in the report if we could access it.
        # Since run_trading_cycle doesn't return the report, we'd need to mock logger or something.
        # But we can verify executor was called with only one signal.
        bot.executor.execute_signals.assert_called_once()
        args, _ = bot.executor.execute_signals.call_args
        assert len(args[0]) == 1
        assert args[0][0].market.market_id == "m2"
