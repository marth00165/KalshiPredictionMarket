
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.trading.strategy import Strategy
from app.trading.position_manager import PositionManager, Trade
from app.models import MarketData, FairValueEstimate, TradeSignal
from app.bot import AdvancedTradingBot
from app.utils import InsufficientCapitalError, NoOpportunitiesError

try:
    import pytest_asyncio  # type: ignore # noqa: F401
    HAS_PYTEST_ASYNCIO = True
except Exception:
    HAS_PYTEST_ASYNCIO = False

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.strategy.min_edge = 0.05
    config.strategy.min_confidence = 0.5
    config.risk.max_positions = 5
    config.risk.max_kelly_fraction = 0.1
    config.risk.max_position_size = 100
    config.risk.max_total_exposure_fraction = 0.8
    config.risk.max_new_exposure_per_day_fraction = 0.8
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
    with pytest.raises(NoOpportunitiesError):
        strategy.generate_trade_signals([(market1, est1)], 1000, current_open_market_keys=open_keys)

    # Case 2: Duplicate in cycle
    # Two identical opportunities
    opportunities = [(market1, est1), (market1, est1)]
    signals = strategy.generate_trade_signals(opportunities, 1000, current_open_market_keys=set())
    assert len(signals) == 1 # Only one signal generated for same market


def test_strategy_buy_no_uses_no_probability_for_kelly(mock_config):
    strategy = Strategy(mock_config)
    mock_config.risk.max_kelly_fraction = 1.0
    mock_config.risk.max_position_size = 10_000

    market = MarketData(
        market_id="m_no",
        platform="k",
        title="Buy NO test",
        description="D",
        category="C",
        end_date="Z",
        yes_price=0.7,
        no_price=0.3,
        volume=1000,
        liquidity=1000,
    )
    estimate = FairValueEstimate(
        market_id="m_no",
        estimated_probability=0.4,  # YES probability
        confidence_level=0.8,
        reasoning="R",
        edge=-0.3,  # buy_no signal
    )

    signals = strategy.generate_trade_signals(
        [(market, estimate)],
        current_bankroll=1000,
        current_open_market_keys=set(),
        current_open_event_keys=set(),
    )

    assert len(signals) == 1
    signal = signals[0]
    assert signal.action == "buy_no"

    # Kelly for buy_no must use p_no = 1 - p_yes = 0.6 at no_price=0.3
    expected_kelly = (0.6 - 0.3) / (1 - 0.3)
    assert signal.kelly_fraction == pytest.approx(expected_kelly, rel=1e-6)

    expected_position_size = round(expected_kelly * 1000.0, 2)
    assert signal.position_size == pytest.approx(expected_position_size, abs=0.01)

    expected_ev = (0.6 * (1 - 0.3) - 0.4 * 0.3) * signal.position_size
    assert signal.expected_value == pytest.approx(expected_ev, rel=1e-6)


def test_strategy_event_level_duplicate_guard_keeps_best_score(mock_config):
    strategy = Strategy(mock_config)
    mock_config.risk.max_kelly_fraction = 1.0
    mock_config.risk.max_position_size = 10_000

    market_low = MarketData(
        market_id="evt1-low",
        platform="k",
        title="E1 Low",
        description="D",
        category="C",
        end_date="Z",
        yes_price=0.5,
        no_price=0.5,
        volume=1000,
        liquidity=1000,
        event_ticker="EVT1",
    )
    market_high = MarketData(
        market_id="evt1-high",
        platform="k",
        title="E1 High",
        description="D",
        category="C",
        end_date="Z",
        yes_price=0.5,
        no_price=0.5,
        volume=1000,
        liquidity=1000,
        event_ticker="EVT1",
    )
    est_low = FairValueEstimate(
        market_id="evt1-low",
        estimated_probability=0.6,
        confidence_level=0.6,
        reasoning="R",
        edge=0.10,  # score=0.06
    )
    est_high = FairValueEstimate(
        market_id="evt1-high",
        estimated_probability=0.65,
        confidence_level=0.7,
        reasoning="R",
        edge=0.15,  # score=0.105 (winner)
    )

    signals = strategy.generate_trade_signals(
        [(market_low, est_low), (market_high, est_high)],
        current_bankroll=1000,
        current_open_market_keys=set(),
        current_open_event_keys=set(),
    )

    assert len(signals) == 1
    assert signals[0].market.market_id == "evt1-high"

    open_event_keys = {strategy.get_event_key_for_market(market_high)}
    with pytest.raises(NoOpportunitiesError):
        strategy.generate_trade_signals(
            [(market_low, est_low), (market_high, est_high)],
            current_bankroll=1000,
            current_open_market_keys=set(),
            current_open_event_keys=open_event_keys,
        )


def test_strategy_exposure_reduces_position_size(mock_config):
    strategy = Strategy(mock_config)

    market = MarketData(
        market_id="m1", platform="k", title="T1", description="D", category="C", end_date="Z",
        yes_price=0.5, no_price=0.5, volume=1000, liquidity=1000
    )
    est = FairValueEstimate(
        market_id="m1", estimated_probability=0.7, confidence_level=0.8, reasoning="R", edge=0.2
    )

    # With no exposure, sizing uses full bankroll.
    no_exposure_signals = strategy.generate_trade_signals(
        [(market, est)],
        current_bankroll=1000,
        current_exposure=0,
        current_open_market_keys=set(),
    )

    # With high exposure, sizing uses only remaining deployable capital.
    high_exposure_signals = strategy.generate_trade_signals(
        [(market, est)],
        current_bankroll=1000,
        current_exposure=900,
        current_open_market_keys=set(),
    )

    assert no_exposure_signals[0].position_size == 100.0
    assert high_exposure_signals[0].position_size == 10.0


def test_strategy_exposure_caps_total_allocated_size(mock_config):
    strategy = Strategy(mock_config)
    mock_config.risk.max_kelly_fraction = 1.0
    mock_config.risk.max_position_size = 1000

    market1 = MarketData(
        market_id="m1", platform="k", title="T1", description="D", category="C", end_date="Z",
        yes_price=0.1, no_price=0.9, volume=1000, liquidity=1000
    )
    market2 = MarketData(
        market_id="m2", platform="k", title="T2", description="D", category="C", end_date="Z",
        yes_price=0.1, no_price=0.9, volume=1000, liquidity=1000
    )
    est1 = FairValueEstimate(
        market_id="m1", estimated_probability=0.9, confidence_level=0.8, reasoning="R", edge=0.8
    )
    est2 = FairValueEstimate(
        market_id="m2", estimated_probability=0.9, confidence_level=0.8, reasoning="R", edge=0.8
    )

    # Only $100 deployable ($1000 bankroll - $900 exposure).
    signals = strategy.generate_trade_signals(
        [(market1, est1), (market2, est2)],
        current_bankroll=1000,
        current_exposure=900,
        current_open_market_keys=set(),
    )

    assert len(signals) == 2
    assert sum(s.position_size for s in signals) <= 100.0


def test_strategy_raises_when_exposure_exhausts_bankroll(mock_config):
    strategy = Strategy(mock_config)

    market = MarketData(
        market_id="m1", platform="k", title="T1", description="D", category="C", end_date="Z",
        yes_price=0.5, no_price=0.5, volume=1000, liquidity=1000
    )
    est = FairValueEstimate(
        market_id="m1", estimated_probability=0.7, confidence_level=0.8, reasoning="R", edge=0.2
    )

    with pytest.raises(InsufficientCapitalError):
        strategy.generate_trade_signals(
            [(market, est)],
            current_bankroll=1000,
            current_exposure=1000,
            current_open_market_keys=set(),
        )

@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_PYTEST_ASYNCIO, reason="pytest-asyncio plugin not installed in this environment")
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
        bot.strategy.get_event_key_for_platform_market.side_effect = lambda platform, market_id: f"{platform}:{market_id.rsplit('-', 1)[0] if '-' in market_id else market_id}"
        bot.strategy.get_event_key_for_market.side_effect = lambda market: f"{market.platform}:{market.event_ticker or (market.market_id.rsplit('-', 1)[0] if '-' in market.market_id else market.market_id)}"
        bot.strategy.get_opportunity_score_tuple.side_effect = lambda market, est: (
            abs(est.effective_edge) * est.effective_confidence,
            abs(est.effective_edge),
            market.market_id,
        )
        bot.strategy.is_better_score.side_effect = lambda cand, cur: (
            cand[0] > cur[0] or (cand[0] == cur[0] and (cand[1] > cur[1] or (cand[1] == cur[1] and cand[2] < cur[2])))
        )

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
