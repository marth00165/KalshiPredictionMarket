---
name: prediction-market-trader
description: Build AI-powered prediction market trading bots for Polymarket and Kalshi. Analyzes 1000+ markets using Claude AI for fair value estimation, detects mispricings >8%, sizes positions with Kelly criterion, executes trades autonomously, and pays its own API bills. Use when users want to create automated trading systems, analyze prediction markets, implement quantitative strategies, or build self-sustaining trading agents.
---

# Prediction Market Trading Bot Skill

This skill helps users build sophisticated AI-powered trading bots for prediction markets (Polymarket, Kalshi). The bot scans markets at scale, uses Claude AI to estimate fair values, finds mispricings, sizes positions mathematically, and executes trades autonomously while tracking its own costs.

## When to Use This Skill

Trigger this skill when the user wants to:
- Build a prediction market trading bot
- Automate trading on Polymarket or Kalshi
- Analyze markets using AI for edge detection
- Implement Kelly criterion position sizing
- Create a self-sustaining trading system
- Backtest prediction market strategies
- Monitor trading performance and profitability

## Core Philosophy

The bot is designed around these principles:

1. **AI-Powered Analysis**: Use Claude to estimate fair values rather than simple heuristics
2. **Mathematical Rigor**: Kelly criterion for optimal position sizing
3. **Cost Awareness**: Track and pay API bills from profits to ensure sustainability
4. **Scale**: Analyze 1000+ markets to find best opportunities
5. **Risk Management**: Built-in limits, stop losses, and position sizing

## Key Components

### 1. Market Scanner
**Purpose**: Efficiently fetch 1000+ markets from Polymarket and Kalshi

**Implementation**:
```python
class MarketScanner:
    async def scan_all_markets(self) -> List[MarketData]:
        # Fetch from both platforms in parallel
        # Handle pagination and rate limits
        # Filter by volume, liquidity, price range
        # Return structured MarketData objects
```

**Critical Features**:
- Async/await for parallel API calls
- Proper rate limiting (100 req/min for Polymarket)
- Pagination handling
- Error recovery

### 2. Claude Analyzer
**Purpose**: Estimate fair probability for each market using Claude AI

**Prompt Structure**:
```
Analyze this prediction market and estimate the TRUE probability.

Market: {title}
Description: {description}
Current Price: {yes_price}%
Volume: ${volume:,}

Your Task:
1. Research what is known about this event
2. Consider base rates, historical data, current trends
3. Estimate the TRUE probability (0-100%)
4. Explain your reasoning step-by-step
5. Rate your confidence (0-100%)

Respond ONLY in JSON:
{
  "probability": <float 0-100>,
  "confidence": <float 0-100>,
  "reasoning": "<detailed explanation>",
  "key_factors": ["factor1", "factor2"],
  "data_sources": ["source1", "source2"]
}
```

**Cost Tracking**:
```python
# Track every API call
input_cost = (input_tokens / 1_000_000) * 3.00
output_cost = (output_tokens / 1_000_000) * 15.00
total_api_cost += input_cost + output_cost
```

### 3. Edge Detection
**Purpose**: Find mispricings where market price differs from fair value

```python
def find_opportunities(estimates, markets, min_edge=0.08):
    opportunities = []

    for estimate in estimates:
        market = get_market(estimate.market_id)
        edge = abs(estimate.probability - market.yes_price)

        # Only trade if:
        # - Edge > threshold (default 8%)
        # - High confidence (>60%)
        # - Sufficient volume
        if edge >= min_edge and estimate.confidence >= 0.6:
            opportunities.append((market, estimate))

    return opportunities
```

### 4. Kelly Criterion Position Sizing
**Purpose**: Calculate mathematically optimal position sizes

**Formula**:
```python
def calculate_kelly(probability: float, market_price: float) -> float:
    """
    For binary outcomes:
    kelly = edge / (1 - market_price)

    Example:
    True prob: 60%, Market: 40%
    Edge: 20%
    Kelly: 0.20 / 0.60 = 33.3% of bankroll
    """
    edge = probability - market_price

    if edge <= 0:
        return 0  # No edge, no bet

    kelly = edge / (1 - market_price)

    # Safety cap (default 25%)
    return min(kelly, max_kelly_fraction)
```

**Position Size**:
```python
position_size = kelly_fraction * bankroll
position_size = min(position_size, max_position)  # Apply limits
```

### 5. Trade Execution
**Purpose**: Place orders on platforms

**Polymarket** (requires blockchain wallet):
```python
from py_clob_client import ClobClient

client = ClobClient(
    host="https://clob.polymarket.com",
    key=private_key,
    chain_id=137  # Polygon
)

order = client.create_market_order(
    token_id=token_id,
    side=BUY,
    amount=position_size
)
```

**Kalshi** (requires API keys):
```python
from kalshi_python import KalshiClient

client = KalshiClient(config)

order = client.create_order(
    ticker=ticker,
    side='yes',
    count=quantity,
    yes_price=price_cents
)
```

### 6. Cost Management
**Purpose**: Ensure profitability after API costs

```python
def pay_api_bills():
    # Deduct from bankroll
    bankroll -= total_api_cost

    # Check profitability
    net_profit = expected_value - total_api_cost

    if net_profit <= 0:
        logger.warning("API costs exceed expected profits!")
        # Consider reducing markets analyzed or increasing min_edge
```

## Implementation Workflow

When user requests a trading bot:

### Step 1: Gather Requirements
Ask about:
- **Platforms**: "Which platform(s)? Polymarket, Kalshi, or both?"
- **Bankroll**: "How much capital to deploy?"
- **Risk**: "Conservative or aggressive?"
- **Automation**: "Fully automated or manual approval?"

### Step 2: Create Core Files

Generate these files in order:

**1. Data structures** (`models.py`):
```python
@dataclass
class MarketData:
    platform: str
    market_id: str
    title: str
    description: str
    yes_price: float
    volume: float
    liquidity: float
    category: str
```

**2. Market scanner** (`market_scanner.py`):
- Async API fetching
- Rate limiting
- Filtering logic

**3. Claude analyzer** (`claude_analyzer.py`):
- API integration
- Batch processing
- Cost tracking

**4. Kelly calculator** (`kelly_calculator.py`):
- Position sizing math
- Safety caps

**5. Trade executor** (`trade_executor.py`):
- Platform integrations
- Order placement
- Dry-run mode

**6. Main bot** (`ai_trading_bot.py`):
- Orchestrates all components
- Main trading loop
- Error handling

**7. Configuration** (`config.json`):
- Strategy parameters
- Risk limits
- API settings

**8. Documentation** (`README.md`):
- Setup instructions
- Usage guide
- Examples

### Step 3: Add Safety Features

Always include:
- **Dry run mode** (default)
- **Position limits** (max per trade, max total)
- **Confidence thresholds** (only high-confidence trades)
- **Volume filters** (avoid illiquid markets)
- **Stop losses**
- **Error handling**

### Step 4: Add Analytics

Include performance monitoring:
- Win rate tracking
- Sharpe ratio calculation
- Maximum drawdown
- API cost efficiency
- Equity curve plotting
- Claude accuracy calibration

## Configuration Template

```json
{
  "trading": {
    "initial_bankroll": 10000
  },

  "platforms": {
    "polymarket": {
      "enabled": true,
      "max_markets": 600,
      "private_key": "ENV:POLYMARKET_KEY"
    },
    "kalshi": {
      "enabled": true,
      "max_markets": 400,
      "api_key_id": "ENV:KALSHI_KEY_ID",
      "private_key_path": "ENV:KALSHI_KEY_PATH"
    }
  },

  "filters": {
    "min_volume": 1000,
    "min_liquidity": 500,
    "max_price": 0.99,
    "min_price": 0.01
  },

  "strategy": {
    "min_edge": 0.08,
    "min_confidence": 0.6,
    "use_claude_analysis": true
  },

  "risk": {
    "max_positions": 10,
    "max_position_size": 1000,
    "max_total_exposure": 5000,
    "max_kelly_fraction": 0.25
  },

  "api": {
    "batch_size": 50,
    "max_requests_per_minute": 100,
    "api_cost_limit_per_cycle": 5.00
  },

  "execution": {
    "dry_run": true,
    "slippage_tolerance": 0.01
  }
}
```

## Main Trading Loop

```python
async def run_trading_cycle():
    logger.info("Starting trading cycle...")

    # 1. SCAN MARKETS
    markets = await scanner.scan_all_markets()
    logger.info(f"Found {len(markets)} markets")

    # 2. FILTER
    filtered = filter_markets(
        markets,
        min_volume=config['filters']['min_volume'],
        min_liquidity=config['filters']['min_liquidity']
    )
    logger.info(f"{len(filtered)} markets passed filters")

    # 3. ANALYZE WITH CLAUDE (in batches)
    estimates = await analyzer.analyze_batch(
        filtered,
        batch_size=config['api']['batch_size']
    )
    logger.info(f"Claude analyzed {len(estimates)} markets")
    logger.info(f"API cost: ${analyzer.total_api_cost:.2f}")

    # 4. FIND OPPORTUNITIES
    opportunities = find_opportunities(
        estimates,
        min_edge=config['strategy']['min_edge'],
        min_confidence=config['strategy']['min_confidence']
    )
    logger.info(f"Found {len(opportunities)} opportunities")

    # 5. CALCULATE POSITIONS (Kelly)
    signals = []
    for market, estimate in opportunities:
        kelly = calculate_kelly(
            estimate.probability,
            market.yes_price,
            max_fraction=config['risk']['max_kelly_fraction']
        )

        position_size = kelly * bankroll
        position_size = min(position_size, config['risk']['max_position_size'])

        signals.append(TradeSignal(
            market=market,
            action='buy_yes' if estimate.edge > 0 else 'buy_no',
            position_size=position_size,
            expected_value=calculate_ev(estimate, position_size)
        ))

    # 6. EXECUTE TRADES
    for signal in signals:
        await executor.execute_trade(signal)

    # 7. PAY API BILLS
    bankroll -= analyzer.total_api_cost
    logger.info(f"Updated bankroll: ${bankroll:,.2f}")
```

## Example Prompts That Trigger This Skill

- "Build me a trading bot for Polymarket and Kalshi"
- "Create an AI-powered prediction market trader"
- "Help me automate trading on prediction markets"
- "Build a bot that uses Claude to analyze markets"
- "Create a Kelly criterion betting system"
- "Build a self-sustaining trading agent"

## Common Customizations

**Conservative Setup**:
```json
{
  "strategy": {
    "min_edge": 0.15,
    "min_confidence": 0.80
  },
  "risk": {
    "max_kelly_fraction": 0.10,
    "max_positions": 5
  }
}
```

**Aggressive Setup**:
```json
{
  "strategy": {
    "min_edge": 0.05,
    "min_confidence": 0.50
  },
  "risk": {
    "max_kelly_fraction": 0.50,
    "max_positions": 20
  }
}
```

**Arbitrage Focus**:
- Scan same event on both platforms
- Buy low platform, sell high platform
- Account for fees (0.5-1%)

## Performance Metrics

Track these KPIs:

```python
metrics = {
    'total_trades': len(trades),
    'win_rate': winning_trades / total_trades,
    'total_profit': sum(trade.profit for trade in trades),
    'sharpe_ratio': calculate_sharpe(returns),
    'max_drawdown': calculate_max_drawdown(equity_curve),
    'avg_edge': mean(trade.edge for trade in trades),
    'edge_captured': mean(trade.actual_profit / trade.position_size),
    'api_cost': total_api_cost,
    'roi_on_api': total_profit / total_api_cost
}
```

## Risk Warnings

Always include these warnings in documentation:

⚠️ **Important Disclaimers**:
- Prediction markets are risky; you can lose money
- Claude's predictions aren't guaranteed accurate
- Markets can remain irrational longer than expected
- API costs can exceed profits if edge is too small
- Platform risks (smart contracts, regulation)
- Start with small amounts
- This is for educational purposes

## Deliverables Checklist

When completing the skill, provide:

- [x] Working Python bot with all components
- [x] Configuration file (JSON)
- [x] Requirements.txt
- [x] README.md with setup guide
- [x] Example showing full workflow
- [x] Performance monitoring tools
- [x] Risk warnings
- [x] Cost analysis
- [x] Dry-run mode enabled by default

## Code Quality Standards

Ensure:
- Type hints on all functions
- Comprehensive error handling
- Async/await properly used
- Logging at appropriate levels
- Configuration validation
- API rate limiting
- Cost tracking accuracy
- Documentation strings

## Testing Recommendations

Suggest users:
1. Run in dry-run mode first (1 week minimum)
2. Start with small bankroll ($500-1000)
3. Monitor Claude's accuracy
4. Track API cost efficiency
5. Validate Kelly sizing is working
6. Check for correlation in positions
7. Scale gradually based on results

## Common Issues & Solutions

**"No opportunities found"**:
- Lower min_edge threshold
- Check volume filters
- Verify markets are being scanned

**"API costs too high"**:
- Reduce markets scanned
- Increase min_volume filter
- Use larger batch sizes

**"Position sizes too small"**:
- Check bankroll is sufficient
- Increase max_kelly_fraction
- Verify edge calculations

**"Authentication failed"**:
- Check API keys are correct
- Verify private key file exists
- Test connection separately

## Advanced Features (Optional)

If user requests:

**News Integration**:
```python
# Monitor RSS/Twitter
# Re-analyze on breaking news
# Fast-reaction trading
```

**Multi-Strategy**:
```python
# Combine value + momentum + arbitrage
# Weight by past performance
# Dynamic strategy allocation
```

**Machine Learning**:
```python
# Train on historical outcomes
# Improve Claude calibration
# Learn market-specific patterns
```

## Skill Output Format

When generating files, use this structure:

```
prediction-market-bot/
├── ai_trading_bot.py          # Main orchestrator
├── market_scanner.py           # Fetch markets
├── claude_analyzer.py          # AI analysis
├── kelly_calculator.py         # Position sizing
├── trade_executor.py           # Order execution
├── risk_manager.py             # Risk controls
├── config.json                 # Configuration
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
└── performance_analyzer.py     # Analytics (optional)
```

## Summary

This skill creates a complete AI-powered trading system that:
1. Scans 1000+ markets efficiently
2. Uses Claude to estimate fair values
3. Finds mispricings >8%
4. Sizes positions with Kelly criterion
5. Executes trades autonomously
6. Pays its own API bills
7. Tracks performance

The result is a self-sustaining trading agent that combines AI intelligence with mathematical rigor.
