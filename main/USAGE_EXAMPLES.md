# Example Usage Patterns

This guide shows common ways to use the prediction market trading bot skill.

## Example 1: Basic Bot Request

**User**: "Help me build a trading bot for Polymarket"

**Response Flow**:
1. Ask clarifying questions (bankroll, risk tolerance, automation level)
2. Generate basic bot with default settings
3. Include Polymarket integration only
4. Set dry_run: true by default
5. Provide simple README

**Files Generated**:
- `ai_trading_bot.py` (simplified version)
- `config.json` (Polymarket only)
- `requirements.txt`
- `README.md`

---

## Example 2: Full-Featured AI Bot

**User**: "I want an AI-powered trading bot that uses Claude to analyze prediction markets on both Polymarket and Kalshi. It should find mispricings above 8%, use Kelly criterion for position sizing, and track its own API costs. Budget is $10,000."

**Response Flow**:
1. Generate complete system with all components
2. Integrate Claude API for analysis
3. Implement Kelly criterion sizing
4. Add API cost tracking
5. Support both platforms
6. Set initial_bankroll: 10000
7. Include comprehensive analytics

**Files Generated**:
- `ai_trading_bot.py` (full version)
- `market_scanner.py`
- `claude_analyzer.py`
- `kelly_calculator.py`
- `trade_executor.py`
- `risk_manager.py`
- `config.json` (both platforms)
- `performance_analyzer.py`
- `requirements.txt`
- `README.md` (comprehensive)

---

## Example 3: Conservative Strategy

**User**: "Create a conservative trading bot for prediction markets. I want minimal risk - only trade when you're very confident and the edge is large."

**Response Flow**:
1. Use `examples/config_conservative.json` as template
2. Set high thresholds (min_edge: 0.15, min_confidence: 0.80)
3. Use small Kelly fraction (0.10)
4. Limit positions (max 5)
5. Document conservative approach
6. Explain trade-offs

**Key Config Settings**:
```json
{
  "strategy": {
    "min_edge": 0.15,
    "min_confidence": 0.80
  },
  "risk": {
    "max_kelly_fraction": 0.10,
    "max_positions": 5,
    "max_position_size": 500
  }
}
```

---

## Example 4: Aggressive Strategy

**User**: "Help me build an aggressive trading bot. I want to find lots of opportunities and size positions optimally for maximum growth."

**Response Flow**:
1. Use `examples/config_aggressive.json` as template
2. Lower thresholds (min_edge: 0.05, min_confidence: 0.50)
3. Higher Kelly fraction (0.50)
4. More positions (max 20)
5. Include clear risk warnings
6. Document aggressive approach

**Key Config Settings**:
```json
{
  "strategy": {
    "min_edge": 0.05,
    "min_confidence": 0.50
  },
  "risk": {
    "max_kelly_fraction": 0.50,
    "max_positions": 20,
    "max_position_size": 2000
  }
}
```

---

## Example 5: Arbitrage Focus

**User**: "Build a trading bot that can do arbitrage between Polymarket and Kalshi when the same event is priced differently."

**Response Flow**:
1. Enable both platforms
2. Add market matching logic
3. Implement price comparison
4. Account for transaction fees
5. Add simultaneous execution
6. Document arbitrage strategy

**Key Features**:
```python
# Match markets across platforms
matched = find_matching_markets(poly_markets, kalshi_markets)

# Calculate arbitrage opportunity
for pair in matched:
    spread = abs(pair.poly_price - pair.kalshi_price)
    net_spread = spread - total_fees
    
    if net_spread > min_arbitrage_spread:
        # Buy low, sell high
        execute_arbitrage(pair)
```

---

## Example 6: Analytics Only

**User**: "I need performance analytics for my trading bot. Show me win rates, profit curves, Sharpe ratio, and how accurate Claude's predictions are."

**Response Flow**:
1. Focus on `performance_analyzer.py`
2. Add backtesting framework
3. Include visualization tools
4. Claude calibration analysis
5. Skip trade execution components

**Key Metrics**:
```python
metrics = {
    'win_rate': wins / total,
    'sharpe_ratio': calculate_sharpe(returns),
    'max_drawdown': calculate_drawdown(equity),
    'claude_calibration': analyze_predictions(results),
    'edge_capture_rate': actual_edge / theoretical_edge
}
```

---

## Example 7: Cost-Focused

**User**: "Create a trading bot but make sure it stays profitable after paying for Claude API costs."

**Response Flow**:
1. Implement detailed cost tracking
2. Calculate break-even analysis
3. Show cost per market
4. Deduct from bankroll
5. Include profitability warnings

**Cost Analysis**:
```
Markets analyzed: 1,000
Cost per market: $0.0105
Total API cost: $10.50

Expected profit: $400
Net profit: $389.50
ROI on API: 37x ✓

Break-even edge: 2.6%
Your edge: 8%
Safety margin: 3.1x
```

---

## Example 8: Platform-Specific

**User**: "Build a trading bot for Kalshi only. Focus on political markets and use medium risk."

**Response Flow**:
1. Enable only Kalshi
2. Filter for political category
3. Use moderate parameters
4. Kalshi-specific documentation
5. Include API setup for Kalshi

**Key Config**:
```json
{
  "platforms": {
    "polymarket": {"enabled": false},
    "kalshi": {"enabled": true}
  },
  "filters": {
    "categories": ["politics"]
  },
  "risk": {
    "max_kelly_fraction": 0.20
  }
}
```

---

## Example 9: Backtesting

**User**: "I want to backtest a trading strategy on historical prediction market data."

**Response Flow**:
1. Create backtesting framework
2. Add historical data loader
3. Simulate trades with strategy
4. Calculate performance metrics
5. Compare vs buy-and-hold

**Backtest Structure**:
```python
def backtest_strategy(historical_markets, strategy):
    bankroll = initial_bankroll
    trades = []
    
    for market in historical_markets:
        signal = strategy.analyze(market)
        if signal:
            result = simulate_trade(signal, market.outcome)
            bankroll += result.profit
            trades.append(result)
    
    return analyze_performance(trades, bankroll)
```

---

## Example 10: Fully Autonomous

**User**: "Create a fully autonomous trading bot that runs continuously, analyzes markets every hour, and pays its own API bills."

**Response Flow**:
1. Add scheduling logic (run every hour)
2. Implement continuous mode
3. Autonomous execution (no manual approval)
4. Automatic cost payment
5. Add monitoring/logging
6. Include start/stop scripts

**Continuous Mode**:
```python
async def run_autonomous():
    while True:
        try:
            await run_trading_cycle()
            logger.info("Cycle complete. Sleeping 1 hour...")
            await asyncio.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            await asyncio.sleep(60)
```

---

## Common Modifications

### Add News Monitoring
```python
# Monitor RSS feeds
# Re-analyze on breaking news
# Fast reaction trading
```

### Multi-Strategy
```python
# Combine value + momentum + arbitrage
# Weight by past performance
# Dynamic allocation
```

### Position Management
```python
# Automatic rebalancing
# Trailing stops
# Take profit levels
```

### Enhanced Analytics
```python
# Correlation analysis
# Drawdown alerts
# Performance attribution
```

---

## Template Selection Guide

Choose template based on request:

| User Request | Template | Key Settings |
|-------------|----------|--------------|
| "Simple bot" | Basic | Single platform, dry run |
| "AI-powered" | Full | Claude analysis, both platforms |
| "Conservative" | Conservative | High thresholds, small Kelly |
| "Aggressive" | Aggressive | Low thresholds, large Kelly |
| "Arbitrage" | Arbitrage | Both platforms, matching |
| "Analytics" | Analytics | No execution, just analysis |
| "Profitable" | Cost-focused | Track costs, profitability |

---

## Customization Tips

1. **Always ask clarifying questions first**
2. **Use appropriate template as starting point**
3. **Customize based on specific needs**
4. **Include relevant examples in README**
5. **Test configuration before delivering**
6. **Provide clear next steps**

---

## Success Criteria

Bot is ready when it has:
- [x] All required files
- [x] Valid configuration
- [x] Clear documentation
- [x] Working dry-run mode
- [x] Risk warnings
- [x] Cost analysis
- [x] Setup instructions
- [x] Example usage
