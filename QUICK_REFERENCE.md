# Prediction Market Trader - Quick Reference

## Skill Trigger Phrases

Use this skill when user says:
- "build a trading bot"
- "prediction market"
- "Polymarket" or "Kalshi"
- "automate trading"
- "Kelly criterion"
- "AI-powered trading"
- "analyze markets with Claude"

## Core Components to Generate

Always create these files:

### 1. Main Bot (`ai_trading_bot.py`)
- MarketScanner class
- ClaudeAnalyzer class  
- KellyCriterion class
- RiskManager class
- Main trading loop
- Async/await structure

### 2. Configuration (`config.json`)
```json
{
  "trading": {"initial_bankroll": 10000},
  "platforms": {"polymarket": {...}, "kalshi": {...}},
  "strategy": {"min_edge": 0.08, "min_confidence": 0.6},
  "risk": {"max_kelly_fraction": 0.25, "max_positions": 10},
  "api": {"batch_size": 50, "api_cost_limit_per_cycle": 5.00}
}
```

### 3. Documentation (`README.md`)
- Installation steps
- API key setup
- Configuration guide
- Usage examples
- Risk warnings
- Cost analysis

### 4. Dependencies (`requirements.txt`)
```
requests>=2.31.0
aiohttp>=3.9.0
kalshi-python>=1.0.0
py-clob-client>=0.25.0
anthropic>=0.18.0
```

## Key Implementation Points

### Claude Analysis Prompt
```python
prompt = f"""
Analyze this prediction market and estimate TRUE probability.

Market: {title}
Description: {description}
Current Price: {yes_price}%

Respond in JSON:
{{
  "probability": <0-100>,
  "confidence": <0-100>,
  "reasoning": "<explanation>"
}}
"""
```

### Kelly Criterion Formula
```python
kelly = edge / (1 - market_price)
kelly_capped = min(kelly, max_fraction)  # Default 25%
position_size = kelly_capped * bankroll
```

### API Cost Tracking
```python
input_cost = (input_tokens / 1_000_000) * 3.00
output_cost = (output_tokens / 1_000_000) * 15.00
total_cost += input_cost + output_cost
bankroll -= total_cost  # Pay from profits
```

## User Request Patterns

### "Conservative"
```json
{
  "strategy": {"min_edge": 0.15, "min_confidence": 0.80},
  "risk": {"max_kelly_fraction": 0.10, "max_positions": 5}
}
```

### "Aggressive"  
```json
{
  "strategy": {"min_edge": 0.05, "min_confidence": 0.50},
  "risk": {"max_kelly_fraction": 0.50, "max_positions": 20}
}
```

### "Arbitrage"
- Scan both platforms
- Compare same events
- Account for fees
- Execute simultaneously

### "Analytics Only"
- Focus on performance_analyzer.py
- Skip trade execution
- Backtest historical data

## Safety Checklist

Always include:
- [x] Dry run mode (default: true)
- [x] Position limits
- [x] Confidence thresholds
- [x] Volume filters
- [x] API cost tracking
- [x] Error handling
- [x] Risk warnings in docs

## Common Issues

**"No opportunities"**: Lower min_edge or min_confidence
**"API costs high"**: Reduce markets scanned or increase batch_size
**"Positions small"**: Check bankroll, increase max_kelly_fraction
**"Auth failed"**: Verify API keys in config

## Performance Metrics

Always track:
```python
{
  'win_rate': wins / total_trades,
  'sharpe_ratio': excess_return / std_dev,
  'max_drawdown': worst_peak_to_trough,
  'api_cost': total_api_spending,
  'roi_on_api': profit / api_cost,
  'edge_captured': actual_edge / theoretical_edge
}
```

## File Structure Output

```
trading-bot/
├── ai_trading_bot.py
├── config.json
├── requirements.txt
├── README.md
└── performance_analyzer.py  # Optional
```

## Platform-Specific Notes

**Polymarket**:
- Blockchain-based (Polygon)
- Requires USDC
- Library: py-clob-client
- Order signing needed

**Kalshi**:
- Regulated exchange
- API key + private key
- Library: kalshi-python
- REST API

## Cost Analysis Template

Include in docs:
```
Markets analyzed: 1,000
API cost per market: $0.0105
Total API cost: $10.50

Trades found: 10
Average position: $500
Average edge: 8%
Expected profit: $400

Net profit: $389.50
ROI on API: 37x ✓
```

## Testing Workflow

1. Generate all files
2. Set dry_run: true
3. Run single cycle
4. Verify outputs created
5. Check cost tracking
6. Validate Kelly math
7. Review documentation

## Documentation Sections

Every README must have:
1. Overview
2. Installation
3. Configuration
4. Usage (dry run first)
5. Cost analysis
6. Risk warnings
7. Troubleshooting
8. Examples

## Templates Location

Use files from `/templates/`:
- ai_trading_bot.py
- config_template.json
- performance_analyzer.py
- requirements.txt
- README_template.md

Customize based on user requirements!
