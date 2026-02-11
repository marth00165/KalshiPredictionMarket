# Kalshi Market Predictor

An AI-powered trading bot that automatically analyzes prediction markets on Polymarket and Kalshi, calculates fair value estimates using Claude AI, and executes trades with intelligent position sizing.

## Quick Start (Beginners)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Configuration

Copy and edit the configuration file with your API keys:

```bash
cp advanced_config.json my_config.json
```

Edit `my_config.json` and add:

- `polymarket.api_key` - Your Polymarket API token
- `kalshi.api_key` - Your Kalshi API key
- `claude.api_key` - Your Claude API key from Anthropic
- `trading.initial_bankroll` - Starting capital (e.g., 1000)

### 3. Run the Bot

**Test run (dry-run mode, no real trades):**

```bash
python main/ai_trading_bot_refactored.py
# Default: uses advanced_config.json with dry_run: true
```

In **dry-run mode**:
- ✅ Scans all markets from Polymarket and Kalshi
- ✅ Analyzes with Claude AI (costs money for API calls)
- ✅ Generates trade signals with position sizing
- ✅ **Blocks execution** - logs signals instead of placing bets
- ✅ **Zero trading losses**

**Live trading (real money):**

First, edit `advanced_config.json` and change the dry_run flag:

```json
{
  "trading": {
    "dry_run": false,    ← Change from true to false
    "initial_bankroll": 1000
  }
}
```

Then run:

```bash
python main/ai_trading_bot_refactored.py
```

⚠️ **WARNING**: `dry_run: false` places **REAL bets** with **REAL money**!

## How It Works

1. **Scans** - Bot checks Polymarket and Kalshi for active markets
2. **Analyzes** - Claude AI estimates fair value for each market
3. **Identifies** - Finds opportunities where market price differs from fair value
4. **Sizes** - Kelly Criterion calculates optimal position size
5. **Executes** - Places trades on profitable opportunities
6. **Reports** - Tracks performance and error logs

## Important Commands

| Command                                                            | Purpose                         |
| ------------------------------------------------------------------ | ------------------------------- |
| `python main/ai_trading_bot_refactored.py`                         | Run bot with default config     |
| `python main/ai_trading_bot_refactored.py --dry-run`               | Test trades without executing   |
| `python main/ai_trading_bot_refactored.py --config my_config.json` | Run with custom config          |
| `python main/ai_trading_bot.py`                                    | Run original monolithic version |

## Configuration Basics

Key settings in `advanced_config.json`:

```json
{
  "trading": {
    "dry_run": true,              ← Set to false for live trading
    "initial_bankroll": 1000,
    "min_edge_percentage": 8.0
  },
  "strategy": {
    "min_volume_usd": 500,
    "min_liquidity_usd": 200
  },
  "polymarket": {
    "api_base_url": "https://clob.polymarket.com"
  },
  "kalshi": {
    "api_base_url": "https://api.kalshi.com/v2"
  }
}
```

### Dry-Run vs Live Mode

| Phase | Dry-Run | Live |
|-------|---------|------|
| Market Scanning | ✅ Real data | ✅ Real data |
| Claude Analysis | ✅ Real calls | ✅ Real calls |
| Signal Generation | ✅ Normal | ✅ Normal |
| **Trade Execution** | **❌ Blocked** | **✅ Places bets** |
| Money Risk | None | **Real money!** |

**Safe Workflow**: Always test with `dry_run: true` first, then change one flag to go live.

## File Structure

- `main/` - Main bot code
  - `ai_trading_bot_refactored.py` - **Start here** (new modular version)
  - `ai_trading_bot.py` - Original version
  - `api_clients/` - API integration (Polymarket, Kalshi)
  - `models/` - Data models
  - `trading/` - Strategy, position management, execution
  - `utils/` - Configuration, error handling, parsing

- `advanced_config.json` - Main configuration file
- `REFACTORING_COMPLETE.md` - Technical refactoring details

## Troubleshooting

**"Module not found" error:**

```bash
# Make sure you're in the right directory
cd /Users/rohitpratti/repos/KalshiMarketPredictor

# Re-install dependencies
pip install --upgrade -r requirements.txt
```

**API authentication fails:**

- Check API keys in config are correct
- Ensure they have proper permissions on their respective platforms
- Check `error_logs.json` for detailed error messages

**"Insufficient bankroll" error:**

- Your `initial_bankroll` is too low for current opportunities
- Lower `min_volume_usd` and `min_liquidity_usd` thresholds, or increase bankroll

## Next Steps

1. **Test Mode First** - Always run with `--dry-run` before live trading
2. **Start Small** - Set low initial bankroll to test strategy
3. **Monitor Logs** - Check `error_logs.json` for issues
4. **Read Docs** - See `main/QUICK_REFERENCE.md` for advanced features

## Need Help?

- Check [QUICK_REFERENCE.md](main/QUICK_REFERENCE.md) for API reference
- Read [REFACTORING_COMPLETE.md](REFACTORING_COMPLETE.md) for architecture overview
- Review [USAGE_EXAMPLES.md](main/USAGE_EXAMPLES.md) for code examples
