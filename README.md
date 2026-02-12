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

| Command                                                            | Purpose                               |
| ------------------------------------------------------------------ | ------------------------------------- |
| `python main/ai_trading_bot_refactored.py`                         | Run bot with default config           |
| `python main/ai_trading_bot_refactored.py --dry-run`               | Test trades without executing         |
| `python main/ai_trading_bot_refactored.py --config my_config.json` | Run with custom config                |
| `python main/series_scanner.py --discover`                         | Discover available Kalshi series      |
| `python main/series_scanner.py --series KXFED`                     | Scan specific series (no rate limits) |
| `python main/ai_trading_bot.py`                                    | Run original monolithic version       |

## Series Scanner (Recommended for Dry Runs)

The series scanner is a **lightweight, rate-limit-friendly** way to fetch and analyze Kalshi markets.

Instead of fetching all 500+ markets and making individual orderbook calls (which hits rate limits), it:

- Fetches markets by `series_ticker` — targeted, minimal API calls
- Uses prices from the market list response — no orderbook calls needed
- Filters by category (Economics, Politics, etc.)

### Discover Available Series

See all series and their categories:

```bash
python main/series_scanner.py --discover
```

Filter by category:

```bash
python main/series_scanner.py --discover --category Economics
python main/series_scanner.py --discover --category Politics
```

### Scan Specific Series

Once you know the series tickers, scan them directly:

```bash
# Scan Fed rate decision markets
python main/series_scanner.py --series KXFED

# Scan multiple series
python main/series_scanner.py --series KXFED KXCPI KXNFP

# Scan and run AI analysis (costs money)
python main/series_scanner.py --series KXFED --analyze --max-analyze 5
```

### Series Scanner Options

| Flag                 | Description                                      |
| -------------------- | ------------------------------------------------ |
| `--discover`         | List all available series                        |
| `--category <name>`  | Filter series by category                        |
| `--series <tickers>` | Series tickers to scan (space-separated)         |
| `--analyze`          | Run AI analysis on filtered markets              |
| `--max-analyze <n>`  | Limit number of markets to analyze (default: 10) |
| `--no-save`          | Don't save report to file                        |

Reports are saved to `reports/series_scan_<timestamp>.json`.

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

| Phase               | Dry-Run        | Live               |
| ------------------- | -------------- | ------------------ |
| Market Scanning     | ✅ Real data   | ✅ Real data       |
| Claude Analysis     | ✅ Real calls  | ✅ Real calls      |
| Signal Generation   | ✅ Normal      | ✅ Normal          |
| **Trade Execution** | **❌ Blocked** | **✅ Places bets** |
| Money Risk          | None           | **Real money!**    |

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

## Viewing Reports (Market Cycle Analyzer)

After running scans, JSON reports are saved to the `reports/` folder. To visualize and analyze these reports, use the **Market Cycle Analyzer** web tool:

👉 **[https://elcurryapps.org/marketCycleAnalyzer/](https://elcurryapps.org/marketCycleAnalyzer/)**

### How to Use

1. Navigate to the link above
2. Either:
   - **Paste JSON** directly into the text area and click "Parse JSON"
   - **Upload a JSON file** by clicking "📁 Upload JSON File"
3. View the parsed data in the dashboard:
   - **Cycle Info** - Cycle number, timestamps, provider, dry run status
   - **Counts** - Scanned, passed filters, analyzed, opportunities, signals, executed
   - **API Cost** - Total cost, requests, tokens used
   - **Configuration** - Min volume, liquidity, edge, confidence, position size
   - **Markets Table** - All markets with prices, stats, filter results, and failure reasons
4. Use the checkbox to filter and show only markets that passed filters
5. Click "📸 Download Screenshot" to save the report as a PNG image

### Expected JSON Format

The analyzer expects the JSON format produced by both `ai_trading_bot_refactored.py` and `series_scanner.py`:

```json
{
  "cycle": 1,
  "started_at": "2026-02-11T03:42:58.486507",
  "finished_at": "2026-02-11T03:43:00.601340",
  "config": { ... },
  "counts": { "scanned": 25, "passed_filters": 0, ... },
  "api_cost": { "total_cost": 0, ... },
  "markets": [ ... ],
  "signals": [],
  "errors": []
}
```

## Need Help?

- Check [QUICK_REFERENCE.md](main/QUICK_REFERENCE.md) for API reference
- Read [REFACTORING_COMPLETE.md](REFACTORING_COMPLETE.md) for architecture overview
- Review [USAGE_EXAMPLES.md](main/USAGE_EXAMPLES.md) for code examples
