# Kalshi NBA Elo Trading Bot (V1)

This repo is now tuned for an NBA-first workflow:

- Elo is the primary probability engine.
- LLM does not set probabilities directly.
- LLM can only provide bounded Elo adjustments for context like injuries/rest/lineups.
- Final trade probability always comes from Elo math.

## Core Pipeline

`kaggleGameData.csv` -> Elo ratings -> base probability -> LLM Elo adjustment -> adjusted Elo -> final probability -> edge/risk checks -> execution

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
```

Set at least:

- `KALSHI_API_KEY`
- `KALSHI_PRIVATE_KEY` or `KALSHI_PRIVATE_KEY_FILE` (live mode)
- `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`

### 3. Config file

```bash
cp advanced_config.template.json advanced_config.json
```

Recommended for NBA dry runs:

- `analysis.provider = "openai"` (or `"claude"`)
- `analysis.nba_elo_enabled = true`
- `analysis.nba_elo_data_path = "context/kaggleGameData.csv"`
- `analysis.context_json_path = "context/llm_context.json"`
- `trading.dry_run = true`
- `platforms.kalshi.enabled = true`
- `platforms.polymarket.enabled = false`

## Quick Start: NBA Dry Run (One Cycle)

### 1. Scope bot to NBA game winner markets

```bash
python -m app --set-allowed-series-tickers KXNBAGAME
```

### 2. (Optional) lower filters for testing

```bash
python -m app --set-config filters.min_volume=0 --set-config filters.min_liquidity=0
```

### 3. Run one non-interactive dry-run cycle

```bash
python -m app --mode trade --once --dry-run --skip-setup-wizard --non-interactive
```

For one-cycle dry runs, the bot writes JSON output to `reports/dry_run_analysis/` and does not print the full table.

## Interactive Dry Run (Optional)

```bash
python -m app --mode trade --once --dry-run
```

This mode supports:

- setup wizard edits
- market picker
- pre-scan NBA scope discovery and manual series selection

## Where Outputs Go

- Dry-run analysis JSON: `reports/dry_run_analysis/`
- Daily trade journal (dry + live): `reports/trade_journal/YYYY-MM-DD.json`
- Heartbeat: `reports/heartbeat.json`
- SQLite DBs:
  - Dry-run: `kalshi_dryrun.sqlite`
  - Live: `kalshi.sqlite`

## V1 Safeguards Implemented

- Duplicate market/event guards (no overlapping same-event positions in cycle).
- Buy-NO Kelly sizing fix (uses NO-side probability).
- Execution-time revalidation:
  - price drift
  - minimum edge at execution
  - submit slippage
- Risk guards:
  - per-cycle order/notional caps
  - daily loss guard
  - optional kill switch via env var
  - market/day frequency cap
- Execution-path exposure enforcement:
  - `risk.max_new_exposure_per_day_fraction`
  - `risk.max_total_exposure_fraction`
- Structured pre-trade logs:
  - `TRADE_DECISION`
  - `MODEL_DIVERGENCE_WARNING` (logging only, no block)
- Live bankroll startup check:
  - verifies Kalshi cash >= `trading.initial_bankroll` before live run

## Useful Commands

### Config and validation

```bash
python -m app --show-config
python -m app --verify-config --mode trade
python -m app --set-config analysis.provider=openai
python -m app --set-config trading.dry_run=true
```

### Scope controls

```bash
python -m app --set-allowed-series-tickers KXNBAGAME
python -m app --set-allowed-market-ids KXNBAGAME-26FEB19DETNYK-DET
python -m app --set-allowed-event-tickers KXNBAGAME-26FEB19DETNYK
```

### Trading runs

```bash
python -m app --mode trade --once --dry-run
python -m app --mode trade --once --dry-run --skip-setup-wizard --non-interactive
python -m app --mode trade --dry-run
python -m app --mode trade
```

### Utility

```bash
python -m app --discover-series --category Sports
python -m app --backup
python scripts/kalshi_user_details.py
```

## Context for LLM

Add custom context in `context/llm_context.json` (or your configured `analysis.context_json_path`).  
This context is loaded into analysis prompts as supplemental information.

## Live/VPS Notes (DigitalOcean-ready)

Before enabling live:

1. Keep strict scope (`KXNBAGAME` or narrower).
2. Start with small bankroll and conservative risk caps.
3. Run at least one full dry-run cycle on VPS using your production command.
4. Confirm journals/logs/heartbeat update correctly.

Kill switch:

```bash
export BOT_DISABLE_TRADING=1
```

Run non-interactive autonomous style:

```bash
python -m app --mode trade --skip-setup-wizard --non-interactive
```

## Troubleshooting

- `ModuleNotFoundError: app`: run from repo root and use the venv.
- `.env parse warning`: fix malformed `.env` line format (`KEY=value`).
- No markets found: verify scope and filters.
- Live startup fails bankroll check: fund account or lower `trading.initial_bankroll`.

## Technical Docs

- `TECHNICAL_DOCS.md`
- `OPERATIONS.md`
- `LIVE_AUTONOMOUS_V1_GAP_LIST.md`
