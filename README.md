# 🤖 Kalshi & Polymarket Trading Bot

An easy-to-use, AI-powered trading bot for prediction markets. It scans markets, uses AI to figure out the "true" odds, and places smart bets automatically.

---

## 🌟 What This Bot Does

- **Scans Markets**: Automatically finds active bets on Kalshi and Polymarket.
- **AI Analysis**: Uses Claude or OpenAI to analyze news and data to find the real probability of an event happening.
- **Smart Sizing**: Uses the "Kelly Criterion" (a math formula) to decide exactly how much to bet based on the bot's confidence.
- **Autonomous**: Once started, it handles everything—tracking your money, managing open bets, and syncing with your exchange.

---

## 🚀 Beginner's Quick Start Guide

### 1. Install the Bot

First, make sure you have Python installed. Then, open your terminal and run:

```bash
pip install -r requirements.txt
```

### 2. Set Up Your API Keys

You need to tell the bot how to access your accounts and the AI.

1. Copy the example file: `cp .env.example .env`
2. Open the new `.env` file in a text editor.
3. Add your keys:
   - `KALSHI_API_KEY`: From your Kalshi settings.
   - `ANTHROPIC_API_KEY`: From Anthropic (for Claude AI).
   - `KALSHI_PRIVATE_KEY`: Needed for placing real bets.

### 3. Configure Your Settings

1. Copy the template: `cp advanced_config.template.json advanced_config.json`
2. Open `advanced_config.json`.
3. **Important for Beginners**: Make sure `"dry_run": true` is set. This lets the bot "pretend" to trade so you don't lose money while learning.
4. On startup, the bot now prompts you to either use defaults from `advanced_config.json` or enter interactive edit mode to update/validate key settings.

---

## 🛡️ Safety First: Testing Your Bot

Before using real money, run these commands to make sure everything is working:

### 📡 Test Data Collection (Free)

This checks if the bot can see the markets.

```bash
python -m app --mode collect --once
```

### 🧠 Test AI Analysis (Costs a few cents in AI tokens)

This runs a full cycle, analyzes a few markets, but **will not** place any real bets.

```bash
python -m app --mode trade --once --dry-run
```

---

## 💸 Going Live

Once you are confident the bot is making good choices in Dry Run mode, you can go live:

1. Edit `advanced_config.json`.
2. Change `"dry_run": false`.
3. Set your `"initial_bankroll": 100` (Start small!).
4. Run the bot:

```bash
python -m app --mode trade
```

#Todo

- Add https://textual.textualize.io/
- Make Application much more user friendly

---

## 📖 Common Commands

### Run Modes

| Command                                       | What it does                                            |
| --------------------------------------------- | ------------------------------------------------------- |
| `python -m app --mode collect --once`         | One collection cycle only (safe test).                  |
| `python -m app --mode collect`                | Continuous collection every hour.                       |
| `python -m app --mode trade --once --dry-run` | One dry-run trade cycle with analysis + no real orders. |
| `python -m app --mode trade --dry-run`        | Continuous dry-run trade cycles.                        |
| `python -m app --mode trade`                  | Live trade mode (real orders if config is live).        |
| `python -m app --mode trade --once --dry-run --skip-setup-wizard --non-interactive` | One autonomous-style dry-run cycle (no prompts). |

### Autonomous Live Run (VPS)

Required `advanced_config.json` keys:

- `trading.dry_run = false`
- `trading.autonomous_mode = true`
- `trading.non_interactive = true`
- `trading.require_scope_in_live = true`
- One scope must be set:
  - `platforms.kalshi.series_tickers`, or
  - `trading.allowed_market_ids`, or
  - `trading.allowed_event_tickers`
- Risk rails:
  - `risk.max_orders_per_cycle`
  - `risk.max_notional_per_cycle`
  - `risk.daily_loss_limit_fraction`
  - `risk.max_trades_per_market_per_day` (optional, `0` disables)
  - `risk.failure_streak_cooldown_threshold` + `risk.failure_cooldown_cycles` (optional, `0` disables)
- Execution rails:
  - `execution.max_price_drift`
  - `execution.min_edge_at_execution`
  - `execution.max_submit_slippage`
  - `execution.pending_not_found_retries`
  - `execution.pending_timeout_minutes`

Autonomous one-cycle run:

```bash
python -m app --mode trade --once --skip-setup-wizard --non-interactive
```

Continuous autonomous run:

```bash
python -m app --mode trade --skip-setup-wizard --non-interactive
```

Kill switch:

```bash
export BOT_DISABLE_TRADING=1
```

### Dry-Run UX Commands

| Command                                                           | What it does                                                                                        |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `python -m app --mode trade --once --dry-run --pick-markets`      | Lets you choose filtered markets interactively before analysis and prints a dry-run analysis table. |
| `python -m app --mode trade --once --dry-run --skip-setup-wizard` | Runs without the startup questionnaire (good for automation).                                       |

Note: In interactive dry-run `trade`/`analyze` runs, the CLI also asks after setup whether you want market picking for that run.
Note: Before scanning starts, interactive `trade`/`analyze` runs now offer a pre-scan scope menu. For discovery-based scopes (`Sports->NBA` or `category+keyword`), you get an indexed ticker-to-title list, can choose up to 10 IDs, and then confirm before scan starts.

### Config + Validation

| Command                                                                                           | What it does                                 |
| ------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| `python -m app --show-config`                                                                     | Prints effective non-secret config values.   |
| `python -m app --verify-config --mode trade`                                                      | Validates config for trade mode.             |
| `python -m app --set-max-positions 3 --set-max-position-size 100 --set-dry-run true`              | Updates core risk/trading settings.          |
| `python -m app --set-max-total-exposure-fraction 0.8 --set-max-new-exposure-per-day-fraction 0.8` | Updates total/day exposure caps.             |
| `python -m app --set-config risk.max_kelly_fraction=0.1`                                          | Updates any config key via `dot.path=value`. |

### Utility

| Command                           | What it does                        |
| --------------------------------- | ----------------------------------- |
| `python -m app --discover-series` | Lists Kalshi series you can target. |
| `python -m app --backup`          | Creates a DB backup.                |
| `python scripts/kalshi_user_details.py` | Prints current Kalshi account details, cash balance, open positions, and recent orders. |

Live safety note:
- In live mode, bot startup now verifies Kalshi available cash is at least `trading.initial_bankroll`. If not, startup fails fast.
- You can disable this only if needed with `trading.enforce_live_cash_check=false` (not recommended).

---

## ❓ Frequently Asked Questions

**"Does this cost money?"**

- Running the bot in `collect` mode is free.
- Using the AI (Claude/OpenAI) costs a small amount per scan (usually a few cents).
- Real trading, of course, uses your Kalshi/Polymarket balance.

**"The bot stopped trading. Why?"**

- Check your bankroll. The bot will automatically stop if your balance hits $0 to protect you.
- Check if you already have an open position in that market. The bot will not open duplicate positions in the same market.
- Check the logs. If your API keys expire, it will stop and tell you why.

---

## 🛠️ Need Technical Details?

Check out [TECHNICAL_DOCS.md](./TECHNICAL_DOCS.md) for a deep dive into how the bot works, the database structure, and the safety systems.
