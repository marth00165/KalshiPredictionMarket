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

---

## 📖 Common Commands

| Command | What it does |
|---------|--------------|
| `python -m app --mode collect` | Just watches markets and saves data. |
| `python -m app --mode trade --dry-run` | Runs full AI analysis without betting real money. |
| `python -m app --discover-series` | Lists all types of events (like Economics, Politics) you can track. |
| `python -m app --backup` | Saves a backup of your trading data. |
| `python -m app --show-config` | Prints active non-secret configuration values. |
| `python -m app --verify-config --mode trade` | Checks that required keys/settings are valid for trade mode. |
| `python -m app --set-max-positions 3 --set-max-position-size 100 --set-dry-run true` | Updates risk settings in your config file. |
| `python -m app --set-max-total-exposure-fraction 0.8 --set-max-new-exposure-per-day-fraction 0.8` | Sets lenient portfolio/day exposure caps. |
| `python -m app --set-config risk.max_kelly_fraction=0.1` | Updates any config key via `dot.path=value`. |
| `python -m app --skip-setup-wizard --mode trade --once` | Skips interactive setup (useful for cron/systemd/autonomous runs). |

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
