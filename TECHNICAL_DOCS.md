# 🛠️ Technical Documentation

This document provides a deep dive into the architecture, state management, and execution logic of the Kalshi/Polymarket Trading Bot.

## 🏗️ Architecture Overview

The bot is structured as a modular Python package located in the `app/` directory.

- **`AdvancedTradingBot` (`app/bot.py`)**: The main orchestrator. It manages the lifecycle of a trading cycle: scan -> filter -> analyze -> execute -> reconcile.
- **`DatabaseManager` (`app/storage/db.py`)**: Handles SQLite persistence using `aiosqlite`. Uses a versioned migration system.
- **`BankrollManager` (`app/trading/bankroll_manager.py`)**: Manages the persistent bankroll, including history of all adjustments (trades, API costs, settlements).
- **`PositionManager` (`app/trading/position_manager.py`)**: Tracks open and historical positions. Reloads state and cumulative stats from DB on startup.
- **`ReconciliationManager` (`app/trading/reconciliation.py`)**: Synchronizes local state with the remote platform (Kalshi) to handle discrepancies.

## 🗄️ Database Schema

The bot uses SQLite with WAL (Write-Ahead Logging) enabled for concurrency.

### Tables:
- `market_snapshots`: Raw JSON data of markets at specific hours.
- `bankroll_history`: Audit log of every dollar moved (timestamp, change, reason, current balance).
- `positions`: State of all trades (market_id, side, entry_price, quantity, status, external_order_id, PnL).
- `executions`: Record of every order attempt and its outcome.
- `pnl_history`: Summary of profit/loss events.
- `status`: Heartbeat and bot metadata (last success, last error).

## 💰 Bankroll & Risk Management

### Persistence
The bankroll is never kept purely in memory. Every adjustment is written to `bankroll_history`. On startup, the bot loads the latest balance. If the table is empty, it initializes with the `initial_bankroll` from the config.

### Safety Guards
- **Duplicate-Market Guard**: The bot will never open a new position in a market if it already has an open position in that same market. This check is performed at both the signal generation stage and the execution stage.
- **Zero-Bankroll Lock**: If the bankroll hits $0 or less, the bot disables the `TradeExecutor`. It will continue to collect data and perform AI analysis, but it will not place orders.
- **API Cost Deduction**: After every cycle, the bot calculates the token cost from the AI provider (Claude/OpenAI) and deducts it from the bankroll to ensure the balance reflects net worth accurately.

## 🔄 Position Management & Reconciliation

### Lifecycle of a Position
1. **Signal**: Strategy generates a `TradeSignal` with Kelly sizing.
2. **Execution**: `TradeExecutor` places the order and captures the `external_order_id`.
3. **Storage**: `PositionManager` saves the new position to the `positions` table with status `open`.
4. **Settlement**: When a position is closed (detected via reconciliation or settlement check), PnL is calculated, the bankroll is replenished (cost + profit/loss), and the status is updated to `closed`.

### Reconciliation System
To handle bot crashes or manual trades, the `ReconciliationManager` runs on startup:
- **Unknown Remote Positions**: If Kalshi has a position the bot doesn't know about, it's added to the local DB with status `recovered`.
- **Missing Local Positions**: If a local position is gone from Kalshi, the bot assumes it was settled/closed and marks it as such in the DB, replenishing the bankroll accordingly.

## 🔌 API Integration

- **Rate Limiting**: The `KalshiClient` uses an asynchronous `RateLimiter` to stay within platform limits (default 8 req/sec).
- **Retries**: The `BaseAPIClient` implements exponential backoff for 429 (Rate Limit) and 5xx (Server Error) responses.
- **Timeouts**: Global request timeouts are enforced to prevent the bot from hanging on stalled connections.

##  UTC Policy
All timestamps are stored in **ISO8601 format with the `Z` suffix** (e.g., `2023-10-27T10:00:00.000Z`) to ensure absolute consistency across timezones.
