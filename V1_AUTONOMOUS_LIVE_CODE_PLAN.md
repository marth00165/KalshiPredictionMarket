# V1 Autonomous Live Code Plan (7-Day VPS Run, Real Money)

This document is the **code-change plan** required to run the bot autonomously for one full week with real money on a VPS, constrained to a specific market scope.

## Success Criteria

1. Bot runs non-interactively for 7 days on VPS.
2. Bot trades only explicitly allowed market scope.
3. Duplicate/overlapping exposure protections hold in live mode.
4. Risk rails prevent runaway losses/exposure.
5. Restarts/crashes do not create duplicate orders.

---

## Phase 1: Enforce Live Market Scope (Blocker)

### Why

Current flow can still scan broadly unless user config/prompt is set correctly. Live autonomous runs need hard scope enforcement in code.

### Code Changes

1. Add explicit scope config fields:
- `trading.allowed_market_ids: []`
- `trading.allowed_event_tickers: []`
- `trading.allowed_series_tickers: []`
- `trading.require_scope_in_live: true`

2. Add validation rules in `app/config.py`:
- In live mode (`dry_run=false`), fail startup if `require_scope_in_live=true` and no allowlist is provided.
- Fail if both platforms enabled and scope is ambiguous.

3. Add scope filtering helper in `app/bot.py`:
- Method: `_is_market_in_allowed_scope(market) -> bool`
- Use `market.market_id`, `market.event_ticker`, and series prefix fallback.

4. Apply scope guard in two places (defense-in-depth):
- Pre-analysis opportunity list
- Pre-execution signal list

5. Add CLI support in `app/__main__.py`:
- `--set-config trading.allowed_series_tickers=...` already possible, but add dedicated flags for usability:
  - `--set-allowed-market-ids`
  - `--set-allowed-event-tickers`
  - `--set-allowed-series-tickers`

### Tests

- `tests/test_live_scope_guard.py`
  - live mode fails without scope when required
  - out-of-scope opportunities are skipped
  - out-of-scope signals are skipped even if generated

---

## Phase 2: Live Safety Rails (Blocker)

### Why

Exposure caps exist, but week-long unattended live requires hard stop controls beyond allocation sizing.

### Code Changes

1. Extend risk config in `app/config.py`:
- `risk.max_orders_per_cycle`
- `risk.max_notional_per_cycle`
- `risk.daily_loss_limit_fraction`
- `risk.kill_switch_env_var` (default: `BOT_DISABLE_TRADING`)

2. Enforce in `app/bot.py` before execution:
- If env kill switch set, skip all live executions.
- Cap number of executable signals per cycle.
- Cap cumulative notional sent per cycle.
- Stop opening new positions when daily loss threshold breached.

3. Add concise logs/reasons:
- `kill_switch_guard`
- `daily_loss_guard`
- `max_orders_per_cycle_guard`
- `max_notional_per_cycle_guard`

### Tests

- `tests/test_live_risk_guards.py`
  - kill switch blocks execution
  - daily loss halt blocks new orders
  - per-cycle caps truncate execution list deterministically

---

## Phase 3: Execution-Time Revalidation (Blocker)

### Why

Signal may be valid at analysis time but stale at order time. Live orders need a final price/edge recheck.

### Code Changes

1. Add execution guard config in `app/config.py`:
- `execution.max_price_drift` (absolute probability or bps)
- `execution.min_edge_at_execution`

2. In `app/trading/executor.py` for live Kalshi orders:
- Fetch latest market price right before `place_order`.
- Recompute side-consistent edge:
  - `buy_yes`: `fair_value - live_yes_price`
  - `buy_no`: `(1 - fair_value) - live_no_price`
- Skip order if price drift too large or edge below minimum.

3. Record explicit skip reason in execution record.

### Tests

- `tests/test_execution_revalidation.py`
  - order skipped on excessive drift
  - order skipped when edge decays below threshold
  - order proceeds when checks pass

---

## Phase 4: Crash/Restart Idempotency (Blocker)

### Why

Autonomous VPS runs will restart eventually. Replays after partial failures can cause duplicate live orders.

### Code Changes

1. Add deterministic `client_order_id` generation in `app/trading/executor.py`:
- Derive from `(cycle_id, market_id, action, rounded_price, rounded_size)`.

2. Add pending execution persistence before network call:
- Write `status=pending_submit` row.
- Update row to `submitted/failed` after API response.

3. Reconcile pending submits on startup in `app/bot.py` / `app/trading/reconciliation.py`:
- Query exchange/local state and finalize ambiguous pending rows.

### Tests

- `tests/test_execution_idempotency.py`
  - repeated submission attempt for same signal yields one effective order path
  - pending records survive restart and reconcile correctly

---

## Phase 5: Separate Live vs Dry-Run State (Required)

### Why

Paper and live state must never mix.

### Code Changes

1. Add database path split logic in `app/config.py` or `app/__main__.py`:
- `kalshi_dryrun.sqlite` when dry-run
- `kalshi_live.sqlite` when live
- allow explicit override

2. Add startup log banner showing active DB file and mode.

3. Add migration/compat fallback:
- if only legacy `kalshi.sqlite` exists, warn and require explicit confirmation flag to reuse in live.

### Tests

- `tests/test_db_mode_separation.py`
  - dry-run and live use different DBs by default
  - live startup warns/fails on unsafe legacy path unless explicitly overridden

---

## Phase 6: Non-Interactive Live Runtime Contract (Required)

### Why

Autonomous runs must never block on prompts.

### Code Changes

1. Add `--non-interactive` in `app/__main__.py`:
- Disables all prompts unconditionally.
- Fails fast if required values are missing.

2. Ensure runtime scope prompts are skipped when:
- `--non-interactive` is set
- stdin is not TTY

3. Optional: add direct scope flags to avoid editing JSON:
- `--series-tickers`
- `--event-tickers`
- `--market-ids`

### Tests

- `tests/test_non_interactive_mode.py`
  - no `input()` path executed under non-interactive mode
  - startup fails clearly if required scope/keys missing

---

## Phase 7: Platform Safety Gate (Required)

### Why

Polymarket execution is currently stubbed; live should not proceed if unsupported platform is enabled.

### Code Changes

1. In `app/config.py` `validate_for_mode('trade')`:
- If `dry_run=false` and `platforms.polymarket.enabled=true`, fail with explicit message unless execution is fully implemented.

2. Add startup summary warning for unsupported live platforms.

### Tests

- `tests/test_platform_live_validation.py`
  - live + polymarket enabled => validation error
  - live + kalshi only => passes

---

## Phase 8: Observability for 7-Day Unattended Run (Recommended)

### Why

A week-long live run needs machine-readable health and simple alerts.

### Code Changes

1. Add cycle heartbeat JSON write in `app/bot.py`:
- `reports/heartbeat.json` with timestamp, cycle status, bankroll, exposure, open positions, errors.

2. Add optional webhook notifications in `app/utils/`:
- send only on critical events (fatal error, kill switch hit, daily loss halt).

3. Add clear per-cycle summary codes for grep-friendly monitoring.

### Tests

- `tests/test_heartbeat_reporting.py`
  - heartbeat file updates each cycle
  - critical events trigger notifier call

---

## Delivery Order (PR Sequence)

1. Phase 1 (scope guard)
2. Phase 2 (live safety rails)
3. Phase 3 (execution-time revalidation)
4. Phase 4 (idempotency)
5. Phase 5 (DB separation)
6. Phase 6 (non-interactive contract)
7. Phase 7 (platform gate)
8. Phase 8 (observability)

Each PR should include:
- Code
- Tests
- README command updates for new flags/settings

---

## Go-Live Checklist After Code Complete

1. All new tests pass.
2. Full suite passes in VPS-like environment.
3. Dry-run burn-in for 24h with exact live config except `dry_run=true`.
4. Live micro-capital run starts with kill switch tested.
5. Daily review cadence and rollback procedure confirmed.

