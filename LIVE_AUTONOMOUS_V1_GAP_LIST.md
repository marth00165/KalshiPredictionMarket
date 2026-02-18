# Live Autonomous V1 Gap List

This is the remaining work needed before running the bot unattended with real money for a week.

## Current State

- Autonomous/non-interactive flags exist.
- Scope guard exists (market/event/series).
- Duplicate guards exist (market + event).
- `buy_no` Kelly sizing bug is fixed.
- Risk rails exist (kill switch, daily loss, per-cycle caps).
- Execution revalidation exists (price drift + minimum edge).
- Tests currently pass locally (`24 passed`).

---

## Must-Fix Before Live Week (Blockers)

1. Fill-aware execution lifecycle (do not treat accepted order as filled)

### COMPLETED

- Files: `app/trading/executor.py`, `app/bot.py`, `app/trading/position_manager.py`, `app/trading/reconciliation.py`
- Problem: current flow can add position + deduct bankroll immediately when an order is accepted, even if unfilled/partially filled.
- Required change:
  - Persist `submitted`/`pending_fill` first.
  - Only call `add_position` and bankroll debit when fill is confirmed (or confirmed filled quantity).
  - Handle partial fills explicitly.

2. Daily loss guard must use true equity, not entry-cost exposure

### COMPLETED

- Files: `app/bot.py`, `app/trading/position_manager.py`, `app/trading/reconciliation.py`
- Problem: guard uses `balance + total_exposure` where exposure is local cost basis, not mark-to-market/live account equity.
- Required change:
  - Derive equity from exchange portfolio + cash (or mark open positions with live prices).
  - Use that value in daily loss stop.

3. Idempotency should not permanently block valid future trades

### COMPLETED

- File: `app/trading/executor.py`
- Problem: short-circuit includes `skipped_drift` and `skipped_edge`; those are retryable and should not suppress future attempts.
- Required change:
  - Only short-circuit terminal states (`submitted`, `executed`, maybe `pending_submit` with timeout policy).
  - Exclude retryable skip states from permanent dedupe.

4. Persist actual execution price/quantity from execution-time values

### COMPLETED

- File: `app/trading/executor.py`
- Problem: execution record currently uses signal-time prices, not revalidated submission price.
- Required change:
  - Record final `price`, `count/quantity`, and submitted notional used in `place_order`.
  - Keep this consistent with exchange payload.

5. Fix DB mode split edge case with CLI `--dry-run`

### COMPLETED

- Files: `app/config.py`, `app/__main__.py`
- Problem: DB default path is chosen during config parse; later CLI dry-run override can mismatch mode vs DB file.
- Required change:
  - Recompute/default DB path after CLI mode overrides, or enforce explicit DB path per mode.
  - Guarantee dry-run never touches live DB unless explicitly overridden.

---

## High-Priority Hardening (Strongly Recommended)

6. Pending order reconciliation robustness

### COMPLETED

- Files: `app/trading/reconciliation.py`, `app/api_clients/kalshi_client.py`
- Add:
  - bounded retry window before marking `failed_not_found`,
  - pagination/time-window handling for `get_orders`,
  - stale pending timeout policy.

7. Improve execution risk controls

### COMPLETED

- Files: `app/config.py`, `app/bot.py`, `app/trading/executor.py`
- Add:
  - max slippage in cents/probability at submit,
  - optional per-market/day trade frequency cap,
  - optional cooldown after N consecutive execution failures.

8. Clarify production logs and docs

### COMPLETED

- Files: `app/__main__.py`, `README.md`, `OPERATIONS.md`
- Remove/update “Trade mode not fully implemented…” log line.
- Document autonomous live commands and required config fields explicitly.

9. Test cleanup + reliability

### COMPLETED

- Files: `tests/test_live_autonomous_v1_v2.py`, `tests/test_live_autonomous_v1_final.py`
- Consolidate duplicate test files into one canonical suite.
- Keep async test behavior deterministic and environment-safe.

---

## Additional TODOs From Design Discussion

TODO 10. Explicit settlement-by-outcome lifecycle (result correctness)

### COMPLETED

- Files: `app/trading/reconciliation.py`, `app/trading/position_manager.py`, `app/api_clients/kalshi_client.py`
- Add:
  - explicit handling for settled market outcomes (`YES`/`NO`) to close positions with deterministic payout logic,
  - payout/PnL accounting assertions for both `buy_yes` and `buy_no`,
  - guardrails so “missing from remote positions” is not the only closure signal.

TODO 11. VPS runtime operations checklist + automation

### COMPLETED

- Files: `OPERATIONS.md`, `README.md`, `deploy/` (new scripts/unit examples if needed)
- Add:
  - system service setup (e.g., `systemd`) with restart policy and startup ordering,
  - log rotation and retention policy,
  - health watchdog expectations (heartbeat freshness + alert path),
  - one-command start/stop/status runbook for unattended week-long operation.

---

## Recommended Config Contract For Live Autonomous Run

In `advanced_config.json`:

- `trading.dry_run = false`
- `trading.autonomous_mode = true`
- `trading.non_interactive = true`
- `trading.require_scope_in_live = true`
- One scope must be set:
  - `platforms.kalshi.series_tickers` OR
  - `trading.allowed_market_ids` OR
  - `trading.allowed_event_tickers`
- Conservative risk:
  - `risk.max_positions`
  - `risk.max_position_size`
  - `risk.max_orders_per_cycle`
  - `risk.max_notional_per_cycle`
  - `risk.daily_loss_limit_fraction`
- Execution:
  - `execution.max_price_drift`
  - `execution.min_edge_at_execution`

---

## Go/No-Go Gates For Week-Long Live Run

All must be true:

1. Blockers 1-5 are completed.
2. Full tests pass after test-suite consolidation.
3. 24h dry-run burn-in passes with exact live config except `dry_run=true`.
4. Kill switch tested in runtime.
5. Manual restart test confirms no duplicate order submission.
6. Live runbook/docs updated and verified.

---

## AI Agent Prompts (Copy/Paste)

Use one prompt at a time in order.

### Prompt 1: Fill-Aware Execution Lifecycle

```text
Implement blocker #1 from LIVE_AUTONOMOUS_V1_GAP_LIST.md.

Goal:
- Do not treat accepted order as filled.
- Only add local position and debit bankroll when fill is confirmed.
- Support partial fills.

Files to update:
- app/trading/executor.py
- app/bot.py
- app/trading/position_manager.py
- app/trading/reconciliation.py

Requirements:
- Persist execution states: pending_submit/submitted/pending_fill/filled/partially_filled/failed.
- If API returns only order acceptance, do not call add_position or adjust_balance yet.
- Add reconciliation logic to convert submitted/pending_fill into filled/partially_filled using exchange order status.
- For partial fill, apply proportional position add and bankroll debit.
- Keep existing dry-run behavior unchanged.

Tests:
- Add deterministic tests for: accepted-not-filled, full fill, partial fill, and failed order.
- Run full test suite and report results.

Constraints:
- Minimal targeted changes only.
- Preserve existing logging; add concise new logs for state transitions.
```

### Prompt 2: True Equity Daily Loss Guard

```text
Implement blocker #2 from LIVE_AUTONOMOUS_V1_GAP_LIST.md.

Goal:
- Daily loss guard must use true equity, not local entry-cost exposure.

Files to update:
- app/bot.py
- app/trading/position_manager.py
- app/trading/reconciliation.py

Requirements:
- Add a method that computes current equity using live portfolio/cash source of truth (or mark-to-market open positions with live prices).
- Replace current balance+exposure approximation in _run_risk_guards with this true equity metric.
- Keep UTC day-start baseline behavior.

Tests:
- Add tests proving guard triggers and does not trigger under controlled equity scenarios.
- Include one regression test that old approximation would have misclassified.

Constraints:
- Keep interfaces stable where possible.
- No unrelated config changes.
```

### Prompt 3: Idempotency Retry Semantics

```text
Implement blocker #3 from LIVE_AUTONOMOUS_V1_GAP_LIST.md.

Goal:
- Idempotency must not permanently block retryable skips.

File:
- app/trading/executor.py

Requirements:
- In idempotency short-circuit, only treat terminal statuses as final.
- Do NOT short-circuit retryable statuses like skipped_drift/skipped_edge.
- Add clear status classification helper (terminal vs retryable).
- Keep duplicate prevention for submitted/executed/pending_submit.

Tests:
- Add tests for short-circuit on executed/submitted.
- Add tests that skipped_drift/skipped_edge can be retried later.

Constraints:
- Keep behavior deterministic.
- Do not remove existing idempotency fields.
```

### Prompt 4: Persist Actual Submitted Price/Quantity

```text
Implement blocker #4 from LIVE_AUTONOMOUS_V1_GAP_LIST.md.

Goal:
- Execution records must store execution-time price/quantity used for order submission.

File:
- app/trading/executor.py

Requirements:
- Capture final price/count/notional after revalidation, before place_order.
- Persist those values in executions table records instead of signal-time estimates.
- Keep consistency between place_order payload and persisted execution fields.

Tests:
- Add tests asserting persisted values match execution-time values after drift/edge checks.

Constraints:
- Keep schema changes minimal; reuse existing columns if possible.
```

### Prompt 5: DB Mode Split CLI Edge Case

```text
Implement blocker #5 from LIVE_AUTONOMOUS_V1_GAP_LIST.md.

Goal:
- Ensure CLI --dry-run cannot accidentally use live DB path.

Files:
- app/config.py
- app/__main__.py

Requirements:
- Resolve DB path after CLI mode overrides are applied.
- Guarantee dry-run defaults to dry-run DB and live defaults to live DB.
- Keep explicit user-provided DB path precedence intact.
- Preserve --allow-live-legacy-db semantics.

Tests:
- Add tests for config+CLI combinations:
  - config live + CLI --dry-run
  - config dry-run + live run
  - explicit DB override path

Constraints:
- No behavior change for users with explicit DB path unless safety checks require it.
```

### Prompt 6: Pending Reconciliation Robustness

```text
Implement hardening task #6 from LIVE_AUTONOMOUS_V1_GAP_LIST.md.

Goal:
- Make pending_submit reconciliation robust against API timing and pagination gaps.

Files:
- app/trading/reconciliation.py
- app/api_clients/kalshi_client.py

Requirements:
- Add bounded retry window/attempt tracking before failed_not_found.
- Ensure get_orders supports pagination/time-window retrieval needed for reconciliation.
- Add stale pending timeout policy with explicit status transitions.

Tests:
- Add tests for delayed visibility, eventual match, and timeout-to-failed flow.

Constraints:
- Keep logs concise and grep-friendly.
```

### Prompt 7: Extra Execution Risk Controls

```text
Implement hardening task #7 from LIVE_AUTONOMOUS_V1_GAP_LIST.md.

Goal:
- Add extra live risk protections: slippage cap, frequency cap, failure cooldown.

Files:
- app/config.py
- app/bot.py
- app/trading/executor.py

Requirements:
- New configurable guards:
  - max slippage at submit
  - optional per-market/day trade frequency cap
  - cooldown after N consecutive execution failures
- Enforce guards pre-execution with explicit skip reasons.

Tests:
- Add deterministic unit tests for each guard and one combined scenario.

Constraints:
- Defaults should be safe but not break current dry-run flows.
```

### Prompt 8: Production Logs + Docs Cleanup

```text
Implement hardening task #8 from LIVE_AUTONOMOUS_V1_GAP_LIST.md.

Goal:
- Remove misleading production messages and document autonomous live usage clearly.

Files:
- app/__main__.py
- README.md
- OPERATIONS.md

Requirements:
- Remove/update “Trade mode not fully implemented...” and similar misleading logs.
- Add explicit autonomous live command examples.
- Document required advanced_config keys for live autonomous mode.
- Add a short operator checklist for startup, kill switch, and restart test.

Tests:
- If any CLI behavior changed, add/update tests accordingly.
```

### Prompt 9: Test Suite Consolidation

```text
Implement hardening task #9 from LIVE_AUTONOMOUS_V1_GAP_LIST.md.

Goal:
- Consolidate duplicate live-autonomous tests into one canonical file and keep async tests reliable.

Files:
- tests/test_live_autonomous_v1_v2.py
- tests/test_live_autonomous_v1_final.py
- (create/keep one canonical file)

Requirements:
- Merge duplicate coverage into one test module.
- Remove duplicated test definitions.
- Ensure async tests work in environments with and without pytest-asyncio plugin (use skip guards where needed).
- Keep test names and intent clear.

Validation:
- Run full test suite and report before/after count.
```
