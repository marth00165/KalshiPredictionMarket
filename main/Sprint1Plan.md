# Hosting Prep Checklist (Kalshi Bot) — Context for Copilot

## Goal

Prepare a Python app to run **unattended on a VPS (DigitalOcean)** with an **hourly schedule**, **SQLite persistence**, **retries**, **logs**, **cost controls**, and **safe defaults**. Sprint 1 focuses on **reliable data collection**; analysis/trading can be split out.

---

## Current App Summary

- Python 3.x, async (`aiohttp`)
- Scans Kalshi markets, filters, optionally uses LLM analysis, generates signals, optional execution
- Writes JSON cycle reports
- SQLite used for persistence (should be file-based on VPS)
- Scheduled hourly (cron/systemd)

---

## Sprint 1 Recommendation: Split Responsibilities

Implement modes so the app can be deployed safely:

### Modes

- **collect**: fetch markets + store raw snapshots to SQLite (NO LLM, NO trading)
- **analyze**: read from SQLite, run LLM in batches, store estimates (optional for Sprint 1)
- **trade**: generate/execute signals (keep disabled by default; require explicit opt-in)

**Default mode should be `collect`.**

---

## Hard Requirements Before Deploying

### 1) Single “run once” entrypoint + exit codes

Provide a command that runs one cycle and exits:

- `python -m app run-once --mode collect --config config.json`
- Return code:
  - `0` on success
  - non-zero on failure

Avoid infinite loops for scheduled execution.

### 2) Overlap prevention (locking)

Prevent multiple instances if a run stalls:

- Use a lock file or SQLite lock table
- Must fail/exit if lock is already held (or wait with timeout)
- Include stale lock cleanup logic (PID + timestamp)

### 3) Secrets management

Do NOT store API keys in the JSON config.

- Load secrets from environment variables:
  - `KALSHI_API_KEY`
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY` (if applicable)
- JSON config should contain only knobs/thresholds.

Add a startup config validation step:

- Validate required env vars exist for enabled providers/platforms
- Validate numeric fields in config (batch_size, limits, etc.)
- Fail fast with clear error messages.

### 4) Resilient HTTP client behavior

Kalshi API:

- Retry on: timeouts, connection errors, 429, 5xx
- Exponential backoff with jitter
- Cap max retries per request (e.g., 5)
- Respect rate limit (~10 req/sec):
  - central limiter (token bucket) or bounded semaphore + sleep

LLM API:

- Retry on 429/5xx/timeouts (cap retries)
- Enforce `api_cost_limit_per_cycle` strictly (stop early if exceeded)
- Log tokens/cost stats per cycle

### 5) SQLite durability + schema hygiene

- Enable WAL mode:
  - `PRAGMA journal_mode=WAL;`
  - `PRAGMA synchronous=NORMAL;` (or FULL if you want max safety)
- Use explicit schema migrations:
  - store `schema_version`
  - run migrations at startup
- Ensure idempotent inserts:
  - Choose a stable `snapshot_hour_utc` (rounded to hour)
  - Unique constraint: `(market_id, snapshot_hour_utc)`
  - Use `INSERT OR IGNORE` or UPSERT

### 6) Logging & observability

- Log to stdout (for systemd/journald)
- Add one structured summary log line per cycle:
  - `cycle=... mode=collect scanned=... inserted=... errors=... duration_s=...`
- Keep JSON cycle reports (good), but logs should be sufficient for quick debugging.

### 7) Heartbeat / last-success tracking

Write a small status artifact each cycle:

- In SQLite table `status` OR a `status.json` file
- Include:
  - `last_success_at_utc`
  - `last_error_at_utc`
  - `last_cycle_id`
  - `last_mode`
  - `last_duration_s`

Optional: notify on failure (webhook/email) later.

### 8) Backups + restore procedure

- Nightly SQLite backup:
  - copy DB file to `backups/kalshi_YYYYMMDD.sqlite`
- Test restore locally:
  - stop job
  - replace DB with backup
  - rerun job and confirm it continues

---

## Implementation Guidance for Copilot

### Suggested Project Layout

- `app/`
  - `main.py` (CLI entry)
  - `config.py` (parse + validate)
  - `kalshi_client.py` (aiohttp client + retries + rate limit)
  - `storage/`
    - `db.py` (SQLite connection, WAL, migrations)
    - `schema.sql` / migrations
  - `modes/`
    - `collect.py`
    - `analyze.py`
    - `trade.py`
  - `logging.py` (structured logging helper)
  - `lock.py` (file lock)
- `reports/` (cycle JSON output)
- `backups/`
- `requirements.txt` (pinned) or `pyproject.toml` (+ lock)

### CLI Requirements

- `--config path`
- `--mode collect|analyze|trade`
- `--dry-run` (for analyze/trade)
- `--once` (run one cycle and exit)
- `--log-level`

### Data Model (minimal)

Tables:

- `market_snapshots`:
  - `market_id TEXT`
  - `snapshot_hour_utc TEXT` (ISO hour)
  - `payload_json TEXT` (raw market JSON blob)
  - `created_at_utc TEXT`
  - UNIQUE `(market_id, snapshot_hour_utc)`

Optional tables for later:

- `estimates` (LLM output)
- `signals`
- `executions`
- `status`

---

## Deployment Assumptions (VPS)

- Ubuntu VPS
- Use `systemd` timer (or cron) to run hourly
- DB stored at fixed path (e.g., `/var/lib/kalshi/kalshi.sqlite`)
- App logs go to journald via stdout

---

## Safety Defaults

- Trading is OFF by default.
- If `trade` mode is enabled, require an explicit flag like `--enable-live-trading`.
- Enforce spend limits (`api_cost_limit_per_cycle`) and stop early.

---

## Definition of Done (Sprint 1)

- Runs hourly without manual intervention
- No overlapping runs
- SQLite file persists and stays consistent
- Retries handle transient API failures
- Logs show success/failure and counts
- Heartbeat indicates last successful run
- Backups exist and restore tested

The plan is updated with:

1. **Market filtering config** for macro focus (Fed/CPI/NFP series)
2. **Gap detection utility** with SQL query and CLI command
3. **Health check options** (file-based and optional HTTP)
4. Expanded **data model** with calibration fields (`probability_bucket`, `event_ticker`, `macro_event_type`)
5. **Refined CLI** with subcommands (`run`, `gaps`, `status`, `migrate`)
6. Your earlier schema suggestions integrated

Ready for your next instruction!The plan is updated with:

1. **Market filtering config** for macro focus (Fed/CPI/NFP series)
2. **Gap detection utility** with SQL query and CLI command
3. **Health check options** (file-based and optional HTTP)
4. Expanded **data model** with calibration fields (`probability_bucket`, `event_ticker`, `macro_event_type`)
5. **Refined CLI** with subcommands (`run`, `gaps`, `status`, `migrate`)
6. Your earlier schema suggestions integrated

Ready for your next instruction!
