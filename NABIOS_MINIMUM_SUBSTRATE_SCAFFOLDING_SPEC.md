# NabiOS Minimum Scaffolding Spec (KalshiPredictionMarket)

**Goal**
Create the smallest viable substrate stack so your buddy can bootstrap `nabi-curator` + `market_reflex_daemon` with Rust services from source, without overbuilding the full platform first.

**Scope**
- This document covers local developer/machine setup, not production hardening.
- It assumes your buddy can build Rust binaries from your private crates.
- It separates what is required for:
  - curator + reflex core behavior, and
  - optional event persistence/observability.

## 0) Choose your track

1. **Track A: Rust-backed pipeline (recommended if he already has private crates)**
   - Keep NATS + kernel + bridge style services.
   - Run curator/reflex to consume raw events, enrich, then emit intents.
2. **Track B: Python-only prototype (faster, fewer moving parts)**
   - Skip Rust services initially.
   - Useful to validate event contracts and curator/reflex behavior first.

**Recommended for buddy handoff**: Track A.

## 1) Required minimum components for Track A

### Must have
1. `nats` with JetStream enabled
2. `nabi-kernel` binary
3. `plexus-bridge` binary (recommended; provides `/events/full` and persistence path)
4. `nabi-curator` daemon (`market_signal_curator.py`)
5. `market_reflex_daemon.py`
6. One raw event source that writes into a curator input path or emits raw events

### Needed for persistence/ops (not mandatory for first smoke test)
1. `surrealdb`
2. optional `market-signal-engine`/`feature-authority` if running broader analytics loop
3. optional Prometheus/Grafana

### Nice-to-have
1. `event-processor` service if you want full canonical event fan-in to Surreal via stream consumers.
2. a portal dashboard.

## 2) Critical contracts and defaults (do not change lightly)

- Default curated input path (used by curator):
  - `~/.local/state/nabi/kernel/event-queue.jsonl`
- Default curated output mode:
  - NATS subject prefix: `nabi.events`
  - NATS URL: `nats://localhost:4222`
  - Subject used by reflex default: `nabi.events.market_curator.*`
- Default curator event types:
  - Input: `signals.<source>.raw`
  - Output: `signals.<source>.curated`
- Reflex output type by default:
  - `trade.intent.hypo`
- Policy updates (if enabled):
  - `nabi.events.reflex_policy.halo`

- Ports commonly used by substrate services:
  - NATS: `4222`
  - NATS monitor: `8222`
  - nabi-kernel: `5380`
  - plexus-bridge: `5381`
  - SurrealDB: `8284`

## 3) Data layout to create

1. Runtime state dirs
   - `~/.local/state/nabi/kernel/`
   - `~/.local/state/nabi/curator/`
   - Optional logs/telemetry in `~/.local/state/nabi/curator/reflex_telemetry.jsonl`

2. Config locations
   - Curator: `~/.config/nabi/curator/config.yaml`
   - Reflex daemon: `~/.config/nabi/market-reflex.toml`
   - Bridge config (rendered): `deploy/trading-substrate/config/plexus-bridge.toml`

3. Event queue file behavior
   - Curator maintains a checkpoint file at:
     - `~/.local/state/nabi/curator/checkpoint.json`

## 4) Runtime sequence (minimum bootstrap)

### Step 1: build Rust binaries
1. Build and artifact these binaries from private crates:
   - `nabi-kernel`
   - `plexus-bridge`
   - (optional) other services you plan to include.
2. Confirm each binary runs and prints help/status.

### Step 2: run NATS
1. Start NATS with JetStream enabled.
2. Confirm:
   - TCP open on `127.0.0.1:4222`
   - monitor on `127.0.0.1:8222`

### Step 3: run `nabi-kernel`
1. Ensure it points to NATS.
2. Confirm health endpoint (if exposed in your build) responds at `localhost:5380`.

### Step 4: run `plexus-bridge`
1. Provide a rendered `plexus-bridge.toml` with NATS + Surreal settings.
2. Start bridge with that config.
3. Confirm health endpoint on `localhost:5381`.

### Step 5: run curator
1. Put minimal `~/.config/nabi/curator/config.yaml` with correct `event_queue_path`, `checkpoint_path`, and NATS/subject settings.
2. Start `market_signal_curator.py` daemon.
3. Confirm it reads events and updates checkpoint.

### Step 6: run reflex daemon
1. Put minimal `~/.config/nabi/market-reflex.toml`.
2. Start `market_reflex_daemon.py`.
3. Confirm it receives curated messages and emits intent events.

### Step 7: run one raw event and validate
1. Inject one raw event with required envelope fields (`id`, `event_type`, `source`, `timestamp`, `vector_clock`, `payload`).
2. Verify curated output appears on subscribed `nabi.events.market_curator.*`.
3. Verify reflex emits `trade.intent.hypo` path and telemetry write (if default path writable).

## 5) Minimal config examples

### 5.1 Curator config skeleton
```yaml
curator:
  emit_mode: nats
  nats_url: nats://localhost:4222
  subject_prefix: nabi.events
  source_agent_id: market_curator
  event_queue_path: ~/.local/state/nabi/kernel/event-queue.jsonl
  checkpoint_path: ~/.local/state/nabi/curator/checkpoint.json
  severity: info
  history_store: jsonl
  price_history_dir: ~/.local/state/nabi/curator/price_history
  history_window: 50
  anomaly_window: 50
  momentum_threshold: 1.0
  spread_threshold: 0.15
  zscore_threshold: 1.5
  min_liquidity: 10000
  model_estimator: ensemble
```

### 5.2 Reflex config skeleton
```toml
nats_url = "nats://localhost:4222"
subscribe_subject = "nabi.events.market_curator.*"
stream_name = "SYNAPSE_EVENTS"
durable_name = "market-reflex-consumer"
subject_prefix = "nabi.events"
source_agent_id = "market_reflex"
intent_event_type = "trade.intent.hypo"
min_confidence = 0.65
min_anomaly = 0.40
policy_subject = "nabi.events.reflex_policy.halo"
```

## 6) Success criteria for “minimum scaffold complete”

1. NATS + kernel + bridge process start and remain running.
2. Raw event is written to `~/.local/state/nabi/kernel/event-queue.jsonl` (or equivalent source path).
3. Curator starts, reads checkpoint, and emits curated event.
4. Reflex subscribes and emits `trade.intent.hypo` events or telemetry entries.
5. No parsing-loop exceptions for valid events in both daemons.

## 7) Failure-first troubleshooting order

1. Event envelope mismatch first.
   - Ensure raw event has required top-level fields and numeric payload values where expected.
2. Subject mismatch second.
   - Confirm subject includes `nabi.events.market_curator.*` for reflex.
3. Stream mismatch third.
   - Confirm stream name and NATS JetStream subject config.
4. Config path mismatch fourth.
   - Ensure home expansion works (e.g., `~` expands in YAML/Python runtime config).
5. Permissions and disk writes.
   - Ensure curator can create checkpoint and telemetry files.

## 8) When to add the bot loop

After this scaffold passes smoke test:
1. Add your bot as a consumer of curated/reflex events.
2. Map `market_reflex` intent payload into internal `TradeSignal` objects.
3. Keep bot execution dry-run until platform connectors are implemented safely.

## 9) Delivery checklist for your buddy’s agents

1. Create NATS and service startup scripts.
2. Add config templates + example env.
3. Add curator+reflex systemd/tmux launch wrappers.
4. Add local raw event injector and event sample fixture.
5. Add CI checks for startup + contract smoke test.
6. Add a short runbook for restart/reseed checkpoint.

## 10) Makefile quickstart (for agents)

This repo now includes a scaffold `Makefile` that automates the minimum build/run path:

- `make check-deps` — verify local prerequisites (`docker`, `python3`, `curl`).
- `make substrate-start` — run the substrate bootstrap + start stack from `deploy/trading-substrate`.
- `make curator-bootstrap` — write local curator + reflex config under `~/.config/nabi`.
- `make full-quickstart` — print the recommended next steps after dependencies and config are in place.
- `make curator-once` — run one curator pass over current queue.
- `make curator-daemon` / `make reflex-daemon` — start each process in background.
- `make agent-start` / `make agent-stop` — manage both daemons together.
- `make smoke-push-event` — append one synthetic `signals.kalshi.raw` event to the queue for smoke checks.

Suggested first run for a friend:

1. `make substrate-start`
2. `make curator-bootstrap`
3. `make agent-start`
4. `make smoke-push-event`
5. `make curator-once` (or tail logs: `tail -f ~/.local/state/nabi/curator/*.log`)

---

This is the minimum viable spec to get from zero to “curator+reflex are running and observable” in a controlled environment.
