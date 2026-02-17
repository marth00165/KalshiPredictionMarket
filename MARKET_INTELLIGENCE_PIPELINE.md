---
title: Market Intelligence Pipeline - Unified Architecture
category: architecture
description: End-to-end map of market data ingestion, curation, and trading across all scattered components
audience: [💻, 🤖, 🔧]
state: published
---

# Market Intelligence Pipeline

Single reference for how market data flows through the NabiOS substrate — from raw API
feeds to curated signals to trade execution. Consolidates scattered components across
5 directories into one coherent map.

## Architecture Overview

```
                        ┌─────────────────────────────────────────────────────────────┐
                        │                   MARKET DATA SOURCES                        │
                        │                                                              │
                        │   Coinbase WS    Polymarket CLOB    Kalshi REST              │
                        │   (BTC-USD)      (binary markets)   (prediction markets)     │
                        └────────┬──────────────┬──────────────────┬───────────────────┘
                                 │              │                  │
                        ┌────────▼──────────────▼──────────────────▼───────────────────┐
                        │              ADAPTERS (L0 → Synapse)                          │
                        │                                                              │
                        │   adapter.ts          polymarket_ingest.sh     [MISSING]     │
                        │   ~/nabia/demos/      ~/nabia/platform/        kalshi        │
                        │   market-feed/        portal/scripts/          adapter       │
                        │                                                              │
                        │   Severity engine:    HTTP polling (20s):      Needs:        │
                        │   z-score, price Δ    CLOB API → POST          REST poll →   │
                        │   → /events/full      → event-queue.jsonl      event-queue   │
                        └────────┬──────────────────────┬──────────────────────────────┘
                                 │                      │
                        ┌────────▼──────────────────────▼──────────────────────────────┐
                        │                    EVENT QUEUE (Canonical Artery)             │
                        │                                                              │
                        │   ~/.local/state/nabi/kernel/event-queue.jsonl               │
                        │   One JSON object per line. STRICT CONTRACT.                 │
                        │                                                              │
                        │   Sources seen: market-feed:*, signals.polymarket.raw,       │
                        │                 market.curator, judgment:*                    │
                        └────────┬─────────────────────────────────────────────────────┘
                                 │
                   ┌─────────────┤
                   │             │
          ┌────────▼─────┐  ┌───▼────────────────────────────────────────────────────┐
          │   CURATOR     │  │   EVENT PROCESSOR (Rust)                               │
          │   (Python)    │  │   nabi-event-processor.service                         │
          │               │  │                                                        │
          │  Enrichment:  │  │   Reads queue → publishes to NATS SYNAPSE_EVENTS       │
          │  momentum,    │  │   Subjects: nabi.events.>                              │
          │  spread,      │  │   DLQ for parse failures                               │
          │  z-score,     │  └───┬────────────────────────────────────────────────────┘
          │  volatility,  │      │
          │  kelly edge   │  ┌───▼────────────────────────────────────────────────────┐
          │               │  │   PLEXUS BRIDGE (Rust)                                 │
          │  Writes back: │  │   plexus-bridge.service                                │
          │  *.curated    │  │                                                        │
          │  events to    │  │   NATS consumer → SurrealDB (plexus_event table)       │
          │  queue        │  │   + gauge feeders + reflex arc governance               │
          └───────────────┘  └───┬────────────────────────────────────────────────────┘
                                 │
                        ┌────────▼─────────────────────────────────────────────────────┐
                        │                    SURREALDB (Persistence)                    │
                        │                                                              │
                        │   localhost:8284  ns=nabi  db=substrate                       │
                        │   Table: plexus_event (191K+ events)                         │
                        │                                                              │
                        │   Queryable by: source, severity, timestamp                  │
                        └────────┬─────────────────────────────────────────────────────┘
                                 │
              ┌──────────────────┼───────────────────────┐
              │                  │                        │
     ┌────────▼────────┐  ┌─────▼──────────┐  ┌─────────▼──────────────┐
     │  PORTAL (Nuxt)  │  │ SIGNAL ENGINE  │  │ STRATEGY / TRADING     │
     │  :4200          │  │ (Bun/TS)       │  │                        │
     │                 │  │                │  │ AI Trading Bot (Python) │
     │  Neural Nexus   │  │ 15min window   │  │ Arb Bot (Python)       │
     │  3D graph       │  │ feature extract│  │ Reflex Daemon (Python) │
     │  Live/Demo      │  │ trade signals  │  │                        │
     └─────────────────┘  └────────────────┘  └────────────────────────┘
```

## Component Inventory

### Adapters (Raw Data → Queue)

| Component | Location | Language | Source API | Status |
|-----------|----------|----------|------------|--------|
| **Market Feed Adapter** | `~/nabia/demos/market-feed/adapter.ts` | Bun/TS (610 LOC) | Coinbase WS, Polymarket WS | Working |
| **Polymarket Ingest** | `~/nabia/platform/portal/scripts/polymarket_ingest.sh` | Shell + curl | Polymarket CLOB REST | Working |
| **Kalshi Adapter** | *Does not exist yet* | — | Kalshi REST v2 | **MISSING** |

### Enrichment (Raw → Curated)

| Component | Location | Language | What It Does | Status |
|-----------|----------|----------|-------------|--------|
| **Market Signal Curator** | `~/nabia/platform/curator/market_signal_curator.py` | Python (28K) | Momentum, spread, z-score, volatility, kelly edge enrichment | Built, needs deploy |
| **Curator Adapters** | `~/nabia/platform/curator/curator_adapters.py` | Python (7.7K) | Price history persistence (JSONL), model estimation (EMA/ensemble) | Built |
| **Feature Authority Daemon** | `~/nabia/platform/curator/feature_authority_daemon.py` | Python (27K) | SurrealDB-backed feature authority with gradient scoring | Built |

### Signal Generation

| Component | Location | Language | What It Does | Status |
|-----------|----------|----------|-------------|--------|
| **Signal Engine** | `~/nabia/demos/market-feed/signal-engine.ts` | Bun/TS (487 LOC) | Queries SurrealDB 15min window, ML features, trade signals | Built |
| **Market Reflex Daemon** | `~/nabia/platform/curator/market_reflex_daemon.py` | Python (9.6K) | Hypothetical trade intents from curated signals | Built |

### Trading / Execution

| Component | Location | Language | What It Does | Status |
|-----------|----------|----------|-------------|--------|
| **AI Trading Bot** | `~/.nabi/kalshi-experiment/KalshiPredictionMarket/` | Python 3.13 | Claude analysis → Kelly sizing → trade signals | Built, dry-run default |
| **BTC Arbitrage Bot** | `~/polymarket-kalshi-btc-arbitrage-bot/` | Python (FastAPI) | Risk-free arb detection: Poly vs Kalshi pricing gaps | Built |
| **Portfolio Tracker** | `~/nabia/demos/market-feed/portfolio.ts` | Bun/TS (475 LOC) | Virtual portfolio management | Built |

### Human-in-the-Loop

| Component | Location | Language | What It Does | Status |
|-----------|----------|----------|-------------|--------|
| **Judgment Capture** | `~/nabia/demos/market-feed/capture.ts` | Bun/TS (268 LOC) | "Send without thinking" CLI — attention, reflection, posture events | Working |

### Visualization

| Component | Location | What It Shows | Status |
|-----------|----------|---------------|--------|
| **Neural Nexus** | `~/nabia/platform/portal/` | 3D substrate graph, live/demo toggle | Running (:4200) |
| **Arb Dashboard** | `~/polymarket-kalshi-btc-arbitrage-bot/frontend/` | Next.js arb opportunity viewer | Built, not connected |

### Observability

| Component | Location | What It Does | Status |
|-----------|----------|-------------|--------|
| **Metrics Exporter** | `~/nabia/demos/market-feed/metrics-exporter.ts` | Prometheus export of market metrics | Built |
| **Reflex Telemetry** | `~/.local/state/nabi/curator/reflex_telemetry.jsonl` | Trade intent audit trail | Path defined |

## Event Sources & Types

Events flowing through the queue use these source identifiers:

| Source | Event Types | Origin |
|--------|------------|--------|
| `market-feed:coinbase` | `market.trade` | adapter.ts (Coinbase WS) |
| `market-feed:polymarket` | `market.trade`, `market.ticker` | adapter.ts (Polymarket WS) |
| `market.curator` | `signals.polymarket.curated`, `signals.*.curated` | curator daemon |
| `judgment:attention` | `attention.captured` | capture.ts (human) |
| `judgment:reflection` | `judgment.reflection` | capture.ts (human) |
| `judgment:posture` | `posture.shift` | capture.ts (governance event) |
| `kalshi.live` | `market.trade` | **PLANNED** — kalshi adapter |
| `kalshi.signals` | `signal.*` | **PLANNED** — signal engine extension |
| `arbitrage.opportunity` | `opportunity.risk-free` | **PLANNED** — arb bot wrapper |
| `kalshi.trading` | `signal.*`, `execution.*` | **FUTURE** — full bot integration |

## Severity Engine

The adapter uses domain-aware severity classification:

| Condition | Severity | Threshold |
|-----------|----------|-----------|
| Volume z-score > 3 sigma | `critical` | Statistical anomaly |
| Price delta > 0.5% | `critical` | Large sudden move |
| Volume z-score > 2 sigma | `warning` | Elevated activity |
| Price delta > 0.2% | `warning` | Notable move |
| Normal activity | `info` | Baseline |

For prediction markets (0-1 scale), probability thresholds replace price thresholds
(5 cent move = 5% = critical in binary market context).

## Curator Enrichment Metrics

The curator adds composite scoring to raw signals:

| Metric | Weight | Range | What It Measures |
|--------|--------|-------|-----------------|
| Momentum | 25% | 0-inf | Volume acceleration (24h / 7d ratio) |
| Spread | 15% | 0-0.5 | Distance from consensus (0.5 for binary) |
| Z-Score | 25% | 0-inf | Statistical anomaly strength |
| Volatility | 15% | 0-0.5 | Price turbulence |
| Liquidity | 20% | 0-inf | Market depth (filters below 10K) |

Output: `signal_confidence` (0-1) and `recommendation` (HIGH / ELEVATED / NORMAL / LOW).

## Governance Integration

### Reflex Arc (Rust, ADR-0050)
The plexus-bridge reflex arc monitors gauge breaches. Market events with `critical` severity
feed into `GaugeBreach` → `BreachAggregator` → `PolicyEvaluator` → authority contraction.
A trading bot that triggers too many critical events gets its authority gradient reduced.

### Feature Authority (Python)
`feature_authority_daemon.py` maintains a SurrealDB-backed authority gradient per feature/source.
Features earn trust through consistent curated signal quality. Schema: `feature_authority.schema.surql`.

### Event Boundary (Python)
`event_boundary.py` enforces publish rate limits per source — prevents runaway emitters from
flooding the queue (the exact problem that caused the 2026-02-16 pipeline corruption).

## What's Missing (Integration Gaps)

### Gap 1: Kalshi Adapter (High Priority)
No adapter exists to bring Kalshi market data into the substrate.

**Recommended**: `~/nabia/demos/market-feed/kalshi-adapter.ts` (Bun/TS)
- Poll Kalshi REST v2 API (public endpoints, no auth needed)
- Source: `kalshi.live`
- Reuse severity engine from adapter.ts
- Prediction market thresholds (probability 0-1 scale)

### Gap 2: Arb Bot → Substrate Bridge
The `polymarket-kalshi-btc-arbitrage-bot` FastAPI server runs at localhost:8000 but
doesn't emit substrate events.

**Recommended**: Add HTTP POST to `/events/full` when `total_cost < 1.00` (arbitrage found).
Source: `arbitrage.opportunity`, severity: `critical`.

### Gap 3: AI Trading Bot → Substrate Bridge
The `KalshiPredictionMarket` bot has Claude analysis and Kelly sizing but no substrate output.

**Recommended**: Emit events for:
- `kalshi.analysis` — Claude fair value estimates (every scan cycle)
- `kalshi.signal` — Trade signals (buy/sell/hold with Kelly fraction)
- `kalshi.execution` — Actual trade execution (when dry_run=false)

### Gap 4: Curator Not Deployed as Service
Curator code exists but isn't running as a systemd service. Currently manual start via
`start_curator.sh --daemon`.

**Recommended**: Create `~/.config/systemd/user/market-curator.service`.

### Gap 5: Signal Engine Doesn't Query Kalshi
`signal-engine.ts` queries SurrealDB for `polymarket.live` events only.

**Recommended**: Extend query to include `kalshi.live` source for cross-market feature extraction.

## Quick Start: Bring It All Live

```bash
# 1. Verify pipeline is healthy (after 2026-02-16 recovery)
systemctl --user status nabi-event-processor plexus-bridge surrealdb

# 2. Start market-feed adapter (Coinbase BTC-USD)
cd ~/nabia/demos/market-feed && bun run adapter.ts --source coinbase

# 3. Start curator daemon
cd ~/nabia/platform/curator && ./start_curator.sh --daemon

# 4. Watch curated signals flow
tail -f ~/.local/state/nabi/kernel/event-queue.jsonl | \
  python3 -c "import sys,json;[print(json.dumps(json.loads(l),indent=2)) for l in sys.stdin if 'curated' in l]"

# 5. Verify in portal
curl -s 'http://localhost:4200/api/substrate/graph?hours=1' | python3 -m json.tool | head -20
```

## File Index (All Locations)

| Category | Path | LOC |
|----------|------|-----|
| **Adapters** | | |
| Coinbase/Poly adapter | `~/nabia/demos/market-feed/adapter.ts` | 610 |
| Judgment capture | `~/nabia/demos/market-feed/capture.ts` | 268 |
| Signal engine | `~/nabia/demos/market-feed/signal-engine.ts` | 487 |
| Portfolio tracker | `~/nabia/demos/market-feed/portfolio.ts` | 475 |
| Metrics exporter | `~/nabia/demos/market-feed/metrics-exporter.ts` | 316 |
| Event schema | `~/nabia/demos/market-feed/schema.json` | — |
| **Curator** | | |
| Main curator | `~/nabia/platform/curator/market_signal_curator.py` | ~800 |
| Curator adapters | `~/nabia/platform/curator/curator_adapters.py` | ~230 |
| Feature authority | `~/nabia/platform/curator/feature_authority_daemon.py` | ~800 |
| Reflex daemon | `~/nabia/platform/curator/market_reflex_daemon.py` | ~290 |
| Event boundary | `~/nabia/platform/curator/event_boundary.py` | ~40 |
| Threshold calibration | `~/nabia/platform/curator/calibrate_regime_thresholds.py` | ~75 |
| Launcher | `~/nabia/platform/curator/start_curator.sh` | ~110 |
| Feature auth launcher | `~/nabia/platform/curator/start_feature_authority.sh` | ~50 |
| SurrealDB schema | `~/nabia/platform/curator/feature_authority.schema.surql` | ~170 |
| Tests | `~/nabia/platform/curator/tests/` | — |
| **Kalshi AI Bot** | | |
| Refactored bot | `~/.nabi/kalshi-experiment/KalshiPredictionMarket/main/ai_trading_bot_refactored.py` | ~500 |
| Series scanner | `~/.nabi/kalshi-experiment/KalshiPredictionMarket/main/series_scanner.py` | — |
| Kalshi client | `~/.nabi/kalshi-experiment/KalshiPredictionMarket/main/api_clients/kalshi_client.py` | ~120 |
| Polymarket client | `~/.nabi/kalshi-experiment/KalshiPredictionMarket/main/api_clients/polymarket_client.py` | ~170 |
| Claude analyzer | `~/.nabi/kalshi-experiment/KalshiPredictionMarket/main/analysis/openai_analyzer.py` | — |
| Kernel analyzer | `~/.nabi/kalshi-experiment/KalshiPredictionMarket/main/analysis/kernel_analyzer.py` | — |
| Config | `~/.nabi/kalshi-experiment/KalshiPredictionMarket/advanced_config.json` | — |
| **Arb Bot** | | |
| API server | `~/polymarket-kalshi-btc-arbitrage-bot/backend/api.py` | 152 |
| Arb bot CLI | `~/polymarket-kalshi-btc-arbitrage-bot/backend/arbitrage_bot.py` | 162 |
| Kalshi client | `~/polymarket-kalshi-btc-arbitrage-bot/backend/fetch_current_kalshi.py` | ~120 |
| Poly client | `~/polymarket-kalshi-btc-arbitrage-bot/backend/fetch_current_polymarket.py` | ~170 |
| Frontend | `~/polymarket-kalshi-btc-arbitrage-bot/frontend/` | Next.js |
| **Infrastructure** | | |
| Event processor | `~/.config/systemd/user/nabi-event-processor.service` | — |
| Plexus bridge | `~/.config/systemd/user/plexus-bridge.service` | — |
| Portal service | `~/.config/systemd/user/nabi-portal.service` | — |
| Event processor config | `~/.config/nabi/event-processor.toml` | — |
| Bridge config | `~/.config/nabi/plexus-bridge.toml` | — |
| Source registry | `~/.config/nabi/gateway/source-registry.toml` | — |
| Curator config | `~/.config/nabi/curator/config.yaml` | — |
| Recovery runbook | `~/docs/runbooks/LIVE_EVENT_PIPELINE_RECOVERY_RUNBOOK.md` | — |

## Related ADRs

| ADR | Relevance |
|-----|-----------|
| ADR-0035 | Event publishing separation (queue as canonical artery) |
| ADR-0039 | Substrate physics paradigm (emergence over orchestration) |
| ADR-0050 | Gradient authority (trading bot trust earned through behavior) |
| ADR-0051 | Formal boundary doctrine (curator = Python/interpretation layer) |
| ADR-0052 | Semantic integrity break (JSONL contract violations) |

## History

- **2026-02-08**: Market feed adapter created (Coinbase BTC-USD + judgment capture)
- **2026-02-10**: Curator built with enrichment pipeline and reflex daemon
- **2026-02-11**: Feature authority daemon + curator adapters added
- **2026-02-14**: Signal engine, portfolio tracker, metrics exporter added
- **2026-02-16**: Pipeline recovery (99.95% queue corruption from multiline JSON emitters)
- **2026-02-16**: This consolidated architecture document created
