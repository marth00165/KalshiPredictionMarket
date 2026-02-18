SHELL := /bin/bash
PROJECT_ROOT := $(shell pwd)
PYTHON := python3

TRADER_DIR := deploy/trading-substrate
TRADER_TEMPLATE := $(TRADER_DIR)/trader.toml.example
TRADER_CFG := $(TRADER_DIR)/trader.toml

NABI_STATE_DIR := $(HOME)/.local/state/nabi
NABI_CONFIG_DIR := $(HOME)/.config/nabi
CURATOR_DIR := $(NABI_CONFIG_DIR)/curator
CURATOR_CFG := $(CURATOR_DIR)/config.yaml
REFLEX_CFG := $(NABI_CONFIG_DIR)/market-reflex.toml
CURATOR_PID_FILE := $(NABI_STATE_DIR)/curator/market_signal_curator.pid
REFLEX_PID_FILE := $(NABI_STATE_DIR)/curator/market_reflex.pid
CURATOR_LOG := $(NABI_STATE_DIR)/curator/market_signal_curator.log
REFLEX_LOG := $(NABI_STATE_DIR)/curator/market_reflex.log

CURATOR_EVENT_QUEUE := $(NABI_STATE_DIR)/kernel/event-queue.jsonl
CURATOR_CHECKPOINT := $(NABI_STATE_DIR)/curator/checkpoint.json
CURATOR_STATE_PRICE_HISTORY := $(NABI_STATE_DIR)/curator/price_history

CURATOR_EMIT_MODE ?= nats
CURATOR_SUBJECT_PREFIX ?= nabi.events
CURATOR_NATS_URL ?= nats://localhost:4222

SUBSTRATE_SERVICES := nats surrealdb nabi-kernel plexus-bridge market-feature-authority

.DEFAULT_GOAL := help

.PHONY: help check-deps substrate-config substrate-init substrate-start substrate-status substrate-health substrate-logs substrate-stop \
	curator-bootstrap curator-once curator-daemon reflex-daemon agent-start agent-stop python-deps \
	smoke-push-event full-quickstart

help:
	@echo "Nabios Minimum Scaffold Makefile"
	@echo ""
	@echo "Primary flows:"
	@echo "  make check-deps             Verify required local tools"
	@echo "  make substrate-config       Install template trader.toml"
	@echo "  make substrate-init         Render .env + bridge config from trader.toml"
	@echo "  make substrate-start        Start NATS, kernel, bridge, feature-authority"
	@echo "  make curator-bootstrap      Write local curator + reflex config"
	@echo "  make curator-once           Run curator one pass (for initial validation)"
	@echo "  make curator-daemon         Start curator daemon in background"
	@echo "  make reflex-daemon          Start reflex daemon in background"
	@echo "  make agent-start            Start curator + reflex daemons in background"
	@echo "  make smoke-push-event       Push 1 synthetic raw event into queue"
	@echo "  make full-quickstart        One-shot minimal setup scaffold"
	@echo ""

check-deps:
	@command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is required"; exit 1; }
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "ERROR: python3 is required"; exit 1; }
	@command -v curl >/dev/null 2>&1 || { echo "ERROR: curl is required"; exit 1; }

python-deps:
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install -r requirements.txt nats pyyaml || true ; \
	else \
		$(PYTHON) -m pip install -r requirements.txt nats pyyaml ; \
	fi

substrate-config:
	@if [ ! -f $(TRADER_CFG) ]; then \
		cp $(TRADER_TEMPLATE) $(TRADER_CFG); \
		echo "Created $(TRADER_CFG) from template. Set SURREAL credentials, then run: make substrate-init"; \
	else \
		echo "$(TRADER_CFG) already exists"; \
	fi

substrate-init: check-deps substrate-config
	@cd $(TRADER_DIR) && ./nabi-init-trading.sh

substrate-start: check-deps substrate-init
	@cd $(TRADER_DIR) && ./nabi-trade.sh start

substrate-status:
	@cd $(TRADER_DIR) && ./nabi-trade.sh status

substrate-health:
	@cd $(TRADER_DIR) && ./nabi-trade.sh health

substrate-logs:
	@cd $(TRADER_DIR) && ./nabi-trade.sh logs

substrate-stop:
	@cd $(TRADER_DIR) && ./nabi-trade.sh stop

curator-bootstrap:
	@mkdir -p "$(CURATOR_DIR)" "$(NABI_STATE_DIR)/kernel" "$(NABI_STATE_DIR)/curator" "$(CURATOR_STATE_PRICE_HISTORY)"
	@cat > "$(CURATOR_CFG)" <<EOF
curator:
  emit_mode: $(CURATOR_EMIT_MODE)
  nats_url: $(CURATOR_NATS_URL)
  subject_prefix: $(CURATOR_SUBJECT_PREFIX)
  source_agent_id: market_curator
  severity: info
  event_queue_path: $(CURATOR_EVENT_QUEUE)
  checkpoint_path: $(CURATOR_CHECKPOINT)
  history_window: 50
  anomaly_window: 50
  momentum_threshold: 1.0
  spread_threshold: 0.15
  zscore_threshold: 1.5
  min_liquidity: 10000
  model_estimator: ensemble
  history_store: jsonl
  price_history_dir: $(CURATOR_STATE_PRICE_HISTORY)
EOF
	@cat > "$(REFLEX_CFG)" <<EOF
nats_url = "$(CURATOR_NATS_URL)"
subscribe_subject = "nabi.events.market_curator.*"
stream_name = "SYNAPSE_EVENTS"
durable_name = "market-reflex-consumer"
subject_prefix = "nabi.events"
source_agent_id = "market_reflex"
severity = "info"
intent_event_type = "trade.intent.hypo"
min_confidence = 0.65
min_anomaly = 0.40
policy_subject = "nabi.events.reflex_policy.halo"
max_cache = 5000
telemetry_path = "~/.local/state/nabi/curator/reflex_telemetry.jsonl"
EOF
	@echo "Curator config written: $(CURATOR_CFG)"
	@echo "Reflex config written: $(REFLEX_CFG)"

curator-once: curator-bootstrap
	@$(PYTHON) nabi-curator/market_signal_curator.py --once

curator-daemon: curator-bootstrap
	@mkdir -p "$(NABI_STATE_DIR)/curator"
	@echo "starting curator daemon..."
	@nohup $(PYTHON) nabi-curator/market_signal_curator.py --daemon > "$(CURATOR_LOG)" 2>&1 & \
		echo $$! > "$(CURATOR_PID_FILE)"
	@echo "curator daemon pid: $$(cat "$(CURATOR_PID_FILE)")"
	@echo "logs: $(CURATOR_LOG)"

reflex-daemon: curator-bootstrap
	@mkdir -p "$(NABI_STATE_DIR)/curator"
	@echo "starting reflex daemon..."
	@nohup $(PYTHON) nabi-curator/market_reflex_daemon.py > "$(REFLEX_LOG)" 2>&1 & \
		echo $$! > "$(REFLEX_PID_FILE)"
	@echo "reflex daemon pid: $$(cat "$(REFLEX_PID_FILE)")"
	@echo "logs: $(REFLEX_LOG)"

agent-start: curator-daemon reflex-daemon
	@echo "agent daemons running"

agent-stop:
	@if [ -f "$(CURATOR_PID_FILE)" ]; then \
		echo "stopping curator daemon $$(cat $(CURATOR_PID_FILE))"; \
		kill "$$(cat "$(CURATOR_PID_FILE)")" 2>/dev/null || true; \
		rm -f "$(CURATOR_PID_FILE)"; \
	fi
	@if [ -f "$(REFLEX_PID_FILE)" ]; then \
		echo "stopping reflex daemon $$(cat $(REFLEX_PID_FILE))"; \
		kill "$$(cat "$(REFLEX_PID_FILE)")" 2>/dev/null || true; \
		rm -f "$(REFLEX_PID_FILE)"; \
	fi

smoke-push-event: curator-bootstrap
	@mkdir -p "$(NABI_STATE_DIR)/kernel"
	@CURATOR_EVENT_QUEUE="$(CURATOR_EVENT_QUEUE)" $(PYTHON) - <<'PY'
from pathlib import Path
import json
import os
from datetime import datetime, timezone
queue_path = Path(os.environ["CURATOR_EVENT_QUEUE"]).expanduser()
queue_path.parent.mkdir(parents=True, exist_ok=True)
event = {
    "id": "smoke-checkpoint-001",
    "event_type": "signals.kalshi.raw",
    "source": "kalshi.live",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "vector_clock": {"smoke": 1},
    "payload": {
        "market_id": "demo-market-001",
        "title": "Demo Prediction: sample event for bootstrap",
        "yes_price": 0.52,
        "no_price": 0.48,
        "liquidity": 50000,
        "volume_24h": 120000,
        "volume_7d": 200000,
        "historical_prices": [0.48, 0.50, 0.49, 0.51, 0.52],
        "category": "demo"
    },
}
with queue_path.open("a") as f:
    f.write(json.dumps(event) + "\n")
print(f"Wrote synthetic raw event to {queue_path}")
print("Now run: make curator-once (or curator-daemon + check logs)")
PY

full-quickstart: check-deps python-deps curator-bootstrap substrate-start agent-start smoke-push-event curator-once
	@echo "Full quickstart completed."
	@echo "1) Substrate services started"
	@echo "2) Curator/reflex daemons started"
	@echo "3) Smoke event pushed"
	@echo "4) Curator one-pass run completed"
