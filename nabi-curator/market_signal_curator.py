#!/usr/bin/env python3
"""
Reusable Market Signal Curator

Enriches raw market events with financial metrics:
- Momentum (volume 24h / volume 7d ratio)
- Spread analysis (distance from fair value 0.5 for binary markets)
- Z-score deviation (statistical significance)
- Volatility and liquidity metrics
- Kelly criterion edge estimation

Sources:
- Reads: signals.{source}.raw events (polymarket, binance, coinbase, etc.)
- Emits: signals.{source}.curated back to event-queue.jsonl

Architecture: Python interpretation layer (ADR-0051)
- Fast, heuristic, probabilistic enrichment
- Can be wrong but useful for pattern detection
- Feeds Lattice Observer and other consumers

Example raw event (Polymarket):
{
  "id": "event-uuid",
  "source": "polymarket.ingest",
  "event_type": "signals.polymarket.raw",
  "timestamp": "2026-02-10T12:34:56Z",
  "vector_clock": {...},
  "payload": {
    "market_id": "0x1234...",
    "title": "Will BTC exceed $100k by EOY 2026?",
    "yes_price": 0.72,
    "no_price": 0.28,
    "volume_24h": 145000,
    "volume_7d": 890000,
    "liquidity": 50000,
    "historical_prices": [0.70, 0.71, 0.72, ...],
    "category": "crypto"
  }
}

Curated output:
{
  "id": "curated-event-uuid",
  "source": "market.curator",
  "event_type": "signals.polymarket.curated",
  "timestamp": "2026-02-10T12:34:56Z",
  "vector_clock": {...},
  "payload": {
    "market_id": "0x1234...",
    "title": "Will BTC exceed $100k by EOY 2026?",
    "original_yes_price": 0.72,
    "enrichment": {
      "momentum_score": 0.65,  # 24h volume / 7d volume
      "spread_distance": 0.22,  # |0.72 - 0.5|
      "zscore": 1.45,  # (0.72 - mean) / stdev
      "volatility": 0.04,  # std dev of price history
      "liquidity_depth": 50000,
      "kelly_edge": 0.15,  # (p * kelly_b - q) for estimation
      "signal_confidence": 0.78,  # composite confidence
      "anomaly_score": 0.12  # deviation from normal distribution
    },
    "recommendation": "ELEVATED_SIGNAL",  # HIGH, ELEVATED, NORMAL, LOW
    "reasoning": "High momentum + reasonable spread + above-average volume"
  }
}
"""

import asyncio
import fnmatch
import json
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from curator_adapters import SynapseEvent, build_history_store, build_model_estimator
from event_boundary import safe_parse_synapse
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MarketEnrichment:
    """Financial metrics enriching raw market data"""
    momentum_score: float  # volume_24h / volume_7d
    spread_distance: float  # distance from fair value (0.5 for binary)
    zscore: float  # statistical deviation
    volatility: float  # price volatility
    liquidity_depth: float  # available liquidity
    kelly_edge: float  # estimated edge for position sizing
    signal_confidence: float  # 0-1 composite confidence
    anomaly_score: float  # 0-1 deviation from normal


@dataclass
class DeltaDynamics:
    delta: float
    delta_velocity: float
    delta_rolling_std: float


class MarketSignalCurator:
    """
    Enriches raw market signals with financial metrics.

    Design (ADR-0051 Boundary Doctrine):
    - Python interpretation layer (fast, heuristic)
    - Reads: signals.{source}.raw from event-queue.jsonl
    - Enriches with momentum, volatility, Z-score
    - Emits: signals.{source}.curated back to event-queue.jsonl
    - Runs as daemon, can handle multiple market sources
    """

    def __init__(self, event_queue_path: str = None, config: Dict = None):
        """Initialize curator with event queue path and config"""
        self.config = config or self._load_config()
        self.event_queue_path = Path(event_queue_path or self.config["event_queue_path"]).expanduser()
        self.checkpoint_path = Path(self.config["checkpoint_path"]).expanduser()
        self.emit_mode = self.config.get("emit_mode", "nats")

        # Price history cache (per market for Z-score calculation)
        self.price_history: Dict[str, deque] = {}
        self.history_window = self.config.get("history_window", 50)
        self.anomaly_window = self.config.get("anomaly_window", 50)
        self.source_filters = self.config.get("sources", [])
        self.history_store = build_history_store(self.config)
        self.model_estimator = build_model_estimator(self.config)

        # NATS publishing config
        self.nats_url = self.config.get("nats_url", "nats://localhost:4222")
        self.subject_prefix = self.config.get("subject_prefix", "nabi.events")
        self.source_agent_id = self.config.get("source_agent_id", "market_curator")
        self.default_severity = self.config.get("severity", "info")
        self._nats = None
        self._nats_js = None
        self._loop = asyncio.new_event_loop()

        # Queue checkpoint for idempotent reads
        self.queue_offset = self._load_checkpoint()
        self.sequence = 0

        logger.info(f"Curator initialized: {self.event_queue_path}")
        logger.info(f"Config: momentum_threshold={self.config['momentum_threshold']}, "
                   f"zscore_threshold={self.config['zscore_threshold']}")

    def _default_config(self) -> Dict:
        """Default curator configuration"""
        return {
            "history_window": 50,  # samples for Z-score calculation
            "anomaly_window": 50,  # window for anomaly z-score
            "momentum_threshold": 1.0,  # 24h/7d ratio (1.0 = normal)
            "spread_threshold": 0.15,  # price deviation from 0.5
            "zscore_threshold": 1.5,  # statistical significance cutoff
            "volatility_window": 20,  # samples for volatility calc
            "min_liquidity": 10000,  # minimum liquidity to curate
            "kelly_kelly_b": 1.0,  # kelly criterion 'b' parameter (odds)
            "emit_mode": "nats",
            "nats_url": "nats://localhost:4222",
            "subject_prefix": "nabi.events",
            "source_agent_id": "market_curator",
            "severity": "info",
            "event_queue_path": str(Path.home() / ".local/state/nabi/kernel/event-queue.jsonl"),
            "checkpoint_path": str(Path.home() / ".local/state/nabi/curator/checkpoint.json"),
            "start_at_end": True,
            "tail_bytes": 500000,
            "max_scan_lines": 2000,
            "history_store": "jsonl",
            "price_history_dir": str(Path.home() / ".local/state/nabi/curator/price_history"),
            "model_estimator": "ensemble",
            "model_ema_window": 50,
            "confidence_weights": {
                "momentum": 0.25,
                "spread": 0.15,
                "zscore": 0.25,
                "volatility": 0.15,
                "liquidity": 0.20
            }
        }

    def _load_config(self) -> Dict:
        """Load config from ~/.config/nabi/curator/config.yaml if present."""
        config = self._default_config()
        config_path = Path.home() / ".config/nabi/curator/config.yaml"
        if config_path.exists():
            try:
                import yaml  # type: ignore
                data = yaml.safe_load(config_path.read_text()) or {}
                if isinstance(data, dict):
                    curator_cfg = data.get("curator", data)
                    if isinstance(curator_cfg, dict):
                        config.update(curator_cfg)
            except Exception as e:
                logger.warning(f"Failed to load config {config_path}: {e}")
        return config

    def _load_checkpoint(self) -> int:
        """Load checkpoint byte offset; default to end of file on first run."""
        try:
            if self.checkpoint_path.exists():
                content = json.loads(self.checkpoint_path.read_text())
                return int(content.get("byte_offset", 0))
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

        if self.event_queue_path.exists():
            try:
                if self.config.get("start_at_end", True):
                    size = self.event_queue_path.stat().st_size
                    tail_bytes = int(self.config.get("tail_bytes", 500000))
                    return max(0, size - max(tail_bytes, 0))
                return 0
            except Exception:
                return 0
        return 0

    def _save_checkpoint(self):
        """Persist checkpoint byte offset."""
        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "byte_offset": self.queue_offset,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            self.checkpoint_path.write_text(json.dumps(payload))
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def curate_raw_event(self, raw_event: SynapseEvent) -> Optional[Dict]:
        """
        Enrich a raw market event with financial metrics.

        Args:
            raw_event: SynapseEvent with event_type="signals.{source}.raw"

        Returns:
            Curated event dict or None if metrics insufficient
        """
        try:
            payload = self._normalize_payload(raw_event)
            if not payload:
                logger.warning("Skipping event: no payload")
                return None

            market_id = payload.get("market_id")

            if not market_id:
                logger.warning("Skipping event: no market_id")
                return None

            # Extract price history (fallback to single price if unavailable)
            prices = payload.get("historical_prices", [])
            if payload.get("yes_price"):
                prices.append(payload["yes_price"])

            if not prices:
                logger.warning(f"Skipping market {market_id}: no price data")
                return None

            # Update persistent history store + in-memory cache
            if payload.get("yes_price") is not None:
                self.history_store.append(
                    market_id,
                    float(payload["yes_price"]),
                    timestamp=raw_event.timestamp
                )
            historic = self.history_store.get_prices(market_id)
            if historic:
                prices = list(historic)

            # Update in-memory price history cache
            if market_id not in self.price_history:
                self.price_history[market_id] = deque(maxlen=self.history_window)
            self.price_history[market_id].extend(prices)

            # Delta dynamics (last 50 deltas)
            delta_dyn = self._calculate_delta_dynamics(list(self.price_history[market_id]))

            # Regime detection (delta-based)
            regime = self._detect_regime(list(self.price_history[market_id]), delta_dyn.delta_rolling_std)

            # Estimate independent model (adapter)
            p_model = self.model_estimator.estimate(
                market_id,
                list(self.price_history[market_id]),
                payload,
                regime
            )

            # Calculate enrichment metrics
            enrichment = self._calculate_enrichment(
                market_id=market_id,
                payload=payload,
                price_history=list(self.price_history[market_id]),
                delta_dyn=delta_dyn,
                p_model=p_model,
                regime=regime
            )

            if not enrichment:
                return None

            # Build curated event
            curated_event = {
                "id": f"evt-{uuid.uuid4()}",
                "source": "market.curator",
                "event_type": raw_event.event_type.replace(".raw", ".curated"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "vector_clock": raw_event.vector_clock,
                "payload": {
                    "market_id": market_id,
                    "title": payload.get("title", ""),
                    "original_yes_price": payload.get("yes_price"),
                    "original_no_price": payload.get("no_price"),
                    "enrichment": asdict(enrichment),
                    "p_model": p_model,
                    "p_market": payload.get("yes_price"),
                    "delta": delta_dyn.delta,
                    "delta_velocity": delta_dyn.delta_velocity,
                    "delta_rolling_std": delta_dyn.delta_rolling_std,
                    "regime": regime,
                    "edge": (p_model - payload.get("yes_price")) if p_model is not None and payload.get("yes_price") is not None else None,
                    "recommendation": self._rate_signal(enrichment),
                    "reasoning": self._generate_reasoning(enrichment, payload)
                }
            }

            # Attach provenance for downstream authority tracking
            curated_event["payload"]["_curator"] = {
                "parent_id": raw_event.id,
                "parent_source": raw_event.source,
            }

            return curated_event

        except Exception as e:
            logger.error(f"Error curating event: {e}", exc_info=True)
            return None

    def _normalize_payload(self, raw_event: SynapseEvent) -> Optional[Dict]:
        """Normalize payload from either payload or metadata schema."""
        payload = raw_event.payload
        if isinstance(payload, dict) and payload.get("market_id"):
            return payload

        metadata = raw_event.metadata or {}
        if not isinstance(metadata, dict):
            return None

        # Extract market_id and title
        market_id = metadata.get("market_id") or metadata.get("id")
        title = metadata.get("title")

        markets = metadata.get("markets") or []
        market0 = markets[0] if isinstance(markets, list) and markets else {}
        if not market_id:
            market_id = market0.get("id") or metadata.get("ticker")
        if not title:
            title = market0.get("question")

        # Volume and liquidity mapping
        volume_24h = metadata.get("volume_24h") or metadata.get("volume24hr") or metadata.get("volume24h")
        if volume_24h is None:
            volume_24h = market0.get("volume24hr") or market0.get("volume24h")

        volume_7d = metadata.get("volume_7d") or metadata.get("volume1wk") or metadata.get("volume1wkAmm")
        if volume_7d is None:
            volume_7d = market0.get("volume1wk") or market0.get("volume1wkAmm")

        liquidity = metadata.get("liquidity") or metadata.get("liquidityNum")
        if liquidity is None:
            liquidity = market0.get("liquidityNum") or market0.get("liquidity")

        # Price extraction
        outcome_prices = (
            metadata.get("outcomePrices") or
            market0.get("outcomePrices") or
            metadata.get("outcome_prices") or
            market0.get("outcome_prices")
        )
        prices = []
        try:
            if isinstance(outcome_prices, str):
                prices = json.loads(outcome_prices)
            elif isinstance(outcome_prices, list):
                prices = outcome_prices
        except Exception:
            prices = []

        yes_price = None
        no_price = None
        if len(prices) >= 2:
            try:
                yes_price = float(prices[0])
                no_price = float(prices[1])
            except Exception:
                pass
        elif len(prices) == 1:
            try:
                yes_price = float(prices[0])
                no_price = 1.0 - yes_price
            except Exception:
                pass

        normalized = {
            "market_id": market_id,
            "title": title or "",
            "yes_price": yes_price,
            "no_price": no_price,
            "volume_24h": volume_24h or 0,
            "volume_7d": volume_7d or 1,
            "liquidity": float(liquidity) if liquidity is not None else 0,
            "historical_prices": [],
            "category": metadata.get("category") or market0.get("category") or "",
        }

        return normalized if normalized.get("market_id") else None

    def _calculate_enrichment(
        self,
        market_id: str,
        payload: Dict,
        price_history: List[float],
        delta_dyn: DeltaDynamics,
        p_model: Optional[float],
        regime: str
    ) -> Optional[MarketEnrichment]:
        """Calculate financial enrichment metrics"""

        # Momentum: 24h volume / 7d volume ratio
        vol_24h = payload.get("volume_24h", 0)
        vol_7d = payload.get("volume_7d", 1)  # avoid division by zero
        momentum = vol_24h / max(vol_7d, 1)

        # Spread distance from fair value (0.5 for binary markets)
        yes_price = payload.get("yes_price", 0.5)
        spread = abs(yes_price - 0.5)

        # Z-score: how many standard deviations from mean (windowed)
        price_window = price_history[-self.anomaly_window:] if price_history else []
        zscore = 0.0
        if len(price_window) >= 2:
            mean_price = statistics.mean(price_window)
            stdev = statistics.stdev(price_window)
            if stdev > 0:
                zscore = abs((yes_price - mean_price) / stdev)

        # Volatility: standard deviation of prices
        volatility = 0.0
        if len(price_history) >= 2:
            volatility = statistics.stdev(price_history[-self.config["volatility_window"]:])

        # Liquidity depth
        liquidity = payload.get("liquidity", 0)

        # Skip low-liquidity markets
        if liquidity < self.config["min_liquidity"]:
            logger.debug(f"Market {market_id}: insufficient liquidity {liquidity}")
            return None

        # Kelly Criterion edge estimation
        # f* = (bp - q) / b, where b=1 (binary odds), p=yes_price, q=1-p
        p = yes_price
        q = 1 - p
        kelly_edge = max(0, (p - q))  # simplified kelly for binary markets

        # Composite confidence score (weighted metrics)
        weights = self.config["confidence_weights"]

        momentum_conf = min(1.0, momentum / 2.0)  # 0-2 momentum → 0-1 confidence
        spread_conf = 1.0 - (spread / 0.5)  # closer to 0.5 → higher confidence
        zscore_conf = max(0, 1.0 - (zscore / 3.0))  # lower Z-score → higher confidence
        volatility_conf = max(0, 1.0 - (volatility * 5))  # lower volatility → higher confidence
        liquidity_conf = min(1.0, liquidity / 100000)  # more liquidity → higher confidence

        composite_confidence = (
            momentum_conf * weights["momentum"] +
            spread_conf * weights["spread"] +
            zscore_conf * weights["zscore"] +
            volatility_conf * weights["volatility"] +
            liquidity_conf * weights["liquidity"]
        )

        # Anomaly score: delta dynamics + z-score
        anomaly = min(1.0, zscore / 3.0)  # base
        if delta_dyn.delta_rolling_std > 0:
            anomaly = min(1.0, max(anomaly, delta_dyn.delta_rolling_std * 10))

        return MarketEnrichment(
            momentum_score=momentum,
            spread_distance=spread,
            zscore=zscore,
            volatility=volatility,
            liquidity_depth=liquidity,
            kelly_edge=kelly_edge,
            signal_confidence=composite_confidence,
            anomaly_score=anomaly
        )

    def _calculate_delta_dynamics(self, price_history: List[float]) -> DeltaDynamics:
        if len(price_history) < 2:
            return DeltaDynamics(0.0, 0.0, 0.0)
        deltas = [price_history[i] - price_history[i - 1] for i in range(1, len(price_history))]
        recent = deltas[-50:] if len(deltas) >= 50 else deltas
        delta = recent[-1] if recent else 0.0
        delta_velocity = recent[-1] - recent[-2] if len(recent) >= 2 else 0.0
        delta_rolling_std = float(statistics.stdev(recent)) if len(recent) > 1 else 0.0
        return DeltaDynamics(delta, delta_velocity, delta_rolling_std)

    def _detect_regime(self, price_history: List[float], delta_rolling_std: float) -> str:
        if len(price_history) < 3:
            return "unknown"
        deltas = [price_history[i] - price_history[i - 1] for i in range(1, len(price_history))]
        recent = deltas[-10:] if len(deltas) >= 10 else deltas
        trend = sum(recent) / max(len(recent), 1)
        if delta_rolling_std > 0.05:
            return "regime_shift" if abs(trend) > 0.02 else "volatile"
        if abs(trend) > 0.01:
            return "trending"
        return "stable"

    def _rate_signal(self, enrichment: MarketEnrichment) -> str:
        """Rate signal quality: HIGH, ELEVATED, NORMAL, or LOW"""
        conf = enrichment.signal_confidence

        if conf >= 0.8 and enrichment.anomaly_score >= 0.5:
            return "HIGH"
        elif conf >= 0.65:
            return "ELEVATED"
        elif conf >= 0.4:
            return "NORMAL"
        else:
            return "LOW"

    def _generate_reasoning(self, enrichment: MarketEnrichment, payload: Dict) -> str:
        """Generate human-readable reasoning for enrichment"""
        factors = []

        if enrichment.momentum_score > 1.5:
            factors.append(f"High momentum (24h/7d={enrichment.momentum_score:.2f})")
        elif enrichment.momentum_score < 0.5:
            factors.append(f"Low momentum (24h/7d={enrichment.momentum_score:.2f})")

        if enrichment.spread_distance > 0.3:
            factors.append(f"Significant price deviation ({enrichment.spread_distance:.2%})")
        elif enrichment.spread_distance < 0.1:
            factors.append("Price near consensus (0.5)")

        if enrichment.zscore > 2.0:
            factors.append(f"Statistical anomaly (Z={enrichment.zscore:.2f})")

        if enrichment.volatility > 0.1:
            factors.append(f"High volatility ({enrichment.volatility:.2%})")

        if enrichment.liquidity_depth > 50000:
            factors.append("Strong liquidity depth")

        if enrichment.kelly_edge > 0.1:
            factors.append(f"Positive kelly edge ({enrichment.kelly_edge:.2%})")

        if not factors:
            return "Market signal within normal parameters"

        return " + ".join(factors)

    def process_event_queue(self, max_events: int = 100):
        """
        Read raw events from event-queue, curate them, write curated events.

        Called periodically by daemon loop.
        """
        if not self.event_queue_path.exists():
            logger.warning(f"Event queue not found: {self.event_queue_path}")
            return

        processed = 0
        scanned = 0
        curated = 0

        try:
            with open(self.event_queue_path, 'r') as f:
                # Seek to last checkpoint offset
                f.seek(self.queue_offset, 0)

                max_scan = int(self.config.get("max_scan_lines", 2000))

                while True:
                    if processed >= max_events or scanned >= max_scan:
                        break

                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break

                    try:
                        event = json.loads(line.strip())
                        # Boundary parse: fail-closed with metrics
                        event = safe_parse_synapse(event, "market_signal_curator")
                        if event is None:
                            continue
                        scanned += 1
                        self.queue_offset = pos + len(line)

                        # Only process raw signal events
                        if not event.event_type.endswith(".raw"):
                            continue
                        processed += 1

                        # Apply source filter if configured
                        if self.source_filters:
                            ev_type = event.event_type
                            if not any(fnmatch.fnmatch(ev_type, pat) for pat in self.source_filters):
                                continue

                        # Curate the event
                        curated_event = self.curate_raw_event(event)

                        if curated_event:
                            self._emit_curated_event(curated_event)
                            curated += 1

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing event: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error reading event queue: {e}")

        self._save_checkpoint()
        logger.info(f"Process cycle: {processed} processed, {curated} curated (scanned {scanned})")

    def _emit_curated_event(self, curated_event: Dict):
        """Emit curated event via configured transport (NATS preferred)."""
        if self.emit_mode == "nats":
            self._emit_curated_event_nats(curated_event)
        else:
            self._emit_curated_event_queue(curated_event)

    def _emit_curated_event_queue(self, curated_event: Dict):
        """Append curated event to event-queue.jsonl (fallback)."""
        try:
            with open(self.event_queue_path, 'a') as f:
                f.write(json.dumps(curated_event) + '\n')
        except Exception as e:
            logger.error(f"Error emitting curated event: {e}")

    def _emit_curated_event_nats(self, curated_event: Dict):
        """Publish curated event to NATS JetStream."""
        try:
            synapse_event = self._to_synapse_event(curated_event)
            payload = json.dumps(synapse_event).encode()
            subject = self._subject_for_event(self.default_severity)
            self._loop.run_until_complete(self._publish_nats(subject, payload))
        except Exception as e:
            logger.error(f"Error publishing to NATS: {e}")

    def _subject_for_event(self, severity: str) -> str:
        context_layer = "halo"
        if severity.lower() in ("warning",):
            context_layer = "aura"
        elif severity.lower() in ("critical", "error"):
            context_layer = "core"
        return f"{self.subject_prefix}.{self.source_agent_id}.{context_layer}"

    def _to_synapse_event(self, curated_event: Dict) -> Dict:
        self.sequence += 1
        return {
            "id": curated_event["id"],
            "event_type": curated_event["event_type"],
            "source": self.source_agent_id,
            "severity": self.default_severity,
            "message": "",
            "timestamp": curated_event["timestamp"],
            "vector_clock": {self.source_agent_id: self.sequence},
            "metadata": {
                "state": "ready",
                "payload": curated_event["payload"],
            },
        }

    async def _publish_nats(self, subject: str, payload: bytes):
        try:
            if self._nats is None or self._nats.is_closed:
                import nats  # type: ignore
                self._nats = await nats.connect(self.nats_url, connect_timeout=2)
                self._nats_js = self._nats.jetstream()
            await self._nats_js.publish(subject, payload)
        except Exception as e:
            logger.error(f"NATS publish failed: {e}")

    def run_daemon(self, interval: int = 30):
        """Run as daemon, processing events periodically"""
        logger.info(f"Curator daemon starting (interval={interval}s)")

        try:
            while True:
                self.process_event_queue(max_events=50)
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Curator daemon stopping")
        except Exception as e:
            logger.error(f"Daemon error: {e}", exc_info=True)


if __name__ == "__main__":
    import sys

    # Usage: python market_signal_curator.py [--daemon|--once]
    mode = sys.argv[1] if len(sys.argv) > 1 else "--once"

    curator = MarketSignalCurator()

    if mode == "--daemon":
        curator.run_daemon(interval=30)
    else:  # --once
        curator.process_event_queue(max_events=100)
        logger.info("One-time curation complete")
