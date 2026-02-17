#!/usr/bin/env python3
"""
Market Reflex Daemon

Subscribes to curated market signals and emits hypothetical trade intents.
This is an interpretation-layer reflex arc for testing decision discipline
without execution.
"""

import asyncio
import json
import logging
from curator_adapters import SynapseEvent
from event_boundary import safe_parse_synapse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import uuid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class ReflexDaemon:
    def __init__(self, config: Dict):
        self.config = config
        self.nats_url = config.get("nats_url", "nats://localhost:4222")
        self.subject = config.get("subscribe_subject", "nabi.events.market_curator.*")
        self.stream_name = config.get("stream_name", "SYNAPSE_EVENTS")
        self.durable_name = config.get("durable_name", "market-reflex-consumer")
        self.subject_prefix = config.get("subject_prefix", "nabi.events")
        self.source_agent_id = config.get("source_agent_id", "market_reflex")
        self.severity = config.get("severity", "info")
        self.intent_event_type = config.get("intent_event_type", "trade.intent.hypo")
        self.min_confidence = float(config.get("min_confidence", 0.65))
        self.min_anomaly = float(config.get("min_anomaly", 0.40))
        self.policy_subject = config.get("policy_subject", "nabi.events.reflex_policy.halo")
        self._feature_weights: Dict[str, float] = {}
        self.max_cache = int(config.get("max_cache", 5000))
        self.telemetry_path = Path(
            config.get(
                "telemetry_path",
                str(Path.home() / ".local/state/nabi/curator/reflex_telemetry.jsonl"),
            )
        ).expanduser()
        self._seen = {}
        self._seq = 0

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _should_emit(self, payload: Dict) -> bool:
        enrichment = payload.get("enrichment", {})
        confidence = float(enrichment.get("signal_confidence", 0))
        anomaly = float(enrichment.get("anomaly_score", 0))

        conf_w = float(self._feature_weights.get("signal_confidence", 1.0))
        anom_w = float(self._feature_weights.get("anomaly_score", 1.0))

        confidence_eff = confidence * conf_w
        anomaly_eff = anomaly * anom_w

        return confidence_eff >= self.min_confidence and anomaly_eff >= self.min_anomaly

    def _dedupe(self, parent_id: Optional[str]) -> bool:
        if not parent_id:
            return False
        if parent_id in self._seen:
            return True
        self._seen[parent_id] = self._now()
        if len(self._seen) > self.max_cache:
            for k in list(self._seen.keys())[: self.max_cache // 5]:
                self._seen.pop(k, None)
        return False

    def _to_intent_event(self, payload: Dict, parent_id: Optional[str]) -> Dict:
        self._seq += 1
        return {
            "id": f"evt-{uuid.uuid4()}",
            "event_type": self.intent_event_type,
            "source": self.source_agent_id,
            "severity": self.severity,
            "message": "",
            "timestamp": self._now(),
            "vector_clock": {self.source_agent_id: self._seq},
            "metadata": {
                "state": "ready",
                "payload": {
                    "market_id": payload.get("market_id"),
                    "title": payload.get("title"),
                    "recommendation": payload.get("recommendation"),
                    "enrichment": payload.get("enrichment"),
                    "parent_id": parent_id,
                },
            },
        }

    def _emit_telemetry(self, payload: Dict, parent_id: Optional[str], fired: bool, reason: str):
        try:
            enrichment = payload.get("enrichment", {})
            entry = {
                "timestamp": self._now(),
                "market_id": payload.get("market_id"),
                "title": payload.get("title"),
                "parent_id": parent_id,
                "signal_confidence": enrichment.get("signal_confidence"),
                "anomaly_score": enrichment.get("anomaly_score"),
                "recommendation": payload.get("recommendation"),
                "fired": fired,
                "reason": reason,
                "thresholds": {
                    "min_confidence": self.min_confidence,
                    "min_anomaly": self.min_anomaly,
                },
            }
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with self.telemetry_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning("Failed to write telemetry: %s", e)

    async def _publish(self, js, subject: str, event: Dict):
        data = json.dumps(event).encode()
        try:
            await js.publish(subject, data)
        except Exception as e:
            logger.warning("JetStream publish failed (%s), falling back to core NATS", e)
            await js._nc.publish(subject, data)

    async def _subscribe_policy(self, nc):
        async def _handler(msg):
            try:
                data = json.loads(msg.data.decode())
                # Boundary parse: fail-closed with metrics
                data = safe_parse_synapse(data, "market_reflex_policy")
                if data is None:
                    return
                payload = (data.metadata or {}).get("payload") or {}
                thresholds = payload.get("thresholds") or {}
                weights = payload.get("feature_weights") or {}
                if "min_confidence" in thresholds:
                    self.min_confidence = float(thresholds.get("min_confidence", self.min_confidence))
                if "min_anomaly" in thresholds:
                    self.min_anomaly = float(thresholds.get("min_anomaly", self.min_anomaly))
                if isinstance(weights, dict):
                    self._feature_weights = {k: float(v) for k, v in weights.items()}
                logger.info(
                    "Policy update: min_conf=%.3f min_anom=%.3f weights=%d",
                    self.min_confidence,
                    self.min_anomaly,
                    len(self._feature_weights),
                )
            except Exception as e:
                logger.warning("Failed to parse policy snapshot: %s", e)

        await nc.subscribe(self.policy_subject, cb=_handler)

    async def run(self):
        import nats  # type: ignore

        logger.info("Connecting to NATS at %s", self.nats_url)
        nc = await nats.connect(self.nats_url, connect_timeout=2)
        js = nc.jetstream()

        await self._subscribe_policy(nc)
        logger.info("Policy subscription active: %s", self.policy_subject)
        logger.info("Subscribing to JetStream %s subject %s", self.stream_name, self.subject)
        sub = await js.pull_subscribe(
            self.subject,
            durable=self.durable_name,
            stream=self.stream_name,
        )

        while True:
            try:
                msgs = await sub.fetch(100, timeout=1)
            except Exception:
                await asyncio.sleep(0.2)
                continue
            if not msgs:
                await asyncio.sleep(0.2)
                continue
            logger.info("Fetched %d curated messages", len(msgs))
            for msg in msgs:
                try:
                    data = json.loads(msg.data.decode())
                    # Boundary parse: fail-closed with metrics
                    data = safe_parse_synapse(data, "market_reflex")
                    if data is None:
                        await msg.ack()
                        continue
                except Exception:
                    await msg.ack()
                    continue

                meta = data.metadata or {}
                payload = meta.get("payload") or {}
                if not isinstance(payload, dict):
                    await msg.ack()
                    continue

                parent_id = None
                curator_meta = payload.get("_curator", {})
                if isinstance(curator_meta, dict):
                    parent_id = curator_meta.get("parent_id")

                if self._dedupe(parent_id):
                    await msg.ack()
                    continue

                enrichment = payload.get("enrichment", {})
                confidence = float(enrichment.get("signal_confidence", 0))
                anomaly = float(enrichment.get("anomaly_score", 0))

                fired = False
                reason = ""
                if self._should_emit(payload):
                    intent = self._to_intent_event(payload, parent_id)
                    await self._publish(js, f"{self.subject_prefix}.market_reflex.intent", intent)
                    fired = True
                    reason = "intent-emitted"
                else:
                    reason = f"below-threshold conf={confidence:.3f} anom={anomaly:.3f}"

                self._emit_telemetry(payload, parent_id, fired, reason)
                await msg.ack()


def main():
    config_path = Path.home() / ".config/nabi/market-reflex.toml"
    if config_path.exists():
        try:
            import tomllib
        except Exception:
            tomllib = None
        config = tomllib.loads(config_path.read_text()) if tomllib else {}
    else:
        config = {}
    daemon = ReflexDaemon(config)
    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
