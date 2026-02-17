#!/usr/bin/env python3
"""Feature Authority Engine (substrate-agnostic).

This daemon:
- Consumes curated snapshots from NATS JetStream (`SYNAPSE_EVENTS`)
- Persists ML-ready `market_observation` rows to SurrealDB
- Computes cheap hourly evidence and updates feature×regime authority
- Emits `reflex.policy.snapshot` events for reflex hot-swap

Interpretation-layer component (ADR-0051): it does not constitute truth.
"""

import asyncio
import json
import logging
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from curator_adapters import SynapseEvent
from event_boundary import safe_parse_synapse

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)


@dataclass
class AuthorityConfig:
    update_interval_minutes: int
    lag_minutes: int
    half_life_hours: float
    k_good: float
    k_bad: float


@dataclass
class ReflexConfig:
    base_min_confidence: float
    base_min_anomaly: float
    alpha: float
    beta: float


@dataclass
class KernelReviewConfig:
    enabled: bool
    endpoint: str
    timeout_seconds: float
    interval_minutes: int
    influence: float
    cooldown_seconds: float


class FeatureAuthorityDaemon:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()

        nats_cfg = self.config.get("nats", {})
        self.nats_url = nats_cfg.get("url", "nats://localhost:4222")
        self.stream = nats_cfg.get("stream", "SYNAPSE_EVENTS")
        self.subjects = nats_cfg.get("subjects", ["nabi.events.market_curator.*"])

        surreal = self.config.get("surreal", {})
        self.surreal_endpoint = surreal.get("endpoint", "http://127.0.0.1:8284")
        self.surreal_namespace = surreal.get("namespace", "nabi")
        self.surreal_database = surreal.get("database", "substrate")
        self.surreal_username = surreal.get("username", "temp123")
        self.surreal_password = surreal.get("password", "Alliswellthatendswell")

        auth = self.config.get("authority", {})
        self.authority = AuthorityConfig(
            update_interval_minutes=int(auth.get("update_interval_minutes", 60)),
            lag_minutes=int(auth.get("lag_minutes", 60)),
            half_life_hours=float(auth.get("half_life_hours", 72)),
            k_good=float(auth.get("k_good", 0.08)),
            k_bad=float(auth.get("k_bad", 0.12)),
        )

        reflex = self.config.get("reflex", {})
        self.reflex = ReflexConfig(
            base_min_confidence=float(reflex.get("base_min_confidence", 0.65)),
            base_min_anomaly=float(reflex.get("base_min_anomaly", 0.40)),
            alpha=float(reflex.get("alpha", 0.15)),
            beta=float(reflex.get("beta", 0.15)),
        )

        self.regime_rules = self.config.get("regime", {}).get("rules", [])
        self.feature_names = self.config.get("features", {}).get("names", [])

        out_cfg = self.config.get("output", {})
        self.emit_policy_snapshot = bool(out_cfg.get("emit_policy_snapshot", True))
        self.policy_subject = out_cfg.get("policy_subject", "nabi.events.reflex_policy.halo")
        kernel_cfg = self.config.get("kernel", {})
        self.kernel_review = KernelReviewConfig(
            enabled=bool(kernel_cfg.get("enabled", False)),
            endpoint=str(kernel_cfg.get("endpoint", "http://127.0.0.1:5380/kernel/policy/review")).strip(),
            timeout_seconds=float(kernel_cfg.get("timeout_seconds", 2.5)),
            interval_minutes=int(kernel_cfg.get("interval_minutes", 60)),
            influence=float(kernel_cfg.get("influence", 0.35)),
            cooldown_seconds=float(kernel_cfg.get("cooldown_seconds", 0)),
        )
        if self.kernel_review.interval_minutes <= 0:
            self.kernel_review.interval_minutes = 60

        self._nc = None
        self._js = None
        self._surreal = None
        self._ran_once = False
        self._last_policy: Dict[str, Dict[str, Any]] = {}
        self._last_kernel_review_at: Optional[datetime] = None

    def _load_config(self) -> dict:
        import tomllib

        if not self.config_path.exists():
            return {}
        return tomllib.loads(self.config_path.read_text()) or {}

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _bucket_start(self, ts: datetime) -> datetime:
        return ts.replace(minute=0, second=0, microsecond=0)

    def _decay_rate_per_hr(self) -> float:
        if self.authority.half_life_hours <= 0:
            return 0.0
        return math.log(2.0) / self.authority.half_life_hours

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _should_request_kernel_policy(self, now: datetime, regime_id: str) -> bool:
        if not self.kernel_review.enabled:
            return False
        if not self.kernel_review.endpoint:
            return False
        if self._last_kernel_review_at is None:
            return True
        min_interval = timedelta(
            minutes=self.kernel_review.interval_minutes,
            seconds=self.kernel_review.cooldown_seconds,
        )
        if now - self._last_kernel_review_at >= min_interval:
            return True
        return False

    async def _post_json(self, url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        def _sync_request() -> Optional[Dict[str, Any]]:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "nabi-feature-authority-daemon",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.kernel_review.timeout_seconds) as resp:
                status = getattr(resp, "status", 200)
                if status < 200 or status >= 300:
                    raise RuntimeError(f"Kernel review endpoint returned HTTP {status}")
                body = resp.read().decode("utf-8", errors="replace")
                return json.loads(body) if body else {}

        try:
            return await asyncio.to_thread(_sync_request)
        except Exception as exc:
            logger.debug("Kernel policy review request failed: %s", exc)
            return None

    def _extract_kernel_override(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}

        state = payload.get("state")
        if state and state != "ready":
            return {}

        body = payload
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("payload"), dict):
            body = metadata["payload"]
        elif isinstance(payload.get("payload"), dict):
            body = payload["payload"]

        if not isinstance(body, dict):
            return {}
        if isinstance(body.get("policy"), dict):
            return body["policy"]
        if isinstance(body.get("suggestions"), dict):
            return body["suggestions"]
        return body

    async def _review_policy_with_kernel(
        self,
        regime_id: str,
        regime_conf: float,
        evidence: Dict[str, dict],
        weights: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        now = self._now()
        if not self._should_request_kernel_policy(now, regime_id):
            return None

        payload = {
            "id": f"evt-kernel-review-{int(now.timestamp())}-{regime_id}",
            "event_type": "feature_authority.policy_review_request",
            "source": "feature_authority",
            "severity": "info",
            "timestamp": now.isoformat(),
            "metadata": {
                "state": "ready",
                "payload": {
                    "regime_id": regime_id,
                    "regime_confidence": float(regime_conf),
                    "evidence": evidence,
                    "weights": weights,
                    "request_type": "policy_tuning",
                },
            },
        }

        logger.info("Requesting kernel policy review for regime=%s", regime_id)
        resp = await self._post_json(self.kernel_review.endpoint, payload)
        if resp is None:
            return None

        override = self._extract_kernel_override(resp)
        if not override:
            logger.info("Kernel policy review returned no override for regime=%s", regime_id)
            return None

        self._last_kernel_review_at = now
        return {
            "source": "kernel",
            "raw": override,
        }

    def _classify_regime(self, features: Dict[str, Any]) -> Tuple[str, float]:
        anomaly = float(features.get("anomaly_score", 0))
        delta_std = float(features.get("delta_rolling_std", 0))

        for rule in self.regime_rules:
            rule_id = rule.get("id")
            min_anomaly = rule.get("min_anomaly")
            min_delta_std = rule.get("min_delta_std")

            if min_anomaly is not None and anomaly < float(min_anomaly):
                continue
            if min_delta_std is not None and delta_std < float(min_delta_std):
                continue

            if rule_id == "shock":
                return "shock", min(1.0, anomaly)
            if rule_id == "high_vol":
                denom = float(min_delta_std or 1.0)
                return "high_vol", min(1.0, delta_std / denom) if denom else 0.5
            if rule_id:
                return str(rule_id), 0.5

        return "baseline", 0.5

    def _extract_features(self, payload: Dict[str, Any]) -> Dict[str, float]:
        enrichment = payload.get("enrichment", {}) if isinstance(payload, dict) else {}
        features: Dict[str, float] = {}
        for name in self.feature_names:
            if name in enrichment:
                features[name] = float(enrichment.get(name) or 0.0)
            elif name in payload:
                features[name] = float(payload.get(name) or 0.0)

        if "edge" not in features:
            p_model = payload.get("p_model")
            p_market = payload.get("p_market") or payload.get("yes_price")
            if p_model is not None and p_market is not None:
                try:
                    features["edge"] = float(p_model) - float(p_market)
                except Exception:
                    pass

        return features

    async def _connect_nats(self):
        import nats  # type: ignore

        self._nc = await nats.connect(self.nats_url, connect_timeout=2)
        self._js = self._nc.jetstream()

    async def _connect_surreal(self):
        import surrealdb  # type: ignore

        self._surreal = surrealdb.Surreal(self.surreal_endpoint)
        # surrealdb==1.0.6 uses blocking http with username/password.
        self._surreal.signin({"username": self.surreal_username, "password": self.surreal_password})
        self._surreal.use(self.surreal_namespace, self.surreal_database)

    async def _write_market_observation(self, event: dict) -> Optional[dict]:
        meta = event.get("metadata") or {}
        payload = meta.get("payload") or {}
        if not isinstance(payload, dict):
            return None

        event_id = event.get("id")
        if not event_id:
            return None

        ts_observed_raw = (event.get("timestamp") or event.get("time") or self._now().isoformat())
        try:
            ts_observed = datetime.fromisoformat(str(ts_observed_raw).replace("Z", "+00:00"))
        except Exception:
            ts_observed = self._now()

        features = self._extract_features(payload)
        regime_id, regime_conf = self._classify_regime(features)

        record = {
            "event_id": str(event_id),
            "parent_id": payload.get("_curator", {}).get("parent_id"),
            "market_id": payload.get("market_id"),
            "ts_observed": ts_observed.isoformat(),
            "ts_ingested": self._now().isoformat(),
            "source": event.get("source", "unknown"),
            "event_type": event.get("event_type", "unknown"),
            "regime_id": regime_id,
            "regime_confidence": float(regime_conf),
            "features": features,
            "raw": payload,
        }

        self._surreal.create("market_observation", record)
        logger.info("Ingested observation event_id=%s market_id=%s regime=%s", event_id, record.get("market_id"), regime_id)
        return record

    async def _fetch_recent_observations(self, since: datetime, until: datetime) -> List[dict]:
        # Use ts_ingested so we can run cycles immediately on new data, while keeping ts_observed
        # for causal modeling and offline training.
        query = "SELECT * FROM market_observation WHERE ts_ingested >= $since AND ts_ingested < $until"
        result = self._surreal.query(query, {"since": since.isoformat(), "until": until.isoformat()})
        # surrealdb blocking client returns a list of rows, not [{result:...}]
        if isinstance(result, list):
            return result
        if result:
            return result[0].get("result", [])
        return []
        logger.info("Fetch obs: since=%s until=%s result_type=%s", since.isoformat(), until.isoformat(), type(result).__name__)
        if isinstance(result, list):
            logger.info("Fetch obs: list_len=%d first_type=%s", len(result), type(result[0]).__name__ if result else "none")

    def _compute_evidence(self, observations: List[dict]) -> Dict[Tuple[str, str], dict]:
        bins: Dict[Tuple[str, str], List[float]] = {}
        for obs in observations:
            regime_id = obs.get("regime_id", "baseline")
            features = obs.get("features", {}) or {}
            if not isinstance(features, dict):
                continue
            for feature_id, value in features.items():
                key = (str(feature_id), str(regime_id))
                bins.setdefault(key, []).append(float(value or 0.0))

        results: Dict[Tuple[str, str], dict] = {}
        for key, values in bins.items():
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
            std = math.sqrt(variance)

            drift = min(1.0, std * 2.0)
            stability = max(0.0, 1.0 - drift)
            contribution = mean
            redundancy = 0.0
            calibration_error = min(1.0, abs(mean)) * 0.1

            results[key] = {
                "contribution": contribution,
                "drift": drift,
                "stability": stability,
                "redundancy": redundancy,
                "calibration_error": calibration_error,
                "sample_n": len(values),
            }

        return results

    async def _upsert_evidence(self, bucket_start: datetime, bucket_end: datetime, evidence: Dict[Tuple[str, str], dict]):
        for (feature_id, regime_id), metrics in evidence.items():
            record = {
                "feature_id": feature_id,
                "regime_id": regime_id,
                "bucket_start": bucket_start.isoformat(),
                "bucket_end": bucket_end.isoformat(),
                "contribution": metrics["contribution"],
                "drift": metrics["drift"],
                "calibration_error": metrics["calibration_error"],
                "redundancy": metrics["redundancy"],
                "stability": metrics["stability"],
                "sample_n": int(metrics["sample_n"]),
            }
            self._surreal.create("feature_evidence", record)

    async def _get_authority_weight(self, feature_id: str, regime_id: str) -> float:
        query = "SELECT weight FROM feature_regime_authority WHERE feature_id = $fid AND regime_id = $rid LIMIT 1"
        result = self._surreal.query(query, {"fid": feature_id, "rid": regime_id})
        rows = result[0].get("result", []) if result else []
        if rows:
            return float(rows[0].get("weight", 0.5))
        return 0.5

    async def _set_authority_weight(self, feature_id: str, regime_id: str, weight: float, stability: float, reason: str):
        record = {
            "feature_id": feature_id,
            "regime_id": regime_id,
            "weight": float(weight),
            "stability_score": float(stability),
            "affinity": 0.0,
            "updated_by": "feature-authority-daemon",
            "reason": reason,
        }

        # simple upsert: try update then create if missing
        self._surreal.query(
            "UPDATE feature_regime_authority SET weight = $weight, stability_score = $stability, "
            "updated_by = $by, reason = $reason, last_updated_at = time::now() "
            "WHERE feature_id = $fid AND regime_id = $rid;",
            {
                "weight": float(weight),
                "stability": float(stability),
                "by": "feature-authority-daemon",
                "reason": reason,
                "fid": feature_id,
                "rid": regime_id,
            },
        )
        self._surreal.create("feature_regime_authority", record)

    async def _emit_policy_snapshot(
        self,
        regime_id: str,
        regime_conf: float,
        weights: Dict[str, float],
        kernel_override: Optional[Dict[str, Any]] = None,
    ):
        prev = self._last_policy.get(regime_id, {})
        prev_thr = prev.get("thresholds") or {}
        prev_w = prev.get("feature_weights") or {}

        trust = sum(weights.values()) / max(1, len(weights))
        trust_adj = trust * regime_conf

        min_conf = min(1.0, max(0.0, self.reflex.base_min_confidence + self.reflex.alpha * (1.0 - trust_adj)))
        min_anom = min(1.0, max(0.0, self.reflex.base_min_anomaly + self.reflex.beta * (1.0 - trust_adj)))

        kernel_payload: Dict[str, Any] = {}
        if kernel_override and isinstance(kernel_override.get("raw"), dict):
            kernel_payload = kernel_override["raw"]

        if kernel_payload:
            overrides_thresholds = kernel_payload.get("thresholds", {})
            overrides_weights = kernel_payload.get("feature_weights", {})

            if isinstance(overrides_thresholds, dict):
                if "min_confidence" in overrides_thresholds:
                    o_val = float(overrides_thresholds.get("min_confidence", min_conf))
                    min_conf = self._clamp01((1.0 - self.kernel_review.influence) * min_conf + self.kernel_review.influence * o_val)
                if "min_anomaly" in overrides_thresholds:
                    o_val = float(overrides_thresholds.get("min_anomaly", min_anom))
                    min_anom = self._clamp01((1.0 - self.kernel_review.influence) * min_anom + self.kernel_review.influence * o_val)

            if isinstance(overrides_weights, dict):
                blended_weights: Dict[str, float] = dict(weights)
                for feature_id, override_value in overrides_weights.items():
                    if not isinstance(feature_id, str):
                        continue
                    try:
                        val = float(override_value)
                    except Exception:
                        continue
                    base = blended_weights.get(feature_id, 0.0)
                    blended_weights[feature_id] = self._clamp01((1.0 - self.kernel_review.influence) * base + self.kernel_review.influence * val)
                weights = blended_weights

        snapshot = {
            "ts_effective": self._now().isoformat(),
            "trigger_reason": "hourly_update",
            "evidence_window": "last_1h_bucket_lagged",
            "regime_id": regime_id,
            "regime_confidence": float(regime_conf),
            "thresholds": {"min_confidence": float(min_conf), "min_anomaly": float(min_anom)},
            "feature_weights": weights,
            "delta_weights": {k: float(weights.get(k, 0.0)) - float(prev_w.get(k, 0.0)) for k in weights.keys()},
            "delta_thresholds": {
                "min_confidence": float(min_conf) - float(prev_thr.get("min_confidence", min_conf)),
                "min_anomaly": float(min_anom) - float(prev_thr.get("min_anomaly", min_anom)),
            },
            "policy_confidence": float(trust_adj),
            "generated_by": "feature-authority-daemon",
            "policy_source": "kernel" if kernel_override else "local",
            "kernel_override": kernel_payload if kernel_override else {},
        }

        self._surreal.create("reflex_policy_snapshot", snapshot)
        self._last_policy[regime_id] = {"thresholds": snapshot["thresholds"], "feature_weights": snapshot["feature_weights"]}

        if self.emit_policy_snapshot:
            env = {
                "id": f"evt-feature-policy-{int(self._now().timestamp())}-{regime_id}",
                "event_type": "reflex.policy.snapshot",
                "source": "feature_authority",
                "severity": "info",
                "message": "",
                "timestamp": self._now().isoformat(),
                "metadata": {"state": "ready", "payload": snapshot},
            }
            try:
                await self._js.publish(self.policy_subject, json.dumps(env).encode())
                logger.info("Published policy snapshot to JetStream subject=%s id=%s", self.policy_subject, env.get("id"))
            except Exception:
                await self._nc.publish(self.policy_subject, json.dumps(env).encode())
            logger.info("Emitted policy snapshot regime=%s min_conf=%.3f min_anom=%.3f policy_conf=%.3f", regime_id, min_conf, min_anom, trust_adj)

    async def _authority_update_once(self):
        now = self._now()
        lag = timedelta(minutes=self.authority.lag_minutes)
        bucket_end = now
        bucket_start = bucket_end - timedelta(hours=1)

        observations = await self._fetch_recent_observations(bucket_start, bucket_end)
        logger.info("Authority cycle: len(observations)=%d", len(observations))
        logger.info("Authority cycle: observations_type=%s", type(observations).__name__)
        logger.info(
            "Authority cycle: bucket_start=%s bucket_end=%s obs=%d",
            bucket_start.isoformat(),
            bucket_end.isoformat(),
            len(observations),
        )
        if not observations:
            logger.info("No observations in this cycle")
            return

        evidence = self._compute_evidence(observations)
        logger.info("Evidence keys=%d", len(evidence))
        await self._upsert_evidence(bucket_start, bucket_end, evidence)

        decay = self._decay_rate_per_hr()
        dt = 1.0

        evidence_by_regime: Dict[str, Dict[str, dict]] = {}
        weights_by_regime: Dict[str, Dict[str, float]] = {}
        for (feature_id, regime_id), metrics in evidence.items():
            current = await self._get_authority_weight(feature_id, regime_id)
            good = metrics["stability"] * (1.0 - metrics["drift"]) * max(0.0, metrics["contribution"])
            bad = metrics["drift"] + (1.0 - metrics["stability"]) + metrics["calibration_error"] + metrics["redundancy"]

            updated = current * math.exp(-decay * dt) + self.authority.k_good * good - self.authority.k_bad * bad
            updated = max(0.0, min(1.0, updated))
            await self._set_authority_weight(feature_id, regime_id, updated, metrics["stability"], "hourly_update")
            weights_by_regime.setdefault(regime_id, {})[feature_id] = float(updated)
            evidence_by_regime.setdefault(regime_id, {})[feature_id] = {
                "contribution": metrics["contribution"],
                "drift": metrics["drift"],
                "stability": metrics["stability"],
                "redundancy": metrics["redundancy"],
                "calibration_error": metrics["calibration_error"],
                "sample_n": metrics["sample_n"],
            }

        for regime_id, weights in weights_by_regime.items():
            kernel_override: Optional[Dict[str, Any]] = await self._review_policy_with_kernel(
                regime_id,
                0.5,
                evidence_by_regime.get(regime_id, {}),
                weights,
            )
            await self._emit_policy_snapshot(regime_id, 0.5, weights, kernel_override=kernel_override)

    async def _consume_loop(self):
        subject = self.subjects[0] if self.subjects else "nabi.events.market_curator.*"
        sub = await self._js.pull_subscribe(subject, durable="feature-authority-consumer", stream=self.stream)

        while True:
            try:
                msgs = await sub.fetch(50, timeout=1)
            except Exception:
                await asyncio.sleep(0.2)
                continue

            for msg in msgs:
                try:
                    raw = json.loads(msg.data.decode())
                except Exception:
                    await msg.ack()
                    continue

                # Boundary parse: fail-closed with metrics
                syn = safe_parse_synapse(raw, "feature_authority")
                if syn is None:
                    await msg.ack()
                    continue

                try:
                    await self._write_market_observation(syn)
                finally:
                    await msg.ack()

    async def _update_loop(self):
        while True:
            if not self._ran_once:
                try:
                    await self._authority_update_once()
                except Exception as e:
                    logger.warning("Authority update failed (startup): %s", e)
                self._ran_once = True

            try:
                await self._authority_update_once()
            except Exception as e:
                logger.warning("Authority update failed: %s", e)

            await asyncio.sleep(self.authority.update_interval_minutes * 60)

    async def run(self):
        logger.info(
            "FeatureAuthorityDaemon starting: nats_url=%s stream=%s subjects=%s surreal=%s/%s",
            self.nats_url,
            self.stream,
            self.subjects,
            self.surreal_namespace,
            self.surreal_database,
        )
        await self._connect_nats()
        await self._connect_surreal()
        logger.info("SurrealDB connected")
        await asyncio.gather(self._consume_loop(), self._update_loop())


def main():
    config_path = Path.home() / ".config/nabi/feature-authority.toml"
    daemon = FeatureAuthorityDaemon(config_path)
    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
