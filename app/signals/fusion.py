"""Optional external signal ingestion and fusion for trading estimates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import logging
import time
from datetime import datetime, timezone
import re

from app.config import ConfigManager
from app.models import FairValueEstimate, MarketData

logger = logging.getLogger(__name__)


def _normalize_recommendation(value: Optional[str]) -> Optional[str]:
    """Normalize curated/market-feed recommendation labels."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip().upper()
    if not cleaned:
        return None
    cleaned = cleaned.replace("_SIGNAL", "")
    if cleaned == "MEDIUM":
        return "ELEVATED"
    return cleaned


def _coerce_probability(raw: Any) -> Optional[float]:
    """Parse confidence-like numbers and normalize to [0, 1]."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return min(value / 100.0, 1.0) if value > 1.0 else value


def _coerce_nonnegative(raw: Any) -> Optional[float]:
    """Parse 0..1 signals while preserving sign when useful."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not (abs(value) >= 0):
        return None
    return value


def _coerce_timestamp(raw: Any) -> float:
    """Parse ISO timestamps to unix time, fallback to now."""
    if isinstance(raw, (int, float)):
        return float(raw)

    if isinstance(raw, str):
        try:
            text = raw.strip().replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
            return parsed.timestamp()
        except Exception:
            pass
    return time.time()


def _safe_list(raw: Any) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [str(raw)]


def _normalize_market_id(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    return value


def _extract_market_id(*candidates: Any) -> Optional[str]:
    for candidate in candidates:
        candidate_id = _normalize_market_id(candidate)
        if candidate_id:
            return candidate_id
    return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clamp_edge(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _parse_recommendation_from_event_type(event_type: str) -> Optional[str]:
    if not event_type:
        return None
    et = event_type.lower()
    if "hold" in et:
        return "LOW"
    if "sell" in et or "long" in et:
        return "HIGH"
    if "buy" in et or "strong" in et:
        return "HIGH"
    return None


def _severity_to_confidence(severity: Optional[str]) -> Optional[float]:
    if not severity:
        return None
    lvl = str(severity).strip().lower()
    if lvl == "critical":
        return 0.95
    if lvl == "warning":
        return 0.70
    if lvl == "info":
        return 0.40
    return None


def _read_jsonl_tail(path: Path, max_lines: int) -> List[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            if max_lines and max_lines > 0:
                return list(deque(handle, maxlen=max_lines))
            return [line for line in handle]
    except FileNotFoundError:
        return []
    except Exception as exc:
        logger.debug("Failed to read signal source file %s: %s", path, exc)
        return []


def _parse_json(value: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(value)
        if isinstance(payload, dict):
            return payload
    except (TypeError, json.JSONDecodeError):
        return None
    return None


@dataclass
class MarketSignalFeature:
    """Normalized signal feature attached to a market id."""

    market_id: str
    provider: str
    confidence: Optional[float] = None
    signal_score: Optional[float] = None
    anomaly: Optional[float] = None
    volatility: Optional[float] = None
    regime: Optional[str] = None
    recommendation: Optional[str] = None
    reason: str = ""
    tags: List[str] = field(default_factory=list)
    captured_at: float = field(default_factory=time.time)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fresh(self) -> bool:
        return self.captured_at > 0

    def as_report_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "provider": self.provider,
            "confidence": self.confidence,
            "signal_score": self.signal_score,
            "anomaly": self.anomaly,
            "volatility": self.volatility,
            "regime": self.regime,
            "recommendation": self.recommendation,
            "reason": self.reason,
            "tags": self.tags,
            "captured_at": self.captured_at,
        }


class _JsonlReader:
    """Base reader for local JSONL event export files."""

    def __init__(self, path: Optional[str], provider: str, cfg: ConfigManager) -> None:
        self.path = Path(path).expanduser() if path else None
        self.provider = provider
        self.cfg = cfg
        self._cache: Dict[str, MarketSignalFeature] = {}
        self._cache_until = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.path) and self.path.exists()

    def read(self, market_ids: Set[str]) -> Dict[str, MarketSignalFeature]:
        now = time.time()
        ttl = int(self.cfg.signal_fusion.ttl_seconds)
        if ttl > 0 and now < self._cache_until and self._cache:
            if not market_ids:
                return dict(self._cache)
            return {
                mid: item
                for mid, item in self._cache.items()
                if mid in market_ids
            }

        if not self.enabled:
            return {}

        max_age = float(self.cfg.signal_fusion.max_event_age_seconds)
        max_lines = int(self.cfg.signal_fusion.max_scan_lines)
        parsed: Dict[str, MarketSignalFeature] = {}

        for raw_line in _read_jsonl_tail(self.path, max_lines=max_lines):
            raw = _parse_json(raw_line)
            if not raw:
                continue

            feature = self.extract_feature(raw)
            if not feature:
                continue
            if max_age and (now - feature.captured_at) > max_age:
                continue
            if market_ids and feature.market_id not in market_ids:
                continue

            existing = parsed.get(feature.market_id)
            if not existing or feature.captured_at >= existing.captured_at:
                parsed[feature.market_id] = feature

        self._cache = parsed
        self._cache_until = now + ttl if ttl > 0 else 0.0
        return dict(parsed)

    def extract_feature(self, event: Dict[str, Any]) -> Optional[MarketSignalFeature]:
        raise NotImplementedError


class CuratorSignalReader(_JsonlReader):
    """Reader for curator enriched outputs and payloads."""

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg.signal_fusion.curator_jsonl_path, "curator", cfg)

    def extract_feature(self, event: Dict[str, Any]) -> Optional[MarketSignalFeature]:
        event_payload = event.get("payload", event)
        if not isinstance(event_payload, dict):
            return None
        event_type = str(event.get("event_type", "")).lower()
        if "curated" not in event_type and "curator" not in str(self.provider):
            # Accept both old and new curator stream shapes.
            return None

        market_id = _extract_market_id(
            event_payload.get("market_id"),
            event_payload.get("ticker"),
            event.get("market_id"),
            event.get("ticker"),
            event_payload.get("id"),
        )
        if not market_id:
            return None

        enrichment = event_payload.get("enrichment", event_payload)
        if not isinstance(enrichment, dict):
            enrichment = {}

        rec_raw = event_payload.get("recommendation")
        if isinstance(rec_raw, str):
            recommendation = _normalize_recommendation(rec_raw)
        else:
            recommendation = _normalize_recommendation(str(rec_raw)) if rec_raw else None
        confidence = _coerce_probability(
            enrichment.get("signal_confidence")
            or enrichment.get("confidence")
            or event_payload.get("signal_confidence")
            or event_payload.get("confidence"),
        )
        anomaly = _coerce_probability(
            enrichment.get("anomaly_score")
            or enrichment.get("anomaly")
            or event.get("anomaly"),
        )
        signal_score = _coerce_nonnegative(
            enrichment.get("kelly_edge")
            or enrichment.get("signal_score")
            or enrichment.get("momentum_score"),
        )
        if signal_score is not None and signal_score > 1:
            signal_score = _coerce_probability(signal_score)
        volatility = _coerce_nonnegative(
            enrichment.get("volatility") or event_payload.get("volatility"),
        )
        regime = _normalize_recommendation(
            event_payload.get("regime")
        ) if isinstance(event_payload.get("regime"), str) else _normalize_recommendation(enrichment.get("regime"))
        if not regime and isinstance(event_payload.get("market_regime"), str):
            regime = _normalize_recommendation(event_payload.get("market_regime"))

        tags = _safe_list(event_payload.get("tags") or enrichment.get("tags"))
        reason = str(event_payload.get("reasoning") or event_payload.get("reason") or "")
        raw = {
            "event_type": event.get("event_type"),
            "source": event.get("source"),
            "message": event.get("message"),
        }

        timestamp = _coerce_timestamp(event.get("timestamp") or event_payload.get("timestamp"))

        return MarketSignalFeature(
            market_id=market_id,
            provider="curator",
            confidence=confidence,
            signal_score=signal_score,
            anomaly=anomaly,
            volatility=volatility,
            regime=regime,
            recommendation=recommendation,
            reason=reason,
            tags=tags,
            captured_at=timestamp,
            raw=raw,
        )


class MarketFeedSignalReader(_JsonlReader):
    """Reader for demo market-feed exports."""

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg.signal_fusion.market_feed_jsonl_path, "market_feed", cfg)

    def extract_feature(self, event: Dict[str, Any]) -> Optional[MarketSignalFeature]:
        event_type = str(event.get("event_type", "")).lower()
        event_payload = event.get("payload", event)
        if not isinstance(event_payload, dict):
            return None

        # Focus on actionable signal events; curator snapshots can be useful for
        # anomaly mapping but they are too noisy for default fusion.
        if (
            "signal.suggestion" not in event_type
            and event.get("source", "").startswith("market-feed")
            and not event_payload.get("signal")
            and event_payload.get("severity") not in {"critical", "warning"}
        ):
            return None

        market_id = _extract_market_id(
            event_payload.get("market_id"),
            event_payload.get("ticker"),
            event_payload.get("condition_id"),
            event_payload.get("conditionId"),
            event_payload.get("market_slug"),
            event_payload.get("ticker_symbol"),
            event_payload.get("event_ticker"),
            event_payload.get("symbol"),
        )
        if not market_id and event_type:
            # Polymarket-like identifiers can be embedded in event type or message.
            match = re.search(r"([A-Za-z0-9_-]{4,})", event_type or "")
            if match:
                market_id = match.group(1)
        if not market_id:
            return None

        confidence = _coerce_probability(
            event_payload.get("confidence")
            or event_payload.get("signal_confidence")
            or event.get("confidence")
            or event.get("signal_confidence"),
        )
        if confidence is None:
            confidence = _severity_to_confidence(event_payload.get("severity"))

        if confidence is None and event_type:
            rec_guess = _parse_recommendation_from_event_type(event_type)
            if rec_guess == "HIGH":
                confidence = 0.60 if "strong" not in event_type else 0.80
            elif rec_guess == "LOW":
                confidence = 0.20
            recommendation = rec_guess
        else:
            recommendation = _normalize_recommendation(_parse_recommendation_from_event_type(event_type))

        anomaly = _coerce_probability(
            event_payload.get("anomaly")
            or event_payload.get("anomaly_score")
            or event_payload.get("derived", {}).get("anomaly_score")
        )
        if anomaly is None and event_payload.get("severity") in {"warning", "critical"}:
            anomaly = 0.85 if event_payload.get("severity") == "critical" else 0.50

        signal_score = _coerce_probability(
            event_payload.get("signal_score")
            or event_payload.get("score")
            or event_payload.get("derived", {}).get("score"),
        )

        tags = _safe_list(
            event_payload.get("tags") or event_payload.get("derived", {}).get("tags")
        )
        reason = str(event.get("message") or event_payload.get("message") or "")
        timestamp = _coerce_timestamp(
            event.get("timestamp")
            or event_payload.get("timestamp")
            or event_payload.get("time")
        )

        return MarketSignalFeature(
            market_id=market_id,
            provider="market_feed",
            confidence=confidence,
            signal_score=signal_score,
            anomaly=anomaly,
            volatility=None,
            regime=None,
            recommendation=recommendation,
            reason=reason,
            tags=tags,
            captured_at=timestamp,
            raw={"event_type": event_type},
        )


class SignalFusionService:
    """Apply curator/feed signals to LLM estimates before strategy decisions."""

    def __init__(self, config: ConfigManager) -> None:
        self.config = config
        self.cfg = config.signal_fusion
        self.enabled = config.signal_fusion_enabled
        self.mode = self.cfg.mode
        self.curator_reader: Optional[CuratorSignalReader] = None
        self.market_feed_reader: Optional[MarketFeedSignalReader] = None

        if self.enabled:
            if self.mode in {"curator", "both"} and self.cfg.curator_jsonl_path:
                self.curator_reader = CuratorSignalReader(config)
                if not self.curator_reader.enabled:
                    logger.debug(
                        "Curator signal reader path missing: %s",
                        self.cfg.curator_jsonl_path,
                    )
            if self.mode in {"market_feed", "both"} and self.cfg.market_feed_jsonl_path:
                self.market_feed_reader = MarketFeedSignalReader(config)
                if not self.market_feed_reader.enabled:
                    logger.debug(
                        "Market feed signal reader path missing: %s",
                        self.cfg.market_feed_jsonl_path,
                    )

    def report_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "curator_enabled": bool(self.curator_reader and self.curator_reader.enabled),
            "market_feed_enabled": bool(
                self.market_feed_reader and self.market_feed_reader.enabled
            ),
        }

    def apply(self, estimates: List[FairValueEstimate], markets_by_id: Optional[Dict[str, MarketData]]) -> Tuple[List[FairValueEstimate], Dict[str, MarketSignalFeature]]:
        """
        Enrich estimates in place and return both output list and matched feature map.

        Returns:
            (enriched_estimates, used_features_by_market_id)
        """
        if not self.enabled:
            return estimates, {}

        estimate_market_ids = {e.market_id for e in estimates}
        if not estimate_market_ids:
            return estimates, {}

        features: Dict[str, MarketSignalFeature] = {}

        if self.curator_reader:
            for mid, feature in self.curator_reader.read(estimate_market_ids).items():
                features[mid] = feature

        if self.market_feed_reader:
            for mid, feature in self.market_feed_reader.read(estimate_market_ids).items():
                # prefer market feed signal only when curator signal is absent
                if mid not in features:
                    features[mid] = feature

        enriched: Dict[str, MarketSignalFeature] = {}
        for estimate in estimates:
            feature = features.get(estimate.market_id)
            if not feature:
                continue
            self._apply_feature_to_estimate(estimate, feature, markets_by_id.get(estimate.market_id) if markets_by_id else None)
            enriched[estimate.market_id] = feature

        return estimates, enriched

    def _apply_feature_to_estimate(
        self,
        estimate: FairValueEstimate,
        feature: MarketSignalFeature,
        market: Optional[MarketData],
    ) -> None:
        base_prob = estimate.estimated_probability
        base_conf = estimate.confidence_level
        base_edge = estimate.edge

        conf = float(base_conf)
        edge = float(base_edge)
        score = feature.confidence
        notes: List[str] = []

        if score is not None:
            weight = float(self.cfg.signal_score_weight)
            conf = (1.0 - weight) * conf + weight * _clamp01(score)
            notes.append(f"score_blend:{weight:.2f}")

        # Recommendation-based adjustment
        recommendation = feature.recommendation or "NORMAL"
        if recommendation == "HIGH":
            boost = (
                self.cfg.curator_boost
                if feature.provider == "curator"
                else self.cfg.market_feed_boost
            )
            conf = _clamp01(conf * (1.0 + boost))
            edge = _clamp_edge(edge * (1.0 + boost))
            notes.append(f"rec_high:+{boost:.2f}")
        elif recommendation == "ELEVATED":
            half_boost = (
                self.cfg.curator_boost * 0.45
                if feature.provider == "curator"
                else self.cfg.market_feed_boost * 0.5
            )
            conf = _clamp01(conf * (1.0 + half_boost))
            notes.append(f"rec_elev:+{half_boost:.2f}")
        elif recommendation == "LOW":
            penalty = self.cfg.low_recommendation_penalty
            conf = _clamp01(conf * (1.0 - penalty))
            edge = _clamp_edge(edge * (1.0 - penalty))
            notes.append(f"rec_low:-{penalty:.2f}")

        # Optional anomaly dampening
        anomaly = feature.anomaly
        if anomaly is not None:
            if anomaly >= self.cfg.max_anomaly_veto:
                conf = min(conf, 0.20)
                edge = _clamp_edge(edge * 0.2)
                notes.append("anomaly_veto")
            else:
                scale = _clamp01(1.0 - (anomaly * self.cfg.anomaly_penalty_factor))
                conf = _clamp01(conf * scale)
                edge = _clamp_edge(edge * scale)
                if anomaly >= 0.35:
                    notes.append(f"anomaly:{anomaly:.2f}")

        # Keep estimates bounded and internally consistent
        conf = _clamp01(conf)
        edge = _clamp_edge(edge)
        fused_prob = _clamp01(base_prob + (edge - base_edge))

        estimate.fused_confidence = conf
        estimate.fused_edge = edge
        estimate.fused_probability = fused_prob
        estimate.feature_confidence = feature.confidence
        estimate.feature_signal_score = feature.signal_score
        estimate.feature_anomaly = feature.anomaly
        estimate.feature_recommendation = recommendation
        estimate.feature_regime = feature.regime
        estimate.feature_provider = feature.provider
        estimate.fusion_metadata = {
            "feature": feature.as_report_dict(),
            "applied_rules": notes,
            "market_id": estimate.market_id,
            "market_ticker": market.market_id if market else None,
        }
        estimate.fusion_tags = list({*estimate.fusion_tags, *feature.tags})
