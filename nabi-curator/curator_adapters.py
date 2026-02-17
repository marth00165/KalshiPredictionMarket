#!/usr/bin/env python3
"""
Curator adapter extensions (self-contained experiment).

Provides pluggable history stores and model estimators without
changing the core curator loop.
"""

import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_market_id(market_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in market_id)


@dataclass(frozen=True)
class SynapseEvent:
    """
    Minimal, schema-aligned event envelope for Python interpretation.

    NOTE: Canonical truth lives in Rust: `NabiEvent` in
    `~/nabia/core/crates/nabi-core/src/event.rs`. Python must treat this
    as a convenience view, not a source of truth.
    """

    id: str
    event_type: str
    timestamp: str
    source: str
    vector_clock: Dict[str, int]
    payload: Dict
    metadata: Optional[Dict] = None

    @classmethod
    def from_dict(cls, raw: Dict) -> "SynapseEvent":
        required = ("id", "event_type", "timestamp", "source")
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise ValueError("Missing required SynapseEvent fields: " + ", ".join(missing))
        return cls(
            id=str(raw.get("id", "")),
            event_type=str(raw.get("event_type", "")),
            timestamp=str(raw.get("timestamp", "")),
            source=str(raw.get("source", "")),
            vector_clock=dict(raw.get("vector_clock", {}) or {}),
            payload=dict(raw.get("payload", {}) or {}),
            metadata=dict(raw.get("metadata", {}) or {}) if raw.get("metadata") is not None else None,
        )


class PriceHistoryStore:
    """Interface for price history persistence."""

    def get_prices(self, market_id: str) -> List[float]:
        raise NotImplementedError

    def append(self, market_id: str, price: float, timestamp: Optional[str] = None) -> None:
        raise NotImplementedError


class MemoryPriceHistoryStore(PriceHistoryStore):
    """In-memory history store (non-persistent)."""

    def __init__(self, max_len: int = 50):
        self.max_len = max_len
        self._cache: Dict[str, Deque[float]] = {}

    def get_prices(self, market_id: str) -> List[float]:
        return list(self._cache.get(market_id, deque()))

    def append(self, market_id: str, price: float, timestamp: Optional[str] = None) -> None:
        if market_id not in self._cache:
            self._cache[market_id] = deque(maxlen=self.max_len)
        self._cache[market_id].append(price)


class JsonlPriceHistoryStore(PriceHistoryStore):
    """
    Persistent JSONL history store.

    Each line: {"timestamp": "...", "price": 0.42}
    """

    def __init__(self, base_dir: Path, max_len: int = 50):
        self.base_dir = base_dir.expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_len = max_len
        self._cache: Dict[str, Deque[float]] = {}

    def _path_for(self, market_id: str) -> Path:
        return self.base_dir / f"{_sanitize_market_id(market_id)}.jsonl"

    def _load_cache(self, market_id: str) -> Deque[float]:
        if market_id in self._cache:
            return self._cache[market_id]
        history = deque(maxlen=self.max_len)
        path = self._path_for(market_id)
        if path.exists():
            try:
                with path.open("r") as f:
                    for line in f:
                        try:
                            rec = json.loads(line.strip())
                            price = rec.get("price")
                            if isinstance(price, (int, float)):
                                history.append(float(price))
                        except Exception:
                            continue
            except Exception:
                pass
        self._cache[market_id] = history
        return history

    def get_prices(self, market_id: str) -> List[float]:
        return list(self._load_cache(market_id))

    def append(self, market_id: str, price: float, timestamp: Optional[str] = None) -> None:
        history = self._load_cache(market_id)
        history.append(price)
        path = self._path_for(market_id)
        rec = {"timestamp": timestamp or _now_iso(), "price": price}
        try:
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass


class ModelEstimator:
    """Interface for independent probability estimation."""

    def estimate(self, market_id: str, prices: List[float], payload: Dict, regime: str) -> Optional[float]:
        raise NotImplementedError


class PayloadModelEstimator(ModelEstimator):
    """Use p_model from payload if present."""

    def estimate(self, market_id: str, prices: List[float], payload: Dict, regime: str) -> Optional[float]:
        p_model = payload.get("p_model")
        if isinstance(p_model, (int, float)):
            return float(p_model)
        return None


class EmaModelEstimator(ModelEstimator):
    """Fallback model: EMA over recent prices (not independent)."""

    def __init__(self, window: int = 50):
        self.window = window

    def estimate(self, market_id: str, prices: List[float], payload: Dict, regime: str) -> Optional[float]:
        if not prices:
            return None
        window_prices = prices[-self.window:]
        n = len(window_prices)
        if n == 1:
            return float(window_prices[0])
        weights = [math.exp(x) for x in [i / max(n - 1, 1) - 1 for i in range(n)]]
        weight_sum = sum(weights)
        ema = sum(p * w for p, w in zip(window_prices, weights)) / weight_sum
        return max(0.01, min(0.99, float(ema)))


class EnsembleModelEstimator(ModelEstimator):
    """Combine multiple estimators with weights."""

    def __init__(self, estimators: List[Tuple[ModelEstimator, float]]):
        self.estimators = estimators

    def estimate(self, market_id: str, prices: List[float], payload: Dict, regime: str) -> Optional[float]:
        weighted = []
        for estimator, weight in self.estimators:
            value = estimator.estimate(market_id, prices, payload, regime)
            if value is not None:
                weighted.append((value, weight))
        if not weighted:
            return None
        weight_sum = sum(w for _, w in weighted)
        if weight_sum <= 0:
            return None
        return sum(v * w for v, w in weighted) / weight_sum


def build_history_store(config: Dict) -> PriceHistoryStore:
    store_type = config.get("history_store", "jsonl")
    max_len = int(config.get("history_window", 50))
    if store_type == "memory":
        return MemoryPriceHistoryStore(max_len=max_len)
    base_dir = Path(config.get("price_history_dir", str(Path.home() / ".local/state/nabi/curator/price_history")))
    return JsonlPriceHistoryStore(base_dir=base_dir, max_len=max_len)


def build_model_estimator(config: Dict) -> ModelEstimator:
    model_cfg = config.get("model_estimator", "ensemble")
    if isinstance(model_cfg, dict):
        model_type = model_cfg.get("type", "ensemble")
    else:
        model_type = model_cfg

    if model_type == "payload":
        return PayloadModelEstimator()
    if model_type == "ema":
        window = int(config.get("model_ema_window", 50))
        return EmaModelEstimator(window=window)

    # Default: ensemble (payload first, EMA fallback)
    window = int(config.get("model_ema_window", 50))
    return EnsembleModelEstimator([
        (PayloadModelEstimator(), 0.7),
        (EmaModelEstimator(window=window), 0.3),
    ])
