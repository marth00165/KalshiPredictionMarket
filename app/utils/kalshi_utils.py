"""Kalshi payload normalization helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional


def normalize_settlement_outcome(market_payload: Dict[str, Any]) -> Optional[str]:
    """
    Normalize settlement outcome to 'yes' or 'no' when determinable.
    """
    if not isinstance(market_payload, dict):
        return None

    direct_fields = (
        market_payload.get("result"),
        market_payload.get("outcome"),
        market_payload.get("settlement_result"),
        market_payload.get("winner"),
        market_payload.get("settlement"),
    )
    for raw in direct_fields:
        norm = str(raw or "").strip().lower()
        if norm in {"yes", "y", "true", "1"}:
            return "yes"
        if norm in {"no", "n", "false", "0"}:
            return "no"

    for key in ("settlement_price", "settle_price", "yes_settlement_price"):
        val = market_payload.get(key)
        if val is None:
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        if num > 1:
            num = num / 100.0
        if 0.0 <= num <= 1.0:
            return "yes" if num >= 0.5 else "no"
    return None

