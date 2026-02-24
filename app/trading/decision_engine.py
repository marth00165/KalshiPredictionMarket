"""Fee-aware shadow decision engine for opportunity evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional


DecisionSide = Literal["yes", "no"]
DecisionResult = Literal["take", "skip"]


@dataclass
class TradeDecisionRecord:
    timestamp_utc: str
    market_id: str
    event_ticker: str
    title: str
    platform: str
    side_considered: DecisionSide
    fair_value_prob: float
    market_price: float
    implied_edge_pre_fee: float
    taker_fee_pct: float
    slippage_pct: float
    total_cost_pct: float
    net_edge_after_costs: float
    confidence: float
    decision: DecisionResult
    reason: str
    attribution: dict
    metadata_version: str = "v1"


def compute_pre_fee_edge(
    side: DecisionSide,
    fair_prob: float,
    yes_price: float,
    no_price: float,
) -> float:
    """Compute side-specific pre-fee edge."""
    side_norm = str(side).strip().lower()
    if side_norm == "yes":
        return float(fair_prob) - float(yes_price)
    if side_norm == "no":
        return (1.0 - float(fair_prob)) - float(no_price)
    raise ValueError(f"unsupported side: {side!r}")


def estimate_total_cost_pct(config: Any, market: Any = None) -> float:
    """
    Estimate total execution cost as a linear pct.

    This keeps costs centralized so future dynamic slippage models can plug in.
    """
    execution_cfg = getattr(config, "execution", None)
    taker_fee_pct = float(getattr(execution_cfg, "kalshi_taker_fee_pct", 0.0075) or 0.0)
    slippage_pct = float(getattr(execution_cfg, "estimated_slippage_pct", 0.005) or 0.0)
    return max(0.0, taker_fee_pct) + max(0.0, slippage_pct)


def compute_net_edge(pre_fee_edge: float, total_cost_pct: float) -> float:
    """Net edge after subtracting estimated costs."""
    return float(pre_fee_edge) - float(total_cost_pct)


def choose_side_from_estimate(fair_prob: float, yes_price: float, no_price: float) -> DecisionSide:
    """Choose side with higher pre-fee edge."""
    yes_edge = compute_pre_fee_edge("yes", fair_prob, yes_price, no_price)
    no_edge = compute_pre_fee_edge("no", fair_prob, yes_price, no_price)
    return "yes" if yes_edge >= no_edge else "no"


def compute_expected_value_per_dollar(
    side: DecisionSide,
    fair_prob: float,
    market_price: float,
    total_cost_pct: float,
) -> float:
    """
    Approximate EV per $1 notional for a binary contract.

    Assumption:
    - Contract payoff expectation for chosen side is p_side dollars per $1 payout.
    - You pay market_price plus estimated execution costs (pct of notional).
    """
    p_side = float(fair_prob) if side == "yes" else (1.0 - float(fair_prob))
    return p_side - float(market_price) - float(total_cost_pct)


def _safe_effective_probability(estimate: Any) -> Optional[float]:
    value = getattr(estimate, "effective_probability", None)
    if value is None:
        value = getattr(estimate, "estimated_probability", None)
    try:
        return float(value)
    except Exception:
        return None


def _safe_effective_confidence(estimate: Any) -> Optional[float]:
    value = getattr(estimate, "effective_confidence", None)
    if value is None:
        value = getattr(estimate, "confidence_level", None)
    try:
        return float(value)
    except Exception:
        return None


def _safe_effective_edge(estimate: Any) -> Optional[float]:
    value = getattr(estimate, "effective_edge", None)
    if value is None:
        value = getattr(estimate, "edge", None)
    try:
        return float(value)
    except Exception:
        return None


def _is_nba_market(market: Any) -> bool:
    platform = str(getattr(market, "platform", "") or "").strip().lower()
    market_id = str(getattr(market, "market_id", "") or "").strip().upper()
    event_ticker = str(getattr(market, "event_ticker", "") or "").strip().upper()
    series_ticker = str(getattr(market, "series_ticker", "") or "").strip().upper()
    if platform != "kalshi":
        return False
    return (
        market_id.startswith("KXNBAGAME-")
        or event_ticker.startswith("KXNBAGAME-")
        or series_ticker == "KXNBAGAME"
    )


def evaluate_opportunity(market: Any, estimate: Any, config: Any) -> TradeDecisionRecord:
    """Evaluate one opportunity and return a shadow take/skip decision record."""
    timestamp_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    market_id = str(getattr(market, "market_id", "") or "")
    event_ticker = str(getattr(market, "event_ticker", "") or "")
    title = str(getattr(market, "title", "") or "")
    platform = str(getattr(market, "platform", "") or "")
    yes_price = float(getattr(market, "yes_price", 0.0) or 0.0)
    no_price = float(getattr(market, "no_price", 0.0) or 0.0)

    fair_prob = _safe_effective_probability(estimate)
    confidence = _safe_effective_confidence(estimate)
    effective_edge = _safe_effective_edge(estimate)
    fusion_metadata = getattr(estimate, "fusion_metadata", {}) if estimate is not None else {}
    if not isinstance(fusion_metadata, dict):
        fusion_metadata = {}
    elo_meta = fusion_metadata.get("elo_adjustment")
    if not isinstance(elo_meta, dict):
        elo_meta = {}
    feature_meta = fusion_metadata.get("feature")
    if not isinstance(feature_meta, dict):
        feature_meta = {}

    execution_cfg = getattr(config, "execution", None)
    strategy_cfg = getattr(config, "strategy", None)
    taker_fee_pct = float(getattr(execution_cfg, "kalshi_taker_fee_pct", 0.0075) or 0.0)
    slippage_pct = float(getattr(execution_cfg, "estimated_slippage_pct", 0.005) or 0.0)
    total_cost_pct = estimate_total_cost_pct(config, market)
    min_edge_net = float(getattr(strategy_cfg, "min_edge_net", 0.025) or 0.0)
    min_confidence = float(getattr(strategy_cfg, "min_confidence", 0.75) or 0.0)

    side: DecisionSide = "yes"
    market_price = yes_price
    pre_fee_edge = 0.0
    net_edge = -total_cost_pct
    decision: DecisionResult = "skip"
    reason = "skip_missing_estimate"

    if estimate is None or fair_prob is None or confidence is None:
        reason = "skip_missing_estimate"
    elif not _is_nba_market(market):
        reason = "skip_non_nba"
    elif (
        yes_price < 0.0
        or yes_price > 1.0
        or no_price < 0.0
        or no_price > 1.0
        or abs((yes_price + no_price) - 1.0) > 0.12
    ):
        reason = "skip_invalid_prices"
    else:
        side = choose_side_from_estimate(fair_prob, yes_price, no_price)
        market_price = yes_price if side == "yes" else no_price
        pre_fee_edge = compute_pre_fee_edge(side, fair_prob, yes_price, no_price)
        net_edge = compute_net_edge(pre_fee_edge, total_cost_pct)

        if confidence < min_confidence:
            reason = "skip_low_confidence"
        elif net_edge < min_edge_net:
            reason = "skip_low_net_edge"
        else:
            decision = "take"
            reason = "take_strong_yes" if side == "yes" else "take_strong_no"

    ev_per_dollar = (
        compute_expected_value_per_dollar(
            side=side,
            fair_prob=float(fair_prob or 0.0),
            market_price=float(market_price),
            total_cost_pct=float(total_cost_pct),
        )
        if fair_prob is not None
        else None
    )

    attribution: Dict[str, Any] = {
        "p_elo_raw": elo_meta.get("p_elo"),
        "p_calibrated": elo_meta.get("p_final"),
        "p_final": elo_meta.get("p_final", elo_meta.get("final_probability")),
        "applied_elo_delta": elo_meta.get("applied_elo_delta"),
        "calibration_bucket": elo_meta.get("calibration_bucket"),
        "calibration_n": elo_meta.get("calibration_n"),
        "calibration_weight_w": elo_meta.get("calibration_weight_w"),
        "effective_probability": fair_prob,
        "effective_confidence": confidence,
        "effective_edge_pre_fee": effective_edge,
        "yes_price": yes_price,
        "no_price": no_price,
        "selected_side_price": market_price,
        "selected_side_pre_fee_edge": pre_fee_edge,
        "selected_side_ev_per_dollar": ev_per_dollar,
        "fee_inputs": {
            "taker_fee_pct": taker_fee_pct,
            "estimated_slippage_pct": slippage_pct,
            "use_dynamic_slippage": bool(getattr(execution_cfg, "use_dynamic_slippage", False)),
            "total_cost_pct": total_cost_pct,
        },
        "thresholds": {
            "min_edge_net": min_edge_net,
            "min_confidence": min_confidence,
        },
        "feature_summary": {
            "provider": getattr(estimate, "feature_provider", None) if estimate is not None else None,
            "recommendation": getattr(estimate, "feature_recommendation", None) if estimate is not None else None,
            "signal_score": getattr(estimate, "feature_signal_score", None) if estimate is not None else None,
            "anomaly": getattr(estimate, "feature_anomaly", None) if estimate is not None else None,
            "reason": feature_meta.get("reason"),
        },
        "source_components": {
            "elo_adjustment": elo_meta,
            "feature": feature_meta,
            "applied_rules": fusion_metadata.get("applied_rules"),
            "fusion_tags": getattr(estimate, "fusion_tags", None) if estimate is not None else None,
        },
    }

    return TradeDecisionRecord(
        timestamp_utc=timestamp_utc,
        market_id=market_id,
        event_ticker=event_ticker,
        title=title,
        platform=platform,
        side_considered=side,
        fair_value_prob=float(fair_prob or 0.0),
        market_price=float(market_price),
        implied_edge_pre_fee=float(pre_fee_edge),
        taker_fee_pct=float(taker_fee_pct),
        slippage_pct=float(slippage_pct),
        total_cost_pct=float(total_cost_pct),
        net_edge_after_costs=float(net_edge),
        confidence=float(confidence or 0.0),
        decision=decision,
        reason=reason,
        attribution=attribution,
    )


def append_decision_jsonl(record: TradeDecisionRecord, path: str) -> None:
    """Append decision record to JSONL with safe local-file handling."""
    try:
        out_path = Path(path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), sort_keys=True, default=str))
            fh.write("\n")
    except Exception:
        # Intentionally swallow errors to avoid impacting trading loop.
        return
