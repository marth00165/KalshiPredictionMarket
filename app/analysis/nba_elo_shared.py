"""Shared NBA Elo helper functions used by multiple analyzers."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from app.analytics import EloEngine
from app.analytics.elo_calibration import (
    EloCalibrationConfig,
    blend_probabilities,
    lookup_empirical_rate,
)
from app.models import FairValueEstimate, MarketData
from app.trading.engine import calculate_adjusted_yes_probability


def is_nba_game_market(market: MarketData) -> bool:
    if str(market.platform).strip().lower() != "kalshi":
        return False
    series = str(getattr(market, "series_ticker", "") or "").strip().upper()
    market_id = str(getattr(market, "market_id", "") or "").strip().upper()
    return series == "KXNBAGAME" or market_id.startswith("KXNBAGAME-")


def get_nba_elo_context(
    *,
    market: MarketData,
    elo_enabled: bool,
    elo_engine: Optional[EloEngine],
    logger: logging.Logger,
) -> Optional[Dict[str, object]]:
    if not elo_enabled or not elo_engine:
        return None
    if not is_nba_game_market(market):
        return None
    try:
        matchup = elo_engine.parse_kalshi_nba_matchup(market.market_id)
        if matchup is None:
            return None
        yes_probability = elo_engine.get_market_yes_probability(market.market_id)
        if yes_probability is None:
            return None
        home_elo = elo_engine.ratings.get(matchup.home_team)
        away_elo = elo_engine.ratings.get(matchup.away_team)
        if home_elo is None or away_elo is None:
            return None
        return {
            "yes_probability": float(yes_probability),
            "home_elo": float(home_elo),
            "away_elo": float(away_elo),
            "home_team": matchup.home_team,
            "away_team": matchup.away_team,
            "yes_team": matchup.yes_team,
            "home_court_bonus": float(elo_engine.home_advantage),
            "elo_edge": float(yes_probability - market.yes_price),
        }
    except Exception as e:
        logger.debug("NBA Elo context unavailable for %s: %s", market.market_id, e)
        return None


def build_elo_estimate_from_delta(
    *,
    market: MarketData,
    elo_ctx: Dict[str, object],
    elo_delta: float,
    confidence: float,
    reason: str,
    source_tag: str,
    llm_suggestion: Optional[Dict[str, object]] = None,
) -> FairValueEstimate:
    adjusted = calculate_adjusted_yes_probability(
        yes_team=str(elo_ctx["yes_team"]),
        home_team=str(elo_ctx["home_team"]),
        away_team=str(elo_ctx["away_team"]),
        home_elo=float(elo_ctx["home_elo"]),
        away_elo=float(elo_ctx["away_elo"]),
        llm_elo_delta=elo_delta,
        home_court_bonus=float(elo_ctx["home_court_bonus"]),
    )
    base = calculate_adjusted_yes_probability(
        yes_team=str(elo_ctx["yes_team"]),
        home_team=str(elo_ctx["home_team"]),
        away_team=str(elo_ctx["away_team"]),
        home_elo=float(elo_ctx["home_elo"]),
        away_elo=float(elo_ctx["away_elo"]),
        llm_elo_delta=0,
        home_court_bonus=float(elo_ctx["home_court_bonus"]),
    )
    base_prob = float(base["yes_probability"])
    final_prob = float(adjusted["yes_probability"])
    edge = final_prob - float(market.yes_price)
    confidence_clamped = float(max(0.0, min(1.0, float(confidence))))
    suggestion_payload = llm_suggestion if isinstance(llm_suggestion, dict) else {}
    if not suggestion_payload:
        suggestion_payload = {
            "raw_elo_delta": float(elo_delta),
            "applied_elo_delta": float(elo_delta),
            "confidence": confidence_clamped,
            "reason": str(reason or ""),
            "key_factors": [],
            "data_sources": [],
            "injury_report": {
                "status": "unknown",
                "impact": "unknown",
                "notes": "",
            },
        }

    return FairValueEstimate(
        market_id=market.market_id,
        estimated_probability=final_prob,
        confidence_level=confidence_clamped,
        reasoning=(
            f"Elo base={base_prob:.3f}, elo_delta={float(elo_delta):+0.0f}. "
            f"{reason}"
        ).strip(),
        data_sources=["nba_elo", "sportradar_injuries"],
        key_factors=["elo_with_injury_delta"],
        edge=edge,
        fusion_metadata={
            "elo_adjustment": {
                "yes_team": str(elo_ctx["yes_team"]),
                "home_team": str(elo_ctx["home_team"]),
                "away_team": str(elo_ctx["away_team"]),
                "home_elo": float(elo_ctx["home_elo"]),
                "away_elo": float(elo_ctx["away_elo"]),
                "base_probability": float(base_prob),
                "applied_elo_delta": float(adjusted["applied_elo_delta"]),
                "yes_adjusted_elo": float(adjusted["yes_adjusted_elo"]),
                "yes_effective_elo": float(adjusted["yes_effective_elo"]),
                "opponent_effective_elo": float(adjusted["opponent_effective_elo"]),
                "final_probability": float(final_prob),
                "market_probability": float(market.yes_price),
                "edge": float(edge),
                "llm_suggestion": suggestion_payload,
                "llm_source_tag": str(source_tag),
            }
        },
    )


def build_elo_fallback_estimate(
    *,
    market: MarketData,
    elo_ctx: Dict[str, object],
    reason: str,
    source_tag: str,
) -> FairValueEstimate:
    return build_elo_estimate_from_delta(
        market=market,
        elo_ctx=elo_ctx,
        elo_delta=0.0,
        confidence=0.60,
        reason=f"Elo-only fallback ({reason}).",
        source_tag=source_tag,
        llm_suggestion={
            "raw_elo_delta": 0.0,
            "applied_elo_delta": 0.0,
            "confidence": 0.60,
            "reason": f"Elo-only fallback ({reason}).",
            "key_factors": ["elo_baseline_only"],
            "data_sources": ["nba_elo"],
            "injury_report": {
                "status": "unknown",
                "impact": "neutral",
                "notes": "",
            },
        },
    )


def apply_elo_calibration(
    *,
    market: MarketData,
    estimate: FairValueEstimate,
    elo_ctx: Dict[str, object],
    calibration_table,
    calibration_config: EloCalibrationConfig,
) -> None:
    if calibration_table is None:
        return

    metadata = dict(estimate.fusion_metadata or {})
    elo_meta = metadata.get("elo_adjustment")
    if not isinstance(elo_meta, dict):
        return

    yes_team = str(elo_ctx.get("yes_team", "")).strip().upper()
    home_team = str(elo_ctx.get("home_team", "")).strip().upper()
    away_team = str(elo_ctx.get("away_team", "")).strip().upper()
    if yes_team not in {home_team, away_team}:
        return

    is_home = 1 if yes_team == home_team else 0
    home_elo = float(elo_ctx.get("home_elo", 0.0))
    away_elo = float(elo_ctx.get("away_elo", 0.0))
    yes_base_elo = home_elo if is_home else away_elo
    opponent_base_elo = away_elo if is_home else home_elo
    applied_delta = float(elo_meta.get("applied_elo_delta", 0.0))
    yes_adjusted_elo = yes_base_elo + applied_delta
    elo_difference = yes_adjusted_elo - opponent_base_elo

    p_elo = float(estimate.estimated_probability)
    p_emp, n, bucket_key = lookup_empirical_rate(
        calibration_table,
        is_home=is_home,
        elo_difference=elo_difference,
        bucket_size=int(calibration_config.bucket_size),
    )
    p_final, w = blend_probabilities(
        p_elo=p_elo,
        p_emp=p_emp,
        n=n,
        prior=float(calibration_config.prior),
    )

    estimate.estimated_probability = float(p_final)
    estimate.edge = float(p_final) - float(market.yes_price)

    elo_meta["p_elo"] = float(p_elo)
    elo_meta["p_emp"] = (None if p_emp is None else float(p_emp))
    elo_meta["p_final"] = float(p_final)
    elo_meta["calibration_bucket"] = bucket_key
    elo_meta["calibration_n"] = float(n)
    elo_meta["calibration_weight_w"] = float(w)
    elo_meta["calibration_min_season"] = calibration_config.min_season
    elo_meta["calibration_recency_mode"] = calibration_config.recency_mode
    elo_meta["calibration_recency_halflife_days"] = calibration_config.recency_halflife_days
    elo_meta["final_probability"] = float(p_final)
    elo_meta["edge"] = float(estimate.edge)

    metadata["elo_adjustment"] = elo_meta
    estimate.fusion_metadata = metadata


def log_elo_decision_fields(
    *,
    logger: logging.Logger,
    market: MarketData,
    estimate: FairValueEstimate,
) -> None:
    meta = (estimate.fusion_metadata or {}).get("elo_adjustment", {})
    if not isinstance(meta, dict):
        return
    try:
        p_elo = float(meta.get("p_elo", meta.get("final_probability", estimate.estimated_probability)))
        p_final = float(meta.get("p_final", meta.get("final_probability", estimate.estimated_probability)))
        p_emp_raw = meta.get("p_emp")
        p_emp = "na" if p_emp_raw is None else f"{float(p_emp_raw):.4f}"
        logger.info(
            "ELO_DECISION | team=%s | opponent=%s | elo_base=%.2f | elo_delta=%+.0f | "
            "elo_adjusted=%.2f | probability_base=%.4f | probability_p_elo=%.4f | "
            "probability_final=%.4f | p_emp=%s | cal_n=%.1f | cal_w=%.4f | "
            "market_probability=%.4f | edge=%+.4f | llm_source=%s | refresh_reason=%s",
            str(meta.get("yes_team") or ""),
            str(meta.get("away_team") if str(meta.get("yes_team")) == str(meta.get("home_team")) else meta.get("home_team")),
            float(meta.get("home_elo") if str(meta.get("yes_team")) == str(meta.get("home_team")) else meta.get("away_elo")),
            float(meta.get("applied_elo_delta", 0.0)),
            float(meta.get("yes_adjusted_elo", 0.0)),
            float(meta.get("base_probability", 0.0)),
            p_elo,
            p_final,
            p_emp,
            float(meta.get("calibration_n", 0.0)),
            float(meta.get("calibration_weight_w", 0.0)),
            float(meta.get("market_probability", market.yes_price)),
            float(meta.get("edge", estimate.edge)),
            str(meta.get("llm_source_tag", "llm_refresh")),
            str(meta.get("llm_refresh_reason", "")),
        )
        suggestion = meta.get("llm_suggestion", {})
        if isinstance(suggestion, dict):
            injury = suggestion.get("injury_report", {})
            if not isinstance(injury, dict):
                injury = {}
            logger.info(
                "ELO_SUGGESTION | market_id=%s | raw_delta=%s | applied_delta=%+.0f | confidence=%.2f | "
                "injury_status=%s | injury_impact=%s | factors=%s | sources=%s",
                market.market_id,
                str(suggestion.get("raw_elo_delta")),
                float(suggestion.get("applied_elo_delta", meta.get("applied_elo_delta", 0.0))),
                float(suggestion.get("confidence", estimate.confidence_level)),
                str(injury.get("status", "unknown")),
                str(injury.get("impact", "unknown")),
                ",".join(str(x) for x in (suggestion.get("key_factors") or [])[:5]),
                ",".join(str(x) for x in (suggestion.get("data_sources") or [])[:5]),
            )
    except Exception:
        logger.debug("Failed logging Elo decision metadata for %s", market.market_id)
