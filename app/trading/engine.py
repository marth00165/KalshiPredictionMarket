"""Elo trading helpers and compatibility wrapper for integration layers."""

from __future__ import annotations

import math
from typing import Dict, Optional

from app.analytics import EloEngine, get_win_probability as _get_win_probability, load_ratings as _load_ratings

MIN_ELO_DELTA = -75
MAX_ELO_DELTA = 75
DEFAULT_HOME_COURT_BONUS = 100.0


def validate_llm_elo_delta(delta: object) -> int:
    """
    Parse and clamp LLM-provided Elo delta.

    Invalid values degrade safely to 0.
    """
    try:
        parsed = int(round(float(delta)))  # type: ignore[arg-type]
    except Exception:
        return 0
    return max(MIN_ELO_DELTA, min(MAX_ELO_DELTA, parsed))


def apply_elo_adjustment(team: str, base_elo: float, elo_delta: object) -> float:
    """
    Apply clamped Elo delta to base team Elo.
    """
    _ = team  # keep signature explicit for logging/call-site clarity
    applied_delta = validate_llm_elo_delta(elo_delta)
    return float(base_elo) + float(applied_delta)


def calculate_probability_from_elo(team_elo: float, opponent_elo: float) -> float:
    """
    Convert Elo difference into win probability.
    """
    prob = 1.0 / (1.0 + 10.0 ** ((float(opponent_elo) - float(team_elo)) / 400.0))
    if math.isnan(prob):
        return 0.5
    return float(max(0.0, min(1.0, prob)))


def calculate_adjusted_yes_probability(
    *,
    yes_team: str,
    home_team: str,
    away_team: str,
    home_elo: float,
    away_elo: float,
    llm_elo_delta: object,
    home_court_bonus: float = DEFAULT_HOME_COURT_BONUS,
) -> Dict[str, float]:
    """
    Build final YES probability from Elo-only math with an LLM Elo adjustment.
    """
    yes_team_norm = str(yes_team).strip().upper()
    home_team_norm = str(home_team).strip().upper()
    away_team_norm = str(away_team).strip().upper()

    if yes_team_norm == home_team_norm:
        yes_base_elo = float(home_elo)
        opp_base_elo = float(away_elo)
        yes_is_home = True
    elif yes_team_norm == away_team_norm:
        yes_base_elo = float(away_elo)
        opp_base_elo = float(home_elo)
        yes_is_home = False
    else:
        raise ValueError(
            f"yes_team '{yes_team_norm}' does not match home/away teams "
            f"('{home_team_norm}', '{away_team_norm}')"
        )

    applied_delta = validate_llm_elo_delta(llm_elo_delta)
    yes_adjusted_elo = apply_elo_adjustment(yes_team_norm, yes_base_elo, applied_delta)

    if yes_is_home:
        yes_effective = yes_adjusted_elo + float(home_court_bonus)
        opp_effective = opp_base_elo
    else:
        yes_effective = yes_adjusted_elo
        opp_effective = opp_base_elo + float(home_court_bonus)

    probability = calculate_probability_from_elo(yes_effective, opp_effective)
    probability = float(max(0.01, min(0.99, probability)))

    return {
        "yes_probability": probability,
        "yes_base_elo": yes_base_elo,
        "yes_adjusted_elo": yes_adjusted_elo,
        "opponent_base_elo": opp_base_elo,
        "opponent_effective_elo": opp_effective,
        "yes_effective_elo": yes_effective,
        "applied_elo_delta": float(applied_delta),
    }


def load_ratings(
    data_path: str = EloEngine.DEFAULT_DATA_PATH,
    output_path: str = EloEngine.DEFAULT_OUTPUT_PATH,
    force_rebuild: bool = False,
) -> Dict[str, float]:
    return _load_ratings(
        data_path=data_path,
        output_path=output_path,
        force_rebuild=force_rebuild,
    )


def get_win_probability(
    team_a: str,
    team_b: str,
    team_a_is_home: bool = True,
    home_advantage: Optional[float] = None,
) -> float:
    return _get_win_probability(
        team_a=team_a,
        team_b=team_b,
        team_a_is_home=team_a_is_home,
        home_advantage=home_advantage,
    )
