"""Deterministic player-impact utilities for injury-driven Elo adjustments."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

PIM_K_FACTOR = 25.0
PIM_MAX_DELTA = 75.0

STATUS_WEIGHTS: Dict[str, float] = {
    "OUT": 1.0,
    "DOUBTFUL": 0.7,
    "QUESTIONABLE": 0.35,
    "PROBABLE": 0.15,
    "AVAILABLE": 0.0,
}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_player_id(profile_json: Dict[str, Any]) -> str:
    if not isinstance(profile_json, dict):
        return ""
    root = profile_json.get("player_profile")
    if not isinstance(root, dict):
        root = profile_json
    player_obj = root.get("player")
    if isinstance(player_obj, dict):
        player_id = str(player_obj.get("id") or "").strip()
        if player_id:
            return player_id
    return str(root.get("id") or "").strip()


def _iter_reg_season_team_averages(profile_json: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if not isinstance(profile_json, dict):
        return []
    root = profile_json.get("player_profile")
    if not isinstance(root, dict):
        root = profile_json
    seasons = root.get("seasons")
    if not isinstance(seasons, list):
        return []

    reg_seasons: List[Dict[str, Any]] = []
    for season in seasons:
        if not isinstance(season, dict):
            continue
        if str(season.get("type") or "").strip().upper() != "REG":
            continue
        reg_seasons.append(season)

    if not reg_seasons:
        return []

    reg_seasons.sort(
        key=lambda s: (
            int(_to_float(s.get("year"), 0)),
            str(s.get("id") or ""),
        ),
        reverse=True,
    )
    latest = reg_seasons[0]
    teams = latest.get("teams")
    if not isinstance(teams, list):
        return []
    return [t for t in teams if isinstance(t, dict)]


def _weighted_average_metric(team_rows: List[Dict[str, Any]], metric_key: str) -> float:
    if not team_rows:
        return 0.0
    weighted_sum = 0.0
    total_games = 0.0
    plain_values: List[float] = []
    for row in team_rows:
        avg = row.get("average")
        if not isinstance(avg, dict):
            continue
        metric = _to_float(avg.get(metric_key), 0.0)
        games = _to_float(avg.get("games_played"), 0.0)
        plain_values.append(metric)
        if games > 0:
            weighted_sum += metric * games
            total_games += games
    if total_games > 0:
        return weighted_sum / total_games
    if plain_values:
        return sum(plain_values) / len(plain_values)
    return 0.0


def compute_player_impact(profile_json: Dict[str, Any]) -> float:
    """
    Compute deterministic Player Impact Metric (PIM) from most recent REG season.

    Formula:
      OFFENSE = PPG + 0.7*APG - 0.9*TPG
      DEFENSE = 0.7*RPG + 1.5*SPG + 1.5*BPG
      MINUTES_FACTOR = MPG / 36
      PIM = (OFFENSE + DEFENSE) * MINUTES_FACTOR
    """
    team_rows = list(_iter_reg_season_team_averages(profile_json))
    if not team_rows:
        return 0.0

    ppg = _weighted_average_metric(team_rows, "points")
    apg = _weighted_average_metric(team_rows, "assists")
    rpg = _weighted_average_metric(team_rows, "rebounds")
    spg = _weighted_average_metric(team_rows, "steals")
    bpg = _weighted_average_metric(team_rows, "blocks")
    tpg = _weighted_average_metric(team_rows, "turnovers")
    mpg = _weighted_average_metric(team_rows, "minutes")

    offense = ppg + (0.7 * apg) - (0.9 * tpg)
    defense = (0.7 * rpg) + (1.5 * spg) + (1.5 * bpg)
    minutes_factor = mpg / 36.0
    pim = (offense + defense) * minutes_factor
    return float(pim)


def compute_team_injury_impact(
    team_players: List[Dict[str, Any]],
    injury_status_map: Dict[str, str],
) -> float:
    """Compute weighted sum of impacted player PIM values for one team."""
    total = 0.0
    for profile in team_players or []:
        player_id = _extract_player_id(profile)
        if not player_id:
            continue
        status = str((injury_status_map or {}).get(player_id, "AVAILABLE") or "AVAILABLE").strip().upper()
        weight = STATUS_WEIGHTS.get(status, 0.0)
        if weight <= 0:
            continue
        total += compute_player_impact(profile) * weight
    return float(total)


def compute_injury_elo_delta(
    yes_team_players: List[Dict[str, Any]],
    opp_team_players: List[Dict[str, Any]],
    injury_status_map: Dict[str, str],
) -> float:
    """
    Compute deterministic Elo delta using PIM-weighted injury impact.

      delta = K * (opp_impact - yes_team_impact), K=25
      clamp to [-75, +75]
    """
    yes_team_impact = compute_team_injury_impact(yes_team_players, injury_status_map)
    opp_team_impact = compute_team_injury_impact(opp_team_players, injury_status_map)
    delta = PIM_K_FACTOR * (opp_team_impact - yes_team_impact)
    delta = max(-PIM_MAX_DELTA, min(PIM_MAX_DELTA, delta))
    return float(delta)


def compute_injury_impact_breakdown(
    yes_team_players: List[Dict[str, Any]],
    opp_team_players: List[Dict[str, Any]],
    injury_status_map: Dict[str, str],
    *,
    k_factor: float = PIM_K_FACTOR,
    max_delta: float = PIM_MAX_DELTA,
) -> Dict[str, float]:
    """Compute yes/opp team impacts and deterministic Elo delta for reporting."""
    yes_team_impact = compute_team_injury_impact(yes_team_players, injury_status_map)
    opp_team_impact = compute_team_injury_impact(opp_team_players, injury_status_map)
    delta = float(k_factor) * (opp_team_impact - yes_team_impact)
    clamp_val = abs(float(max_delta))
    if clamp_val > 0:
        delta = max(-clamp_val, min(clamp_val, delta))
    return {
        "yes_team_impact": float(yes_team_impact),
        "opp_team_impact": float(opp_team_impact),
        "delta_pim": float(delta),
    }

