#!/usr/bin/env python3
"""Walk-back NBA matchup analysis using season-only Elo + last-20 form."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.analytics.elo_engine import EloEngine, GameRecord


@dataclass(frozen=True)
class EventMarket:
    market_id: str
    away_team: str
    home_team: str
    yes_team: str
    date_code: str
    game_date: date
    yes_price: float


@dataclass(frozen=True)
class TeamForm:
    games: int
    wins: int
    losses: int
    win_pct: float
    avg_point_diff: float
    avg_points_for: float
    avg_points_against: float
    last5_win_pct: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze NBA Kalshi matchups by rebuilding Elo from only current-season "
            "games and adding last-20 team form."
        )
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to dry-run cycle JSON report (default: latest in reports/dry_run_analysis).",
    )
    parser.add_argument(
        "--date-code",
        default=None,
        help="Filter event date code like 26FEB20 (default: derived from report timestamp UTC).",
    )
    parser.add_argument(
        "--data-path",
        default="context/kaggleGameData.csv",
        help="Path to Kaggle NBA game CSV.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path for structured analysis results.",
    )
    return parser.parse_args()


def _latest_report_path() -> Path:
    report_dir = Path("reports/dry_run_analysis")
    candidates = sorted(report_dir.glob("cycle_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No cycle report JSON found under {report_dir}")
    return candidates[0]


def _date_to_code(value: date) -> str:
    return value.strftime("%y%b%d").upper()


def _parse_market_row(row: Dict[str, Any]) -> Optional[EventMarket]:
    market_id = str(row.get("market_id") or "").strip().upper()
    if not market_id.startswith("KXNBAGAME-"):
        return None
    parts = market_id.split("-")
    if len(parts) < 3:
        return None

    matchup_chunk = parts[1]
    if len(matchup_chunk) < 13:
        return None

    date_code = matchup_chunk[:7]
    away_team = matchup_chunk[7:10]
    home_team = matchup_chunk[10:13]
    yes_team = parts[2][:3]
    try:
        game_date = datetime.strptime(date_code.title(), "%y%b%d").date()
    except ValueError:
        return None

    try:
        yes_price = float(row.get("yes_price"))
    except Exception:
        return None

    return EventMarket(
        market_id=market_id,
        away_team=away_team,
        home_team=home_team,
        yes_team=yes_team,
        date_code=date_code,
        game_date=game_date,
        yes_price=yes_price,
    )


def _season_start_for_game_date(game_day: date) -> date:
    # NBA regular season starts in October and spans two calendar years.
    season_year = game_day.year if game_day.month >= 10 else game_day.year - 1
    return date(season_year, 10, 1)


def _elo_probability(team_a: str, team_b: str, team_a_is_home: bool, ratings: Dict[str, float], home_adv: float) -> float:
    elo_a = float(ratings.get(team_a, EloEngine.DEFAULT_INITIAL_RATING))
    elo_b = float(ratings.get(team_b, EloEngine.DEFAULT_INITIAL_RATING))
    rating_a = elo_a + (home_adv if team_a_is_home else 0.0)
    rating_b = elo_b + (0.0 if team_a_is_home else home_adv)
    prob = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    return max(0.0, min(1.0, prob))


def _team_last_n_form(games: Iterable[GameRecord], team: str, n: int = 20) -> TeamForm:
    team_games = [g for g in games if g.home_team == team or g.away_team == team]
    team_games.sort(key=lambda g: g.game_date, reverse=True)
    sample = team_games[:n]
    if not sample:
        return TeamForm(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    wins = 0
    points_for = 0
    points_against = 0
    outcomes: List[int] = []
    for g in sample:
        is_home = g.home_team == team
        pf = g.home_score if is_home else g.away_score
        pa = g.away_score if is_home else g.home_score
        won = 1 if pf > pa else 0
        wins += won
        points_for += pf
        points_against += pa
        outcomes.append(won)

    games_count = len(sample)
    losses = games_count - wins
    win_pct = wins / games_count
    avg_pf = points_for / games_count
    avg_pa = points_against / games_count
    avg_diff = (points_for - points_against) / games_count
    last5 = outcomes[:5]
    last5_win_pct = (sum(last5) / len(last5)) if last5 else 0.0
    return TeamForm(
        games=games_count,
        wins=wins,
        losses=losses,
        win_pct=win_pct,
        avg_point_diff=avg_diff,
        avg_points_for=avg_pf,
        avg_points_against=avg_pa,
        last5_win_pct=last5_win_pct,
    )


def _blend_with_form(away_prob_elo: float, away_form: TeamForm, home_form: TeamForm) -> float:
    # Small bounded adjustment so Elo remains primary.
    net_diff = away_form.avg_point_diff - home_form.avg_point_diff
    trend_diff = away_form.last5_win_pct - home_form.last5_win_pct
    adjustment = (0.012 * net_diff) + (0.05 * trend_diff)
    adjustment = max(-0.08, min(0.08, adjustment))
    return max(0.02, min(0.98, away_prob_elo + adjustment))


def _round(v: float) -> float:
    return round(float(v), 4)


def main() -> int:
    args = _parse_args()
    report_path = Path(args.report) if args.report else _latest_report_path()
    payload = json.loads(report_path.read_text())

    ts = str(payload.get("timestamp_utc") or "").strip()
    report_date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date() if ts else datetime.utcnow().date()
    target_date_code = (args.date_code or _date_to_code(report_date)).upper()

    markets: List[EventMarket] = []
    for row in payload.get("results", []):
        parsed = _parse_market_row(row)
        if parsed and parsed.date_code == target_date_code:
            markets.append(parsed)

    if not markets:
        print(f"No NBA markets found in {report_path} for date code {target_date_code}.")
        return 1

    # Build event map keyed by date+away+home.
    events: Dict[Tuple[date, str, str], Dict[str, Any]] = {}
    for m in markets:
        key = (m.game_date, m.away_team, m.home_team)
        entry = events.setdefault(
            key,
            {
                "game_date": m.game_date.isoformat(),
                "date_code": m.date_code,
                "away_team": m.away_team,
                "home_team": m.home_team,
                "market_yes_prices": {},
            },
        )
        entry["market_yes_prices"][m.yes_team] = m.yes_price

    engine = EloEngine(data_path=args.data_path, output_path="app/outputs/elo_ratings.json")
    all_games = engine.load_games()
    analyses: List[Dict[str, Any]] = []

    for (game_day, away_team, home_team), event in sorted(events.items()):
        season_start = _season_start_for_game_date(game_day)
        season_games = [g for g in all_games if season_start <= g.game_date.date() < game_day]

        teams = {g.home_team for g in season_games} | {g.away_team for g in season_games}
        engine.initialize_ratings(teams)
        engine.games_processed = 0
        engine.last_game_date = None
        engine.update_ratings(season_games)

        away_prob_elo = _elo_probability(
            team_a=away_team,
            team_b=home_team,
            team_a_is_home=False,
            ratings=engine.ratings,
            home_adv=engine.home_advantage,
        )
        home_prob_elo = 1.0 - away_prob_elo

        away_form = _team_last_n_form(season_games, away_team, n=20)
        home_form = _team_last_n_form(season_games, home_team, n=20)

        away_prob_blended = _blend_with_form(away_prob_elo, away_form, home_form)
        home_prob_blended = 1.0 - away_prob_blended

        away_market_yes = event["market_yes_prices"].get(away_team)
        home_market_yes = event["market_yes_prices"].get(home_team)

        away_edge = (away_prob_blended - float(away_market_yes)) if away_market_yes is not None else None
        home_edge = (home_prob_blended - float(home_market_yes)) if home_market_yes is not None else None

        recommendation = "no_edge"
        if away_edge is not None and away_edge >= 0.05:
            recommendation = f"lean_{away_team}_yes"
        if home_edge is not None and home_edge >= 0.05:
            if recommendation == "no_edge" or (away_edge is not None and home_edge > away_edge):
                recommendation = f"lean_{home_team}_yes"

        analyses.append(
            {
                "game_date": event["game_date"],
                "away_team": away_team,
                "home_team": home_team,
                "season_start": season_start.isoformat(),
                "season_games_used": len(season_games),
                "elo": {
                    "away_rating": _round(engine.ratings.get(away_team, engine.initial_rating)),
                    "home_rating": _round(engine.ratings.get(home_team, engine.initial_rating)),
                    "away_win_prob": _round(away_prob_elo),
                    "home_win_prob": _round(home_prob_elo),
                },
                "last20_form": {
                    "away": {
                        "games": away_form.games,
                        "record": f"{away_form.wins}-{away_form.losses}",
                        "win_pct": _round(away_form.win_pct),
                        "avg_point_diff": _round(away_form.avg_point_diff),
                        "last5_win_pct": _round(away_form.last5_win_pct),
                    },
                    "home": {
                        "games": home_form.games,
                        "record": f"{home_form.wins}-{home_form.losses}",
                        "win_pct": _round(home_form.win_pct),
                        "avg_point_diff": _round(home_form.avg_point_diff),
                        "last5_win_pct": _round(home_form.last5_win_pct),
                    },
                },
                "blend": {
                    "away_win_prob": _round(away_prob_blended),
                    "home_win_prob": _round(home_prob_blended),
                },
                "market_yes_prices": {
                    away_team: away_market_yes,
                    home_team: home_market_yes,
                },
                "edges_vs_market_yes": {
                    away_team: _round(away_edge) if away_edge is not None else None,
                    home_team: _round(home_edge) if home_edge is not None else None,
                },
                "recommendation": recommendation,
            }
        )

    result = {
        "report_path": str(report_path),
        "target_date_code": target_date_code,
        "games_analyzed": len(analyses),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "analysis": analyses,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote analysis JSON to {out_path}")

    print(
        f"Walk-back analysis for {target_date_code}: "
        f"{len(analyses)} games | report={report_path.name}"
    )
    print(
        "away@home | elo_away | blend_away | mkt_away_yes | edge_away | "
        "mkt_home_yes | edge_home | rec"
    )
    for row in analyses:
        away = row["away_team"]
        home = row["home_team"]
        elo_away = row["elo"]["away_win_prob"]
        blend_away = row["blend"]["away_win_prob"]
        mkt_away = row["market_yes_prices"][away]
        mkt_home = row["market_yes_prices"][home]
        edge_away = row["edges_vs_market_yes"][away]
        edge_home = row["edges_vs_market_yes"][home]
        rec = row["recommendation"]
        print(
            f"{away}@{home} | {elo_away:>7.4f} | {blend_away:>9.4f} | "
            f"{mkt_away:>12.4f} | {edge_away:>8.4f} | {mkt_home:>11.4f} | {edge_home:>8.4f} | {rec}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
