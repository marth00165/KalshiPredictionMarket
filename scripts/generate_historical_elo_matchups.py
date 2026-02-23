#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd


OUTPUT_COLUMNS = [
    "date",
    "season",
    "team",
    "opponent",
    "is_home",
    "team_elo_before_game",
    "opponent_elo_before_game",
    "elo_difference",
    "result",
    "team_points",
    "opponent_points",
    "team_elo_bucket",
    "opponent_elo_bucket",
]

SEASON_RATINGS_COLUMNS = ["season", "team", "elo_final"]


@dataclass
class EloArtifacts:
    matchups: pd.DataFrame
    season_ratings: pd.DataFrame
    latest_ratings: Dict[str, float]


def _pick_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _round_elo(value: float) -> float:
    return round(float(value), 1)


def _bucket(value: float, bucket_size: int) -> int:
    # Use true floor for stable negative-bucket behavior.
    return int(floor(float(value) / float(bucket_size)) * bucket_size)


def _parse_seasons_arg(value: Optional[str]) -> Optional[Set[int]]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    out: Set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            season = int(token)
        except Exception as exc:
            raise ValueError(f"Invalid season in --seasons: {token!r}") from exc
        if season < 1:
            raise ValueError(f"Invalid season in --seasons: {season}")
        out.add(season)
    return out or None


def load_games(path: str) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(input_path, low_memory=False)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    date_col = _pick_column(
        df.columns,
        ["date", "GAME_DATE", "GAME_DATE_EST", "GAME_DATE_TIME_EST", "gameDateTimeEst", "Date"],
    )
    home_team_col = _pick_column(
        df.columns,
        [
            "home_team",
            "HOME_TEAM",
            "homeTeam",
            "hometeamName",
            "home_team_name",
            "HOME_TEAM_NAME",
            "hometeamId",
        ],
    )
    away_team_col = _pick_column(
        df.columns,
        [
            "away_team",
            "AWAY_TEAM",
            "awayTeam",
            "awayteamName",
            "away_team_name",
            "AWAY_TEAM_NAME",
            "awayteamId",
        ],
    )
    home_score_col = _pick_column(
        df.columns,
        ["home_score", "HOME_SCORE", "homeScore", "home_points", "home_pts", "PTS_home"],
    )
    away_score_col = _pick_column(
        df.columns,
        ["away_score", "AWAY_SCORE", "awayScore", "away_points", "away_pts", "PTS_away"],
    )
    season_col = _pick_column(df.columns, ["season", "SEASON", "Season"])
    game_id_col = _pick_column(df.columns, ["game_id", "GAME_ID", "gameId", "GameID"])

    missing = []
    if date_col is None:
        missing.append("date")
    if home_team_col is None:
        missing.append("home_team")
    if away_team_col is None:
        missing.append("away_team")
    if home_score_col is None:
        missing.append("home_score")
    if away_score_col is None:
        missing.append("away_score")
    if missing:
        raise ValueError(f"Missing required columns (or known variants): {missing}")

    rename_map = {
        date_col: "date",
        home_team_col: "home_team",
        away_team_col: "away_team",
        home_score_col: "home_score",
        away_score_col: "away_score",
    }
    if season_col is not None:
        rename_map[season_col] = "season"
    if game_id_col is not None:
        rename_map[game_id_col] = "game_id"

    df = df.rename(columns=rename_map).copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    df.loc[df["home_team"] == "", "home_team"] = pd.NA
    df.loc[df["away_team"] == "", "away_team"] = pd.NA

    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"]).copy()
    if df.empty:
        raise ValueError("No valid rows after cleaning required fields.")

    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].astype(str)

    return df


def derive_season(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce")
        out = out.dropna(subset=["season"]).copy()
        out["season"] = out["season"].astype(int)
    else:
        out["season"] = out["date"].dt.year + (out["date"].dt.month >= 10).astype(int)
        out["season"] = out["season"].astype(int)
    return out


def _calculate_elo_artifacts(
    df: pd.DataFrame,
    initial_elo: float,
    k_factor: float,
    home_advantage: float,
    bucket_size: int,
    regression_factor: float,
) -> EloArtifacts:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be > 0")
    if not 0.0 <= regression_factor <= 1.0:
        raise ValueError("regression_factor must be between 0.0 and 1.0")

    sort_cols = ["date"]
    if "game_id" in df.columns:
        sort_cols.append("game_id")
    games = df.sort_values(sort_cols, ascending=True, kind="mergesort").copy()

    elo_by_team: Dict[str, float] = {}
    season_teams: Dict[int, Set[str]] = {}
    season_final_ratings: Dict[int, Dict[str, float]] = {}
    current_season: Optional[int] = None
    rows: List[dict] = []

    def _finalize_season(season: int) -> None:
        teams = season_teams.get(int(season), set())
        if not teams:
            return
        season_final_ratings[int(season)] = {
            team: _round_elo(elo_by_team.get(team, float(initial_elo)))
            for team in sorted(teams)
        }

    for game in games.itertuples(index=False):
        season = int(getattr(game, "season"))
        if current_season is None:
            current_season = season
        elif season != current_season:
            _finalize_season(int(current_season))
            season_steps = max(1, season - current_season) if season > current_season else 1
            for _ in range(season_steps):
                elo_by_team = {
                    team: _round_elo(
                        float(regression_factor) * float(elo)
                        + (1.0 - float(regression_factor)) * float(initial_elo)
                    )
                    for team, elo in elo_by_team.items()
                }
            current_season = season

        home_team = str(getattr(game, "home_team")).strip()
        away_team = str(getattr(game, "away_team")).strip()
        season_teams.setdefault(season, set()).update({home_team, away_team})

        home_score = int(getattr(game, "home_score"))
        away_score = int(getattr(game, "away_score"))

        home_elo = _round_elo(float(elo_by_team.get(home_team, initial_elo)))
        away_elo = _round_elo(float(elo_by_team.get(away_team, initial_elo)))

        expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + home_advantage)) / 400.0))
        expected_away = 1.0 - expected_home

        home_result = 1 if home_score > away_score else 0
        away_result = 1 - home_result

        mov = abs(home_score - away_score)
        elo_diff = abs(home_elo - away_elo)
        mov_multiplier = ((mov + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff)
        effective_k = float(k_factor) * float(mov_multiplier)

        game_date_str = getattr(game, "date").strftime("%Y-%m-%d")

        home_bucket = _bucket(home_elo, bucket_size)
        away_bucket = _bucket(away_elo, bucket_size)

        rows.append(
            {
                "date": game_date_str,
                "season": season,
                "team": home_team,
                "opponent": away_team,
                "is_home": 1,
                "team_elo_before_game": home_elo,
                "opponent_elo_before_game": away_elo,
                "elo_difference": _round_elo(home_elo - away_elo),
                "result": int(home_result),
                "team_points": int(home_score),
                "opponent_points": int(away_score),
                "team_elo_bucket": int(home_bucket),
                "opponent_elo_bucket": int(away_bucket),
            }
        )
        rows.append(
            {
                "date": game_date_str,
                "season": season,
                "team": away_team,
                "opponent": home_team,
                "is_home": 0,
                "team_elo_before_game": away_elo,
                "opponent_elo_before_game": home_elo,
                "elo_difference": _round_elo(away_elo - home_elo),
                "result": int(away_result),
                "team_points": int(away_score),
                "opponent_points": int(home_score),
                "team_elo_bucket": int(away_bucket),
                "opponent_elo_bucket": int(home_bucket),
            }
        )

        updated_home_elo = home_elo + effective_k * (home_result - expected_home)
        updated_away_elo = away_elo + effective_k * (away_result - expected_away)
        elo_by_team[home_team] = _round_elo(updated_home_elo)
        elo_by_team[away_team] = _round_elo(updated_away_elo)

    if current_season is not None:
        _finalize_season(int(current_season))

    matchup_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    season_rows: List[dict] = []
    for season in sorted(season_final_ratings):
        for team in sorted(season_final_ratings[season]):
            season_rows.append(
                {
                    "season": int(season),
                    "team": str(team),
                    "elo_final": _round_elo(float(season_final_ratings[season][team])),
                }
            )
    season_df = pd.DataFrame(season_rows, columns=SEASON_RATINGS_COLUMNS)
    latest = {team: _round_elo(float(elo)) for team, elo in sorted(elo_by_team.items())}

    return EloArtifacts(matchups=matchup_df, season_ratings=season_df, latest_ratings=latest)


def calculate_elo_matchups(
    df: pd.DataFrame,
    initial_elo: float,
    k_factor: float,
    home_advantage: float,
    bucket_size: int,
    regression_factor: float = 0.75,
) -> pd.DataFrame:
    artifacts = _calculate_elo_artifacts(
        df=df,
        initial_elo=initial_elo,
        k_factor=k_factor,
        home_advantage=home_advantage,
        bucket_size=bucket_size,
        regression_factor=regression_factor,
    )
    return artifacts.matchups


def save_results(df: pd.DataFrame, path: str) -> None:
    output_path = Path(path)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_columns = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Output dataframe missing required columns: {missing_columns}")
    df.loc[:, OUTPUT_COLUMNS].to_csv(output_path, index=False)


def _save_season_ratings(df: pd.DataFrame, path: str) -> None:
    output_path = Path(path)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_columns = [c for c in SEASON_RATINGS_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Season ratings dataframe missing required columns: {missing_columns}")
    df.loc[:, SEASON_RATINGS_COLUMNS].to_csv(output_path, index=False)


def _save_latest_ratings_json(
    ratings: Dict[str, float],
    *,
    path: str,
    input_csv: str,
    games_processed: int,
    last_game_date: Optional[str],
    params: Dict[str, object],
) -> None:
    output_path = Path(path)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = {
        "model_version": "seasonal_mov_v2",
        "generated_by": "scripts/generate_historical_elo_matchups.py",
        "build_completed_at_utc": now_utc,
        "last_updated": now_utc,
        "source_csv": str(input_csv),
        "games_processed": int(games_processed),
        "last_game_date": last_game_date,
        "params": dict(params),
        "ratings": {team: _round_elo(value) for team, value in sorted(ratings.items())},
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate historical NBA Elo datasets.")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="historical_elo_matchups.csv")
    parser.add_argument("--season-ratings-output", default="elo_ratings_by_season.csv")
    parser.add_argument("--ratings-json-output", default=None)
    parser.add_argument("--initial-elo", type=float, default=1500.0)
    parser.add_argument("--k-factor", type=float, default=20.0)
    parser.add_argument("--home-advantage", type=float, default=100.0)
    parser.add_argument("--regression-factor", type=float, default=0.75)
    parser.add_argument("--bucket-size", type=int, default=25)
    parser.add_argument("--min-season", type=int, default=None)
    parser.add_argument("--seasons", type=str, default=None, help="Comma-separated season years to include")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.bucket_size <= 0:
        parser.error("--bucket-size must be a positive integer.")
    if not 0.0 <= float(args.regression_factor) <= 1.0:
        parser.error("--regression-factor must be between 0.0 and 1.0.")
    if args.min_season is not None and int(args.min_season) < 1:
        parser.error("--min-season must be >= 1 when provided.")

    try:
        explicit_seasons = _parse_seasons_arg(args.seasons)

        games = load_games(args.input)
        games = derive_season(games)

        if args.min_season is not None:
            games = games[games["season"] >= int(args.min_season)].copy()
        if explicit_seasons:
            games = games[games["season"].isin(sorted(explicit_seasons))].copy()
        if games.empty:
            raise ValueError("No games left after season filters.")

        artifacts = _calculate_elo_artifacts(
            df=games,
            initial_elo=float(args.initial_elo),
            k_factor=float(args.k_factor),
            home_advantage=float(args.home_advantage),
            bucket_size=int(args.bucket_size),
            regression_factor=float(args.regression_factor),
        )

        save_results(artifacts.matchups, args.output)
        _save_season_ratings(artifacts.season_ratings, args.season_ratings_output)

        if args.ratings_json_output:
            sorted_games = games.sort_values(
                ["date", "game_id"] if "game_id" in games.columns else ["date"],
                ascending=True,
                kind="mergesort",
            )
            last_game_date = sorted_games["date"].max()
            _save_latest_ratings_json(
                artifacts.latest_ratings,
                path=str(args.ratings_json_output),
                input_csv=str(args.input),
                games_processed=int(len(sorted_games)),
                last_game_date=(last_game_date.strftime("%Y-%m-%d") if pd.notna(last_game_date) else None),
                params={
                    "initial_elo": float(args.initial_elo),
                    "k_factor": float(args.k_factor),
                    "home_advantage": float(args.home_advantage),
                    "regression_factor": float(args.regression_factor),
                    "bucket_size": int(args.bucket_size),
                    "min_season": int(args.min_season) if args.min_season is not None else None,
                    "allowed_seasons": sorted(explicit_seasons) if explicit_seasons else None,
                },
            )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
