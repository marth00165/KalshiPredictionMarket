"""Deterministic NBA Elo engine for primary game winner probabilities."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    pd = None

logger = logging.getLogger(__name__)


# Current franchise abbreviations by NBA team_id (stable across historical names).
NBA_TEAM_ID_TO_ABBR: Dict[int, str] = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612739: "CLE",
    1610612740: "NOP",
    1610612741: "CHI",
    1610612742: "DAL",
    1610612743: "DEN",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612751: "BKN",
    1610612752: "NYK",
    1610612753: "ORL",
    1610612754: "IND",
    1610612755: "PHI",
    1610612756: "PHX",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612760: "OKC",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612763: "MEM",
    1610612764: "WAS",
    1610612765: "DET",
    1610612766: "CHA",
}

ABBR_ALIASES: Dict[str, str] = {
    "NOH": "NOP",
    "NOK": "NOP",
    "NJN": "BKN",
    "SEA": "OKC",
    "VAN": "MEM",
    "CHH": "CHA",
    "WSB": "WAS",
    "PHO": "PHX",
    "BRK": "BKN",
}


@dataclass(frozen=True)
class GameRecord:
    game_date: datetime
    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int


@dataclass(frozen=True)
class MarketMatchup:
    away_team: str
    home_team: str
    yes_team: str


class EloEngine:
    """
    NBA Elo engine built from historical games in chronological order.

    Design goals:
    - deterministic/reproducible
    - idempotent rebuilds
    - fast full rebuilds for ~100k rows
    """

    DEFAULT_INITIAL_RATING = 1300.0
    DEFAULT_HOME_ADVANTAGE = 100.0
    DEFAULT_K_FACTOR = 20.0
    DEFAULT_MAX_AGE_HOURS = 24
    DEFAULT_DATA_PATH = "context/kaggleGameData.csv"
    DEFAULT_OUTPUT_PATH = "app/outputs/elo_ratings.json"

    _KALSHI_NBA_SERIES = "KXNBAGAME"

    def __init__(
        self,
        data_path: str = DEFAULT_DATA_PATH,
        output_path: str = DEFAULT_OUTPUT_PATH,
        initial_rating: float = DEFAULT_INITIAL_RATING,
        home_advantage: float = DEFAULT_HOME_ADVANTAGE,
        k_factor: float = DEFAULT_K_FACTOR,
        max_age_hours: int = DEFAULT_MAX_AGE_HOURS,
        include_game_types: Optional[Iterable[str]] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.initial_rating = float(initial_rating)
        self.home_advantage = float(home_advantage)
        self.k_factor = float(k_factor)
        self.max_age_hours = max(1, int(max_age_hours))
        self.include_game_types = set(include_game_types or {
            "Regular Season",
            "Playoffs",
            "Play-in Tournament",
            "NBA Cup",
            "NBA Emirates Cup",
            "Emirates NBA Cup",
        })
        self.ratings: Dict[str, float] = {}
        self.games_processed: int = 0
        self.last_game_date: Optional[datetime] = None

    def load_games(self, as_of_date: Optional[str] = None) -> List[GameRecord]:
        """
        Load and normalize historical game rows sorted oldest -> newest.
        """
        if pd is not None:
            return self._load_games_with_pandas(as_of_date=as_of_date)
        logger.warning("pandas not installed; using CSV fallback loader for Elo.")
        return self._load_games_with_csv(as_of_date=as_of_date)

    def initialize_ratings(self, teams: Iterable[str]) -> None:
        """
        Reset ratings dictionary with default initial Elo for all teams.
        """
        unique = sorted({self._normalize_team_code(t) for t in teams if t})
        self.ratings = {team: float(self.initial_rating) for team in unique if team}

    def update_ratings(self, games: Iterable[GameRecord]) -> None:
        """
        Apply Elo updates in chronological order.
        """
        for game in games:
            home = game.home_team
            away = game.away_team
            if home not in self.ratings:
                self.ratings[home] = float(self.initial_rating)
            if away not in self.ratings:
                self.ratings[away] = float(self.initial_rating)

            home_elo = self.ratings[home]
            away_elo = self.ratings[away]

            expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + self.home_advantage)) / 400.0))
            expected_away = 1.0 - expected_home

            home_win = 1.0 if game.home_score > game.away_score else 0.0
            away_win = 1.0 - home_win

            new_home = home_elo + self.k_factor * (home_win - expected_home)
            new_away = away_elo + self.k_factor * (away_win - expected_away)

            self.ratings[home] = float(new_home)
            self.ratings[away] = float(new_away)
            self.games_processed += 1
            self.last_game_date = game.game_date

    def export_ratings(self) -> None:
        """
        Persist current ratings snapshot to JSON.
        """
        now_utc = datetime.now(timezone.utc)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_updated": now_utc.isoformat().replace("+00:00", "Z"),
            "last_game_date": (self.last_game_date.strftime("%Y-%m-%d") if self.last_game_date else None),
            "build_completed_at_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "games_processed": int(self.games_processed),
            "params": {
                "initial_rating": self.initial_rating,
                "home_advantage": self.home_advantage,
                "k_factor": self.k_factor,
            },
            "ratings": {
                team: round(value, 6)
                for team, value in sorted(self.ratings.items())
            },
        }
        with self.output_path.open("w") as f:
            json.dump(payload, f, indent=2)

    def rebuild(self, as_of_date: Optional[str] = None) -> Dict[str, float]:
        """
        Full deterministic rebuild from source CSV.
        """
        games = self.load_games(as_of_date=as_of_date)
        self.games_processed = 0
        self.last_game_date = None
        teams = set()
        for g in games:
            teams.add(g.home_team)
            teams.add(g.away_team)
        self.initialize_ratings(teams)
        self.update_ratings(games)
        self.export_ratings()
        return dict(self.ratings)

    def load_ratings(
        self,
        rebuild_if_missing: bool = True,
        as_of_date: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Load ratings from memory/file or rebuild if needed.
        """
        if self.ratings:
            return dict(self.ratings)

        if self.output_path.exists() and not self._output_is_stale():
            try:
                with self.output_path.open("r") as f:
                    payload = json.load(f)
                rating_map = payload.get("ratings", {})
                self.ratings = {
                    self._normalize_team_code(k): float(v)
                    for k, v in rating_map.items()
                    if self._normalize_team_code(k)
                }
                self.games_processed = int(payload.get("games_processed", 0) or 0)
                # Backward compatibility: older payloads used last_updated as last game date.
                last_game_date = payload.get("last_game_date", payload.get("last_updated"))
                self.last_game_date = (
                    datetime.strptime(last_game_date, "%Y-%m-%d")
                    if isinstance(last_game_date, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", last_game_date)
                    else None
                )
                return dict(self.ratings)
            except Exception as e:
                logger.warning("Failed loading Elo ratings JSON (%s), rebuilding.", e)

        if rebuild_if_missing:
            return self.rebuild(as_of_date=as_of_date)

        return {}

    def get_win_probability(
        self,
        team_a: str,
        team_b: str,
        team_a_is_home: bool = True,
        home_advantage: Optional[float] = None,
    ) -> float:
        """
        Probability that team_a wins using current Elo ratings.
        """
        self.load_ratings(rebuild_if_missing=True)
        a = self._normalize_team_code(team_a)
        b = self._normalize_team_code(team_b)
        if a not in self.ratings or b not in self.ratings:
            missing = [t for t in (a, b) if t not in self.ratings]
            raise KeyError(f"Missing Elo rating for team(s): {missing}")

        ha = self.home_advantage if home_advantage is None else float(home_advantage)
        a_elo = self.ratings[a]
        b_elo = self.ratings[b]
        rating_a = a_elo + ha if team_a_is_home else a_elo
        rating_b = b_elo if team_a_is_home else b_elo + ha
        prob = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
        return float(max(0.0, min(1.0, prob)))

    def parse_kalshi_nba_matchup(self, market_id: str) -> Optional[MarketMatchup]:
        """
        Parse Kalshi NBA market_id into away/home/yes team abbreviations.

        Example:
          KXNBAGAME-26FEB19BKNCLE-BKN
        """
        parts = str(market_id or "").strip().upper().split("-")
        if len(parts) < 3:
            return None
        if parts[0] != self._KALSHI_NBA_SERIES:
            return None

        matchup_chunk = parts[1]
        yes_team = self._normalize_team_code(parts[2])
        if len(matchup_chunk) < 6 or not yes_team:
            return None

        tail = matchup_chunk[-6:]
        if not re.fullmatch(r"[A-Z]{6}", tail):
            return None
        away_team = self._normalize_team_code(tail[:3])
        home_team = self._normalize_team_code(tail[3:])
        if not away_team or not home_team:
            return None
        if yes_team not in {away_team, home_team}:
            return None

        return MarketMatchup(
            away_team=away_team,
            home_team=home_team,
            yes_team=yes_team,
        )

    def get_market_yes_probability(self, market_id: str) -> Optional[float]:
        """
        Compute YES probability for a Kalshi NBA game winner market_id.
        """
        matchup = self.parse_kalshi_nba_matchup(market_id)
        if not matchup:
            return None

        home_win_prob = self.get_win_probability(
            team_a=matchup.home_team,
            team_b=matchup.away_team,
            team_a_is_home=True,
        )
        if matchup.yes_team == matchup.home_team:
            return home_win_prob
        return 1.0 - home_win_prob

    def _output_is_stale(self) -> bool:
        if not self.output_path.exists():
            return True
        if not self.data_path.exists():
            data_is_newer = False
        else:
            data_is_newer = self.output_path.stat().st_mtime < self.data_path.stat().st_mtime
        if data_is_newer:
            return True

        try:
            with self.output_path.open("r") as f:
                payload = json.load(f)
            last_updated = self._parse_last_updated_utc(payload.get("last_updated"))
            if last_updated is None:
                return True
            return datetime.now(timezone.utc) - last_updated > timedelta(hours=self.max_age_hours)
        except Exception:
            return True

    @staticmethod
    def _parse_last_updated_utc(value: object) -> Optional[datetime]:
        if not isinstance(value, str) or not value.strip():
            return None
        raw = value.strip()
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
                return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None

    def _load_games_with_pandas(self, as_of_date: Optional[str]) -> List[GameRecord]:
        required = {
            "date": ("GAME_DATE", "gameDateTimeEst", "gameDate"),
            "home_team": ("HOME_TEAM", "hometeamId", "homeTeam"),
            "away_team": ("VISITOR_TEAM", "awayteamId", "visitorTeam"),
            "home_score": ("HOME_SCORE", "homeScore"),
            "away_score": ("VISITOR_SCORE", "awayScore", "visitorScore"),
            "game_id": ("gameId", "GAME_ID"),
            "game_type": ("gameType", "GAME_TYPE"),
        }

        # Disable chunked dtype inference to avoid noisy mixed-type warnings on Kaggle exports.
        df = pd.read_csv(self.data_path, low_memory=False)  # type: ignore[union-attr]
        if df.empty:
            return []

        col_date = self._pick_column(df.columns, required["date"])
        col_home = self._pick_column(df.columns, required["home_team"])
        col_away = self._pick_column(df.columns, required["away_team"])
        col_home_score = self._pick_column(df.columns, required["home_score"])
        col_away_score = self._pick_column(df.columns, required["away_score"])
        col_game_id = self._pick_column(df.columns, required["game_id"])
        col_game_type = self._pick_column(df.columns, required["game_type"], required=False)

        if not (col_date and col_home and col_away and col_home_score and col_away_score):
            missing = [
                name for name, value in (
                    ("date", col_date),
                    ("home_team", col_home),
                    ("away_team", col_away),
                    ("home_score", col_home_score),
                    ("away_score", col_away_score),
                ) if not value
            ]
            raise ValueError(f"Elo dataset missing required columns: {missing}")

        if col_game_type:
            df = df[df[col_game_type].isin(self.include_game_types)]

        # Avoid leading-underscore temporary column names here because pandas
        # may remap those names in namedtuple iteration (e.g. "_22"), which can
        # break downstream key lookups.
        norm_date = "game_date_norm"
        norm_home = "home_team_norm"
        norm_away = "away_team_norm"
        norm_home_score = "home_score_norm"
        norm_away_score = "away_score_norm"

        df = df.assign(
            **{
                norm_date: pd.to_datetime(df[col_date], errors="coerce"),
                norm_home: df[col_home].map(self._normalize_team_code),
                norm_away: df[col_away].map(self._normalize_team_code),
                norm_home_score: pd.to_numeric(df[col_home_score], errors="coerce"),
                norm_away_score: pd.to_numeric(df[col_away_score], errors="coerce"),
            }
        )

        df = df.dropna(subset=[norm_date, norm_home, norm_away, norm_home_score, norm_away_score])
        if as_of_date:
            cutoff = pd.to_datetime(as_of_date, errors="coerce")
            if pd.notna(cutoff):
                df = df[df[norm_date] < cutoff]

        sort_cols = [norm_date]
        if col_game_id:
            sort_cols.append(col_game_id)
        df = df.sort_values(sort_cols, kind="mergesort")

        selected_cols = [norm_date, norm_home, norm_away, norm_home_score, norm_away_score]
        if col_game_id:
            selected_cols.append(col_game_id)

        records: List[GameRecord] = []
        for row in df[selected_cols].itertuples(index=False, name=None):
            if col_game_id:
                game_date, home_team, away_team, home_score, away_score, game_id_raw = row
                game_id = str(game_id_raw)
            else:
                game_date, home_team, away_team, home_score, away_score = row
                game_id = ""
            records.append(
                GameRecord(
                    game_date=game_date.to_pydatetime(),
                    game_id=game_id,
                    home_team=str(home_team),
                    away_team=str(away_team),
                    home_score=int(home_score),
                    away_score=int(away_score),
                )
            )
        return records

    def _load_games_with_csv(self, as_of_date: Optional[str]) -> List[GameRecord]:
        with self.data_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            fields = reader.fieldnames
            col_date = self._pick_column(fields, ("GAME_DATE", "gameDateTimeEst", "gameDate"))
            col_home = self._pick_column(fields, ("HOME_TEAM", "hometeamId", "homeTeam"))
            col_away = self._pick_column(fields, ("VISITOR_TEAM", "awayteamId", "visitorTeam"))
            col_home_score = self._pick_column(fields, ("HOME_SCORE", "homeScore"))
            col_away_score = self._pick_column(fields, ("VISITOR_SCORE", "awayScore", "visitorScore"))
            col_game_id = self._pick_column(fields, ("gameId", "GAME_ID"), required=False)
            col_game_type = self._pick_column(fields, ("gameType", "GAME_TYPE"), required=False)

            if not (col_date and col_home and col_away and col_home_score and col_away_score):
                raise ValueError("Elo dataset missing required columns")

            cutoff = self._parse_datetime(as_of_date) if as_of_date else None
            records: List[GameRecord] = []
            for row in reader:
                if col_game_type and row.get(col_game_type) not in self.include_game_types:
                    continue
                game_dt = self._parse_datetime(row.get(col_date))
                if game_dt is None:
                    continue
                if cutoff is not None and not (game_dt < cutoff):
                    continue
                home_team = self._normalize_team_code(row.get(col_home))
                away_team = self._normalize_team_code(row.get(col_away))
                if not home_team or not away_team:
                    continue
                try:
                    home_score = int(float(row.get(col_home_score, "")))
                    away_score = int(float(row.get(col_away_score, "")))
                except Exception:
                    continue
                records.append(
                    GameRecord(
                        game_date=game_dt,
                        game_id=str(row.get(col_game_id, "")) if col_game_id else "",
                        home_team=home_team,
                        away_team=away_team,
                        home_score=home_score,
                        away_score=away_score,
                    )
                )

        records.sort(key=lambda g: (g.game_date, g.game_id))
        return records

    @staticmethod
    def _pick_column(
        columns: Iterable[str],
        candidates: Iterable[str],
        required: bool = True,
    ) -> Optional[str]:
        normalized = {str(c).strip().lower(): str(c) for c in columns}
        for candidate in candidates:
            key = str(candidate).strip().lower()
            if key in normalized:
                return normalized[key]
        if required:
            return None
        return None

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(raw, fmt)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return None

    @staticmethod
    def _normalize_team_code(token: object) -> str:
        if token is None:
            return ""

        # Numeric team_id
        try:
            if isinstance(token, (int, float)) and not math.isnan(float(token)):  # type: ignore[arg-type]
                team_id = int(float(token))
                mapped = NBA_TEAM_ID_TO_ABBR.get(team_id, "")
                return mapped
        except Exception:
            pass

        raw = str(token).strip().upper()
        if not raw:
            return ""

        # Team ID encoded as string.
        if raw.isdigit():
            mapped = NBA_TEAM_ID_TO_ABBR.get(int(raw), "")
            if mapped:
                return mapped
            return ""

        # Already abbreviation (or alias).
        if re.fullmatch(r"[A-Z]{2,4}", raw):
            canonical = ABBR_ALIASES.get(raw, raw)
            return canonical if re.fullmatch(r"[A-Z]{3}", canonical) else ""

        return ""


_DEFAULT_ENGINE: Optional[EloEngine] = None


def load_ratings(
    data_path: str = EloEngine.DEFAULT_DATA_PATH,
    output_path: str = EloEngine.DEFAULT_OUTPUT_PATH,
    force_rebuild: bool = False,
) -> Dict[str, float]:
    """
    Convenience loader for integration paths that expect module-level functions.
    """
    global _DEFAULT_ENGINE
    if (
        _DEFAULT_ENGINE is None
        or str(_DEFAULT_ENGINE.data_path) != str(Path(data_path))
        or str(_DEFAULT_ENGINE.output_path) != str(Path(output_path))
    ):
        _DEFAULT_ENGINE = EloEngine(data_path=data_path, output_path=output_path)

    if force_rebuild:
        return _DEFAULT_ENGINE.rebuild()
    return _DEFAULT_ENGINE.load_ratings(rebuild_if_missing=True)


def get_win_probability(
    team_a: str,
    team_b: str,
    team_a_is_home: bool = True,
    home_advantage: Optional[float] = None,
) -> float:
    """
    Convenience probability helper backed by module singleton.
    """
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        load_ratings()
    if _DEFAULT_ENGINE is None:
        raise RuntimeError("Failed to initialize Elo engine")
    return _DEFAULT_ENGINE.get_win_probability(
        team_a=team_a,
        team_b=team_b,
        team_a_is_home=team_a_is_home,
        home_advantage=home_advantage,
    )
