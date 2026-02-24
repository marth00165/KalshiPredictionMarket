"""Injury-aware LLM refresh and caching utilities for NBA analyzers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.api_clients.base_client import RateLimitError
from app.api_clients.sportradar_client import SportsRadarClient, SportsRadarConfig

logger = logging.getLogger(__name__)


_TEAM_ALIASES = {
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

_PIM_STATUS_RANK = {
    "OUT": 5,
    "DOUBTFUL": 4,
    "QUESTIONABLE": 3,
    "PROBABLE": 2,
    "AVAILABLE": 1,
    "UNKNOWN": 0,
}

_TEAM_PROFILE_STATUS_MAP = {
    "SUS": "OUT",
    "SUSPENDED": "OUT",
    "OUT": "OUT",
    "INJ": "OUT",
    "INACTIVE": "OUT",
    "DOUBTFUL": "DOUBTFUL",
    "DOUBT": "DOUBTFUL",
    "DOUT": "DOUBTFUL",
    "QUESTIONABLE": "QUESTIONABLE",
    "Q": "QUESTIONABLE",
    "DTD": "QUESTIONABLE",
    "GTD": "QUESTIONABLE",
    "PROB": "PROBABLE",
    "PROBABLE": "PROBABLE",
    "AVAILABLE": "AVAILABLE",
    "ACTIVE": "AVAILABLE",
}

_LOW_MINUTES_PERCENTILE = 0.10

# NBA team_id -> current franchise abbreviation
_NBA_TEAM_ID_TO_ABBR = {
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


def normalize_team_code(value: object) -> str:
    raw = re.sub(r"[^A-Za-z0-9]", "", str(value or "").upper())
    if not raw:
        return ""
    if raw.isdigit():
        # SportsRadar frequently exposes NBA team references as numeric team IDs.
        try:
            mapped = _NBA_TEAM_ID_TO_ABBR.get(int(raw))
        except Exception:
            mapped = None
        return mapped or ""
    if raw in _TEAM_ALIASES:
        return _TEAM_ALIASES[raw]
    return raw[:3]


def normalize_player_name(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text)


def normalize_injury_status(value: object) -> str:
    """Map provider-specific text to a compact, deterministic status enum."""
    raw = str(value or "").strip().lower()
    if not raw:
        return "UNKNOWN"

    if re.search(r"\bout\b", raw) or "inactive" in raw or "suspended" in raw or "ir" == raw:
        return "OUT"
    if "doubt" in raw:
        return "DOUBTFUL"
    if "question" in raw or "gtd" in raw or "game-time decision" in raw:
        return "QUESTIONABLE"
    if "probable" in raw or "day-to-day" in raw or "day to day" in raw:
        return "PROBABLE"
    if "available" in raw or "active" in raw or "healthy" in raw:
        return "AVAILABLE"
    return "UNKNOWN"


def _normalize_injury_text(value: object, *, max_len: int = 120) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text[:max_len]


def build_matchup_injury_snapshot(
    injuries_payload: Dict[str, Any],
    *,
    home_team: str,
    away_team: str,
) -> Dict[str, Any]:
    """
    Build a canonical injury snapshot for a single matchup from SportsRadar payload.

    Snapshot includes only semantic fields and stable ordering for deterministic hashing.
    """
    home = normalize_team_code(home_team)
    away = normalize_team_code(away_team)

    teams = injuries_payload.get("teams")
    if not isinstance(teams, list):
        teams = []

    team_index: Dict[str, Dict[str, Any]] = {}
    for team in teams:
        if not isinstance(team, dict):
            continue
        ref = normalize_team_code(team.get("reference") or team.get("abbr") or team.get("alias"))
        if not ref:
            continue
        team_index[ref] = team

    snapshot_teams: List[Dict[str, Any]] = []
    for team_code in sorted({home, away}):
        team_data = team_index.get(team_code, {})
        players_raw = team_data.get("players")
        if not isinstance(players_raw, list):
            players_raw = []

        canonical_players: List[Dict[str, Any]] = []
        for player in players_raw:
            if not isinstance(player, dict):
                continue
            injuries = player.get("injuries")
            if not isinstance(injuries, list) or not injuries:
                continue

            canonical_injuries: List[Dict[str, str]] = []
            for injury in injuries:
                if not isinstance(injury, dict):
                    continue
                status = normalize_injury_status(injury.get("status"))
                desc = _normalize_injury_text(injury.get("desc"))
                comment = _normalize_injury_text(injury.get("comment"))
                canonical_injuries.append(
                    {
                        "status": status,
                        "desc": desc,
                        "comment": comment,
                    }
                )
            if not canonical_injuries:
                continue

            canonical_injuries = sorted(
                canonical_injuries,
                key=lambda row: (
                    str(row.get("status") or ""),
                    str(row.get("desc") or ""),
                    str(row.get("comment") or ""),
                ),
            )

            player_id = (
                str(player.get("id") or "")
                or str(player.get("reference") or "")
                or str(player.get("sr_id") or "")
            )
            canonical_players.append(
                {
                    "player_id": player_id,
                    "player_name": normalize_player_name(player.get("full_name")),
                    "injuries": canonical_injuries,
                }
            )

        canonical_players = sorted(
            canonical_players,
            key=lambda row: (
                str(row.get("player_id") or ""),
                str(row.get("player_name") or ""),
            ),
        )
        snapshot_teams.append(
            {
                "team": team_code,
                "players": canonical_players,
            }
        )

    return {
        "home_team": home,
        "away_team": away,
        "teams": snapshot_teams,
    }


def fingerprint_injury_snapshot(snapshot: Dict[str, Any]) -> str:
    serialized = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LLMRefreshDecision:
    should_refresh: bool
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _parse_timestamp(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _to_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def should_refresh_llm(
    *,
    cache_entry: Optional[Dict[str, Any]],
    current_injury_fingerprint: str,
    now: Optional[datetime] = None,
    tipoff_time: Optional[datetime] = None,
    current_market_price: Optional[float] = None,
    max_age_seconds: int = 1800,
    near_tipoff_minutes: int = 45,
    near_tipoff_stale_seconds: int = 600,
    price_move_threshold_pct: float = 0.03,
    force_refresh: bool = False,
) -> LLMRefreshDecision:
    now_utc = _to_utc(now) or datetime.now(timezone.utc)
    tipoff_utc = _to_utc(tipoff_time)

    if force_refresh:
        return LLMRefreshDecision(True, "force_refresh", {})

    if not isinstance(cache_entry, dict) or not cache_entry:
        return LLMRefreshDecision(True, "cache_miss", {"cached_delta_age_seconds": None})

    cached_fingerprint = str(cache_entry.get("injury_fingerprint", "") or "")
    if cached_fingerprint != current_injury_fingerprint:
        return LLMRefreshDecision(
            True,
            "fingerprint_changed",
            {
                "cached_fingerprint": cached_fingerprint,
                "current_fingerprint": current_injury_fingerprint,
            },
        )

    generated_at = _parse_timestamp(cache_entry.get("generated_at"))
    if generated_at is None:
        age_seconds = math.inf
    else:
        age_seconds = max((now_utc - generated_at).total_seconds(), 0.0)

    if age_seconds > float(max_age_seconds):
        return LLMRefreshDecision(
            True,
            "ttl_expired",
            {"cached_delta_age_seconds": float(age_seconds)},
        )

    if tipoff_utc is not None:
        seconds_to_tipoff = (tipoff_utc - now_utc).total_seconds()
        near_tipoff_seconds = float(max(near_tipoff_minutes, 0)) * 60.0
        if seconds_to_tipoff <= near_tipoff_seconds and age_seconds > float(near_tipoff_stale_seconds):
            return LLMRefreshDecision(
                True,
                "near_tipoff_stale",
                {
                    "seconds_to_tipoff": float(seconds_to_tipoff),
                    "cached_delta_age_seconds": float(age_seconds),
                },
            )

    if current_market_price is not None and float(price_move_threshold_pct) > 0:
        try:
            current_price = float(current_market_price)
            prev_price = float(cache_entry.get("last_market_price_seen"))
            denom = max(abs(prev_price), 1e-6)
            move = abs(current_price - prev_price) / denom
            if move >= float(price_move_threshold_pct):
                return LLMRefreshDecision(
                    True,
                    "large_price_move",
                    {
                        "price_move_pct": float(move),
                        "price_move_threshold_pct": float(price_move_threshold_pct),
                    },
                )
        except Exception:
            pass

    return LLMRefreshDecision(
        False,
        "cache_hit_reuse",
        {"cached_delta_age_seconds": float(age_seconds)},
    )


class InjuryLLMCacheStore:
    """Local JSON cache for per-matchup LLM injury deltas."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._loaded = False
        self._entries: Dict[str, Dict[str, Any]] = {}

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            self._entries = {}
            return
        try:
            payload = json.loads(self.path.read_text())
            entries = payload.get("entries", {})
            self._entries = entries if isinstance(entries, dict) else {}
        except Exception as e:
            logger.warning("Injury LLM cache read failed (%s). Resetting cache in-memory.", e)
            self._entries = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "entries": self._entries,
        }
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(self.path)

    def get(self, matchup_key: str) -> Optional[Dict[str, Any]]:
        self._load()
        entry = self._entries.get(str(matchup_key))
        if not isinstance(entry, dict):
            return None
        return dict(entry)

    def upsert(self, matchup_key: str, entry: Dict[str, Any]) -> None:
        self._load()
        self._entries[str(matchup_key)] = dict(entry)
        try:
            self._save()
        except Exception as e:
            logger.warning("Injury LLM cache write failed (%s). Continuing without persistence.", e)


class SportsRadarFeedCacheStore:
    """Local JSON cache for SportsRadar league injuries payload."""

    def __init__(self, path: str):
        self.path = Path(path)

    def load(
        self,
        *,
        max_age_seconds: Optional[int] = None,
        allow_stale: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        if not self.path.exists():
            return (None, None)

        try:
            payload = json.loads(self.path.read_text())
        except Exception as e:
            logger.warning("SportsRadar feed cache read failed (%s). Ignoring cached feed.", e)
            return (None, None)

        injuries_payload = payload.get("payload")
        if not isinstance(injuries_payload, dict):
            return (None, None)

        updated_at = _parse_timestamp(payload.get("updated_at"))
        age_seconds: Optional[float] = None
        if updated_at is not None:
            age_seconds = max((datetime.now(timezone.utc) - updated_at).total_seconds(), 0.0)

        if not allow_stale and max_age_seconds is not None and age_seconds is not None:
            if age_seconds > float(max_age_seconds):
                return (None, age_seconds)

        return (injuries_payload, age_seconds)

    def save(self, injuries_payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "payload": injuries_payload,
        }
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(self.path)


class SportsRadarProfileCacheStore:
    """Local JSON cache for SportsRadar team/player profile payloads."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._loaded = False
        self._team_entries: Dict[str, Dict[str, Any]] = {}
        self._player_entries: Dict[str, Dict[str, Any]] = {}

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            self._team_entries = {}
            self._player_entries = {}
            return
        try:
            payload = json.loads(self.path.read_text())
            team_entries = payload.get("team_profiles", {})
            player_entries = payload.get("player_profiles", {})
            self._team_entries = team_entries if isinstance(team_entries, dict) else {}
            self._player_entries = player_entries if isinstance(player_entries, dict) else {}
        except Exception as e:
            logger.warning("SportsRadar profile cache read failed (%s). Resetting cache in-memory.", e)
            self._team_entries = {}
            self._player_entries = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "team_profiles": self._team_entries,
            "player_profiles": self._player_entries,
        }
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(self.path)

    def snapshot(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        self._load()
        return (dict(self._team_entries), dict(self._player_entries))

    def upsert_team(self, team_id: str, payload: Dict[str, Any], *, updated_at: Optional[datetime] = None) -> None:
        self._load()
        team_key = str(team_id or "").strip()
        if not team_key or not isinstance(payload, dict):
            return
        ts = (updated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        self._team_entries[team_key] = {
            "updated_at": ts.isoformat().replace("+00:00", "Z"),
            "payload": payload,
        }
        try:
            self._save()
        except Exception as e:
            logger.warning("SportsRadar profile cache write failed (%s). Continuing without persistence.", e)

    def upsert_player(self, player_id: str, payload: Dict[str, Any], *, updated_at: Optional[datetime] = None) -> None:
        self._load()
        player_key = str(player_id or "").strip()
        if not player_key or not isinstance(payload, dict):
            return
        ts = (updated_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        self._player_entries[player_key] = {
            "updated_at": ts.isoformat().replace("+00:00", "Z"),
            "payload": payload,
        }
        try:
            self._save()
        except Exception as e:
            logger.warning("SportsRadar profile cache write failed (%s). Continuing without persistence.", e)


@dataclass
class InjuryLLMRefreshConfig:
    enabled: bool = False
    enable_injury_llm_cache: bool = True
    injury_cache_file: str = "context/injury_llm_cache.json"
    llm_refresh_max_age_seconds: int = 1800
    force_llm_refresh_near_tipoff_minutes: int = 45
    near_tipoff_llm_stale_seconds: int = 600
    llm_refresh_on_price_move_pct: float = 0.03
    injury_analysis_version: str = "injury-v2"
    injury_prompt_version: str = "injury-prompt-v1"
    force_injury_llm_refresh: bool = False
    injury_feed_cache_ttl_seconds: int = 120
    injury_profile_cache_ttl_seconds: int = 14400
    team_profile_budget_window_seconds: int = 120
    max_team_profiles_per_cycle: int = 12
    show_injury_snapshot_in_prompt: bool = True


@dataclass
class MarketInjuryContext:
    matchup_key: str
    snapshot: Dict[str, Any]
    fingerprint: str
    fingerprint_short: str
    yes_team_code: str
    decision: LLMRefreshDecision
    cache_entry: Optional[Dict[str, Any]]
    tipoff_time: Optional[datetime]


class InjuryLLMRefreshService:
    """Coordinates SportsRadar feed, fingerprinting, and LLM-delta cache policy."""

    def __init__(
        self,
        *,
        config: InjuryLLMRefreshConfig,
        sportsradar_api_key: Optional[str],
        sportsradar_base_url: str = "https://api.sportradar.com/nba/trial/v8/en",
    ):
        self.config = config
        self.sportsradar_api_key = str(sportsradar_api_key or "").strip()
        self.enabled = bool(self.config.enabled and self.sportsradar_api_key)
        self._cache = InjuryLLMCacheStore(self.config.injury_cache_file)
        self._feed_cache = SportsRadarFeedCacheStore(
            self._derive_feed_cache_file(self.config.injury_cache_file)
        )
        self._profile_cache = SportsRadarProfileCacheStore(
            self._derive_profile_cache_file(self.config.injury_cache_file)
        )
        self._feed_lock = asyncio.Lock()
        self._feed_payload: Dict[str, Any] = {}
        self._feed_updated_at: float = 0.0
        self._sportsradar_base_url = sportsradar_base_url
        self._team_id_by_code: Dict[str, str] = {}
        self._active_slate_team_codes: Optional[Set[str]] = None
        self._team_profile_cache: Dict[str, Dict[str, Any]] = {}
        self._team_profile_updated_at: Dict[str, float] = {}
        self._team_profile_inflight: Dict[str, asyncio.Task] = {}
        self._player_profile_cache: Dict[str, Dict[str, Any]] = {}
        self._player_profile_updated_at: Dict[str, float] = {}
        self._player_profile_inflight: Dict[str, asyncio.Task] = {}
        self._rate_limited_until: float = 0.0
        self._rate_limited_retry_after_seconds: int = 0
        self._team_profile_budget_window_started_at: float = time.time()
        self._team_profile_fetch_count_in_window: int = 0

        self._load_feed_cache_if_available(
            max_age_seconds=max(int(self.config.injury_feed_cache_ttl_seconds), 1),
            allow_stale=False,
            reason="startup",
        )
        self._load_profile_cache_if_available()

        if self.config.enabled and not self.sportsradar_api_key:
            logger.warning("Live injury news enabled but SportsRadar API key missing; proceeding without feed.")

    @staticmethod
    def _derive_feed_cache_file(injury_cache_file: str) -> str:
        path = Path(str(injury_cache_file or "").strip() or "context/injury_llm_cache.json")
        suffix = path.suffix or ".json"
        return str(path.with_name(f"{path.stem}_feed{suffix}"))

    @staticmethod
    def _derive_profile_cache_file(injury_cache_file: str) -> str:
        path = Path(str(injury_cache_file or "").strip() or "context/injury_llm_cache.json")
        suffix = path.suffix or ".json"
        return str(path.with_name(f"{path.stem}_profiles{suffix}"))

    def _load_feed_cache_if_available(
        self,
        *,
        max_age_seconds: Optional[int],
        allow_stale: bool,
        reason: str,
    ) -> bool:
        payload, age_seconds = self._feed_cache.load(
            max_age_seconds=max_age_seconds,
            allow_stale=allow_stale,
        )
        if not isinstance(payload, dict) or not payload:
            return False
        self._feed_payload = payload
        if age_seconds is not None and age_seconds >= 0:
            self._feed_updated_at = max(time.time() - float(age_seconds), 0.0)
        else:
            self._feed_updated_at = time.time()
        self._index_team_ids_from_injuries_payload(payload)
        logger.info(
            "Loaded SportsRadar injury feed cache (%s; age=%ss; teams=%d)",
            reason,
            "unknown" if age_seconds is None else str(int(round(age_seconds))),
            len(payload.get("teams") or []) if isinstance(payload.get("teams"), list) else 0,
        )
        return True

    def _persist_feed_cache(self, payload: Dict[str, Any]) -> None:
        try:
            self._feed_cache.save(payload)
        except Exception as e:
            logger.warning("SportsRadar feed cache write failed (%s). Continuing without persistence.", e)

    def _load_profile_cache_if_available(self) -> None:
        team_entries, player_entries = self._profile_cache.snapshot()

        loaded_team = 0
        for team_id, entry in team_entries.items():
            if not isinstance(entry, dict):
                continue
            payload = entry.get("payload")
            if not isinstance(payload, dict) or not payload:
                continue
            updated_at = _parse_timestamp(entry.get("updated_at"))
            updated_ts = updated_at.timestamp() if updated_at is not None else 0.0
            team_id_norm = str(team_id or "").strip()
            if not team_id_norm:
                continue
            self._team_profile_cache[team_id_norm] = payload
            self._team_profile_updated_at[team_id_norm] = float(updated_ts)
            profile_team_code = self._team_code_from_profile_payload(payload)
            if profile_team_code:
                self._team_id_by_code[profile_team_code] = team_id_norm
            loaded_team += 1

        loaded_player = 0
        for player_id, entry in player_entries.items():
            if not isinstance(entry, dict):
                continue
            payload = entry.get("payload")
            if not isinstance(payload, dict) or not payload:
                continue
            updated_at = _parse_timestamp(entry.get("updated_at"))
            updated_ts = updated_at.timestamp() if updated_at is not None else 0.0
            player_id_norm = str(player_id or "").strip()
            if not player_id_norm:
                continue
            self._player_profile_cache[player_id_norm] = payload
            self._player_profile_updated_at[player_id_norm] = float(updated_ts)
            loaded_player += 1

        if loaded_team or loaded_player:
            logger.info(
                "Loaded SportsRadar profile cache (teams=%d, players=%d)",
                loaded_team,
                loaded_player,
            )

    def _maybe_reset_team_profile_budget(self) -> None:
        window_seconds = max(int(self.config.team_profile_budget_window_seconds), 1)
        now = time.time()
        if (now - self._team_profile_budget_window_started_at) >= float(window_seconds):
            self._team_profile_budget_window_started_at = now
            self._team_profile_fetch_count_in_window = 0

    def _can_fetch_team_profile_now(self) -> bool:
        self._maybe_reset_team_profile_budget()
        budget = max(int(self.config.max_team_profiles_per_cycle), 1)
        if self._team_profile_fetch_count_in_window >= budget:
            return False
        self._team_profile_fetch_count_in_window += 1
        return True

    def _is_rate_limited(self) -> bool:
        return time.time() < self._rate_limited_until

    def _remaining_rate_limit_seconds(self) -> int:
        remaining = int(round(self._rate_limited_until - time.time()))
        return max(0, remaining)

    def _mark_rate_limited(self, retry_after_seconds: Optional[int]) -> None:
        retry_after = max(1, int(retry_after_seconds or 60))
        self._rate_limited_retry_after_seconds = retry_after
        self._rate_limited_until = max(self._rate_limited_until, time.time() + float(retry_after))
        logger.warning(
            "SportsRadar rate limited; using cached injury data when available for ~%ds",
            self._remaining_rate_limit_seconds(),
        )

    @staticmethod
    def _is_nba_kalshi_market_id(market_id: str, series_ticker: object = "") -> bool:
        series = str(series_ticker or "").strip().upper()
        market = str(market_id or "").strip().upper()
        return series == "KXNBAGAME" or market.startswith("KXNBAGAME-")

    @classmethod
    def _extract_team_codes_from_market_id(cls, market_id: str, series_ticker: object = "") -> Tuple[str, str]:
        if not cls._is_nba_kalshi_market_id(market_id=market_id, series_ticker=series_ticker):
            return ("", "")
        parts = str(market_id or "").strip().upper().split("-")
        if len(parts) < 2:
            return ("", "")
        matchup_chunk = parts[1]
        if len(matchup_chunk) < 6:
            return ("", "")
        tail = matchup_chunk[-6:]
        if not re.fullmatch(r"[A-Z]{6}", tail):
            return ("", "")
        away_team = normalize_team_code(tail[:3])
        home_team = normalize_team_code(tail[3:])
        if not away_team or not home_team:
            return ("", "")
        return (away_team, home_team)

    @classmethod
    def _extract_market_date_from_market_id(
        cls,
        market_id: str,
        series_ticker: object = "",
    ) -> Optional[date]:
        if not cls._is_nba_kalshi_market_id(market_id=market_id, series_ticker=series_ticker):
            return None
        token = str(market_id or "").strip().upper()
        match = re.search(r"-(\d{2}[A-Z]{3}\d{2})", token)
        if not match:
            return None
        code = match.group(1).title()
        try:
            return datetime.strptime(code, "%y%b%d").date()
        except Exception:
            return None

    def build_slate_team_set_from_markets(
        self,
        markets: Iterable[object],
        *,
        today_only: bool = True,
    ) -> Set[str]:
        team_set: Set[str] = set()
        today_utc = datetime.now(timezone.utc).date()
        for market in markets:
            market_id = getattr(market, "market_id", "")
            series_ticker = getattr(market, "series_ticker", "")
            market_date = self._extract_market_date_from_market_id(
                market_id=str(market_id),
                series_ticker=series_ticker,
            )
            if today_only and market_date != today_utc:
                continue
            away_team, home_team = self._extract_team_codes_from_market_id(
                market_id=str(market_id),
                series_ticker=series_ticker,
            )
            if away_team:
                team_set.add(away_team)
            if home_team:
                team_set.add(home_team)
        return team_set

    @classmethod
    def build_game_cache_key(
        cls,
        *,
        market_id: str,
        series_ticker: object = "",
        home_team: str = "",
        away_team: str = "",
    ) -> str:
        away_from_id, home_from_id = cls._extract_team_codes_from_market_id(
            market_id=str(market_id or ""),
            series_ticker=series_ticker,
        )
        market_date = cls._extract_market_date_from_market_id(
            market_id=str(market_id or ""),
            series_ticker=series_ticker,
        )
        away_code = away_from_id or normalize_team_code(away_team)
        home_code = home_from_id or normalize_team_code(home_team)

        if away_code and home_code and market_date is not None:
            return f"KXNBAGAME-{market_date.isoformat()}-{away_code}-{home_code}"
        if away_code and home_code:
            return f"KXNBAGAME-{away_code}-{home_code}"
        return str(market_id or "").strip()

    @staticmethod
    def _snapshot_has_actionable_absence(snapshot: Dict[str, Any], team_code: str) -> bool:
        teams = snapshot.get("teams")
        if not isinstance(teams, list):
            return False
        target = normalize_team_code(team_code)
        if not target:
            return False
        for team in teams:
            if not isinstance(team, dict):
                continue
            if normalize_team_code(team.get("team")) != target:
                continue
            players = team.get("players")
            if not isinstance(players, list):
                return False
            for player in players:
                if not isinstance(player, dict):
                    continue
                injuries = player.get("injuries")
                if not isinstance(injuries, list):
                    continue
                for injury in injuries:
                    if not isinstance(injury, dict):
                        continue
                    status = str(injury.get("status") or "").strip().upper()
                    if status in {"OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE"}:
                        return True
            return False
        return False

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _canonicalize_game_cache_entry(
        self,
        *,
        entry: Dict[str, Any],
        yes_team_code: str,
        home_team_code: str,
        away_team_code: str,
    ) -> Dict[str, Any]:
        """Normalize cache entry so it stores home-side delta canonically."""
        canonical = dict(entry or {})
        canonical["home_team_code"] = home_team_code
        canonical["away_team_code"] = away_team_code

        if "canonical_home_delta" in canonical:
            canonical["canonical_home_delta"] = self._safe_float(canonical.get("canonical_home_delta"), 0.0)
            return canonical

        legacy_delta = self._safe_float(canonical.get("llm_delta"), 0.0)
        if yes_team_code == home_team_code:
            canonical["canonical_home_delta"] = legacy_delta
        elif yes_team_code == away_team_code:
            canonical["canonical_home_delta"] = -legacy_delta
        else:
            canonical["canonical_home_delta"] = legacy_delta
        return canonical

    def _project_cache_entry_for_yes_team(
        self,
        *,
        entry: Optional[Dict[str, Any]],
        yes_team_code: str,
        home_team_code: str,
        away_team_code: str,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(entry, dict):
            return None
        canonical_home_delta = self._safe_float(entry.get("canonical_home_delta"), 0.0)
        projected = dict(entry)
        if yes_team_code == home_team_code:
            projected["llm_delta"] = canonical_home_delta
        elif yes_team_code == away_team_code:
            projected["llm_delta"] = -canonical_home_delta
        else:
            projected["llm_delta"] = self._safe_float(projected.get("llm_delta"), 0.0)
        return projected

    def _index_team_ids_from_injuries_payload(self, payload: Dict[str, Any]) -> None:
        teams = payload.get("teams")
        if not isinstance(teams, list):
            return
        for team in teams:
            if not isinstance(team, dict):
                continue
            team_code = normalize_team_code(team.get("reference") or team.get("alias") or team.get("abbr"))
            team_id = str(team.get("id") or "").strip()
            if not team_code or not team_id:
                continue
            self._team_id_by_code[team_code] = team_id

    async def refresh_injury_feed_if_needed(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        if self._is_rate_limited():
            if not self._feed_payload:
                self._load_feed_cache_if_available(
                    max_age_seconds=None,
                    allow_stale=True,
                    reason="rate_limited",
                )
            return

        now = time.time()
        ttl = max(int(self.config.injury_feed_cache_ttl_seconds), 1)
        if not force and not self._feed_payload:
            self._load_feed_cache_if_available(
                max_age_seconds=ttl,
                allow_stale=False,
                reason="ttl_reuse",
            )
        if not force and self._feed_payload and (now - self._feed_updated_at) < ttl:
            return

        async with self._feed_lock:
            now2 = time.time()
            if not force and not self._feed_payload:
                self._load_feed_cache_if_available(
                    max_age_seconds=ttl,
                    allow_stale=False,
                    reason="lock_ttl_reuse",
                )
            if not force and self._feed_payload and (now2 - self._feed_updated_at) < ttl:
                return
            try:
                client = SportsRadarClient(
                    SportsRadarConfig(
                        api_key=self.sportsradar_api_key,
                        base_url=self._sportsradar_base_url,
                        max_retries=0,
                    )
                )
                async with client:
                    payload = await client.fetch_league_injuries()
                if isinstance(payload, dict):
                    self._feed_payload = payload
                    self._feed_updated_at = time.time()
                    self._index_team_ids_from_injuries_payload(payload)
                    self._persist_feed_cache(payload)
                    logger.info(
                        "Injury feed refreshed from SportsRadar (teams=%d)",
                        len(payload.get("teams") or []) if isinstance(payload.get("teams"), list) else 0,
                    )
            except RateLimitError as e:
                self._mark_rate_limited(getattr(e, "retry_after", None))
                if not self._feed_payload:
                    self._load_feed_cache_if_available(
                        max_age_seconds=None,
                        allow_stale=True,
                        reason="rate_limit_fallback",
                    )
            except Exception as e:
                logger.warning("Failed to refresh SportsRadar injuries feed: %s", e)
                if not self._feed_payload:
                    self._load_feed_cache_if_available(
                        max_age_seconds=None,
                        allow_stale=True,
                        reason="error_fallback",
                    )

    def _parse_tipoff(self, end_date_raw: object) -> Optional[datetime]:
        text = str(end_date_raw or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    async def _resolve_team_profile_id(
        self,
        *,
        team_code: str,
    ) -> Optional[str]:
        team_norm = normalize_team_code(team_code)
        if not team_norm:
            return None

        team_id = str(self._team_id_by_code.get(team_norm) or "").strip()
        if team_id:
            return team_id
        self._index_team_ids_from_injuries_payload(self._feed_payload)
        team_id = str(self._team_id_by_code.get(team_norm) or "").strip()
        return team_id or None

    @staticmethod
    def _team_code_from_profile_payload(payload: Dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        candidate = normalize_team_code(payload.get("reference") or payload.get("alias") or payload.get("abbr"))
        if candidate:
            return candidate
        team_obj = payload.get("team")
        if isinstance(team_obj, dict):
            candidate = normalize_team_code(
                team_obj.get("reference") or team_obj.get("alias") or team_obj.get("abbr")
            )
        return candidate

    @classmethod
    def _team_profile_has_actionable_absence(cls, profile_payload: Dict[str, Any]) -> bool:
        if not isinstance(profile_payload, dict):
            return False
        players = profile_payload.get("players")
        if not isinstance(players, list):
            return False
        for player in players:
            if not isinstance(player, dict):
                continue
            if not cls._is_official_nba_roster_player(player):
                continue
            status_norm = cls._status_from_profile_player(player)
            if _PIM_STATUS_RANK.get(status_norm, 0) > _PIM_STATUS_RANK.get("AVAILABLE", 0):
                return True
        return False

    def _cached_team_profile_has_actionable_absence(self, team_code: str) -> bool:
        team_norm = normalize_team_code(team_code)
        if not team_norm:
            return False
        team_id = str(self._team_id_by_code.get(team_norm) or "").strip()
        if not team_id:
            return False
        payload = self._team_profile_cache.get(team_id)
        if not isinstance(payload, dict) or not payload:
            return False
        return self._team_profile_has_actionable_absence(payload)

    async def _get_team_profile(
        self,
        *,
        team_code: str,
    ) -> Dict[str, Any]:
        team_norm = normalize_team_code(team_code)
        if not team_norm:
            return {}
        if self._is_rate_limited():
            team_id_cached = str(self._team_id_by_code.get(team_norm) or "").strip()
            if team_id_cached:
                cached = self._team_profile_cache.get(team_id_cached)
                if isinstance(cached, dict):
                    return cached
            return {}
        if self._active_slate_team_codes is not None and team_norm not in self._active_slate_team_codes:
            team_id_cached = str(self._team_id_by_code.get(team_norm) or "").strip()
            if team_id_cached:
                cached_payload = self._team_profile_cache.get(team_id_cached)
                if isinstance(cached_payload, dict) and cached_payload:
                    return cached_payload
            return {}

        team_id = await self._resolve_team_profile_id(team_code=team_norm)
        if not team_id:
            return {}

        ttl = max(int(self.config.injury_profile_cache_ttl_seconds), 1)
        now = time.time()
        cached = self._team_profile_cache.get(team_id)
        updated = self._team_profile_updated_at.get(team_id, 0.0)
        if isinstance(cached, dict) and (now - updated) < ttl:
            return cached

        inflight = self._team_profile_inflight.get(team_id)
        if inflight is not None:
            payload = await inflight
        else:
            if not self._can_fetch_team_profile_now():
                if isinstance(cached, dict) and cached:
                    return cached
                logger.info(
                    "Team profile fetch budget reached (window=%ds, max=%d); reusing cached/stale data",
                    max(int(self.config.team_profile_budget_window_seconds), 1),
                    max(int(self.config.max_team_profiles_per_cycle), 1),
                )
                return {}

            async def _fetch() -> Dict[str, Any]:
                client = SportsRadarClient(
                    SportsRadarConfig(
                        api_key=self.sportsradar_api_key,
                        base_url=self._sportsradar_base_url,
                        max_retries=0,
                    )
                )
                async with client:
                    fetched = await client.fetch_team_profile(team_id)
                return fetched if isinstance(fetched, dict) else {}

            task = asyncio.create_task(_fetch())
            self._team_profile_inflight[team_id] = task
            try:
                payload = await task
            except RateLimitError as e:
                self._mark_rate_limited(getattr(e, "retry_after", None))
                payload = {}
            finally:
                self._team_profile_inflight.pop(team_id, None)
        if isinstance(payload, dict):
            self._team_profile_cache[team_id] = payload
            self._team_profile_updated_at[team_id] = now
            self._profile_cache.upsert_team(team_id, payload, updated_at=datetime.now(timezone.utc))
            profile_team_code = self._team_code_from_profile_payload(payload)
            if profile_team_code:
                self._team_id_by_code[profile_team_code] = team_id
            return payload
        return {}

    async def _get_player_profile(self, player_id: str) -> Dict[str, Any]:
        player_id_norm = str(player_id or "").strip()
        if not player_id_norm:
            return {}
        if self._is_rate_limited():
            cached = self._player_profile_cache.get(player_id_norm)
            if isinstance(cached, dict):
                return cached
            return {}
        ttl = max(int(self.config.injury_profile_cache_ttl_seconds), 1)
        now = time.time()
        cached = self._player_profile_cache.get(player_id_norm)
        updated = self._player_profile_updated_at.get(player_id_norm, 0.0)
        if isinstance(cached, dict) and (now - updated) < ttl:
            return cached

        inflight = self._player_profile_inflight.get(player_id_norm)
        if inflight is not None:
            payload = await inflight
        else:
            async def _fetch() -> Dict[str, Any]:
                client = SportsRadarClient(
                    SportsRadarConfig(
                        api_key=self.sportsradar_api_key,
                        base_url=self._sportsradar_base_url,
                        max_retries=0,
                    )
                )
                async with client:
                    fetched = await client.fetch_player_profile(player_id_norm)
                return fetched if isinstance(fetched, dict) else {}

            task = asyncio.create_task(_fetch())
            self._player_profile_inflight[player_id_norm] = task
            try:
                payload = await task
            except RateLimitError as e:
                self._mark_rate_limited(getattr(e, "retry_after", None))
                payload = {}
            finally:
                self._player_profile_inflight.pop(player_id_norm, None)
        if isinstance(payload, dict):
            self._player_profile_cache[player_id_norm] = payload
            self._player_profile_updated_at[player_id_norm] = now
            self._profile_cache.upsert_player(
                player_id_norm,
                payload,
                updated_at=datetime.now(timezone.utc),
            )
            return payload
        return {}

    async def prime_for_markets(self, markets: Iterable[object]) -> None:
        if not self.enabled:
            return
        team_codes = sorted(self.build_slate_team_set_from_markets(markets, today_only=True))
        self._active_slate_team_codes = set(team_codes)
        if not team_codes:
            logger.info("SportsRadar slate prefetch skipped (no NBA teams detected for today's market slate)")
            return
        mapped = sum(1 for team in team_codes if team in self._team_id_by_code)
        logger.info(
            "SportsRadar slate prepared (teams=%d, mapped_ids=%d, profile_cache_ttl=%ds; profiles fetched on-demand)",
            len(team_codes),
            mapped,
            max(int(self.config.injury_profile_cache_ttl_seconds), 1),
        )

    @staticmethod
    def _is_official_nba_roster_player(profile_player: Dict[str, Any]) -> bool:
        status_raw = str(profile_player.get("status") or "").strip().upper()
        if "TWO-WAY" in status_raw or "G LEAGUE" in status_raw or "G-LEAGUE" in status_raw:
            return False
        return True

    @staticmethod
    def _status_from_profile_player(profile_player: Dict[str, Any]) -> str:
        injuries = profile_player.get("injuries")
        if isinstance(injuries, list):
            best = "AVAILABLE"
            for injury in injuries:
                if not isinstance(injury, dict):
                    continue
                norm = normalize_injury_status(injury.get("status"))
                if _PIM_STATUS_RANK.get(norm, 0) > _PIM_STATUS_RANK.get(best, 0):
                    best = norm
            if best != "AVAILABLE":
                return best

        status_raw = str(profile_player.get("status") or "").strip().upper()
        mapped = _TEAM_PROFILE_STATUS_MAP.get(status_raw)
        if mapped:
            return mapped

        # Fallback for verbose provider values that do not match shorthand codes.
        normalized = normalize_injury_status(status_raw)
        if normalized in {"OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE", "AVAILABLE"}:
            return normalized
        return "AVAILABLE"

    @staticmethod
    def _parse_minutes_value(value: object) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            parsed = float(value)
            return parsed if parsed >= 0 else None

        text = str(value).strip()
        if not text:
            return None
        if ":" in text:
            try:
                minutes_part, seconds_part = text.split(":", 1)
                minutes = float(minutes_part)
                seconds = float(seconds_part)
                if minutes < 0 or seconds < 0:
                    return None
                return minutes + (seconds / 60.0)
            except Exception:
                return None
        try:
            parsed = float(text)
            return parsed if parsed >= 0 else None
        except Exception:
            return None

    @classmethod
    def _extract_profile_player_minutes(cls, profile_player: Dict[str, Any]) -> Optional[float]:
        if not isinstance(profile_player, dict):
            return None
        candidate_values: List[object] = []
        candidate_values.append(profile_player.get("minutes"))
        candidate_values.append(profile_player.get("minutes_played"))

        avg = profile_player.get("average")
        if isinstance(avg, dict):
            candidate_values.append(avg.get("minutes"))
            candidate_values.append(avg.get("minutes_played"))

        total = profile_player.get("total")
        if isinstance(total, dict):
            candidate_values.append(total.get("minutes"))
            candidate_values.append(total.get("minutes_played"))

        stats = profile_player.get("statistics")
        if isinstance(stats, dict):
            avg_stats = stats.get("average")
            if isinstance(avg_stats, dict):
                candidate_values.append(avg_stats.get("minutes"))
                candidate_values.append(avg_stats.get("minutes_played"))
            total_stats = stats.get("total")
            if isinstance(total_stats, dict):
                candidate_values.append(total_stats.get("minutes"))
                candidate_values.append(total_stats.get("minutes_played"))

        for candidate in candidate_values:
            parsed = cls._parse_minutes_value(candidate)
            if parsed is not None:
                return parsed
        return None

    @classmethod
    def _compute_low_minutes_threshold(
        cls,
        players: List[Dict[str, Any]],
        *,
        percentile: float = _LOW_MINUTES_PERCENTILE,
    ) -> Optional[float]:
        if not players:
            return None
        minutes_values: List[float] = []
        for player in players:
            minutes = cls._extract_profile_player_minutes(player)
            if minutes is not None:
                minutes_values.append(float(minutes))
        if not minutes_values:
            return None

        minutes_values.sort()
        pct = max(0.0, min(1.0, float(percentile)))
        cutoff_count = max(1, int(math.ceil(len(minutes_values) * pct)))
        return float(minutes_values[cutoff_count - 1])

    async def build_pim_inputs_for_market(
        self,
        *,
        context: MarketInjuryContext,
        yes_team: str,
    ) -> Dict[str, Any]:
        """
        Build deterministic PIM inputs from team profile status + player profiles.

        Returns:
            {
              "yes_team_players": [...player_profile_json...],
              "opp_team_players": [...player_profile_json...],
              "injury_status_map": {player_id: STATUS},
              "yes_team_code": "...",
              "opp_team_code": "...",
              "considered_players": [...debug rows...],
              "low_minutes_filtered_players": [...debug rows...],
            }
        """
        yes_team_code = normalize_team_code(yes_team)
        home = normalize_team_code(context.snapshot.get("home_team"))
        away = normalize_team_code(context.snapshot.get("away_team"))
        if yes_team_code not in {home, away}:
            return {
                "yes_team_players": [],
                "opp_team_players": [],
                "injury_status_map": {},
                "yes_team_code": yes_team_code,
                "opp_team_code": "",
                "considered_players": [],
                "low_minutes_filtered_players": [],
                "low_minutes_thresholds": {},
            }
        opp_team_code = away if yes_team_code == home else home

        yes_should_fetch_profile = (
            self._snapshot_has_actionable_absence(context.snapshot, yes_team_code)
            or self._cached_team_profile_has_actionable_absence(yes_team_code)
        )
        opp_should_fetch_profile = (
            self._snapshot_has_actionable_absence(context.snapshot, opp_team_code)
            or self._cached_team_profile_has_actionable_absence(opp_team_code)
        )

        yes_team_profile: Dict[str, Any] = {}
        opp_team_profile: Dict[str, Any] = {}
        fetch_tasks: List[Tuple[str, asyncio.Task]] = []
        if yes_should_fetch_profile:
            fetch_tasks.append(("yes", asyncio.create_task(self._get_team_profile(team_code=yes_team_code))))
        if opp_should_fetch_profile:
            fetch_tasks.append(("opp", asyncio.create_task(self._get_team_profile(team_code=opp_team_code))))
        if fetch_tasks:
            fetched = await asyncio.gather(*(task for _, task in fetch_tasks))
            for (label, _), payload in zip(fetch_tasks, fetched):
                payload_dict = payload if isinstance(payload, dict) else {}
                if label == "yes":
                    yes_team_profile = payload_dict
                else:
                    opp_team_profile = payload_dict

        injury_status_map: Dict[str, str] = {}
        player_ids_by_team: Dict[str, List[str]] = {
            yes_team_code: [],
            opp_team_code: [],
        }
        considered_players: List[Dict[str, Any]] = []
        low_minutes_filtered_players: List[Dict[str, Any]] = []
        low_minutes_thresholds: Dict[str, float] = {}

        for team_code, profile_payload in (
            (yes_team_code, yes_team_profile),
            (opp_team_code, opp_team_profile),
        ):
            players = profile_payload.get("players")
            if not isinstance(players, list):
                continue
            official_players = [
                player
                for player in players
                if isinstance(player, dict) and self._is_official_nba_roster_player(player)
            ]
            low_minutes_threshold = self._compute_low_minutes_threshold(
                official_players,
                percentile=_LOW_MINUTES_PERCENTILE,
            )
            if low_minutes_threshold is not None:
                low_minutes_thresholds[team_code] = float(low_minutes_threshold)

            for player in official_players:
                player_id = str(player.get("id") or "").strip()
                if not player_id:
                    continue
                status_norm = self._status_from_profile_player(player)
                if _PIM_STATUS_RANK.get(status_norm, 0) <= _PIM_STATUS_RANK.get("AVAILABLE", 0):
                    continue

                minutes_played = self._extract_profile_player_minutes(player)
                if (
                    low_minutes_threshold is not None
                    and minutes_played is not None
                    and float(minutes_played) <= float(low_minutes_threshold)
                ):
                    low_minutes_filtered_players.append(
                        {
                            "team": team_code,
                            "player_id": player_id,
                            "player_name": str(player.get("full_name") or "").strip(),
                            "status": status_norm,
                            "minutes": float(minutes_played),
                            "threshold_minutes": float(low_minutes_threshold),
                        }
                    )
                    continue

                injury_status_map[player_id] = status_norm
                player_ids_by_team.setdefault(team_code, []).append(player_id)
                considered_players.append(
                    {
                        "team": team_code,
                        "player_id": player_id,
                        "player_name": str(player.get("full_name") or "").strip(),
                        "status": status_norm,
                    }
                )

        unique_player_ids = sorted(set(injury_status_map.keys()))
        profile_tasks = [asyncio.create_task(self._get_player_profile(pid)) for pid in unique_player_ids]
        profiles_list = await asyncio.gather(*profile_tasks) if profile_tasks else []
        profiles_by_id: Dict[str, Dict[str, Any]] = {}
        for pid, payload in zip(unique_player_ids, profiles_list):
            if isinstance(payload, dict) and payload:
                profiles_by_id[pid] = payload

        yes_team_players = [
            profiles_by_id[pid]
            for pid in player_ids_by_team.get(yes_team_code, [])
            if pid in profiles_by_id
        ]
        opp_team_players = [
            profiles_by_id[pid]
            for pid in player_ids_by_team.get(opp_team_code, [])
            if pid in profiles_by_id
        ]

        return {
            "yes_team_players": yes_team_players,
            "opp_team_players": opp_team_players,
            "injury_status_map": injury_status_map,
            "yes_team_code": yes_team_code,
            "opp_team_code": opp_team_code,
            "considered_players": considered_players,
            "low_minutes_filtered_players": low_minutes_filtered_players,
            "low_minutes_thresholds": low_minutes_thresholds,
            "profile_fetch_flags": {
                "yes_team_profile_requested": bool(yes_should_fetch_profile),
                "opp_team_profile_requested": bool(opp_should_fetch_profile),
            },
        }

    def build_context_for_market(
        self,
        *,
        matchup_key: str,
        legacy_matchup_key: Optional[str] = None,
        yes_team: str,
        home_team: str,
        away_team: str,
        end_date: object,
        market_yes_price: Optional[float],
    ) -> MarketInjuryContext:
        yes_team_code = normalize_team_code(yes_team)
        home_team_code = normalize_team_code(home_team)
        away_team_code = normalize_team_code(away_team)
        snapshot = build_matchup_injury_snapshot(
            self._feed_payload if isinstance(self._feed_payload, dict) else {},
            home_team=home_team_code,
            away_team=away_team_code,
        )
        fingerprint = fingerprint_injury_snapshot(snapshot)
        root_entry = self._cache.get(matchup_key) if self.config.enable_injury_llm_cache else None
        if root_entry is None and self.config.enable_injury_llm_cache:
            legacy_key = str(legacy_matchup_key or "").strip()
            if legacy_key and legacy_key != str(matchup_key):
                legacy_entry = self._cache.get(legacy_key)
                if isinstance(legacy_entry, dict):
                    root_entry = self._canonicalize_game_cache_entry(
                        entry=dict(legacy_entry),
                        yes_team_code=yes_team_code,
                        home_team_code=home_team_code,
                        away_team_code=away_team_code,
                    )
                    self._cache.upsert(str(matchup_key), dict(root_entry))
        elif isinstance(root_entry, dict):
            canonical_entry = self._canonicalize_game_cache_entry(
                entry=dict(root_entry),
                yes_team_code=yes_team_code,
                home_team_code=home_team_code,
                away_team_code=away_team_code,
            )
            if canonical_entry != root_entry and self.config.enable_injury_llm_cache:
                self._cache.upsert(str(matchup_key), dict(canonical_entry))
            root_entry = canonical_entry

        cache_entry = self._project_cache_entry_for_yes_team(
            entry=root_entry,
            yes_team_code=yes_team_code,
            home_team_code=home_team_code,
            away_team_code=away_team_code,
        )
        tipoff_time = self._parse_tipoff(end_date)
        decision = should_refresh_llm(
            cache_entry=cache_entry,
            current_injury_fingerprint=fingerprint,
            now=datetime.now(timezone.utc),
            tipoff_time=tipoff_time,
            current_market_price=market_yes_price,
            max_age_seconds=int(self.config.llm_refresh_max_age_seconds),
            near_tipoff_minutes=int(self.config.force_llm_refresh_near_tipoff_minutes),
            near_tipoff_stale_seconds=int(self.config.near_tipoff_llm_stale_seconds),
            price_move_threshold_pct=float(self.config.llm_refresh_on_price_move_pct),
            force_refresh=bool(self.config.force_injury_llm_refresh),
        )
        if self._is_rate_limited() and isinstance(cache_entry, dict):
            generated_at = _parse_timestamp(cache_entry.get("generated_at"))
            now_utc = datetime.now(timezone.utc)
            age_seconds = None
            if generated_at is not None:
                age_seconds = max((now_utc - generated_at).total_seconds(), 0.0)
            decision = LLMRefreshDecision(
                should_refresh=False,
                reason="rate_limited_cache_reuse",
                metadata={
                    "cached_delta_age_seconds": (None if age_seconds is None else float(age_seconds)),
                    "rate_limited_for_seconds": self._remaining_rate_limit_seconds(),
                },
            )
        elif self._is_rate_limited() and not isinstance(cache_entry, dict):
            decision = LLMRefreshDecision(
                should_refresh=True,
                reason="rate_limited_no_cache",
                metadata={
                    "cached_delta_age_seconds": None,
                    "rate_limited_for_seconds": self._remaining_rate_limit_seconds(),
                },
            )
        return MarketInjuryContext(
            matchup_key=str(matchup_key),
            snapshot=snapshot,
            fingerprint=fingerprint,
            fingerprint_short=fingerprint[:12],
            yes_team_code=yes_team_code,
            decision=decision,
            cache_entry=cache_entry,
            tipoff_time=tipoff_time,
        )

    def build_prompt_block(self, context: MarketInjuryContext) -> str:
        if not self.config.show_injury_snapshot_in_prompt:
            return ""
        teams = context.snapshot.get("teams")
        if not isinstance(teams, list) or not teams:
            return "\nLIVE INJURY SNAPSHOT: unavailable\n"
        injury_json = json.dumps(context.snapshot, indent=2, sort_keys=True)
        return (
            "\nLIVE STRUCTURED INJURY SNAPSHOT (SportsRadar):\n"
            f"{injury_json}\n"
            "Use this structured data to assess which absences materially change team strength.\n"
        )

    def persist_result(
        self,
        *,
        context: MarketInjuryContext,
        llm_delta: float,
        llm_confidence: Optional[float],
        llm_model: Optional[str],
        prompt_version: str,
        analysis_version: str,
        source_tag: str,
        market_yes_price: Optional[float],
        player_impact: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.config.enable_injury_llm_cache:
            return
        home_team_code = normalize_team_code(context.snapshot.get("home_team"))
        away_team_code = normalize_team_code(context.snapshot.get("away_team"))
        yes_team_code = normalize_team_code(context.yes_team_code)
        delta_yes = self._safe_float(llm_delta, 0.0)
        if yes_team_code == home_team_code:
            canonical_home_delta = delta_yes
        elif yes_team_code == away_team_code:
            canonical_home_delta = -delta_yes
        else:
            canonical_home_delta = delta_yes

        entry = {
            "matchup_key": context.matchup_key,
            "injury_fingerprint": context.fingerprint,
            "llm_delta": float(delta_yes),
            "canonical_home_delta": float(canonical_home_delta),
            "home_team_code": home_team_code,
            "away_team_code": away_team_code,
            "llm_confidence": (None if llm_confidence is None else float(llm_confidence)),
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_tag": str(source_tag),
            "llm_model": str(llm_model or ""),
            "prompt_version": str(prompt_version),
            "analysis_version": str(analysis_version),
            "injury_snapshot_canonical": context.snapshot,
            "last_market_price_seen": (
                None if market_yes_price is None else float(market_yes_price)
            ),
            "player_impact": (
                dict(player_impact)
                if isinstance(player_impact, dict)
                else None
            ),
        }
        self._cache.upsert(context.matchup_key, entry)

    def touch_market_price(self, *, context: MarketInjuryContext, market_yes_price: Optional[float]) -> None:
        if not self.config.enable_injury_llm_cache:
            return
        if context.cache_entry is None:
            return
        entry = dict(context.cache_entry)
        if market_yes_price is not None:
            entry["last_market_price_seen"] = float(market_yes_price)
        self._cache.upsert(context.matchup_key, entry)
