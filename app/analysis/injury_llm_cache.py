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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    show_injury_snapshot_in_prompt: bool = True


@dataclass
class MarketInjuryContext:
    matchup_key: str
    snapshot: Dict[str, Any]
    fingerprint: str
    fingerprint_short: str
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
        self._feed_lock = asyncio.Lock()
        self._feed_payload: Dict[str, Any] = {}
        self._feed_updated_at: float = 0.0
        self._sportsradar_base_url = sportsradar_base_url
        self._team_profile_cache: Dict[str, Dict[str, Any]] = {}
        self._team_profile_updated_at: Dict[str, float] = {}
        self._player_profile_cache: Dict[str, Dict[str, Any]] = {}
        self._player_profile_updated_at: Dict[str, float] = {}
        self._schedule_cache: Dict[str, Dict[str, Any]] = {}
        self._schedule_updated_at: Dict[str, float] = {}

        if self.config.enabled and not self.sportsradar_api_key:
            logger.warning("Live injury news enabled but SportsRadar API key missing; proceeding without feed.")

    async def refresh_injury_feed_if_needed(self, *, force: bool = False) -> None:
        if not self.enabled:
            return

        now = time.time()
        ttl = max(int(self.config.injury_feed_cache_ttl_seconds), 1)
        if not force and self._feed_payload and (now - self._feed_updated_at) < ttl:
            return

        async with self._feed_lock:
            now2 = time.time()
            if not force and self._feed_payload and (now2 - self._feed_updated_at) < ttl:
                return
            try:
                client = SportsRadarClient(
                    SportsRadarConfig(
                        api_key=self.sportsradar_api_key,
                        base_url=self._sportsradar_base_url,
                    )
                )
                async with client:
                    payload = await client.fetch_league_injuries()
                if isinstance(payload, dict):
                    self._feed_payload = payload
                    self._feed_updated_at = time.time()
                    logger.info(
                        "Injury feed refreshed from SportsRadar (teams=%d)",
                        len(payload.get("teams") or []) if isinstance(payload.get("teams"), list) else 0,
                    )
            except Exception as e:
                logger.warning("Failed to refresh SportsRadar injuries feed: %s", e)

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

    def _parse_market_date_from_matchup_key(self, matchup_key: str) -> Optional[date]:
        # Example market id: KXNBAGAME-26FEB23SACMEM-MEM
        token = str(matchup_key or "").strip().upper()
        match = re.search(r"-(\d{2}[A-Z]{3}\d{2})", token)
        if not match:
            return None
        code = match.group(1).title()
        try:
            return datetime.strptime(code, "%y%b%d").date()
        except Exception:
            return None

    def _candidate_schedule_dates(self, context: MarketInjuryContext) -> List[date]:
        dates: List[date] = []
        market_date = self._parse_market_date_from_matchup_key(context.matchup_key)
        if isinstance(market_date, date):
            dates.append(market_date)
        if context.tipoff_time is not None:
            tipoff_date = context.tipoff_time.astimezone(timezone.utc).date()
            if tipoff_date not in dates:
                dates.append(tipoff_date)
            prev = tipoff_date - timedelta(days=1)
            if prev not in dates:
                dates.append(prev)
        return dates

    async def _get_schedule_payload(self, game_date: date) -> Dict[str, Any]:
        key = game_date.isoformat()
        ttl = max(int(self.config.injury_feed_cache_ttl_seconds), 1)
        now = time.time()
        cached = self._schedule_cache.get(key)
        updated = self._schedule_updated_at.get(key, 0.0)
        if isinstance(cached, dict) and (now - updated) < ttl:
            return cached

        client = SportsRadarClient(
            SportsRadarConfig(
                api_key=self.sportsradar_api_key,
                base_url=self._sportsradar_base_url,
            )
        )
        async with client:
            payload = await client.fetch_daily_schedule(game_date)
        if isinstance(payload, dict):
            self._schedule_cache[key] = payload
            self._schedule_updated_at[key] = now
            return payload
        return {}

    async def _resolve_team_profile_id(
        self,
        *,
        team_code: str,
        context: MarketInjuryContext,
    ) -> Optional[str]:
        team_norm = normalize_team_code(team_code)
        if not team_norm:
            return None

        teams = self._feed_payload.get("teams")
        if isinstance(teams, list):
            for team in teams:
                if not isinstance(team, dict):
                    continue
                code = normalize_team_code(team.get("reference") or team.get("alias") or team.get("abbr"))
                if code != team_norm:
                    continue
                team_id = str(team.get("id") or "").strip()
                if team_id:
                    return team_id

        for game_date in self._candidate_schedule_dates(context):
            try:
                sched = await self._get_schedule_payload(game_date)
            except Exception as e:
                logger.debug("Schedule lookup failed for %s: %s", game_date.isoformat(), e)
                continue
            games = sched.get("games")
            if not isinstance(games, list):
                continue
            for game in games:
                if not isinstance(game, dict):
                    continue
                for side in ("home", "away"):
                    team_obj = game.get(side)
                    if not isinstance(team_obj, dict):
                        continue
                    code = normalize_team_code(
                        team_obj.get("alias") or team_obj.get("reference") or team_obj.get("abbr")
                    )
                    if code != team_norm:
                        continue
                    team_id = str(team_obj.get("id") or "").strip()
                    if team_id:
                        return team_id
        return None

    async def _get_team_profile(
        self,
        *,
        team_code: str,
        context: MarketInjuryContext,
    ) -> Dict[str, Any]:
        team_id = await self._resolve_team_profile_id(team_code=team_code, context=context)
        if not team_id:
            return {}

        ttl = max(int(self.config.injury_feed_cache_ttl_seconds), 1)
        now = time.time()
        cached = self._team_profile_cache.get(team_id)
        updated = self._team_profile_updated_at.get(team_id, 0.0)
        if isinstance(cached, dict) and (now - updated) < ttl:
            return cached

        client = SportsRadarClient(
            SportsRadarConfig(
                api_key=self.sportsradar_api_key,
                base_url=self._sportsradar_base_url,
            )
        )
        async with client:
            payload = await client.fetch_team_profile(team_id)
        if isinstance(payload, dict):
            self._team_profile_cache[team_id] = payload
            self._team_profile_updated_at[team_id] = now
            return payload
        return {}

    async def _get_player_profile(self, player_id: str) -> Dict[str, Any]:
        player_id_norm = str(player_id or "").strip()
        if not player_id_norm:
            return {}
        ttl = max(int(self.config.injury_feed_cache_ttl_seconds), 1)
        now = time.time()
        cached = self._player_profile_cache.get(player_id_norm)
        updated = self._player_profile_updated_at.get(player_id_norm, 0.0)
        if isinstance(cached, dict) and (now - updated) < ttl:
            return cached

        client = SportsRadarClient(
            SportsRadarConfig(
                api_key=self.sportsradar_api_key,
                base_url=self._sportsradar_base_url,
            )
        )
        async with client:
            payload = await client.fetch_player_profile(player_id_norm)
        if isinstance(payload, dict):
            self._player_profile_cache[player_id_norm] = payload
            self._player_profile_updated_at[player_id_norm] = now
            return payload
        return {}

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
            }
        opp_team_code = away if yes_team_code == home else home

        yes_profile_task = asyncio.create_task(
            self._get_team_profile(team_code=yes_team_code, context=context)
        )
        opp_profile_task = asyncio.create_task(
            self._get_team_profile(team_code=opp_team_code, context=context)
        )
        yes_team_profile, opp_team_profile = await asyncio.gather(yes_profile_task, opp_profile_task)

        injury_status_map: Dict[str, str] = {}
        player_ids_by_team: Dict[str, List[str]] = {
            yes_team_code: [],
            opp_team_code: [],
        }
        considered_players: List[Dict[str, str]] = []

        for team_code, profile_payload in (
            (yes_team_code, yes_team_profile),
            (opp_team_code, opp_team_profile),
        ):
            players = profile_payload.get("players")
            if not isinstance(players, list):
                continue
            for player in players:
                if not isinstance(player, dict):
                    continue
                if not self._is_official_nba_roster_player(player):
                    continue
                player_id = str(player.get("id") or "").strip()
                if not player_id:
                    continue
                status_norm = self._status_from_profile_player(player)
                if _PIM_STATUS_RANK.get(status_norm, 0) <= _PIM_STATUS_RANK.get("AVAILABLE", 0):
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
        }

    def build_context_for_market(
        self,
        *,
        matchup_key: str,
        home_team: str,
        away_team: str,
        end_date: object,
        market_yes_price: Optional[float],
    ) -> MarketInjuryContext:
        snapshot = build_matchup_injury_snapshot(
            self._feed_payload if isinstance(self._feed_payload, dict) else {},
            home_team=home_team,
            away_team=away_team,
        )
        fingerprint = fingerprint_injury_snapshot(snapshot)
        cache_entry = self._cache.get(matchup_key) if self.config.enable_injury_llm_cache else None
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
        return MarketInjuryContext(
            matchup_key=str(matchup_key),
            snapshot=snapshot,
            fingerprint=fingerprint,
            fingerprint_short=fingerprint[:12],
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
        entry = {
            "matchup_key": context.matchup_key,
            "injury_fingerprint": context.fingerprint,
            "llm_delta": float(llm_delta),
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
