"""SportsRadar NBA client for structured injury/profile ingestion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional

from .base_client import BaseAPIClient

logger = logging.getLogger(__name__)


@dataclass
class SportsRadarConfig:
    """Configuration for SportsRadar NBA API."""

    api_key: Optional[str] = None
    base_url: str = "https://api.sportradar.com/nba/trial/v8/en"


class SportsRadarClient(BaseAPIClient):
    """Client for SportsRadar NBA endpoints used by analysis."""

    def __init__(self, config: SportsRadarConfig):
        super().__init__(
            platform_name="sportradar",
            api_key=config.api_key,
            base_url=config.base_url.rstrip("/"),
        )
        self.config = config

    async def fetch_league_injuries(self) -> Dict[str, Any]:
        """
        Fetch league-wide NBA injuries feed.

        Endpoint:
            GET /league/injuries.json
            Header: x-api-key: <key>
        """
        if not self.config.api_key:
            raise ValueError("SportsRadar API key is not configured")
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")

        url = f"{self.base_url}/league/injuries.json"

        async def make_request():
            async with self.session.get(
                url,
                headers=self._build_headers(
                    auth_type="",
                    auth_header_name="x-api-key",
                    additional_headers={"Accept": "application/json"},
                ),
            ) as response:
                await self._handle_response_status(response)
                return await response.json()

        payload = await self._call_with_retry(
            make_request,
            operation_name="SportsRadar league injuries",
        )
        if not isinstance(payload, dict):
            logger.warning("SportsRadar injuries payload was not a JSON object")
            return {}
        return payload

    async def fetch_team_profile(self, team_id: str) -> Dict[str, Any]:
        """Fetch team profile including roster-level status metadata."""
        if not self.config.api_key:
            raise ValueError("SportsRadar API key is not configured")
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")

        team_id_norm = str(team_id or "").strip()
        if not team_id_norm:
            raise ValueError("team_id is required for team profile")

        url = f"{self.base_url}/teams/{team_id_norm}/profile.json"

        async def make_request():
            async with self.session.get(
                url,
                headers=self._build_headers(
                    auth_type="",
                    auth_header_name="x-api-key",
                    additional_headers={"Accept": "application/json"},
                ),
            ) as response:
                await self._handle_response_status(response)
                return await response.json()

        payload = await self._call_with_retry(
            make_request,
            operation_name=f"SportsRadar team profile ({team_id_norm})",
        )
        if not isinstance(payload, dict):
            logger.warning("SportsRadar team profile payload was not a JSON object for %s", team_id_norm)
            return {}
        return payload

    async def fetch_player_profile(self, player_id: str) -> Dict[str, Any]:
        """Fetch player profile including season/team averages for PIM calculation."""
        if not self.config.api_key:
            raise ValueError("SportsRadar API key is not configured")
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")

        player_id_norm = str(player_id or "").strip()
        if not player_id_norm:
            raise ValueError("player_id is required for player profile")

        url = f"{self.base_url}/players/{player_id_norm}/profile.json"

        async def make_request():
            async with self.session.get(
                url,
                headers=self._build_headers(
                    auth_type="",
                    auth_header_name="x-api-key",
                    additional_headers={"Accept": "application/json"},
                ),
            ) as response:
                await self._handle_response_status(response)
                return await response.json()

        payload = await self._call_with_retry(
            make_request,
            operation_name=f"SportsRadar player profile ({player_id_norm})",
        )
        if not isinstance(payload, dict):
            logger.warning("SportsRadar player profile payload was not a JSON object for %s", player_id_norm)
            return {}
        return payload

    async def fetch_daily_schedule(self, game_date: date) -> Dict[str, Any]:
        """Fetch schedule for a calendar date (used to resolve team UUIDs)."""
        if not self.config.api_key:
            raise ValueError("SportsRadar API key is not configured")
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")

        if not isinstance(game_date, date):
            raise ValueError("game_date must be a datetime.date")

        url = f"{self.base_url}/games/{game_date.year:04d}/{game_date.month:02d}/{game_date.day:02d}/schedule.json"

        async def make_request():
            async with self.session.get(
                url,
                headers=self._build_headers(
                    auth_type="",
                    auth_header_name="x-api-key",
                    additional_headers={"Accept": "application/json"},
                ),
            ) as response:
                await self._handle_response_status(response)
                return await response.json()

        payload = await self._call_with_retry(
            make_request,
            operation_name=f"SportsRadar schedule ({game_date.isoformat()})",
        )
        if not isinstance(payload, dict):
            logger.warning("SportsRadar schedule payload was not a JSON object for %s", game_date.isoformat())
            return {}
        return payload

    async def fetch_daily_injuries(self) -> Dict[str, Any]:
        """
        Backward-compatible alias.

        Historically this client used the name ``fetch_daily_injuries`` while
        calling the league injuries endpoint.
        """
        return await self.fetch_league_injuries()
