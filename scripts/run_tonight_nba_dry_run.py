#!/usr/bin/env python3
"""Run a one-cycle dry run scoped to tonight's NBA Kalshi games."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.api_clients.kalshi_client import KalshiClient, KalshiConfig  # noqa: E402
from app.config import ConfigManager  # noqa: E402


DATE_CODE_RE = re.compile(r"^[A-Z0-9]+-(\d{2}[A-Z]{3}\d{2})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover tonight's NBA event tickers on Kalshi and run one scoped "
            "non-interactive dry-run trade cycle."
        )
    )
    parser.add_argument(
        "--config",
        default="advanced_config.json",
        help="Path to base config JSON (default: advanced_config.json).",
    )
    parser.add_argument(
        "--series",
        default="KXNBAGAME",
        help="Kalshi series ticker to scan (default: KXNBAGAME).",
    )
    parser.add_argument(
        "--timezone",
        default="America/New_York",
        help="IANA timezone used for 'tonight' (default: America/New_York).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override target local date as YYYY-MM-DD.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print discovered tonight event tickers; do not run the bot.",
    )
    parser.add_argument(
        "--keep-temp-config",
        action="store_true",
        help="Keep generated temp config file for debugging.",
    )
    return parser.parse_args()


def _parse_date_code(value: str) -> Optional[date]:
    match = DATE_CODE_RE.match(value.upper())
    if not match:
        return None
    date_code = match.group(1).title()  # e.g. 26FEB19 -> 26Feb19
    try:
        return datetime.strptime(date_code, "%y%b%d").date()
    except ValueError:
        return None


def _parse_iso_to_local_date(value: str, tz: ZoneInfo) -> Optional[date]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(tz).date()


def _extract_market_local_date(market: dict[str, Any], tz: ZoneInfo) -> Optional[date]:
    for key in ("event_ticker", "market_id"):
        raw = str(market.get(key, "")).strip()
        parsed = _parse_date_code(raw)
        if parsed:
            return parsed
    return _parse_iso_to_local_date(str(market.get("end_date", "")), tz)


async def _fetch_open_series_markets(cfg: ConfigManager, series: str) -> list[dict[str, Any]]:
    private_key = cfg.platforms.kalshi.private_key
    if not private_key and cfg.platforms.kalshi.private_key_file:
        try:
            private_key = Path(cfg.platforms.kalshi.private_key_file).expanduser().read_text()
        except OSError:
            private_key = None

    client = KalshiClient(
        KalshiConfig(
            api_key=cfg.platforms.kalshi.api_key,
            private_key=private_key,
            private_key_file=cfg.platforms.kalshi.private_key_file,
            base_url="https://api.elections.kalshi.com/trade-api/v2",
            max_markets=cfg.platforms.kalshi.max_markets,
            use_orderbooks=False,
            series_tickers=[series],
        )
    )
    return await client.fetch_markets_by_series([series], status="open")


def _unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted({v for v in values if str(v).strip()})


def _build_temp_config(
    base_config_path: Path,
    event_tickers: list[str],
    series: str,
) -> Path:
    if base_config_path.exists():
        raw = json.loads(base_config_path.read_text())
    else:
        raw = {}

    raw.setdefault("trading", {})
    raw.setdefault("platforms", {})
    raw["platforms"].setdefault("kalshi", {})

    raw["trading"]["dry_run"] = True
    raw["trading"]["non_interactive"] = True
    raw["trading"]["allowed_market_ids"] = []
    raw["trading"]["allowed_event_tickers"] = event_tickers
    raw["platforms"]["kalshi"]["series_tickers"] = [series]

    fd, tmp_path = tempfile.mkstemp(prefix="nba_tonight_dryrun_", suffix=".json")
    os.close(fd)
    tmp_file = Path(tmp_path)
    tmp_file.write_text(json.dumps(raw, indent=2) + "\n")
    return tmp_file


def main() -> int:
    args = _parse_args()
    tz = ZoneInfo(args.timezone)
    target_date = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date
        else datetime.now(tz).date()
    )

    cfg = ConfigManager(args.config)
    if not cfg.kalshi_enabled:
        print("Kalshi is disabled in config. Enable platforms.kalshi.enabled first.", file=sys.stderr)
        return 2

    markets = asyncio.run(_fetch_open_series_markets(cfg, args.series))
    events_by_date: dict[date, set[str]] = defaultdict(set)

    for market in markets:
        event_ticker = str(market.get("event_ticker") or "").strip()
        if not event_ticker:
            continue
        market_date = _extract_market_local_date(market, tz)
        if market_date:
            events_by_date[market_date].add(event_ticker)

    tonight_events = _unique_sorted(events_by_date.get(target_date, set()))

    print(
        f"Scanned {len(markets)} open {args.series} markets. "
        f"Target date ({args.timezone}): {target_date.isoformat()}"
    )
    if tonight_events:
        print(f"Tonight event tickers ({len(tonight_events)}):")
        for ev in tonight_events:
            print(f"  - {ev}")
    else:
        print("No event tickers found for the target date.")
        if events_by_date:
            print("Available dates with open events:")
            for dt in sorted(events_by_date):
                print(f"  - {dt.isoformat()}: {len(events_by_date[dt])} events")
        return 1

    if args.list_only:
        return 0

    tmp_config = _build_temp_config(Path(args.config), tonight_events, args.series)
    cmd = [
        sys.executable,
        "-m",
        "app",
        "--config",
        str(tmp_config),
        "--mode",
        "trade",
        "--once",
        "--dry-run",
        "--skip-setup-wizard",
        "--non-interactive",
    ]

    print("\nRunning one-cycle dry run:")
    print(" ".join(cmd))
    try:
        return subprocess.call(cmd)
    finally:
        if args.keep_temp_config:
            print(f"Temp config kept: {tmp_config}")
        else:
            try:
                tmp_config.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
