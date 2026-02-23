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
import time
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
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously at a fixed interval.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=300,
        help="Sleep interval between loop iterations (default: 300).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional max loop iterations (useful for testing).",
    )
    parser.add_argument(
        "--summary-top",
        type=int,
        default=8,
        help="How many top abs-edge rows to print in each report summary (default: 8).",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable compact report summary output after each cycle.",
    )
    parser.add_argument(
        "--price-refresh-only",
        action="store_true",
        help=(
            "Reuse fair values from a cached report and only refresh Kalshi prices "
            "for console output."
        ),
    )
    parser.add_argument(
        "--baseline-report",
        default=None,
        help=(
            "Optional path to a dry-run report JSON used as cached baseline for "
            "--price-refresh-only. Defaults to latest report."
        ),
    )
    parser.add_argument(
        "--injury-cache-ttl-seconds",
        type=int,
        default=120,
        help=(
            "TTL for cached analysis/injury deltas in --price-refresh-only mode "
            "(default: 120)."
        ),
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


def _latest_dry_run_report_path() -> Optional[Path]:
    reports_dir = REPO_ROOT / "reports" / "dry_run_analysis"
    if not reports_dir.exists():
        return None
    candidates = sorted(reports_dir.glob("cycle_*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _format_money(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"${value:.2f}"


def _normalize_min_edge(raw_value: Any, fallback: float = 0.10) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return fallback
    if value > 1.0:
        value = value / 100.0
    if value <= 0.0:
        return fallback
    return value


def _print_rows_table(rows: list[dict[str, Any]], top_n: int) -> None:
    if not rows:
        print("No rows to display.")
        return

    def _abs_edge(row: dict[str, Any]) -> float:
        try:
            return abs(float(row.get("edge", 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0

    top_rows = sorted(rows, key=_abs_edge, reverse=True)[: max(top_n, 1)]
    print(
        f"{'#':<3} {'Matchup':<36} {'Side':<16} {'Mkt':>6} {'Fair':>6} "
        f"{'Edge':>7} {'Signal':<6} {'Size':>8}  Market ID"
    )
    print("-" * 150)
    for idx, row in enumerate(top_rows, start=1):
        edge = float(row.get("edge", 0.0) or 0.0)
        fair = float(row.get("fair", 0.0) or 0.0)
        price = float(row.get("yes_price", 0.0) or 0.0)
        signal = str(row.get("signal") or "-")
        question = str(row.get("question") or "-").replace(" Winner?", "")[:36]
        selection = str(row.get("selection") or "-")[:16]
        market_id = str(row.get("market_id") or "-")
        size_raw = row.get("size")
        size = _format_money(float(size_raw)) if isinstance(size_raw, (int, float)) else "-"
        print(
            f"{idx:<3} {question:<36} {selection:<16} {price:>6.3f} {fair:>6.3f} "
            f"{edge:>+7.3f} {signal:<6} {size:>8}  {market_id}"
        )


def _print_compact_report_summary(report_path: Path, target_date: date, top_n: int) -> None:
    try:
        payload = json.loads(report_path.read_text())
    except Exception as exc:
        print(f"Could not read report summary ({report_path}): {exc}", file=sys.stderr)
        return

    rows = list(payload.get("results", []))
    date_code = target_date.strftime("%y%b%d").upper()
    scoped = [r for r in rows if date_code in str(r.get("market_id", "")).upper()]
    selected = scoped if scoped else rows

    print("\n=== REPORT SUMMARY ===")
    print(f"Report: {report_path}")
    print(f"Timestamp UTC: {payload.get('timestamp_utc')}")
    print(
        f"Rows: total={len(rows)} scoped_to_{date_code}={len(scoped)} "
        f"(showing top {max(top_n, 1)} by abs(edge))"
    )

    if not selected:
        print("No rows found in report.")
        return

    _print_rows_table(selected, top_n=top_n)

    signal_rows = [r for r in selected if str(r.get("signal") or "-") != "-"]
    if signal_rows:
        print(f"Signal rows in scope: {len(signal_rows)}")
    else:
        print("Signal rows in scope: 0")


def _resolve_target_date(date_override: Optional[str], tz: ZoneInfo) -> date:
    if date_override:
        return datetime.strptime(date_override, "%Y-%m-%d").date()
    return datetime.now(tz).date()


def _load_baseline_report(args: argparse.Namespace) -> Optional[Path]:
    if args.baseline_report:
        baseline = Path(args.baseline_report).expanduser()
        return baseline if baseline.exists() else None
    return _latest_dry_run_report_path()


def _report_age_seconds(report_path: Path) -> Optional[float]:
    try:
        payload = json.loads(report_path.read_text())
        ts = payload.get("timestamp_utc")
        if isinstance(ts, str) and ts.strip():
            report_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if report_dt.tzinfo is None:
                report_dt = report_dt.replace(tzinfo=ZoneInfo("UTC"))
            now_utc = datetime.now(ZoneInfo("UTC"))
            return max(0.0, (now_utc - report_dt.astimezone(ZoneInfo("UTC"))).total_seconds())
    except Exception:
        pass
    try:
        return max(0.0, time.time() - report_path.stat().st_mtime)
    except OSError:
        return None


def _build_price_refreshed_rows(
    baseline_rows: list[dict[str, Any]],
    live_markets: list[dict[str, Any]],
    min_edge: float,
) -> list[dict[str, Any]]:
    live_by_market_id: dict[str, dict[str, Any]] = {}
    for market in live_markets:
        market_id = str(market.get("market_id") or "").strip()
        if market_id:
            live_by_market_id[market_id] = market

    refreshed: list[dict[str, Any]] = []
    for row in baseline_rows:
        market_id = str(row.get("market_id") or "").strip()
        if not market_id:
            continue
        live = live_by_market_id.get(market_id)
        if live is None:
            continue

        fair = float(row.get("fair", 0.0) or 0.0)
        yes_price = float(live.get("yes_price", row.get("yes_price", 0.0)) or 0.0)
        edge = fair - yes_price

        signal = "-"
        if edge >= min_edge:
            signal = "buy_yes"
        elif edge <= -min_edge:
            signal = "buy_no"

        updated = dict(row)
        updated["yes_price"] = yes_price
        updated["edge"] = edge
        updated["signal"] = signal
        updated["size"] = None
        refreshed.append(updated)

    return refreshed


def _run_full_cycle_once(args: argparse.Namespace, tz: ZoneInfo) -> int:
    target_date = _resolve_target_date(args.date, tz)

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

    report_before = _latest_dry_run_report_path()
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
        exit_code = subprocess.call(cmd)
        if exit_code == 0 and not args.no_summary:
            report_after = _latest_dry_run_report_path()
            if report_after is not None and (report_before is None or report_after != report_before):
                _print_compact_report_summary(report_after, target_date=target_date, top_n=args.summary_top)
            elif report_after is not None:
                _print_compact_report_summary(report_after, target_date=target_date, top_n=args.summary_top)
        return exit_code
    finally:
        if args.keep_temp_config:
            print(f"Temp config kept: {tmp_config}")
        else:
            try:
                tmp_config.unlink(missing_ok=True)
            except OSError:
                pass


def _run_price_refresh_once(args: argparse.Namespace, tz: ZoneInfo) -> int:
    target_date = _resolve_target_date(args.date, tz)
    cfg = ConfigManager(args.config)
    if not cfg.kalshi_enabled:
        print("Kalshi is disabled in config. Enable platforms.kalshi.enabled first.", file=sys.stderr)
        return 2

    baseline_report = _load_baseline_report(args)
    ttl_seconds = max(0, int(args.injury_cache_ttl_seconds))
    baseline_age_seconds = _report_age_seconds(baseline_report) if baseline_report is not None else None

    needs_refresh = baseline_report is None
    if not needs_refresh and ttl_seconds > 0 and baseline_age_seconds is not None:
        needs_refresh = baseline_age_seconds > ttl_seconds

    if needs_refresh:
        reason = "missing" if baseline_report is None else f"stale ({baseline_age_seconds:.0f}s old)"
        print(
            f"Baseline cache is {reason}; running full dry-run analysis refresh "
            f"(TTL={ttl_seconds}s)."
        )
        refresh_rc = _run_full_cycle_once(args, tz)
        if refresh_rc != 0:
            print(f"Full refresh failed with exit code {refresh_rc}.", file=sys.stderr)
            return refresh_rc
        baseline_report = _latest_dry_run_report_path()
        baseline_age_seconds = _report_age_seconds(baseline_report) if baseline_report is not None else None

    if baseline_report is None:
        print("Unable to locate baseline report after refresh.", file=sys.stderr)
        return 2

    try:
        payload = json.loads(baseline_report.read_text())
    except Exception as exc:
        print(f"Could not read baseline report {baseline_report}: {exc}", file=sys.stderr)
        return 2

    all_rows = list(payload.get("results", []))
    if not all_rows:
        print(f"Baseline report has no rows: {baseline_report}", file=sys.stderr)
        return 2

    date_code = target_date.strftime("%y%b%d").upper()
    scoped_rows = [r for r in all_rows if date_code in str(r.get("market_id", "")).upper()]
    baseline_rows = scoped_rows if scoped_rows else all_rows

    live_markets = asyncio.run(_fetch_open_series_markets(cfg, args.series))
    min_edge = _normalize_min_edge(getattr(cfg.strategy, "min_edge", 0.10), fallback=0.10)
    refreshed = _build_price_refreshed_rows(baseline_rows, live_markets, min_edge=min_edge)

    print("\n=== PRICE REFRESH SUMMARY ===")
    print(f"Baseline report: {baseline_report}")
    print(f"Baseline timestamp UTC: {payload.get('timestamp_utc')}")
    if baseline_age_seconds is not None:
        print(f"Baseline age: {baseline_age_seconds:.0f}s (TTL {ttl_seconds}s)")
    print(f"Target date ({args.timezone}): {target_date.isoformat()} [{date_code}]")
    print(
        f"Markets: live_series={len(live_markets)} baseline_rows={len(baseline_rows)} "
        f"matched={len(refreshed)} min_edge={min_edge:.3f}"
    )
    _print_rows_table(refreshed, top_n=args.summary_top)
    signal_rows = [r for r in refreshed if str(r.get("signal") or "-") != "-"]
    print(f"Signal rows in scope: {len(signal_rows)}")
    return 0


def _run_once(args: argparse.Namespace, tz: ZoneInfo) -> int:
    if args.price_refresh_only:
        return _run_price_refresh_once(args, tz)
    return _run_full_cycle_once(args, tz)


def _run_loop(args: argparse.Namespace, tz: ZoneInfo) -> int:
    interval = max(1, int(args.interval_seconds))
    iteration = 0
    last_nonzero_exit = 0

    while True:
        iteration += 1
        now_local = datetime.now(tz).isoformat(timespec="seconds")
        print(f"\n=== LOOP ITERATION {iteration} ({args.timezone} {now_local}) ===")
        rc = _run_once(args, tz)
        if rc != 0:
            last_nonzero_exit = rc
            print(f"Iteration {iteration} exited with non-zero code: {rc}", file=sys.stderr)

        if args.max_iterations is not None and iteration >= args.max_iterations:
            print(f"Reached max iterations: {args.max_iterations}. Exiting loop.")
            break

        print(f"Sleeping for {interval} seconds before next iteration...")
        time.sleep(interval)

    return last_nonzero_exit


def main() -> int:
    args = _parse_args()
    tz = ZoneInfo(args.timezone)

    if args.loop:
        return _run_loop(args, tz)
    return _run_once(args, tz)


if __name__ == "__main__":
    raise SystemExit(main())
