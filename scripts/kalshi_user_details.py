#!/usr/bin/env python3
"""
Show current Kalshi account details for the configured user.

Usage:
  python scripts/kalshi_user_details.py
  python scripts/kalshi_user_details.py --config advanced_config.json --orders-limit 25
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path for direct script execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import ConfigManager
from app.api_clients import KalshiClient, KalshiConfig


async def _collect_details(config_path: str, orders_limit: int, include_raw_account: bool) -> Dict[str, Any]:
    cfg = ConfigManager(config_path)
    if not cfg.kalshi_enabled:
        raise RuntimeError("Kalshi is disabled in config")

    client = KalshiClient(
        KalshiConfig(
            api_key=cfg.platforms.kalshi.api_key,
            private_key=cfg.platforms.kalshi.private_key,
            private_key_file=cfg.platforms.kalshi.private_key_file,
            base_url="https://api.elections.kalshi.com/trade-api/v2",
            max_markets=cfg.platforms.kalshi.max_markets,
            use_orderbooks=False,
            series_tickers=cfg.platforms.kalshi.series_tickers,
        )
    )

    account_details: Dict[str, Any] = {}
    cash_balance = None
    positions: List[Dict[str, Any]] = []
    recent_orders: List[Dict[str, Any]] = []
    errors: List[str] = []

    if include_raw_account:
        # Raw account-details endpoint can vary by API plan/version.
        # Keep this best-effort and non-fatal.
        try:
            account_details = await client.get_account_details()
        except Exception:
            account_details = {}

    try:
        cash_balance = await client.get_cash_balance()
    except Exception as e:
        errors.append(f"cash_balance: {e}")

    try:
        positions = await client.get_positions()
    except Exception as e:
        errors.append(f"positions: {e}")

    try:
        recent_orders = await client.get_orders(limit=max(1, orders_limit), max_pages=1)
    except Exception as e:
        errors.append(f"orders: {e}")

    open_notional_estimate = 0.0
    for p in positions:
        qty = float(p.get("quantity", 0.0) or 0.0)
        entry = float(p.get("entry_price", 0.0) or 0.0)
        open_notional_estimate += qty * entry

    base_url = "https://api.elections.kalshi.com/trade-api/v2"
    success = len(errors) == 0
    message = (
        "Success: fetched Kalshi portfolio details from PRODUCTION."
        if success
        else "Partial/failed fetch: see errors for endpoint details."
    )

    return {
        "success": success,
        "message": message,
        "environment": "production",
        "portfolio_source": f"{base_url}/portfolio/*",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config_path": config_path,
        "kalshi_enabled": bool(cfg.kalshi_enabled),
        "dry_run_config": bool(cfg.is_dry_run),
        "cash_balance": cash_balance,
        "positions_count": len(positions),
        "open_notional_estimate": open_notional_estimate,
        "recent_orders_count": len(recent_orders),
        "positions": positions,
        "recent_orders": recent_orders,
        "account_details": account_details,
        "errors": errors,
    }


async def _amain() -> int:
    parser = argparse.ArgumentParser(description="Show current Kalshi user/account details")
    parser.add_argument("--config", type=str, default="advanced_config.json", help="Path to config file")
    parser.add_argument("--orders-limit", type=int, default=20, help="Number of recent orders to fetch")
    parser.add_argument(
        "--include-raw-account",
        action="store_true",
        help="Also try raw account details endpoints (may return 404 depending on API/version)",
    )
    args = parser.parse_args()

    details = await _collect_details(args.config, args.orders_limit, args.include_raw_account)
    print(json.dumps(details, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
