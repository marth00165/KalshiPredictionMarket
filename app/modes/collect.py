import logging
import asyncio
import time
from datetime import datetime
from typing import List, Optional
from dataclasses import asdict

from app.bot import AdvancedTradingBot

logger = logging.getLogger(__name__)

async def run_collect(bot: AdvancedTradingBot, series_tickers: Optional[List[str]] = None):
    """
    Run collection mode: fetch markets and store snapshots.
    For Sprint 1, this focuses on gathering data without LLM or trading.
    """
    start_time = time.time()
    logger.info("🚀 Starting collection mode...")

    try:
        # Initialize bot (DB, etc.)
        await bot.initialize()

        if series_tickers:
            logger.info(f"Scanning specific series: {series_tickers}")
            markets = await bot.scan_series_markets(series_tickers)
        else:
            logger.info("Scanning all markets...")
            markets = await bot.scanner.scan_all_markets()

        logger.info(f"Collected {len(markets)} markets.")


        # Persist to SQLite and save JSON report
        if markets:
            market_dicts = [asdict(m) for m in markets]
            await bot.db.save_market_snapshots(market_dicts)
            logger.info(f"✅ Successfully stored {len(markets)} snapshots to database.")

            # Save JSON report
            import json
            import os
            reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(reports_dir, f"collect_{timestamp}.json")
            with open(report_path, "w") as f:
                json.dump(market_dicts, f, indent=2)
            logger.info(f"📝 JSON report saved to {report_path}")

        duration = time.time() - start_time

        # Update heartbeat status
        await bot.db.update_status("last_success_at_utc", datetime.utcnow().isoformat() + "Z")
        await bot.db.update_status("last_mode", "collect")
        await bot.db.update_status("last_collect_count", len(markets))
        await bot.db.update_status("last_duration_s", f"{duration:.2f}")

        # Structured summary log line
        logger.info(f"cycle_summary mode=collect scanned={len(markets)} inserted={len(markets)} errors=0 duration_s={duration:.2f}")

        return markets

    except Exception as e:
        logger.error(f"❌ Collection mode failed: {e}")
        await bot.db.update_status("last_error_at_utc", datetime.utcnow().isoformat() + "Z")
        await bot.db.update_status("last_error_message", str(e))
        raise

async def discover_kalshi_series(bot: AdvancedTradingBot, category: Optional[str] = None):
    """Discover available series on Kalshi."""
    logger.info(f"Discovering Kalshi series (category: {category})...")
    series = await bot.discover_kalshi_series(category)

    for s in series:
        print(f"  {s['ticker']:30} | {s['category']:20} | {s['title']}")

    return series
