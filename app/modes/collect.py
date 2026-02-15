import logging
import asyncio
from datetime import datetime
from typing import List, Optional

from app.bot import AdvancedTradingBot

logger = logging.getLogger(__name__)

async def run_collect(bot: AdvancedTradingBot, series_tickers: Optional[List[str]] = None):
    """
    Run collection mode: fetch markets and store snapshots.
    For Sprint 1, this focuses on gathering data without LLM or trading.
    """
    logger.info("Starting collection mode...")

    if series_tickers:
        logger.info(f"Scanning specific series: {series_tickers}")
        markets = await bot.scan_series_markets(series_tickers)
    else:
        logger.info("Scanning all markets...")
        markets = await bot.scanner.scan_all_markets()

    logger.info(f"Collected {len(markets)} markets.")

    # TODO: In future steps, implement SQLite storage here
    # For now, we just fetch them to verify the flow

    return markets

async def discover_kalshi_series(bot: AdvancedTradingBot, category: Optional[str] = None):
    """Discover available series on Kalshi."""
    logger.info(f"Discovering Kalshi series (category: {category})...")
    series = await bot.discover_kalshi_series(category)

    for s in series:
        print(f"  {s['ticker']:30} | {s['category']:20} | {s['title']}")

    return series
