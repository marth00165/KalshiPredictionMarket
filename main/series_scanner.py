#!/usr/bin/env python3
"""
Lightweight series-based market scanner for Kalshi.

This script demonstrates the rate-limit-friendly approach:
- Fetches markets by series_ticker (no orderbook calls)
- Uses prices from market list response
- Minimal API calls = no rate limiting

Usage:
    # Discover available series in Economics category
    python series_scanner.py --discover --category Economics
    
    # Scan specific series
    python series_scanner.py --series KXFED KXCPI
    
    # Scan and analyze (uses AI, costs money)
    python series_scanner.py --series KXFED --analyze --max-analyze 5
"""

import asyncio
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from ai_trading_bot_refactored import AdvancedTradingBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def discover_series(bot: AdvancedTradingBot, category: str = None):
    """Discover available series on Kalshi."""
    logger.info(f"🔍 Discovering series" + (f" in category '{category}'" if category else ""))
    
    series_list = await bot.discover_kalshi_series(category)
    
    print("\n" + "=" * 60)
    print(f"Found {len(series_list)} series:")
    print("=" * 60)
    
    for s in sorted(series_list, key=lambda x: x['category']):
        print(f"  {s['ticker']:30} | {s['category']:20} | {s['title'][:40]}")
    
    return series_list


async def scan_series(
    bot: AdvancedTradingBot,
    series_tickers: list,
    analyze: bool = False,
    max_analyze: int = 10,
    save_report: bool = True,
):
    """Scan markets by series ticker."""
    report = await bot.run_series_scan(
        series_tickers=series_tickers,
        analyze=analyze,
        max_analyze=max_analyze,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SCAN RESULTS")
    print("=" * 60)
    print(f"Series: {report['series_tickers']}")
    print(f"Scanned: {report['counts']['scanned']}")
    print(f"Passed filters: {report['counts']['passed_filters']}")
    if analyze:
        print(f"Analyzed: {report['counts']['analyzed']}")
    
    # Print markets
    if report['markets']:
        print("\n📊 Markets found:")
        for m in report['markets'][:20]:  # Limit display
            print(f"  {m['market_id']:30} | ${m['yes_price']:.2f} | vol:{m['volume']:,.0f} | {m['title'][:40]}")
        if len(report['markets']) > 20:
            print(f"  ... and {len(report['markets']) - 20} more")
    
    # Print analyses if any
    if report['analyses']:
        print("\n🤖 AI Analyses:")
        for a in report['analyses']:
            edge = a['edge']
            edge_str = f"+{edge:.1%}" if edge > 0 else f"{edge:.1%}"
            print(f"  {a['market_id'][:30]} | Market: {a['market_price']:.2f} | AI: {a['estimated_probability']:.2f} | Edge: {edge_str}")
    
    # Save report
    if save_report:
        reports_dir = Path(__file__).parent.parent / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"series_scan_{timestamp}.json"
        filepath = reports_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📁 Report saved to: {filepath}")
    
    return report


async def main():
    parser = argparse.ArgumentParser(description="Kalshi series-based market scanner")
    parser.add_argument('--discover', action='store_true', help='Discover available series')
    parser.add_argument('--category', type=str, help='Filter series by category')
    parser.add_argument('--series', nargs='+', help='Series tickers to scan')
    parser.add_argument('--analyze', action='store_true', help='Run AI analysis on filtered markets')
    parser.add_argument('--max-analyze', type=int, default=10, help='Max markets to analyze')
    parser.add_argument('--no-save', action='store_true', help='Do not save report to file')
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = AdvancedTradingBot()
    
    if args.discover:
        await discover_series(bot, args.category)
    elif args.series:
        await scan_series(
            bot,
            series_tickers=args.series,
            analyze=args.analyze,
            max_analyze=args.max_analyze,
            save_report=not args.no_save,
        )
    else:
        parser.print_help()
        print("\n💡 Examples:")
        print("  python series_scanner.py --discover --category Economics")
        print("  python series_scanner.py --series KXFED KXCPI")
        print("  python series_scanner.py --series KXFED --analyze --max-analyze 5")


if __name__ == "__main__":
    asyncio.run(main())
