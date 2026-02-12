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


async def discover_series(bot: AdvancedTradingBot, category: str = None, check_active: bool = False, save_map: bool = True):
    """Discover available series on Kalshi and save as a map."""
    logger.info(f"🔍 Discovering series" + (f" in category '{category}'" if category else ""))
    
    series_list = await bot.discover_kalshi_series(category)
    
    print("\n" + "=" * 60)
    print(f"Found {len(series_list)} series" + (" (checking for active markets...)" if check_active else ":"))
    print("=" * 60)
    
    # Build map organized by category
    series_map = {
        "discovered_at": datetime.now().isoformat(),
        "filter_category": category,
        "check_active": check_active,
        "total_series": len(series_list),
        "active_series": 0,
        "by_category": {},
        "series": [],
    }
    
    # If checking active, we need to query each series for open markets
    active_tickers = set()
    if check_active:
        print("\nChecking which series have open markets...")
        for i, s in enumerate(series_list):
            ticker = s['ticker']
            try:
                markets = await bot.scanner.kalshi_client.fetch_markets_by_series([ticker], status='open')
                market_count = len(markets)
                if market_count > 0:
                    active_tickers.add(ticker)
                    s['open_markets'] = market_count
                    print(f"  ✅ {ticker}: {market_count} open markets")
                # Progress indicator for inactive (don't spam)
                elif (i + 1) % 20 == 0:
                    print(f"  ... checked {i + 1}/{len(series_list)} series")
            except Exception as e:
                logger.warning(f"Error checking {ticker}: {e}")
        
        series_map["active_series"] = len(active_tickers)
        print(f"\n📊 {len(active_tickers)} series have open markets out of {len(series_list)} total")
    
    print("\n" + "=" * 60)
    if check_active:
        print(f"Active Series ({len(active_tickers)}):")
    else:
        print("All Series:")
    print("=" * 60)
    
    for s in sorted(series_list, key=lambda x: (x['category'], x['ticker'])):
        # Skip inactive series if check_active is enabled
        if check_active and s['ticker'] not in active_tickers:
            continue
            
        market_info = f" ({s.get('open_markets', '?')} markets)" if check_active else ""
        print(f"  {s['ticker']:30} | {s['category']:20} | {s['title'][:35]}{market_info}")
        
        # Add to series list
        series_map["series"].append(s)
        
        # Organize by category
        cat = s['category'] or 'Uncategorized'
        if cat not in series_map["by_category"]:
            series_map["by_category"][cat] = []
        series_map["by_category"][cat].append({
            "ticker": s['ticker'],
            "title": s['title'],
            "open_markets": s.get('open_markets'),
        })
    
    # Print category summary
    print("\n" + "=" * 60)
    print("Categories:")
    print("=" * 60)
    for cat, tickers in sorted(series_map["by_category"].items()):
        print(f"  {cat}: {len(tickers)} series")
    
    # Save map
    if save_map:
        reports_dir = Path(__file__).parent.parent / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{category.lower().replace(' ', '_')}" if category else ""
        filename = f"series_map{suffix}_{timestamp}.json"
        filepath = reports_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(series_map, f, indent=2)
        
        print(f"\n📁 Series map saved to: {filepath}")
    
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
    print(f"SCAN RESULTS - Cycle {report['cycle']}")
    print("=" * 60)
    print(f"Series: {report['config']['series_tickers']}")
    print(f"Started: {report['started_at']}")
    print(f"Finished: {report['finished_at']}")
    print(f"\nCounts:")
    print(f"  Scanned: {report['counts']['scanned']}")
    print(f"  Passed filters: {report['counts']['passed_filters']}")
    if analyze:
        print(f"  Analyzed: {report['counts']['analyzed']}")
        print(f"  Opportunities: {report['counts']['opportunities']}")
    
    # Print markets summary
    if report['markets']:
        print(f"\n📊 Markets ({len(report['markets'])} total):")
        for m in report['markets'][:20]:  # Limit display
            prices = m.get('prices', {})
            stats = m.get('stats', {})
            yes_price = prices.get('yes', 0)
            volume = stats.get('volume', 0)
            
            # Show filter status
            filters = m.get('filters', {})
            passed = filters.get('passed', False) if filters else False
            status = "✓" if passed else "✗"
            
            # Show analysis if present
            analysis = m.get('analysis')
            if analysis:
                edge = analysis.get('edge', 0)
                edge_str = f" | edge:{edge:+.1%}" if edge else ""
            else:
                edge_str = ""
            
            print(f"  {status} {m['market_id']:28} | ${yes_price:.2f} | vol:{volume:>8,.0f}{edge_str} | {m['title'][:35]}")
        
        if len(report['markets']) > 20:
            print(f"  ... and {len(report['markets']) - 20} more")
    
    # Print opportunities (markets with analysis that meet threshold)
    opportunities = [m for m in report['markets'] if m.get('opportunity')]
    if opportunities:
        print(f"\n🎯 Opportunities ({len(opportunities)}):")
        for m in opportunities:
            analysis = m['analysis']
            opp = m['opportunity']
            print(f"  {m['market_id'][:28]} | Market: {m['prices']['yes']:.2f} | AI: {analysis['estimated_probability']:.2f} | Edge: {opp['edge']:+.1%}")
    
    # Print errors if any
    if report['errors']:
        print(f"\n⚠️ Errors ({len(report['errors'])}):")
        for err in report['errors'][:5]:
            print(f"  {err}")
    
    # Print API cost if available
    if report.get('api_cost'):
        print(f"\n💰 API Cost: ${report['api_cost'].get('total_cost', 0):.4f}")
    
    # Save report
    if save_report:
        reports_dir = Path(__file__).parent.parent / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Use same naming convention as trading cycle: cycle_N_timestamp.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cycle_{report['cycle']}_{timestamp}.json"
        filepath = reports_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📁 Report saved to: {filepath}")
    
    return report


async def main():
    parser = argparse.ArgumentParser(description="Kalshi series-based market scanner")
    parser.add_argument('--discover', action='store_true', help='Discover available series')
    parser.add_argument('--category', type=str, help='Filter series by category')
    parser.add_argument('--check-active', action='store_true', help='Only show series with open markets (slower)')
    parser.add_argument('--series', nargs='+', help='Series tickers to scan')
    parser.add_argument('--analyze', action='store_true', help='Run AI analysis on filtered markets')
    parser.add_argument('--max-analyze', type=int, default=10, help='Max markets to analyze')
    parser.add_argument('--no-save', action='store_true', help='Do not save report to file')
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = AdvancedTradingBot()
    
    if args.discover:
        await discover_series(bot, args.category, check_active=args.check_active)
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
        print("  python series_scanner.py --discover                         # List all series")
        print("  python series_scanner.py --discover --check-active          # Only show series with open markets")
        print("  python series_scanner.py --discover --category Sports --check-active")
        print("  python series_scanner.py --series KXNBA KXCPI")
        print("  python series_scanner.py --series KXFED --analyze --max-analyze 5")


if __name__ == "__main__":
    asyncio.run(main())
