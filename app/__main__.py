import argparse
import asyncio
import logging
import sys
from pathlib import Path
from app.bot import AdvancedTradingBot
from app.modes.collect import run_collect, discover_kalshi_series
from app.utils import LockManager

def setup_logging(log_level):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

async def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot CLI")
    parser.add_argument('--config', type=str, default='advanced_config.json', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['collect', 'analyze', 'trade'], default='collect', help='Execution mode')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
    parser.add_argument('--once', action='store_true', help='Run one cycle and exit')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--lock-file', type=str, help='Custom path to lock file')

    # Discovery/Utility arguments
    parser.add_argument('--backup', action='store_true', help='Create a backup of the database')
    parser.add_argument('--discover-series', action='store_true', help='Discover Kalshi series')
    parser.add_argument('--category', type=str, help='Category for series discovery')
    parser.add_argument('--series', nargs='+', help='Specific series tickers to collect')

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("app")

    # Acquire lock before proceeding
    lock_manager = LockManager(args.lock_file)
    if not lock_manager.acquire():
        logger.error("Failed to acquire lock. Another instance might be running. Exiting.")
        sys.exit(1)

    try:
        bot = AdvancedTradingBot(args.config)

        # Override dry-run if specified
        if args.dry_run:
            bot.config.trading.dry_run = True
            logger.info("Dry-run mode enabled via CLI")

        if args.backup:
            logger.info("Creating database backup...")
            # Initialize DB if it doesn't exist yet
            if not Path(bot.db.db_path).exists():
                logger.info("Database does not exist. Initializing empty database first...")
                await bot.db.initialize()

            backup_path = await bot.db.backup()
            print(f"Database backup created at: {backup_path}")
            return

        # Validate config for the selected mode (discovery/collect/analyze/trade)
        bot.config.validate_for_mode(args.mode)

        if args.discover_series:
            await discover_kalshi_series(bot, args.category)
            return

        if args.mode == 'collect':
            if args.once:
                await run_collect(bot, args.series)
            else:
                logger.info("Starting continuous collection mode (every 1 hour)...")
                while True:
                    await run_collect(bot, args.series)
                    logger.info("Sleeping for 1 hour...")
                    await asyncio.sleep(3600)

        elif args.mode == 'analyze':
            logger.info("Analyze mode not fully implemented in this restructuring step.")
            # In a real scenario, we'd call run_analyze(bot)
            if args.once:
                await bot.run_trading_cycle() # Fallback to existing cycle for now
            else:
                while True:
                    await bot.run_trading_cycle()
                    await asyncio.sleep(3600)

        elif args.mode == 'trade':
            logger.info("Trade mode not fully implemented in this restructuring step.")
            # In a real scenario, we'd call run_trade(bot)
            if args.once:
                await bot.run_trading_cycle()
            else:
                while True:
                    await bot.run_trading_cycle()
                    await asyncio.sleep(3600)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        lock_manager.release()

if __name__ == "__main__":
    asyncio.run(main())
