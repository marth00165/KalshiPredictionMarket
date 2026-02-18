import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from app.bot import AdvancedTradingBot
from app.config import ConfigManager
from app.modes.collect import run_collect, discover_kalshi_series
from app.utils import LockManager


ALLOWED_CONFIG_TOP_LEVEL_KEYS = {
    "database",
    "analysis",
    "api",
    "openai",
    "claude",
    "platforms",
    "trading",
    "strategy",
    "risk",
    "filters",
}


def setup_logging(log_level):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )


def _parse_bool(raw: str) -> bool:
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value for --set-dry-run, got: {raw!r}"
    )


def _parse_cli_value(raw: str):
    val = raw.strip()
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() == "null":
        return None
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return val


def _set_nested(config: dict, path: str, value) -> None:
    keys = path.split(".")
    if not keys or not keys[0]:
        raise ValueError(f"Invalid config path: {path!r}")

    top = keys[0]
    if top not in ALLOWED_CONFIG_TOP_LEVEL_KEYS:
        raise ValueError(
            f"Unknown top-level config key '{top}'. "
            f"Allowed: {sorted(ALLOWED_CONFIG_TOP_LEVEL_KEYS)}"
        )

    cur = config
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _collect_config_updates(args) -> list:
    updates = []

    for assignment in args.set_config or []:
        if "=" not in assignment:
            raise ValueError(
                f"Invalid --set-config value: {assignment!r}. Expected KEY=VALUE"
            )
        path, raw_value = assignment.split("=", 1)
        updates.append((path.strip(), _parse_cli_value(raw_value)))

    if args.set_min_edge is not None:
        updates.append(("strategy.min_edge", args.set_min_edge))
    if args.set_min_confidence is not None:
        updates.append(("strategy.min_confidence", args.set_min_confidence))
    if args.set_max_kelly_fraction is not None:
        updates.append(("risk.max_kelly_fraction", args.set_max_kelly_fraction))
    if args.set_max_total_exposure_fraction is not None:
        updates.append(("risk.max_total_exposure_fraction", args.set_max_total_exposure_fraction))
    if args.set_max_new_exposure_per_day_fraction is not None:
        updates.append(("risk.max_new_exposure_per_day_fraction", args.set_max_new_exposure_per_day_fraction))
    if args.set_max_positions is not None:
        updates.append(("risk.max_positions", args.set_max_positions))
    if args.set_max_position_size is not None:
        updates.append(("risk.max_position_size", args.set_max_position_size))
    if args.set_dry_run is not None:
        updates.append(("trading.dry_run", args.set_dry_run))

    return updates


def _apply_config_updates(config_path: str, updates: list) -> list:
    path = Path(config_path)
    if path.exists():
        with path.open("r") as f:
            raw_config = json.load(f)
    else:
        raw_config = {}

    applied = []
    for key_path, value in updates:
        _set_nested(raw_config, key_path, value)
        applied.append((key_path, value))

    with path.open("w") as f:
        json.dump(raw_config, f, indent=2)
        f.write("\n")

    return applied


def _build_safe_config_view(cfg: ConfigManager) -> dict:
    return {
        "database": {"path": cfg.db.path},
        "analysis": {
            "provider": cfg.analysis.provider,
            "allow_runtime_override": cfg.analysis.allow_runtime_override,
        },
        "api": {
            "batch_size": cfg.api.batch_size,
            "api_cost_limit_per_cycle": cfg.api.api_cost_limit_per_cycle,
            "claude_api_key_set": bool(cfg.api.claude_api_key),
            "openai_api_key_set": bool(cfg.api.openai_api_key),
        },
        "openai": {
            "model": cfg.openai.model,
            "temperature": cfg.openai.temperature,
            "max_tokens": cfg.openai.max_tokens,
            "base_url": cfg.openai.base_url,
            "input_cost_per_mtok": cfg.openai.input_cost_per_mtok,
            "output_cost_per_mtok": cfg.openai.output_cost_per_mtok,
        },
        "claude": {
            "model": cfg.claude.model,
            "temperature": cfg.claude.temperature,
            "max_tokens": cfg.claude.max_tokens,
            "input_cost_per_mtok": cfg.claude.input_cost_per_mtok,
            "output_cost_per_mtok": cfg.claude.output_cost_per_mtok,
        },
        "platforms": {
            "polymarket": {
                "enabled": cfg.platforms.polymarket.enabled,
                "max_markets": cfg.platforms.polymarket.max_markets,
                "api_key_set": bool(cfg.platforms.polymarket.api_key),
                "private_key_set": bool(cfg.platforms.polymarket.private_key),
                "private_key_file": cfg.platforms.polymarket.private_key_file,
            },
            "kalshi": {
                "enabled": cfg.platforms.kalshi.enabled,
                "max_markets": cfg.platforms.kalshi.max_markets,
                "series_tickers": cfg.platforms.kalshi.series_tickers,
                "api_key_set": bool(cfg.platforms.kalshi.api_key),
                "private_key_set": bool(cfg.platforms.kalshi.private_key),
                "private_key_file": cfg.platforms.kalshi.private_key_file,
            },
        },
        "trading": {
            "initial_bankroll": cfg.trading.initial_bankroll,
            "dry_run": cfg.trading.dry_run,
        },
        "strategy": {
            "min_edge": cfg.strategy.min_edge,
            "min_confidence": cfg.strategy.min_confidence,
        },
        "risk": {
            "max_kelly_fraction": cfg.risk.max_kelly_fraction,
            "max_positions": cfg.risk.max_positions,
            "max_position_size": cfg.risk.max_position_size,
            "max_total_exposure_fraction": cfg.risk.max_total_exposure_fraction,
            "max_new_exposure_per_day_fraction": cfg.risk.max_new_exposure_per_day_fraction,
        },
        "filters": {
            "min_volume": cfg.filters.min_volume,
            "min_liquidity": cfg.filters.min_liquidity,
            "volume_tiers": cfg.filters.volume_tiers,
            "focus_categories": cfg.filters.focus_categories,
        },
    }


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

    # Config utility arguments
    parser.add_argument('--show-config', action='store_true', help='Print effective non-secret config and exit')
    parser.add_argument('--verify-config', action='store_true', help='Validate config for selected mode and exit')
    parser.add_argument('--set-config', action='append', default=[], metavar='KEY=VALUE',
                        help='Set config value by dot path (example: risk.max_positions=3). Can be repeated.')
    parser.add_argument('--set-min-edge', type=float, help='Set strategy.min_edge (0-1) and exit')
    parser.add_argument('--set-min-confidence', type=float, help='Set strategy.min_confidence (0-1) and exit')
    parser.add_argument('--set-max-kelly-fraction', type=float, help='Set risk.max_kelly_fraction (0-1) and exit')
    parser.add_argument('--set-max-total-exposure-fraction', type=float,
                        help='Set risk.max_total_exposure_fraction (0-1) and exit')
    parser.add_argument('--set-max-new-exposure-per-day-fraction', type=float,
                        help='Set risk.max_new_exposure_per_day_fraction (0-1) and exit')
    parser.add_argument('--set-max-positions', type=int, help='Set risk.max_positions and exit')
    parser.add_argument('--set-max-position-size', type=float, help='Set risk.max_position_size and exit')
    parser.add_argument('--set-dry-run', type=_parse_bool, help='Set trading.dry_run (true/false) and exit')

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("app")

    try:
        updates = _collect_config_updates(args)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    config_utility_mode = bool(updates or args.show_config or args.verify_config)
    if config_utility_mode:
        try:
            if updates:
                applied = _apply_config_updates(args.config, updates)
                print(f"Updated {args.config}:")
                for key_path, value in applied:
                    print(f"  {key_path} = {value!r}")

            cfg = ConfigManager(args.config)
            if args.dry_run:
                cfg.trading.dry_run = True

            if args.verify_config:
                cfg.validate_for_mode(args.mode)
                print(f"Config validation passed for mode='{args.mode}' using {args.config}")

            if args.show_config:
                print(json.dumps(_build_safe_config_view(cfg), indent=2))
            return
        except Exception as e:
            logger.error(f"Config utility operation failed: {e}", exc_info=True)
            sys.exit(1)

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
                await bot.initialize()

            backup_path = await bot.db.backup()
            print(f"Database backup created at: {backup_path}")
            return

        # Validate config for the selected mode (discovery/collect/analyze/trade)
        bot.config.validate_for_mode(args.mode)

        # Initialize bot (DB, Bankroll, Positions)
        await bot.initialize()

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
