import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional

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
    "signal_fusion",
}

DEFAULT_DRYRUN_DB_PATH = "kalshi_dryrun.sqlite"
DEFAULT_LIVE_DB_PATH = "kalshi_live.sqlite"


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
    if args.set_max_orders_per_cycle is not None:
        updates.append(("risk.max_orders_per_cycle", args.set_max_orders_per_cycle))
    if args.set_max_notional_per_cycle is not None:
        updates.append(("risk.max_notional_per_cycle", args.set_max_notional_per_cycle))
    if args.set_daily_loss_limit_fraction is not None:
        updates.append(("risk.daily_loss_limit_fraction", args.set_daily_loss_limit_fraction))
    if args.set_dry_run is not None:
        updates.append(("trading.dry_run", args.set_dry_run))
    if args.set_allowed_market_ids is not None:
        updates.append(("trading.allowed_market_ids", args.set_allowed_market_ids))
    if args.set_allowed_event_tickers is not None:
        updates.append(("trading.allowed_event_tickers", args.set_allowed_event_tickers))
    if args.set_allowed_series_tickers is not None:
        updates.append(("platforms.kalshi.series_tickers", args.set_allowed_series_tickers))

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


def _config_has_explicit_db_path(config_path: str) -> bool:
    path = Path(config_path)
    if not path.exists():
        return False
    try:
        with path.open("r") as f:
            raw = json.load(f)
    except Exception:
        return False
    return isinstance(raw, dict) and isinstance(raw.get("database"), dict) and "path" in raw["database"]


def _resolve_runtime_db_path(
    current_path: str,
    effective_dry_run: bool,
    has_explicit_db_path: bool,
) -> str:
    """
    Resolve runtime DB path after CLI mode overrides.

    Explicit config path always takes precedence.
    """
    if has_explicit_db_path:
        return current_path
    return DEFAULT_DRYRUN_DB_PATH if effective_dry_run else DEFAULT_LIVE_DB_PATH


def _build_safe_config_view(cfg: ConfigManager) -> dict:
    return {
        "database": {"path": cfg.db.path},
        "analysis": {
            "provider": cfg.analysis.provider,
            "allow_runtime_override": cfg.analysis.allow_runtime_override,
            "context_json_path": cfg.analysis.context_json_path,
            "context_max_chars": cfg.analysis.context_max_chars,
            "nba_elo_enabled": cfg.analysis.nba_elo_enabled,
            "nba_elo_data_path": cfg.analysis.nba_elo_data_path,
            "nba_elo_output_path": cfg.analysis.nba_elo_output_path,
            "nba_elo_initial_rating": cfg.analysis.nba_elo_initial_rating,
            "nba_elo_home_advantage": cfg.analysis.nba_elo_home_advantage,
            "nba_elo_k_factor": cfg.analysis.nba_elo_k_factor,
            "nba_elo_regression_factor": cfg.analysis.nba_elo_regression_factor,
            "nba_elo_use_mov_multiplier": cfg.analysis.nba_elo_use_mov_multiplier,
            "nba_elo_round_decimals": cfg.analysis.nba_elo_round_decimals,
            "nba_elo_min_season": cfg.analysis.nba_elo_min_season,
            "nba_elo_allowed_seasons": cfg.analysis.nba_elo_allowed_seasons,
            "nba_elo_season_ratings_output_path": cfg.analysis.nba_elo_season_ratings_output_path,
            "enable_elo_calibration": cfg.analysis.enable_elo_calibration,
            "calibration_bucket_size": cfg.analysis.calibration_bucket_size,
            "calibration_prior": cfg.analysis.calibration_prior,
            "calibration_csv_path": cfg.analysis.calibration_csv_path,
            "calibration_min_season": cfg.analysis.calibration_min_season,
            "calibration_recency_mode": cfg.analysis.calibration_recency_mode,
            "calibration_recency_halflife_days": cfg.analysis.calibration_recency_halflife_days,
            "enable_live_injury_news": cfg.analysis.enable_live_injury_news,
            "enable_injury_llm_cache": cfg.analysis.enable_injury_llm_cache,
            "injury_cache_file": cfg.analysis.injury_cache_file,
            "llm_refresh_max_age_seconds": cfg.analysis.llm_refresh_max_age_seconds,
            "force_llm_refresh_near_tipoff_minutes": cfg.analysis.force_llm_refresh_near_tipoff_minutes,
            "near_tipoff_llm_stale_seconds": cfg.analysis.near_tipoff_llm_stale_seconds,
            "llm_refresh_on_price_move_pct": cfg.analysis.llm_refresh_on_price_move_pct,
            "injury_analysis_version": cfg.analysis.injury_analysis_version,
            "injury_prompt_version": cfg.analysis.injury_prompt_version,
            "force_injury_llm_refresh": cfg.analysis.force_injury_llm_refresh,
            "injury_feed_cache_ttl_seconds": cfg.analysis.injury_feed_cache_ttl_seconds,
            "injury_profile_cache_ttl_seconds": cfg.analysis.injury_profile_cache_ttl_seconds,
            "team_profile_budget_window_seconds": cfg.analysis.team_profile_budget_window_seconds,
            "max_team_profiles_per_cycle": cfg.analysis.max_team_profiles_per_cycle,
            "pim_k_factor": cfg.analysis.pim_k_factor,
            "pim_max_delta": cfg.analysis.pim_max_delta,
            "llm_adjustment_max_delta": cfg.analysis.llm_adjustment_max_delta,
            "persist_reasoning_to_db": cfg.analysis.persist_reasoning_to_db,
            "use_recent_reasoning_context": cfg.analysis.use_recent_reasoning_context,
        },
        "api": {
            "batch_size": cfg.api.batch_size,
            "api_cost_limit_per_cycle": cfg.api.api_cost_limit_per_cycle,
            "claude_api_key_set": bool(cfg.api.claude_api_key),
            "openai_api_key_set": bool(cfg.api.openai_api_key),
            "sportradar_api_key_set": bool(cfg.api.sportradar_api_key),
            "sportradar_access_level": cfg.api.sportradar_access_level,
            "sportradar_base_url": cfg.api.sportradar_base_url,
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
            "enforce_live_cash_check": cfg.trading.enforce_live_cash_check,
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
            "max_orders_per_cycle": cfg.risk.max_orders_per_cycle,
            "max_notional_per_cycle": cfg.risk.max_notional_per_cycle,
            "daily_loss_limit_fraction": cfg.risk.daily_loss_limit_fraction,
            "max_trades_per_market_per_day": cfg.risk.max_trades_per_market_per_day,
            "failure_streak_cooldown_threshold": cfg.risk.failure_streak_cooldown_threshold,
            "failure_cooldown_cycles": cfg.risk.failure_cooldown_cycles,
        },
        "execution": {
            "max_price_drift": cfg.execution.max_price_drift,
            "min_edge_at_execution": cfg.execution.min_edge_at_execution,
            "max_submit_slippage": cfg.execution.max_submit_slippage,
            "apply_kalshi_fees_to_edge": cfg.execution.apply_kalshi_fees_to_edge,
            "kalshi_fee_rate": cfg.execution.kalshi_fee_rate,
            "pending_not_found_retries": cfg.execution.pending_not_found_retries,
            "pending_timeout_minutes": cfg.execution.pending_timeout_minutes,
            "order_reconciliation_max_pages": cfg.execution.order_reconciliation_max_pages,
            "order_reconciliation_page_limit": cfg.execution.order_reconciliation_page_limit,
        },
        "filters": {
            "min_volume": cfg.filters.min_volume,
            "min_liquidity": cfg.filters.min_liquidity,
            "volume_tiers": cfg.filters.volume_tiers,
            "focus_categories": cfg.filters.focus_categories,
        },
    }


def _get_nested(config: dict, path: str, default=None):
    cur = config
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _is_effectively_set(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return False
        if cleaned.startswith("YOUR_") or "YOUR_" in cleaned:
            return False
    return True


def _parse_choice(raw: str, choices: set[str]) -> str:
    value = raw.strip().lower()
    if value not in choices:
        raise ValueError(f"expected one of {sorted(choices)}, got {raw!r}")
    return value


def _parse_float(raw: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    try:
        value = float(raw)
    except ValueError as e:
        raise ValueError("expected a number") from e
    if min_value is not None and value < min_value:
        raise ValueError(f"must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"must be <= {max_value}")
    return value


def _parse_int(raw: str, min_value: Optional[int] = None) -> int:
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError("expected an integer") from e
    if min_value is not None and value < min_value:
        raise ValueError(f"must be >= {min_value}")
    return value


def _parse_nonempty(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("cannot be empty")
    return value


def _prompt_setting(
    raw_config: dict,
    path: str,
    label: str,
    parser: Callable[[str], object],
    required: bool = False,
) -> object:
    current = _get_nested(raw_config, path, None)
    is_secret = path.endswith("api_key")
    if is_secret:
        current_display = "set" if _is_effectively_set(current) else "unset"
    else:
        current_display = current
    while True:
        suffix = f" [{current_display}]" if current_display is not None else ""
        user_input = input(f"{label}{suffix}: ").strip()
        if user_input == "":
            if current is not None and (not required or _is_effectively_set(current)):
                return current
            if not required:
                return None
            print("This value is required.")
            continue
        try:
            return parser(user_input)
        except Exception as e:
            print(f"Invalid value: {e}")


def _write_config_file(config_path: str, raw_config: dict) -> None:
    path = Path(config_path)
    with path.open("w") as f:
        json.dump(raw_config, f, indent=2)
        f.write("\n")


def _missing_required_env_vars(cfg: ConfigManager, mode: str, cli_dry_run: bool) -> list[str]:
    missing: list[str] = []
    effective_dry_run = bool(cli_dry_run or cfg.is_dry_run)

    if mode in {"collect", "trade"}:
        if cfg.kalshi_enabled and not _is_effectively_set(os.getenv("KALSHI_API_KEY")):
            missing.append("KALSHI_API_KEY")
        if cfg.polymarket_enabled and not _is_effectively_set(os.getenv("POLYMARKET_API_KEY")):
            missing.append("POLYMARKET_API_KEY")

    if mode in {"analyze", "trade"}:
        provider = cfg.analysis_provider
        if provider == "claude" and not _is_effectively_set(os.getenv("ANTHROPIC_API_KEY")):
            missing.append("ANTHROPIC_API_KEY")
        if provider == "openai" and not _is_effectively_set(os.getenv("OPENAI_API_KEY")):
            missing.append("OPENAI_API_KEY")

    if mode == "trade" and not effective_dry_run:
        if cfg.kalshi_enabled and not (
            _is_effectively_set(os.getenv("KALSHI_PRIVATE_KEY"))
            or _is_effectively_set(cfg.platforms.kalshi.private_key_file)
        ):
            missing.append("KALSHI_PRIVATE_KEY (or platforms.kalshi.private_key_file)")
        if cfg.polymarket_enabled and not (
            _is_effectively_set(os.getenv("POLYMARKET_PRIVATE_KEY"))
            or _is_effectively_set(cfg.platforms.polymarket.private_key_file)
        ):
            missing.append("POLYMARKET_PRIVATE_KEY (or platforms.polymarket.private_key_file)")

    return missing


def _run_edit_mode(raw_config: dict, mode: str, cli_dry_run: bool) -> None:
    print("\nEntering config edit mode. Press Enter to keep current value.\n")
    print("Credentials are loaded from .env and are not prompted here.\n")

    base_fields = [
        (
            "analysis.provider",
            "Analysis provider (claude/openai)",
            lambda s: _parse_choice(s, {"claude", "openai"}),
        ),
        ("platforms.kalshi.enabled", "Enable Kalshi (true/false)", _parse_bool),
        ("platforms.polymarket.enabled", "Enable Polymarket (true/false)", _parse_bool),
        ("trading.dry_run", "Dry run mode (true/false)", _parse_bool),
        ("trading.initial_bankroll", "Initial bankroll (>0)", lambda s: _parse_float(s, min_value=0.01)),
        ("api.batch_size", "LLM batch size (>=1)", lambda s: _parse_int(s, min_value=1)),
        ("api.api_cost_limit_per_cycle", "API cost limit per cycle (>=0)", lambda s: _parse_float(s, min_value=0)),
        ("strategy.min_edge", "Minimum edge (0-1)", lambda s: _parse_float(s, min_value=0, max_value=1)),
        ("strategy.min_confidence", "Minimum confidence (0-1)", lambda s: _parse_float(s, min_value=0, max_value=1)),
        ("risk.max_kelly_fraction", "Max Kelly fraction (0-1)", lambda s: _parse_float(s, min_value=0, max_value=1)),
        ("risk.max_positions", "Max open positions (>=1)", lambda s: _parse_int(s, min_value=1)),
        ("risk.max_position_size", "Max position size (>0)", lambda s: _parse_float(s, min_value=0.01)),
        ("risk.max_total_exposure_fraction", "Max total exposure fraction (0-1)", lambda s: _parse_float(s, min_value=0, max_value=1)),
        ("risk.max_new_exposure_per_day_fraction", "Max new exposure per day fraction (0-1)", lambda s: _parse_float(s, min_value=0, max_value=1)),
        ("filters.min_volume", "Minimum volume filter (>=0)", lambda s: _parse_float(s, min_value=0)),
        ("filters.min_liquidity", "Minimum liquidity filter (>=0)", lambda s: _parse_float(s, min_value=0)),
    ]

    for path, label, parser in base_fields:
        value = _prompt_setting(raw_config, path, label, parser, required=False)
        if value is not None:
            _set_nested(raw_config, path, value)

    provider = str(_get_nested(raw_config, "analysis.provider", "claude")).lower()
    kalshi_enabled = bool(_get_nested(raw_config, "platforms.kalshi.enabled", True))
    polymarket_enabled = bool(_get_nested(raw_config, "platforms.polymarket.enabled", False))
    effective_dry_run = bool(cli_dry_run or _get_nested(raw_config, "trading.dry_run", True))

    # Optional key-file paths for live mode; actual key material should come from env vars.
    if mode == "trade" and not effective_dry_run and kalshi_enabled:
        has_env_private = bool(os.getenv("KALSHI_PRIVATE_KEY"))
        required = not (has_env_private or bool(_get_nested(raw_config, "platforms.kalshi.private_key_file", None)))
        value = _prompt_setting(
            raw_config,
            "platforms.kalshi.private_key_file",
            "Kalshi private key file path (required if no KALSHI_PRIVATE_KEY env)",
            _parse_nonempty,
            required=required
        )
        if value is not None:
            _set_nested(raw_config, "platforms.kalshi.private_key_file", value)

    if mode == "trade" and not effective_dry_run and polymarket_enabled:
        has_env_private = bool(os.getenv("POLYMARKET_PRIVATE_KEY"))
        required = not (has_env_private or bool(_get_nested(raw_config, "platforms.polymarket.private_key_file", None)))
        value = _prompt_setting(
            raw_config,
            "platforms.polymarket.private_key_file",
            "Polymarket private key file path (required if no POLYMARKET_PRIVATE_KEY env)",
            _parse_nonempty,
            required=required
        )
        if value is not None:
            _set_nested(raw_config, "platforms.polymarket.private_key_file", value)


def _maybe_run_startup_setup_wizard(args, logger) -> None:
    if not sys.stdin.isatty():
        return

    path = Path(args.config)
    cfg_error = None
    if path.exists():
        try:
            with path.open("r") as f:
                raw_config = json.load(f)
        except Exception as e:
            cfg_error = f"Config file parse error: {e}"
            raw_config = {}
    else:
        raw_config = {}

    missing_env = []
    try:
        cfg = ConfigManager(args.config)
        missing_env = _missing_required_env_vars(cfg, args.mode, cli_dry_run=args.dry_run)
    except Exception as e:
        cfg_error = cfg_error or str(e)

    if missing_env:
        print("\nMissing required credentials in environment (.env):")
        for item in missing_env:
            print(f"  - {item}")
        print("\nAdd these to .env, then restart the bot.")
        raise SystemExit(1)

    force_edit = bool(cfg_error)
    if force_edit:
        print("\nConfig setup is required before continuing.")
        if cfg_error:
            print(f"- Current config is invalid: {cfg_error}")
        choice = input("Start edit mode now? [Y/n]: ").strip().lower()
        if choice in {"n", "no"}:
            raise SystemExit(1)
    else:
        choice = input(
            f"\nUse defaults from {args.config}? [Y/e] (type 'e' to edit settings): "
        ).strip().lower()
        if choice not in {"e", "edit"}:
            return

    while True:
        _run_edit_mode(raw_config, mode=args.mode, cli_dry_run=args.dry_run)
        _write_config_file(args.config, raw_config)

        try:
            cfg = ConfigManager(args.config)
            if args.dry_run:
                cfg.trading.dry_run = True
            cfg.validate_for_mode(args.mode)
            logger.info(f"Setup wizard completed and saved {args.config}")
            break
        except Exception as e:
            print(f"\nUpdated config is still invalid: {e}")
            retry = input("Retry edit mode? [Y/n]: ").strip().lower()
            if retry in {"n", "no"}:
                raise SystemExit(1)


def _prompt_enable_market_picker_after_setup(args, effective_dry_run: bool) -> bool:
    """
    Ask whether to enable interactive market picking for this run.

    This prompt appears only for interactive dry-run analyze/trade sessions.
    """
    if args.mode not in {"analyze", "trade"}:
        return False
    if not effective_dry_run:
        return False
    if args.backup or args.discover_series:
        return False
    if not sys.stdin.isatty():
        return False

    choice = input(
        "\nAfter setup, choose markets from a list before analysis? [Y/n]: "
    ).strip().lower()
    return choice not in {"n", "no"}


def _parse_csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _truncate(raw: str, max_len: int) -> str:
    text = str(raw or "")
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def _parse_index_selection(raw: str, max_index: int, max_selected: int = 10) -> list[int]:
    selected = set()
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("selection cannot be empty")

    for token in tokens:
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            if start > end:
                raise ValueError(f"invalid range '{token}'")
            if start < 1 or end > max_index:
                raise ValueError(f"range '{token}' is out of bounds 1-{max_index}")
            for idx in range(start, end + 1):
                selected.add(idx)
        else:
            idx = int(token)
            if idx < 1 or idx > max_index:
                raise ValueError(f"index '{idx}' is out of bounds 1-{max_index}")
            selected.add(idx)

    ordered = sorted(selected)
    if len(ordered) > max_selected:
        raise ValueError(f"select at most {max_selected} series, got {len(ordered)}")
    return ordered


def _prompt_select_series_tickers(series_items: list[dict], max_selected: int = 10) -> list[str]:
    if not series_items:
        return []

    print(f"\nDiscovered {len(series_items)} series. Select up to {max_selected}.")
    print("Index | Ticker                     | Title")
    print("------|----------------------------|----------------------------------------------")
    for i, item in enumerate(series_items, start=1):
        ticker = str(item.get("ticker", "")).strip()
        title = _truncate(str(item.get("title", "")).strip(), 46)
        print(f"{i:>5} | {ticker:<26} | {title}")

    print("\nEnter IDs using comma/range format (example: 1,3,5-8)")
    while True:
        raw = input(f"Series IDs [max {max_selected}]: ").strip().lower()
        if raw in {"", "none", "n"}:
            return []
        try:
            selected_ids = _parse_index_selection(raw, len(series_items), max_selected=max_selected)
            selected = [series_items[idx - 1] for idx in selected_ids]
            selected_tickers = [str(item.get("ticker", "")).strip() for item in selected if item.get("ticker")]
            print("\nSelected series:")
            for item in selected:
                print(f"  - {item.get('ticker', '')}: {item.get('title', '')}")
            confirm = input("Confirm selection and continue? [Y/n]: ").strip().lower()
            if confirm in {"", "y", "yes"}:
                return selected_tickers
        except Exception as e:
            print(f"Invalid selection: {e}")


async def _discover_series_candidates_for_scope(
    bot: AdvancedTradingBot,
    category: Optional[str] = None,
    keyword: Optional[str] = None,
) -> list[dict]:
    if not bot.scanner.kalshi_client:
        return []

    series = await bot.discover_kalshi_series(category)
    keyword_norm = (keyword or "").strip().lower()

    candidates = []
    seen = set()
    for item in series:
        ticker = str(item.get("ticker", "")).strip()
        title = str(item.get("title", "")).strip()
        category_val = str(item.get("category", "")).strip()
        if not ticker:
            continue
        title_norm = title.lower()
        category_norm = category_val.lower()
        if keyword_norm and keyword_norm not in ticker.lower() and keyword_norm not in title_norm and keyword_norm not in category_norm:
            continue
        if ticker in seen:
            continue
        seen.add(ticker)
        candidates.append({
            "ticker": ticker,
            "title": title,
            "category": category_val,
        })

    return candidates


async def _maybe_prompt_runtime_scan_scope(args, bot: AdvancedTradingBot, logger) -> None:
    """
    Interactive pre-scan scope selection for trade/analyze runs.

    Lets user narrow fetches before API scanning begins.
    """
    if args.mode not in {"trade", "analyze"}:
        return
    if args.backup or args.discover_series:
        return
    if not sys.stdin.isatty():
        return
    if not bot.scanner.kalshi_client:
        return

    print("\nPre-scan scope (before market fetch):")
    print("  1) All markets (default)")
    print("  2) Sports -> NBA preset (Kalshi series discovery)")
    print("  3) Category + optional keyword (Kalshi series discovery)")
    print("  4) Manual Kalshi series tickers")
    choice = input("Choose scope [1]: ").strip().lower()
    if choice in {"", "1", "all", "a"}:
        bot.runtime_scan_series_tickers = None
        bot.runtime_scan_scope_description = "all_markets"
        logger.info("Runtime scan scope: all markets")
        return

    try:
        if choice in {"2", "sports", "nba"}:
            candidates = await _discover_series_candidates_for_scope(
                bot,
                category="Sports",
                keyword="nba",
            )
            if not candidates:
                logger.warning("No Sports/NBA series found. Falling back to all markets.")
                return
            tickers = _prompt_select_series_tickers(candidates, max_selected=10)
            if not tickers:
                logger.warning("No series selected. Falling back to all markets.")
                return
            bot.runtime_scan_series_tickers = tickers
            bot.runtime_scan_scope_description = "sports_nba_series_selected"
            logger.info(f"Runtime scan scope set: sports_nba_series_selected ({len(tickers)} series)")
            return

        if choice in {"3", "category", "c"}:
            category = input("Category (example: Sports, Economics) [Sports]: ").strip() or "Sports"
            keyword = input("Optional keyword filter (example: nba) [none]: ").strip()
            candidates = await _discover_series_candidates_for_scope(
                bot,
                category=category,
                keyword=keyword or None,
            )
            if not candidates:
                logger.warning(
                    f"No series found for category='{category}'"
                    + (f", keyword='{keyword}'" if keyword else "")
                    + ". Falling back to all markets."
                )
                return
            tickers = _prompt_select_series_tickers(candidates, max_selected=10)
            if not tickers:
                logger.warning("No series selected. Falling back to all markets.")
                return
            bot.runtime_scan_series_tickers = tickers
            bot.runtime_scan_scope_description = (
                f"category_selected:{category}" + (f":keyword:{keyword}" if keyword else "")
            )
            logger.info(f"Runtime scan scope set: {bot.runtime_scan_scope_description} ({len(tickers)} series)")
            return

        if choice in {"4", "series", "s"}:
            while True:
                raw = input("Enter series tickers (comma separated, max 10): ").strip()
                tickers = list(dict.fromkeys(_parse_csv_values(raw)))
                if not tickers:
                    logger.warning("No valid series tickers entered. Falling back to all markets.")
                    return
                if len(tickers) > 10:
                    print(f"Please enter at most 10 series tickers (you entered {len(tickers)}).")
                    continue
                print("Selected manual series:")
                for ticker in tickers:
                    print(f"  - {ticker}")
                confirm = input("Confirm selection and continue? [Y/n]: ").strip().lower()
                if confirm in {"", "y", "yes"}:
                    bot.runtime_scan_series_tickers = tickers
                    bot.runtime_scan_scope_description = "manual_series_tickers"
                    logger.info(f"Runtime scan scope set: manual_series_tickers ({len(bot.runtime_scan_series_tickers)} series)")
                    return

        logger.info("Unrecognized scope option. Using all markets.")
    except Exception as e:
        logger.warning(f"Failed to configure pre-scan scope: {e}. Using all markets.")


async def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot CLI")
    parser.add_argument('--config', type=str, default='advanced_config.json', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['collect', 'analyze', 'trade'], default='collect', help='Execution mode')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
    parser.add_argument('--non-interactive', action='store_true', help='Disable all interactive prompts')
    parser.add_argument('--autonomous', action='store_true', help='Enable autonomous mode (implies non-interactive)')
    parser.add_argument('--allow-live-legacy-db', action='store_true',
                        help='Allow using legacy kalshi.sqlite in live mode')
    parser.add_argument('--once', action='store_true', help='Run one cycle and exit')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--lock-file', type=str, help='Custom path to lock file')
    parser.add_argument('--skip-setup-wizard', action='store_true',
                        help='Skip interactive startup setup wizard')
    parser.add_argument('--pick-markets', action='store_true',
                        help='In dry-run trade/analyze modes, interactively choose filtered markets before analysis')

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
    parser.add_argument('--set-max-orders-per-cycle', type=int, help='Set risk.max_orders_per_cycle and exit')
    parser.add_argument('--set-max-notional-per-cycle', type=float, help='Set risk.max_notional_per_cycle and exit')
    parser.add_argument('--set-daily-loss-limit-fraction', type=float, help='Set risk.daily_loss_limit_fraction (0-1) and exit')
    parser.add_argument('--set-dry-run', type=_parse_bool, help='Set trading.dry_run (true/false) and exit')
    parser.add_argument('--set-allowed-market-ids', type=_parse_csv_values, help='Set trading.allowed_market_ids (comma-separated) and exit')
    parser.add_argument('--set-allowed-event-tickers', type=_parse_csv_values, help='Set trading.allowed_event_tickers (comma-separated) and exit')
    parser.add_argument('--set-allowed-series-tickers', type=_parse_csv_values, help='Set platforms.kalshi.series_tickers (comma-separated) and exit')

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

    # Pre-load config to check autonomous/non-interactive settings
    try:
        temp_cfg = ConfigManager(args.config)
        is_autonomous = args.autonomous or temp_cfg.is_autonomous
        is_non_interactive = args.non_interactive or args.autonomous or temp_cfg.is_non_interactive
    except Exception:
        is_autonomous = args.autonomous
        is_non_interactive = args.non_interactive or args.autonomous

    if not args.skip_setup_wizard and not args.backup and not is_non_interactive:
        try:
            _maybe_run_startup_setup_wizard(args, logger)
        except SystemExit:
            raise
        except Exception as e:
            logger.error(f"Startup setup wizard failed: {e}", exc_info=True)
            sys.exit(1)

    # Acquire lock before proceeding
    lock_manager = LockManager(args.lock_file)
    if not lock_manager.acquire():
        logger.error("Failed to acquire lock. Another instance might be running. Exiting.")
        sys.exit(1)

    try:
        bot = AdvancedTradingBot(args.config)

        # Override config with CLI flags
        if args.dry_run:
            bot.config.trading.dry_run = True
            logger.info("Dry-run mode enabled via CLI")

        if args.autonomous:
            bot.config.trading.autonomous_mode = True
            bot.config.trading.non_interactive = True
            logger.info("Autonomous mode enabled via CLI")

        if args.non_interactive:
            bot.config.trading.non_interactive = True
            logger.info("Non-interactive mode enabled via CLI")

        effective_dry_run = bool(bot.config.is_dry_run)
        is_non_interactive = bool(bot.config.is_non_interactive)
        bot.single_run_dry_output_json = bool(effective_dry_run and args.once)

        # DB path safety: if the config did not explicitly set database.path, enforce mode-specific defaults
        # after CLI mode overrides (e.g., --dry-run).
        has_explicit_db_path = _config_has_explicit_db_path(args.config)
        desired_db_path = _resolve_runtime_db_path(
            current_path=bot.config.db.path,
            effective_dry_run=effective_dry_run,
            has_explicit_db_path=has_explicit_db_path,
        )
        if bot.config.db.path != desired_db_path:
            logger.info(
                "Adjusting DB path for runtime mode override: "
                f"{bot.config.db.path} -> {desired_db_path}"
            )
            bot.config.db.path = desired_db_path
            bot.db.db_path = desired_db_path

        # DB path safety check
        if not effective_dry_run and bot.config.db_path == "kalshi.sqlite" and not args.allow_live_legacy_db:
            logger.error(
                "Detected legacy 'kalshi.sqlite' in live mode. "
                "For safety, live mode now defaults to 'kalshi_live.sqlite'. "
                "Use --allow-live-legacy-db to override this safety check."
            )
            sys.exit(1)

        if args.pick_markets:
            if effective_dry_run:
                bot.interactive_market_pick = True
                logger.info("Interactive market picker enabled")
            else:
                logger.warning("--pick-markets ignored because bot is not in dry-run mode")
        elif not is_non_interactive and _prompt_enable_market_picker_after_setup(args, effective_dry_run):
            bot.interactive_market_pick = True
            logger.info("Interactive market picker enabled for this run")

        if not is_non_interactive:
            await _maybe_prompt_runtime_scan_scope(args, bot, logger)

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
            logger.info("Starting analyze mode.")
            if args.once:
                await bot.run_trading_cycle() # Fallback to existing cycle for now
            else:
                while True:
                    await bot.run_trading_cycle()
                    await asyncio.sleep(3600)

        elif args.mode == 'trade':
            logger.info("Starting trade mode.")
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
