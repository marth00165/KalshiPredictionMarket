"""Configuration management for trading bot with validation and typed access"""

import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class DatabaseConfig:
    """Configuration for SQLite database"""
    path: str = "kalshi.sqlite"

    def validate(self) -> None:
        if not self.path:
            raise ValueError("database path cannot be empty")


@dataclass
class APIConfig:
    """Configuration for API services (Claude, OpenAI)"""
    
    claude_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    batch_size: int = 5
    api_cost_limit_per_cycle: float = 5.0
    
    def validate(self) -> None:
        """Validate API configuration"""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.api_cost_limit_per_cycle < 0:
            raise ValueError(f"api_cost_limit_per_cycle must be >= 0")


@dataclass
class AnalysisConfig:
    """Configuration for selecting which LLM provider performs analysis"""

    provider: str = "claude"  # "claude" or "openai"
    allow_runtime_override: bool = True

    def validate(self) -> None:
        provider_normalized = (self.provider or "").strip().lower()
        if provider_normalized not in {"claude", "openai"}:
            raise ValueError(f"provider must be 'claude' or 'openai', got {self.provider!r}")
        self.provider = provider_normalized


@dataclass
class SignalFusionConfig:
    """Configuration for optional external signal ingestion + fusion."""

    enabled: bool = False
    mode: str = "none"  # none|curator|market_feed|both
    curator_jsonl_path: Optional[str] = None
    market_feed_jsonl_path: Optional[str] = None
    ttl_seconds: int = 300
    max_event_age_seconds: int = 7200
    max_scan_lines: int = 4000
    curator_boost: float = 0.08
    market_feed_boost: float = 0.06
    low_recommendation_penalty: float = 0.22
    anomaly_penalty_factor: float = 0.70
    max_anomaly_veto: float = 0.95
    signal_score_weight: float = 0.20

    def validate(self) -> None:
        mode_normalized = (self.mode or "").strip().lower()
        if not mode_normalized:
            mode_normalized = "none"

        if mode_normalized not in {"none", "curator", "market_feed", "both"}:
            raise ValueError(
                f"signal_fusion.mode must be one of "
                f"['none', 'curator', 'market_feed', 'both'], got {self.mode!r}"
            )
        self.mode = mode_normalized

        if self.ttl_seconds < 0:
            raise ValueError(f"ttl_seconds must be >= 0, got {self.ttl_seconds}")
        if self.max_event_age_seconds < 0:
            raise ValueError(
                f"max_event_age_seconds must be >= 0, got {self.max_event_age_seconds}"
            )
        if self.max_scan_lines <= 0:
            raise ValueError(f"max_scan_lines must be > 0, got {self.max_scan_lines}")
        if not (0 <= self.curator_boost <= 1):
            raise ValueError(f"curator_boost must be between 0 and 1, got {self.curator_boost}")
        if not (0 <= self.market_feed_boost <= 1):
            raise ValueError(
                f"market_feed_boost must be between 0 and 1, got {self.market_feed_boost}"
            )
        if not (0 <= self.low_recommendation_penalty <= 1):
            raise ValueError(
                f"low_recommendation_penalty must be between 0 and 1, got {self.low_recommendation_penalty}"
            )
        if not (0 <= self.anomaly_penalty_factor <= 1):
            raise ValueError(
                f"anomaly_penalty_factor must be between 0 and 1, got {self.anomaly_penalty_factor}"
            )
        if not (0 <= self.max_anomaly_veto <= 1):
            raise ValueError(
                f"max_anomaly_veto must be between 0 and 1, got {self.max_anomaly_veto}"
            )
        if not (0 <= self.signal_score_weight <= 1):
            raise ValueError(
                f"signal_score_weight must be between 0 and 1, got {self.signal_score_weight}"
            )


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI models"""

    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 1000
    base_url: str = "https://api.openai.com/v1"
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0

    def validate(self) -> None:
        if not self.model:
            raise ValueError("model name cannot be empty")
        if not (0 <= self.temperature <= 2):
            raise ValueError(f"temperature must be between 0 and 2, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if not self.base_url:
            raise ValueError("base_url cannot be empty")
        if self.input_cost_per_mtok < 0 or self.output_cost_per_mtok < 0:
            raise ValueError("Pricing costs cannot be negative")


@dataclass
class PlatformConfig:
    """Configuration for a single prediction market platform"""
    
    api_key: Optional[str] = None
    private_key: Optional[str] = None
    private_key_file: Optional[str] = None
    enabled: bool = True
    max_markets: int = 25
    series_tickers: Optional[List[str]] = None  # For Kalshi: fetch only from these series
    allowed_market_ids: Optional[List[str]] = None
    allowed_event_tickers: Optional[List[str]] = None
    
    def validate(self) -> None:
        """Validate platform configuration"""
        if self.max_markets < 1:
            raise ValueError(f"max_markets must be >= 1, got {self.max_markets}")


@dataclass
class PlatformsConfig:
    """Configuration for all prediction market platforms"""
    
    polymarket: PlatformConfig = field(default_factory=PlatformConfig)
    kalshi: PlatformConfig = field(default_factory=PlatformConfig)
    
    def validate(self) -> None:
        """Validate platforms configuration"""
        self.polymarket.validate()
        self.kalshi.validate()


@dataclass
class TradingConfig:
    """Configuration for trading parameters"""
    
    initial_bankroll: float = 10000
    dry_run: bool = True
    autonomous_mode: bool = False
    non_interactive: Optional[bool] = None
    enforce_live_cash_check: bool = True
    require_scope_in_live: bool = True
    allowed_market_ids: List[str] = field(default_factory=list)
    allowed_event_tickers: List[str] = field(default_factory=list)
    
    def validate(self) -> None:
        """Validate trading configuration"""
        if self.initial_bankroll <= 0:
            raise ValueError(f"initial_bankroll must be > 0, got {self.initial_bankroll}")


@dataclass
class StrategyConfig:
    """Configuration for trading strategy"""
    
    min_edge: float = 0.08  # 8% edge minimum
    min_confidence: float = 0.60  # 60% confidence minimum
    
    def validate(self) -> None:
        """Validate strategy configuration"""
        if not (0 <= self.min_edge <= 1):
            raise ValueError(f"min_edge must be between 0 and 1, got {self.min_edge}")
        if not (0 <= self.min_confidence <= 1):
            raise ValueError(f"min_confidence must be between 0 and 1, got {self.min_confidence}")


@dataclass
class RiskConfig:
    """Configuration for risk management"""
    
    max_kelly_fraction: float = 0.25  # Max 25% of bankroll per trade
    max_positions: int = 10  # Max simultaneous positions
    max_position_size: float = 1000  # Max dollars per position
    max_total_exposure_fraction: float = 0.80  # Lenient: max 80% capital at risk
    max_new_exposure_per_day_fraction: float = 0.80  # Lenient: max 80% newly deployed per UTC day
    max_orders_per_cycle: int = 5
    max_notional_per_cycle: float = 2000
    daily_loss_limit_fraction: float = 0.10  # Stop if 10% of daily starting bankroll lost
    max_trades_per_market_per_day: int = 0  # 0 disables cap
    failure_streak_cooldown_threshold: int = 0  # 0 disables cooldown logic
    failure_cooldown_cycles: int = 0  # 0 disables cooldown logic
    kill_switch_env_var: str = "BOT_DISABLE_TRADING"
    critical_webhook_url: Optional[str] = None
    
    def validate(self) -> None:
        """Validate risk configuration"""
        if not (0 <= self.max_kelly_fraction <= 1):
            raise ValueError(f"max_kelly_fraction must be between 0 and 1, got {self.max_kelly_fraction}")
        if self.max_positions < 1:
            raise ValueError(f"max_positions must be >= 1, got {self.max_positions}")
        if self.max_position_size <= 0:
            raise ValueError(f"max_position_size must be > 0, got {self.max_position_size}")
        if not (0 <= self.max_total_exposure_fraction <= 1):
            raise ValueError(
                f"max_total_exposure_fraction must be between 0 and 1, got {self.max_total_exposure_fraction}"
            )
        if not (0 <= self.max_new_exposure_per_day_fraction <= 1):
            raise ValueError(
                "max_new_exposure_per_day_fraction must be between 0 and 1, "
                f"got {self.max_new_exposure_per_day_fraction}"
            )
        if self.max_orders_per_cycle < 1:
            raise ValueError(f"max_orders_per_cycle must be >= 1, got {self.max_orders_per_cycle}")
        if self.max_notional_per_cycle <= 0:
            raise ValueError(f"max_notional_per_cycle must be > 0, got {self.max_notional_per_cycle}")
        if not (0 <= self.daily_loss_limit_fraction <= 1):
            raise ValueError(f"daily_loss_limit_fraction must be between 0 and 1, got {self.daily_loss_limit_fraction}")
        if self.max_trades_per_market_per_day < 0:
            raise ValueError(
                "max_trades_per_market_per_day must be >= 0 "
                f"(0 disables), got {self.max_trades_per_market_per_day}"
            )
        if self.failure_streak_cooldown_threshold < 0:
            raise ValueError(
                "failure_streak_cooldown_threshold must be >= 0 "
                f"(0 disables), got {self.failure_streak_cooldown_threshold}"
            )
        if self.failure_cooldown_cycles < 0:
            raise ValueError(
                "failure_cooldown_cycles must be >= 0 "
                f"(0 disables), got {self.failure_cooldown_cycles}"
            )
        if not self.kill_switch_env_var:
            raise ValueError("kill_switch_env_var cannot be empty")


@dataclass
class ExecutionConfig:
    """Configuration for order execution revalidation"""

    max_price_drift: float = 0.05  # Absolute probability diff (5%)
    min_edge_at_execution: float = 0.02  # 2% minimum edge right before order
    max_submit_slippage: float = 0.10  # Absolute probability diff allowed at submit
    pending_not_found_retries: int = 3
    pending_timeout_minutes: int = 30
    order_reconciliation_max_pages: int = 5
    order_reconciliation_page_limit: int = 200

    def validate(self) -> None:
        """Validate execution configuration"""
        if self.max_price_drift < 0:
            raise ValueError(f"max_price_drift must be >= 0, got {self.max_price_drift}")
        if self.min_edge_at_execution < 0:
            raise ValueError(f"min_edge_at_execution must be >= 0, got {self.min_edge_at_execution}")
        if self.max_submit_slippage < 0:
            raise ValueError(f"max_submit_slippage must be >= 0, got {self.max_submit_slippage}")
        if self.pending_not_found_retries < 1:
            raise ValueError(f"pending_not_found_retries must be >= 1, got {self.pending_not_found_retries}")
        if self.pending_timeout_minutes < 1:
            raise ValueError(f"pending_timeout_minutes must be >= 1, got {self.pending_timeout_minutes}")
        if self.order_reconciliation_max_pages < 1:
            raise ValueError(f"order_reconciliation_max_pages must be >= 1, got {self.order_reconciliation_max_pages}")
        if self.order_reconciliation_page_limit < 1:
            raise ValueError(f"order_reconciliation_page_limit must be >= 1, got {self.order_reconciliation_page_limit}")


@dataclass
class FiltersConfig:
    """Configuration for market filtering"""
    
    min_volume: float = 1000  # Minimum market volume
    min_liquidity: float = 500  # Minimum liquidity
    volume_tiers: Optional[Dict[str, Dict[str, Any]]] = None  # Volume-based polling tiers
    focus_categories: Optional[List[str]] = None  # Categories to prioritize
    
    def validate(self) -> None:
        """Validate filters configuration"""
        if self.min_volume < 0:
            raise ValueError(f"min_volume must be >= 0, got {self.min_volume}")
        if self.min_liquidity < 0:
            raise ValueError(f"min_liquidity must be >= 0, got {self.min_liquidity}")
        # Validate volume_tiers structure if provided
        if self.volume_tiers:
            for tier_name, tier_config in self.volume_tiers.items():
                if not isinstance(tier_config, dict):
                    raise ValueError(f"volume_tiers.{tier_name} must be a dict")


@dataclass
class ClaudeConfig:
    """Configuration for Claude AI model"""
    
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3  # Lower = more consistent
    max_tokens: int = 1000
    input_cost_per_mtok: float = 3.00  # $3 per million tokens
    output_cost_per_mtok: float = 15.00  # $15 per million tokens
    
    def validate(self) -> None:
        """Validate Claude configuration"""
        if not self.model:
            raise ValueError("model name cannot be empty")
        if not (0 <= self.temperature <= 2):
            raise ValueError(f"temperature must be between 0 and 2, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.input_cost_per_mtok < 0 or self.output_cost_per_mtok < 0:
            raise ValueError("Pricing costs cannot be negative")


# ============================================================================
# MAIN CONFIG MANAGER
# ============================================================================

class ConfigManager:
    """
    Central configuration management with validation and typed access
    
    Replaces scattered config.get() calls throughout codebase with
    typed properties that validate values and provide sensible defaults.
    """
    
    def __init__(self, config_file: str):
        """
        Load and validate configuration from JSON file
        
        Args:
            config_file: Path to config JSON file
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
            ValueError: If configuration validation fails
        """
        # Load environment variables from .env file if it exists
        load_dotenv()

        self.config_path = Path(config_file)
        
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {config_file}. Using defaults and environment variables.")
            raw_config = {}
        else:
            # Load JSON
            with open(self.config_path) as f:
                raw_config = json.load(f)
        
        # Parse into typed configs
        self._parse_config(raw_config)
    
    def _get_secret(self, env_var: str, json_value: Optional[str] = None) -> Optional[str]:
        """
        Get secret from environment variable, falling back to JSON value.
        Filters out placeholder values starting with 'YOUR_'.
        """
        val = os.getenv(env_var)
        if not val:
            val = json_value

        if val and isinstance(val, str) and (val.startswith('YOUR_') or 'YOUR_' in val):
            return None
        return val

    def _parse_config(self, raw_config: Dict[str, Any]) -> None:
        """Parse raw JSON config into typed dataclasses"""

        # Trading configuration (parsed early to help with defaults)
        trading_raw = raw_config.get('trading', {})
        self.trading = TradingConfig(
            initial_bankroll=trading_raw.get('initial_bankroll', 10000),
            dry_run=trading_raw.get('dry_run', True),
            autonomous_mode=trading_raw.get('autonomous_mode', False),
            non_interactive=trading_raw.get('non_interactive'),
            enforce_live_cash_check=trading_raw.get('enforce_live_cash_check', True),
            require_scope_in_live=trading_raw.get('require_scope_in_live', True),
            allowed_market_ids=trading_raw.get('allowed_market_ids', []),
            allowed_event_tickers=trading_raw.get('allowed_event_tickers', []),
        )
        # Handle non_interactive default based on autonomous_mode
        if self.trading.non_interactive is None:
            self.trading.non_interactive = self.trading.autonomous_mode

        # Database configuration
        db_raw = raw_config.get('database', {})
        default_db = 'kalshi_dryrun.sqlite' if self.trading.dry_run else 'kalshi_live.sqlite'
        self.db = DatabaseConfig(
            path=db_raw.get('path', default_db)
        )
        try:
            self.db.validate()
        except ValueError as e:
            raise ValueError(f"Invalid database config: {e}")

        # Analysis provider selection
        analysis_raw = raw_config.get('analysis', {})
        self.analysis = AnalysisConfig(
            provider=analysis_raw.get('provider', 'claude'),
            allow_runtime_override=analysis_raw.get('allow_runtime_override', True),
        )
        try:
            self.analysis.validate()
        except ValueError as e:
            raise ValueError(f"Invalid analysis config: {e}")
        
        # API configuration (required)
        api_raw = raw_config.get('api', {})
        try:
            self.api = APIConfig(
                claude_api_key=self._get_secret('ANTHROPIC_API_KEY', api_raw.get('claude_api_key')),
                openai_api_key=self._get_secret('OPENAI_API_KEY', api_raw.get('openai_api_key')),
                batch_size=api_raw.get('batch_size', 50),
                api_cost_limit_per_cycle=api_raw.get('api_cost_limit_per_cycle', 5.0),
            )
            self.api.validate()
        except ValueError as e:
            raise ValueError(f"Invalid API config: {e}")

        # Claude configuration
        claude_raw = raw_config.get('claude', {})
        self.claude = ClaudeConfig(
            model=claude_raw.get('model', 'claude-sonnet-4-20250514'),
            temperature=claude_raw.get('temperature', 0.3),
            max_tokens=claude_raw.get('max_tokens', 1000),
            input_cost_per_mtok=claude_raw.get('input_cost_per_mtok', 3.00),
            output_cost_per_mtok=claude_raw.get('output_cost_per_mtok', 15.00),
        )
        try:
            self.claude.validate()
        except ValueError as e:
            raise ValueError(f"Invalid Claude config: {e}")

        # OpenAI configuration
        openai_raw = raw_config.get('openai', {})
        self.openai = OpenAIConfig(
            model=openai_raw.get('model', 'gpt-4o-mini'),
            temperature=openai_raw.get('temperature', 0.3),
            max_tokens=openai_raw.get('max_tokens', 1000),
            base_url=openai_raw.get('base_url', 'https://api.openai.com/v1'),
            input_cost_per_mtok=openai_raw.get('input_cost_per_mtok', 0.0),
            output_cost_per_mtok=openai_raw.get('output_cost_per_mtok', 0.0),
        )
        try:
            self.openai.validate()
        except ValueError as e:
            raise ValueError(f"Invalid OpenAI config: {e}")
        
        # Platforms configuration
        platforms_raw = raw_config.get('platforms', {})
        polymarket_raw = platforms_raw.get('polymarket', {})
        kalshi_raw = platforms_raw.get('kalshi', {})
        
        self.platforms = PlatformsConfig(
            polymarket=PlatformConfig(
                api_key=self._get_secret('POLYMARKET_API_KEY', polymarket_raw.get('api_key')),
                private_key=self._get_secret('POLYMARKET_PRIVATE_KEY'),
                private_key_file=polymarket_raw.get('private_key_file'),
                enabled=polymarket_raw.get('enabled', True), # Default to True if not in JSON
                max_markets=polymarket_raw.get('max_markets', 500),
            ),
            kalshi=PlatformConfig(
                api_key=self._get_secret('KALSHI_API_KEY', kalshi_raw.get('api_key')),
                private_key=self._get_secret('KALSHI_PRIVATE_KEY'),
                private_key_file=kalshi_raw.get('private_key_file'),
                enabled=kalshi_raw.get('enabled', True), # Default to True if not in JSON
                max_markets=kalshi_raw.get('max_markets', 500),
                series_tickers=kalshi_raw.get('series_tickers'),  # Optional list of series to fetch
                allowed_market_ids=kalshi_raw.get('allowed_market_ids'),
                allowed_event_tickers=kalshi_raw.get('allowed_event_tickers'),
            ),
        )
        try:
            self.platforms.validate()
        except ValueError as e:
            raise ValueError(f"Invalid platforms config: {e}")
        
        try:
            self.trading.validate()
        except ValueError as e:
            raise ValueError(f"Invalid trading config: {e}")
        
        # Strategy configuration
        strategy_raw = raw_config.get('strategy', {})
        self.strategy = StrategyConfig(
            min_edge=strategy_raw.get('min_edge', 0.08),
            min_confidence=strategy_raw.get('min_confidence', 0.60),
        )
        try:
            self.strategy.validate()
        except ValueError as e:
            raise ValueError(f"Invalid strategy config: {e}")
        
        # Risk configuration
        risk_raw = raw_config.get('risk', {})
        self.risk = RiskConfig(
            max_kelly_fraction=risk_raw.get('max_kelly_fraction', 0.25),
            max_positions=risk_raw.get('max_positions', 10),
            max_position_size=risk_raw.get('max_position_size', 1000),
            max_total_exposure_fraction=risk_raw.get('max_total_exposure_fraction', 0.80),
            max_new_exposure_per_day_fraction=risk_raw.get('max_new_exposure_per_day_fraction', 0.80),
            max_orders_per_cycle=risk_raw.get('max_orders_per_cycle', 5),
            max_notional_per_cycle=risk_raw.get('max_notional_per_cycle', 2000),
            daily_loss_limit_fraction=risk_raw.get('daily_loss_limit_fraction', 0.10),
            max_trades_per_market_per_day=risk_raw.get('max_trades_per_market_per_day', 0),
            failure_streak_cooldown_threshold=risk_raw.get('failure_streak_cooldown_threshold', 0),
            failure_cooldown_cycles=risk_raw.get('failure_cooldown_cycles', 0),
            kill_switch_env_var=risk_raw.get('kill_switch_env_var', 'BOT_DISABLE_TRADING'),
            critical_webhook_url=risk_raw.get('critical_webhook_url'),
        )
        try:
            self.risk.validate()
        except ValueError as e:
            raise ValueError(f"Invalid risk config: {e}")
        
        # Execution configuration
        execution_raw = raw_config.get('execution', {})
        self.execution = ExecutionConfig(
            max_price_drift=execution_raw.get('max_price_drift', 0.05),
            min_edge_at_execution=execution_raw.get('min_edge_at_execution', 0.02),
            max_submit_slippage=execution_raw.get('max_submit_slippage', 0.10),
            pending_not_found_retries=execution_raw.get('pending_not_found_retries', 3),
            pending_timeout_minutes=execution_raw.get('pending_timeout_minutes', 30),
            order_reconciliation_max_pages=execution_raw.get('order_reconciliation_max_pages', 5),
            order_reconciliation_page_limit=execution_raw.get('order_reconciliation_page_limit', 200),
        )
        try:
            self.execution.validate()
        except ValueError as e:
            raise ValueError(f"Invalid execution config: {e}")

        # Filters configuration
        filters_raw = raw_config.get('filters', {})
        self.filters = FiltersConfig(
            min_volume=filters_raw.get('min_volume', 1000),
            min_liquidity=filters_raw.get('min_liquidity', 500),
            volume_tiers=filters_raw.get('volume_tiers'),
            focus_categories=filters_raw.get('focus_categories'),
        )
        try:
            self.filters.validate()
        except ValueError as e:
            raise ValueError(f"Invalid filters config: {e}")

        # Signal fusion configuration
        signal_fusion_raw = raw_config.get('signal_fusion', {})
        self.signal_fusion = SignalFusionConfig(
            enabled=bool(signal_fusion_raw.get('enabled', False)),
            mode=signal_fusion_raw.get('mode', 'none'),
            curator_jsonl_path=signal_fusion_raw.get('curator_jsonl_path'),
            market_feed_jsonl_path=signal_fusion_raw.get('market_feed_jsonl_path'),
            ttl_seconds=signal_fusion_raw.get('ttl_seconds', 300),
            max_event_age_seconds=signal_fusion_raw.get('max_event_age_seconds', 7200),
            max_scan_lines=signal_fusion_raw.get('max_scan_lines', 4000),
            curator_boost=signal_fusion_raw.get('curator_boost', 0.08),
            market_feed_boost=signal_fusion_raw.get('market_feed_boost', 0.06),
            low_recommendation_penalty=signal_fusion_raw.get(
                'low_recommendation_penalty',
                0.22,
            ),
            anomaly_penalty_factor=signal_fusion_raw.get('anomaly_penalty_factor', 0.70),
            max_anomaly_veto=signal_fusion_raw.get('max_anomaly_veto', 0.95),
            signal_score_weight=signal_fusion_raw.get('signal_score_weight', 0.20),
        )
        try:
            self.signal_fusion.validate()
        except ValueError as e:
            raise ValueError(f"Invalid signal_fusion config: {e}")
        
        logger.info(f"✅ Configuration loaded and validated from {self.config_path}")
    
    # ========================================================================
    # CONVENIENCE PROPERTIES (reduce boilerplate)
    # ========================================================================
    
    @property
    def db_path(self) -> str:
        """Get path to SQLite database"""
        return self.db.path

    @property
    def claude_api_key(self) -> str:
        """Get Claude API key"""
        if not self.api.claude_api_key:
            raise ValueError("Claude API key not configured")
        return str(self.api.claude_api_key)

    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key"""
        if not self.api.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        return str(self.api.openai_api_key)

    @property
    def analysis_provider(self) -> str:
        """Get analysis provider ('claude' or 'openai')"""
        return self.analysis.provider

    @property
    def allow_runtime_override(self) -> bool:
        """Whether runtime analyzer override is allowed"""
        return self.analysis.allow_runtime_override

    @property
    def signal_fusion_enabled(self) -> bool:
        """Whether external signal fusion is enabled"""
        return self.signal_fusion.enabled and self.signal_fusion.mode != "none"
    
    @property
    def polymarket_enabled(self) -> bool:
        """Check if Polymarket is enabled"""
        return self.platforms.polymarket.enabled
    
    @property
    def kalshi_enabled(self) -> bool:
        """Check if Kalshi is enabled"""
        return self.platforms.kalshi.enabled
    
    @property
    def is_dry_run(self) -> bool:
        """Check if trading is in dry-run mode"""
        return self.trading.dry_run

    @property
    def is_autonomous(self) -> bool:
        """Check if bot is in autonomous mode"""
        return self.trading.autonomous_mode

    @property
    def is_non_interactive(self) -> bool:
        """Check if bot is in non-interactive mode"""
        return bool(self.trading.non_interactive)

    @property
    def max_price_drift(self) -> float:
        """Get maximum allowed price drift at execution"""
        return self.execution.max_price_drift

    @property
    def min_edge_at_execution(self) -> float:
        """Get minimum edge required at execution"""
        return self.execution.min_edge_at_execution
    
    @property
    def min_edge_percentage(self) -> float:
        """Get minimum edge as percentage (0-100)"""
        return self.strategy.min_edge * 100
    
    @property
    def min_confidence_percentage(self) -> float:
        """Get minimum confidence as percentage (0-100)"""
        return self.strategy.min_confidence * 100
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def validate_for_mode(self, mode: str) -> None:
        """
        Perform strict validation of required keys for a specific execution mode.

        Args:
            mode: 'collect', 'analyze', or 'trade'

        Raises:
            ValueError: If required configuration for the mode is missing
        """
        logger.info(f"🔍 Validating configuration for mode: {mode}")

        if mode == 'collect':
            # Collect mode requires platform keys for enabled platforms
            if self.kalshi_enabled and not self.platforms.kalshi.api_key:
                raise ValueError("KALSHI_API_KEY is required for collect mode when Kalshi is enabled")
            if self.polymarket_enabled and not self.platforms.polymarket.api_key:
                raise ValueError("POLYMARKET_API_KEY is required for collect mode when Polymarket is enabled")

        elif mode == 'analyze':
            # Analyze mode requires LLM keys
            provider = self.analysis_provider
            if provider == 'claude' and not self.api.claude_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required for analyze mode with Claude provider")
            if provider == 'openai' and not self.api.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for analyze mode with OpenAI provider")

        elif mode == 'trade':
            # Trade mode requires everything
            self.validate_for_mode('collect')
            self.validate_for_mode('analyze')

            # Plus private keys if not dry run
            if not self.is_dry_run:
                if self.polymarket_enabled:
                    # Platform safety gate: fail if Polymarket enabled in live
                    raise ValueError(
                        "Polymarket execution is not yet implemented. "
                        "Disable Polymarket or use dry_run=true for live trading."
                    )

                if self.kalshi_enabled:
                    if not self.platforms.kalshi.private_key and not self.platforms.kalshi.private_key_file:
                        raise ValueError("Kalshi private key (env or file) is required for live trading")

    def log_config_summary(self) -> None:
        """Log a summary of the loaded configuration"""
        logger.info("\n" + "=" * 80)
        logger.info("📋 CONFIGURATION SUMMARY")
        logger.info("=" * 80)

        provider = self.analysis_provider
        logger.info("\n🧠 Analysis:")
        logger.info(f"   Provider: {provider}")
        if provider == "openai":
            logger.info(f"   Model: {self.openai.model}")
            logger.info(f"   Temperature: {self.openai.temperature}")
            logger.info(f"   Max tokens: {self.openai.max_tokens}")
            logger.info(
                f"   Pricing: ${self.openai.input_cost_per_mtok}/Mtok (input), "
                f"${self.openai.output_cost_per_mtok}/Mtok (output)"
            )
        else:
            logger.info(f"   Model: {self.claude.model}")
            logger.info(f"   Temperature: {self.claude.temperature}")
            logger.info(f"   Max tokens: {self.claude.max_tokens}")
            logger.info(
                f"   Pricing: ${self.claude.input_cost_per_mtok}/Mtok (input), "
                f"${self.claude.output_cost_per_mtok}/Mtok (output)"
            )
        
        logger.info(f"\n🏢 Platforms:")
        logger.info(f"   Polymarket: {'✅ Enabled' if self.polymarket_enabled else '❌ Disabled'} "
                    f"(max {self.platforms.polymarket.max_markets} markets)")
        logger.info(f"   Kalshi: {'✅ Enabled' if self.kalshi_enabled else '❌ Disabled'} "
                    f"(max {self.platforms.kalshi.max_markets} markets)")
        
        logger.info(f"\n💰 Trading:")
        logger.info(f"   Initial bankroll: ${self.trading.initial_bankroll:,.2f}")
        logger.info(f"   Mode: {'🏁 DRY RUN' if self.is_dry_run else '💸 LIVE TRADING'}")
        logger.info(f"   Enforce live cash check: {'✅' if self.trading.enforce_live_cash_check else '❌'}")
        
        logger.info(f"\n📈 Strategy:")
        logger.info(f"   Min edge: {self.min_edge_percentage:.1f}%")
        logger.info(f"   Min confidence: {self.min_confidence_percentage:.1f}%")
        
        logger.info(f"\n⚠️  Risk Management:")
        logger.info(f"   Max Kelly fraction: {self.risk.max_kelly_fraction:.1%}")
        logger.info(f"   Max positions: {self.risk.max_positions}")
        logger.info(f"   Max position size: ${self.risk.max_position_size:,.2f}")
        logger.info(f"   Max total exposure: {self.risk.max_total_exposure_fraction:.1%}")
        logger.info(f"   Max new exposure/day: {self.risk.max_new_exposure_per_day_fraction:.1%}")
        logger.info(f"   Max orders/cycle: {self.risk.max_orders_per_cycle}")
        logger.info(f"   Max notional/cycle: ${self.risk.max_notional_per_cycle:,.2f}")
        logger.info(f"   Daily loss limit: {self.risk.daily_loss_limit_fraction:.1%}")
        if self.risk.max_trades_per_market_per_day > 0:
            logger.info(f"   Max trades/market/day: {self.risk.max_trades_per_market_per_day}")
        if (
            self.risk.failure_streak_cooldown_threshold > 0
            and self.risk.failure_cooldown_cycles > 0
        ):
            logger.info(
                "   Failure cooldown: "
                f"threshold={self.risk.failure_streak_cooldown_threshold}, "
                f"cooldown_cycles={self.risk.failure_cooldown_cycles}"
            )

        logger.info(f"\n⚙️  Execution:")
        logger.info(f"   Max price drift: {self.execution.max_price_drift:.1%}")
        logger.info(f"   Min edge at execution: {self.execution.min_edge_at_execution:.1%}")
        logger.info(f"   Max submit slippage: {self.execution.max_submit_slippage:.1%}")
        logger.info(
            "   Pending reconcile: "
            f"retries={self.execution.pending_not_found_retries}, "
            f"timeout={self.execution.pending_timeout_minutes}m, "
            f"pages={self.execution.order_reconciliation_max_pages}"
        )
        
        logger.info(f"\n🔍 Filters:")
        logger.info(f"   Min volume: ${self.filters.min_volume:,.2f}")
        logger.info(f"   Min liquidity: ${self.filters.min_liquidity:,.2f}")
        
        logger.info("=" * 80 + "\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for debugging/serialization"""
        return {
            'api': {
                'claude_api_key': self.api.claude_api_key[:20] + '...',  # redact
                'batch_size': self.api.batch_size,
                'api_cost_limit_per_cycle': self.api.api_cost_limit_per_cycle,
            },
            'claude': {
                'model': self.claude.model,
                'temperature': self.claude.temperature,
                'max_tokens': self.claude.max_tokens,
                'input_cost_per_mtok': self.claude.input_cost_per_mtok,
                'output_cost_per_mtok': self.claude.output_cost_per_mtok,
            },
            'platforms': {
                'polymarket': {
                    'enabled': self.platforms.polymarket.enabled,
                    'max_markets': self.platforms.polymarket.max_markets,
                },
                'kalshi': {
                    'enabled': self.platforms.kalshi.enabled,
                    'max_markets': self.platforms.kalshi.max_markets,
                },
            },
            'trading': {
                'initial_bankroll': self.trading.initial_bankroll,
                'dry_run': self.trading.dry_run,
                'enforce_live_cash_check': self.trading.enforce_live_cash_check,
            },
            'strategy': {
                'min_edge': self.strategy.min_edge,
                'min_confidence': self.strategy.min_confidence,
            },
            'risk': {
                'max_kelly_fraction': self.risk.max_kelly_fraction,
                'max_positions': self.risk.max_positions,
                'max_position_size': self.risk.max_position_size,
                'max_total_exposure_fraction': self.risk.max_total_exposure_fraction,
                'max_new_exposure_per_day_fraction': self.risk.max_new_exposure_per_day_fraction,
                'max_orders_per_cycle': self.risk.max_orders_per_cycle,
                'max_notional_per_cycle': self.risk.max_notional_per_cycle,
                'daily_loss_limit_fraction': self.risk.daily_loss_limit_fraction,
                'max_trades_per_market_per_day': self.risk.max_trades_per_market_per_day,
                'failure_streak_cooldown_threshold': self.risk.failure_streak_cooldown_threshold,
                'failure_cooldown_cycles': self.risk.failure_cooldown_cycles,
            },
            'execution': {
                'max_price_drift': self.execution.max_price_drift,
                'min_edge_at_execution': self.execution.min_edge_at_execution,
                'max_submit_slippage': self.execution.max_submit_slippage,
                'pending_not_found_retries': self.execution.pending_not_found_retries,
                'pending_timeout_minutes': self.execution.pending_timeout_minutes,
                'order_reconciliation_max_pages': self.execution.order_reconciliation_max_pages,
                'order_reconciliation_page_limit': self.execution.order_reconciliation_page_limit,
            },
            'filters': {
                'min_volume': self.filters.min_volume,
                'min_liquidity': self.filters.min_liquidity,
            },
            'signal_fusion': {
                'enabled': self.signal_fusion.enabled,
                'mode': self.signal_fusion.mode,
            },
        }
