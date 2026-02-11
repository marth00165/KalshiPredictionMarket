"""Configuration management for trading bot with validation and typed access"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class APIConfig:
    """Configuration for API services (Claude, etc.)"""
    
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
    private_key_file: Optional[str] = None
    enabled: bool = True
    max_markets: int = 25
    
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
    
    def validate(self) -> None:
        """Validate risk configuration"""
        if not (0 <= self.max_kelly_fraction <= 1):
            raise ValueError(f"max_kelly_fraction must be between 0 and 1, got {self.max_kelly_fraction}")
        if self.max_positions < 1:
            raise ValueError(f"max_positions must be >= 1, got {self.max_positions}")
        if self.max_position_size <= 0:
            raise ValueError(f"max_position_size must be > 0, got {self.max_position_size}")


@dataclass
class FiltersConfig:
    """Configuration for market filtering"""
    
    min_volume: float = 1000  # Minimum market volume
    min_liquidity: float = 500  # Minimum liquidity
    
    def validate(self) -> None:
        """Validate filters configuration"""
        if self.min_volume < 0:
            raise ValueError(f"min_volume must be >= 0, got {self.min_volume}")
        if self.min_liquidity < 0:
            raise ValueError(f"min_liquidity must be >= 0, got {self.min_liquidity}")


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
        self.config_path = Path(config_file)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # Load JSON
        with open(self.config_path) as f:
            raw_config = json.load(f)
        
        # Parse into typed configs
        self._parse_config(raw_config)
    
    def _parse_config(self, raw_config: Dict[str, Any]) -> None:
        """Parse raw JSON config into typed dataclasses"""

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
                claude_api_key=api_raw.get('claude_api_key'),
                openai_api_key=api_raw.get('openai_api_key'),
                batch_size=api_raw.get('batch_size', 50),
                api_cost_limit_per_cycle=api_raw.get('api_cost_limit_per_cycle', 5.0),
            )
            self.api.validate()
        except ValueError as e:
            raise ValueError(f"Invalid API config: {e}")

        # Provider-specific key validation
        if self.analysis.provider == 'claude':
            if not self.api.claude_api_key or str(self.api.claude_api_key).startswith('sk-ant-YOUR_'):
                raise ValueError(
                    "Invalid claude_api_key: Please set your actual Claude API key in config"
                )
        elif self.analysis.provider == 'openai':
            if not self.api.openai_api_key or str(self.api.openai_api_key).startswith('sk-proj-YOUR_'):
                raise ValueError(
                    "Invalid openai_api_key: Please set your actual OpenAI API key in config"
                )
        
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
                api_key=polymarket_raw.get('api_key'),
                private_key_file=polymarket_raw.get('private_key_file'),
                enabled=polymarket_raw.get('enabled', True),
                max_markets=polymarket_raw.get('max_markets', 500),
            ),
            kalshi=PlatformConfig(
                api_key=kalshi_raw.get('api_key'),
                private_key_file=kalshi_raw.get('private_key_file'),
                enabled=kalshi_raw.get('enabled', True),
                max_markets=kalshi_raw.get('max_markets', 500),
            ),
        )
        try:
            self.platforms.validate()
        except ValueError as e:
            raise ValueError(f"Invalid platforms config: {e}")
        
        # Trading configuration
        trading_raw = raw_config.get('trading', {})
        self.trading = TradingConfig(
            initial_bankroll=trading_raw.get('initial_bankroll', 10000),
            dry_run=trading_raw.get('dry_run', True),
        )
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
        )
        try:
            self.risk.validate()
        except ValueError as e:
            raise ValueError(f"Invalid risk config: {e}")
        
        # Filters configuration
        filters_raw = raw_config.get('filters', {})
        self.filters = FiltersConfig(
            min_volume=filters_raw.get('min_volume', 1000),
            min_liquidity=filters_raw.get('min_liquidity', 500),
        )
        try:
            self.filters.validate()
        except ValueError as e:
            raise ValueError(f"Invalid filters config: {e}")
        
        logger.info(f"✅ Configuration loaded and validated from {self.config_path}")
    
    # ========================================================================
    # CONVENIENCE PROPERTIES (reduce boilerplate)
    # ========================================================================
    
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
    
    def log_config_summary(self) -> None:
        """Log a summary of the loaded configuration"""
        logger.info("\n" + "=" * 80)
        logger.info("📋 CONFIGURATION SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"\n🤖 Claude AI:")
        logger.info(f"   Model: {self.claude.model}")
        logger.info(f"   Temperature: {self.claude.temperature}")
        logger.info(f"   Max tokens: {self.claude.max_tokens}")
        logger.info(f"   Pricing: ${self.claude.input_cost_per_mtok}/Mtok (input), "
                    f"${self.claude.output_cost_per_mtok}/Mtok (output)")
        
        logger.info(f"\n🏢 Platforms:")
        logger.info(f"   Polymarket: {'✅ Enabled' if self.polymarket_enabled else '❌ Disabled'} "
                    f"(max {self.platforms.polymarket.max_markets} markets)")
        logger.info(f"   Kalshi: {'✅ Enabled' if self.kalshi_enabled else '❌ Disabled'} "
                    f"(max {self.platforms.kalshi.max_markets} markets)")
        
        logger.info(f"\n💰 Trading:")
        logger.info(f"   Initial bankroll: ${self.trading.initial_bankroll:,.2f}")
        logger.info(f"   Mode: {'🏁 DRY RUN' if self.is_dry_run else '💸 LIVE TRADING'}")
        
        logger.info(f"\n📈 Strategy:")
        logger.info(f"   Min edge: {self.min_edge_percentage:.1f}%")
        logger.info(f"   Min confidence: {self.min_confidence_percentage:.1f}%")
        
        logger.info(f"\n⚠️  Risk Management:")
        logger.info(f"   Max Kelly fraction: {self.risk.max_kelly_fraction:.1%}")
        logger.info(f"   Max positions: {self.risk.max_positions}")
        logger.info(f"   Max position size: ${self.risk.max_position_size:,.2f}")
        
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
            },
            'strategy': {
                'min_edge': self.strategy.min_edge,
                'min_confidence': self.strategy.min_confidence,
            },
            'risk': {
                'max_kelly_fraction': self.risk.max_kelly_fraction,
                'max_positions': self.risk.max_positions,
                'max_position_size': self.risk.max_position_size,
            },
            'filters': {
                'min_volume': self.filters.min_volume,
                'min_liquidity': self.filters.min_liquidity,
            },
        }
