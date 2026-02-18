import logging
import asyncio
from typing import List, Optional
from pathlib import Path

from app.config import ConfigManager
from app.models import MarketData
from app.api_clients import (
    PolymarketClient,
    PolymarketConfig,
    KalshiClient,
    KalshiConfig,
    APIError,
)
from app.utils import BatchParser

logger = logging.getLogger(__name__)


def _load_private_key_from_config(
    private_key: Optional[str],
    private_key_file: Optional[str],
) -> Optional[str]:
    """Resolve private key from env value or configured file path."""
    if private_key:
        return private_key
    if private_key_file:
        try:
            return Path(private_key_file).expanduser().read_text()
        except OSError as e:
            logger.warning(f"Failed to read private key file '{private_key_file}': {e}")
    return None

class MarketScanner:
    """
    Efficiently scan 1000+ markets across platforms

    Uses modular platform-specific clients:
    - PolymarketClient for offset-based pagination
    - KalshiClient for cursor-based pagination
    """

    def __init__(self, config: ConfigManager):
        """
        Initialize market scanner

        Args:
            config: ConfigManager with platform settings
        """
        self.config = config

        # Initialize platform clients based on config
        self.polymarket_client = (
            PolymarketClient(
                PolymarketConfig(
                    api_key=config.platforms.polymarket.api_key,
                    base_url="https://gamma-api.polymarket.com",
                    max_markets=config.platforms.polymarket.max_markets,
                )
            )
            if config.platforms.polymarket.enabled
            else None
        )

        self.kalshi_client = (
            KalshiClient(
                KalshiConfig(
                    api_key=config.platforms.kalshi.api_key,
                    private_key=_load_private_key_from_config(
                        config.platforms.kalshi.private_key,
                        config.platforms.kalshi.private_key_file,
                    ),
                    private_key_file=config.platforms.kalshi.private_key_file,
                    base_url="https://api.elections.kalshi.com/trade-api/v2",
                    max_markets=config.platforms.kalshi.max_markets,
                    use_orderbooks=not config.is_dry_run,  # Skip orderbooks in dry run to avoid rate limits
                    series_tickers=config.platforms.kalshi.series_tickers,  # Fetch from specific series if configured
                )
            )
            if config.platforms.kalshi.enabled
            else None
        )

    async def scan_all_markets(self) -> List[MarketData]:
        """
        Scan all markets from enabled platforms in parallel

        Returns:
            List of standardized MarketData objects
        """
        logger.info("🔍 Scanning markets across all platforms...")

        tasks = []

        if self.polymarket_client:
            tasks.append(self._fetch_platform_markets(
                'polymarket',
                self.polymarket_client.fetch_markets()
            ))

        if self.kalshi_client:
            tasks.append(self._fetch_platform_markets(
                'kalshi',
                self.kalshi_client.fetch_markets()
            ))

        if not tasks:
            logger.error("No platforms enabled!")
            return []

        # Fetch from all platforms in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors and flatten results
        all_markets = []
        for result in results:
            if isinstance(result, list):
                all_markets.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error fetching markets: {result}")

        logger.info(f"✅ Found {len(all_markets)} total markets")

        return all_markets

    async def _fetch_platform_markets(
        self,
        platform_name: str,
        fetch_coro
    ) -> List[MarketData]:
        """
        Fetch and convert markets from a platform

        Args:
            platform_name: Name of platform (for logging)
            fetch_coro: Coroutine that returns list of market dicts

        Returns:
            List of MarketData objects
        """
        try:
            market_dicts = await fetch_coro

            # Convert dicts to MarketData via BatchParser
            markets = BatchParser.parse_markets_batch(market_dicts)

            logger.info(f"   {platform_name}: {len(markets)} markets")
            return markets

        except APIError as e:
            logger.error(f"API error fetching {platform_name}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching {platform_name} markets: {e}")
            return []
