"""Kalshi API client for fetching prediction market data"""

import logging
from typing import List, Dict, Optional, Any, Tuple

from dataclasses import dataclass

import asyncio

from .base_client import BaseAPIClient, RateLimitError

logger = logging.getLogger(__name__)


# Many Kalshi markets are auto-generated "bundle" style entries with extremely long
# titles (e.g., multi-game extended sports slips). These are typically not useful
# for this bot's market-level analysis and are hard to locate on the website.
EXCLUDED_TICKER_PREFIXES = (
    "KXMVESPORTSMULTIGAMEEXTENDED",
)


@dataclass
class KalshiConfig:
    """Configuration for Kalshi API"""
    api_key: Optional[str] = None
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    max_markets: int = 500
    pagination_limit: int = 100


class KalshiClient(BaseAPIClient):
    """
    Client for Kalshi prediction market API
    
    Handles:
    - Fetching markets with cursor-based pagination
    - Fetching current market prices from orderbook
    - Parsing Kalshi-specific JSON structure
    - Converting to standardized MarketData format
    """
    
    def __init__(self, config: KalshiConfig):
        """
        Initialize Kalshi client
        
        Args:
            config: KalshiConfig with API key and settings
        """
        super().__init__(
            platform_name="kalshi",
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.config = config
    
    async def fetch_markets(self) -> List[Dict[str, Any]]:
        """
        Fetch all open markets from Kalshi
        
        Returns:
            List of standardized market data dicts with structure:
            {
                'platform': 'kalshi',
                'market_id': str (ticker),
                'title': str,
                'description': str,
                'yes_price': float (0-1),
                'no_price': float (0-1),
                'volume': float,
                'liquidity': float (open_interest),
                'end_date': str,
                'category': str
            }
        
        Raises:
            APIError: If API request fails
        """
        
        url = f"{self.base_url}/markets"
        
        def params_builder(offset: Optional[int], cursor: Optional[str]) -> Dict[str, Any]:
            """Build query parameters for cursor-based pagination"""
            params = {
                'limit': self.config.pagination_limit,
                'status': 'open'
            }
            if cursor:
                params['cursor'] = cursor
            return params
        
        def response_parser(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
            """
            Parse Kalshi API response and extract market data
            
            Returns:
                Tuple of (parsed_markets, next_cursor_or_none)
            """
            parsed_markets = []
            excluded = 0
            
            market_list = data.get('markets', [])
            if not market_list:
                return [], None
            
            for market in market_list:
                try:
                    # Kalshi requires fetching price from orderbook separately
                    # This will be done in _parse_market_with_price
                    ticker = market.get('ticker', '') or ''
                    if any(ticker.startswith(pfx) for pfx in EXCLUDED_TICKER_PREFIXES):
                        excluded += 1
                        continue
                    parsed = self._parse_market_metadata(market)
                    if parsed:
                        parsed_markets.append(parsed)
                except Exception as e:
                    logger.debug(f"[kalshi] Error parsing market {market.get('ticker', 'unknown')}: {e}")
                    continue

            if excluded:
                logger.debug(f"[kalshi] Excluded {excluded} bundle markets by ticker prefix")
            
            # Get cursor for next page
            next_cursor = data.get('cursor')
            
            return parsed_markets, next_cursor
        
        headers = self._build_headers()

        async with self:
            markets = await self._get_paginated(
                url=url,
                params_builder=params_builder,
                response_parser=response_parser,
                max_items=self.config.max_markets,
                pagination_type="cursor",
                headers=headers
            )

            # Fetch prices for each market (TODO: optimize with batch price fetching)
            for market in markets:
                try:
                    ticker = market['market_id']
                    yes_price = await self._get_market_price(ticker)
                    market['yes_price'] = yes_price
                    market['no_price'] = 1 - yes_price
                    # Throttle to reduce likelihood of 429s during scans.
                    await asyncio.sleep(0.15)
                except RateLimitError as e:
                    logger.warning(f"[kalshi] Rate limited fetching prices; using default 0.5 for remaining markets ({e})")
                    market['yes_price'] = 0.5
                    market['no_price'] = 0.5
                except Exception as e:
                    logger.warning(f"[kalshi] Failed to fetch price for {market.get('market_id')}: {e}")
                    # Use default prices
                    market['yes_price'] = 0.5
                    market['no_price'] = 0.5
        
        logger.info(f"[kalshi] Successfully fetched {len(markets)} markets with prices")
        return markets
    
    def _parse_market_metadata(self, market_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a single Kalshi market JSON metadata (without price)
        
        Note: Price is fetched separately via orderbook API
        
        Args:
            market_json: Raw market data from Kalshi API
        
        Returns:
            Market dict with placeholder prices (0.5) to be updated later
        """
        try:
            return {
                'platform': 'kalshi',
                'market_id': market_json.get('ticker', ''),
                'title': market_json.get('title', ''),
                'description': market_json.get('subtitle', ''),
                'yes_price': 0.5,  # Will be updated by fetch_markets()
                'no_price': 0.5,
                'volume': float(market_json.get('volume', 0)),
                'liquidity': float(market_json.get('open_interest', 0)),
                'end_date': market_json.get('expiration_time', ''),
                'category': market_json.get('category', 'other')
            }
        
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"[kalshi] Error parsing market metadata: {e}")
            return None
    
    async def _get_market_price(self, ticker: str) -> float:
        """
        Fetch current YES price for a Kalshi market from its orderbook
        
        Args:
            ticker: Market ticker symbol
        
        Returns:
            YES price as float (0-100 in cents from API, converted to 0-1)
        
        Raises:
            APIError: If orderbook fetch fails
        """
        
        url = f"{self.base_url}/markets/{ticker}/orderbook"
        headers = self._build_headers()
        
        async def fetch_orderbook():
            if not self.session:
                raise RuntimeError("Session not initialized")
            async with self.session.get(url, headers=headers) as response:
                await self._handle_response_status(response)
                return await response.json()
        
        try:
            data = await self._call_with_retry(
                fetch_orderbook,
                f"Fetch orderbook for {ticker}",
                max_retries=0
            )
            
            orderbook = data.get('orderbook', {})
            yes_bids = orderbook.get('yes', [])
            
            if yes_bids:
                # Best bid price in cents (e.g., 65 = 65 cents = 0.65)
                price_cents = yes_bids[0][0]
                return price_cents / 100
            
            # No bids, return midpoint
            logger.debug(f"[kalshi] No YES bids for {ticker}, using default 0.5")
            return 0.5
        
        except RateLimitError:
            # Bubble up so caller can decide how to handle (default to 0.5 and continue).
            raise
        except Exception as e:
            logger.warning(f"[kalshi] Error fetching price for {ticker}: {e}")
            return 0.5
