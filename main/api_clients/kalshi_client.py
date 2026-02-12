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
    async def fetch_markets_by_series(
        self,
        series_tickers: List[str],
        status: str = "open",
    ) -> List[Dict[str, Any]]:
        """
        Fetch markets by series ticker(s) — lightweight, no orderbook calls.
        
        This method is optimized for dry runs and data collection:
        - Uses series_ticker filter to fetch only relevant markets
        - Uses yes_price/no_price from market list (last trade price)
        - No individual orderbook API calls = no rate limit issues
        
        Args:
            series_tickers: List of series tickers (e.g., ['KXFED', 'KXCPI'])
            status: Market status filter ('open', 'closed', 'settled')
        
        Returns:
            List of standardized market data dicts
        """
        all_markets = []
        headers = self._build_headers()
        
        async with self:
            for series_ticker in series_tickers:
                cursor = None
                series_markets = []
                
                while True:
                    # Build URL with series filter
                    url = f"{self.base_url}/markets"
                    params = {
                        'series_ticker': series_ticker,
                        'limit': self.config.pagination_limit,
                        'status': status,
                    }
                    if cursor:
                        params['cursor'] = cursor
                    
                    try:
                        async def fetch_page():
                            if not self.session:
                                raise RuntimeError("Session not initialized")
                            async with self.session.get(url, headers=headers, params=params) as response:
                                await self._handle_response_status(response)
                                return await response.json()
                        
                        data = await self._call_with_retry(
                            fetch_page,
                            f"Fetch markets for series {series_ticker}",
                            max_retries=2
                        )
                        
                        market_list = data.get('markets', [])
                        
                        for market in market_list:
                            parsed = self._parse_market_with_price(market)
                            if parsed:
                                series_markets.append(parsed)
                        
                        # Check for next page
                        cursor = data.get('cursor')
                        if not cursor:
                            break
                        
                        logger.debug(f"[kalshi] Fetched {len(market_list)} markets for {series_ticker}, continuing...")
                        
                    except Exception as e:
                        logger.warning(f"[kalshi] Error fetching series {series_ticker}: {e}")
                        break
                
                logger.info(f"[kalshi] Fetched {len(series_markets)} markets for series {series_ticker}")
                all_markets.extend(series_markets)
        
        logger.info(f"[kalshi] Total: {len(all_markets)} markets from {len(series_tickers)} series")
        return all_markets

    def _parse_market_with_price(self, market_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a Kalshi market JSON including price from the response.
        
        Unlike _parse_market_metadata, this uses yes_price/no_price directly
        from the market list response (last trade price), avoiding orderbook calls.
        
        Args:
            market_json: Raw market data from Kalshi API
        
        Returns:
            Standardized market dict with prices
        """
        try:
            ticker = market_json.get('ticker', '')
            
            # Skip excluded tickers
            if any(ticker.startswith(pfx) for pfx in EXCLUDED_TICKER_PREFIXES):
                return None
            
            # Get price from response (in cents, 0-100)
            # yes_price is the last trade price, or we can use yes_bid/yes_ask
            yes_price_cents = market_json.get('yes_price') or market_json.get('last_price') or 50
            yes_price = yes_price_cents / 100 if yes_price_cents > 1 else yes_price_cents
            
            # Clamp to valid range
            yes_price = max(0.01, min(0.99, yes_price))
            
            return {
                'platform': 'kalshi',
                'market_id': ticker,
                'title': market_json.get('title', ''),
                'description': market_json.get('subtitle', ''),
                'yes_price': yes_price,
                'no_price': 1 - yes_price,
                'volume': float(market_json.get('volume', 0)),
                'liquidity': float(market_json.get('open_interest', 0)),
                'end_date': market_json.get('expiration_time', ''),
                'category': market_json.get('category', 'other'),
                'series_ticker': market_json.get('series_ticker', ''),
                # Additional fields useful for analysis
                'yes_bid': market_json.get('yes_bid', 0) / 100 if market_json.get('yes_bid') else None,
                'yes_ask': market_json.get('yes_ask', 0) / 100 if market_json.get('yes_ask') else None,
                'no_bid': market_json.get('no_bid', 0) / 100 if market_json.get('no_bid') else None,
                'no_ask': market_json.get('no_ask', 0) / 100 if market_json.get('no_ask') else None,
            }
        
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"[kalshi] Error parsing market with price: {e}")
            return None

    async def discover_series(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Discover available series tickers on Kalshi.
        
        Useful for finding the right series_ticker values for macro markets.
        
        Args:
            category: Optional category filter (e.g., 'Economics', 'Politics')
        
        Returns:
            List of dicts with series info: {ticker, title, category}
        """
        url = f"{self.base_url}/series"
        headers = self._build_headers()
        series_list = []
        cursor = None
        
        async with self:
            while True:
                params = {'limit': 100}
                if cursor:
                    params['cursor'] = cursor
                
                try:
                    async def fetch_series():
                        if not self.session:
                            raise RuntimeError("Session not initialized")
                        async with self.session.get(url, headers=headers, params=params) as response:
                            await self._handle_response_status(response)
                            return await response.json()
                    
                    data = await self._call_with_retry(fetch_series, "Fetch series", max_retries=2)
                    
                    for series in data.get('series', []):
                        series_info = {
                            'ticker': series.get('ticker', ''),
                            'title': series.get('title', ''),
                            'category': series.get('category', ''),
                        }
                        # Filter by category if specified
                        if category is None or series_info['category'].lower() == category.lower():
                            series_list.append(series_info)
                    
                    cursor = data.get('cursor')
                    if not cursor:
                        break
                        
                except Exception as e:
                    logger.warning(f"[kalshi] Error fetching series: {e}")
                    break
        
        logger.info(f"[kalshi] Discovered {len(series_list)} series" + (f" in category '{category}'" if category else ""))
        return series_list