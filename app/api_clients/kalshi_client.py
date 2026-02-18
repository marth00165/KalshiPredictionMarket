"""Kalshi API client for fetching prediction market data"""

import logging
import time
from typing import List, Dict, Optional, Any, Tuple

from dataclasses import dataclass

import asyncio

from .base_client import BaseAPIClient, RateLimitError

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter to stay under API limits"""
    
    def __init__(self, max_per_sec: float = 10):
        """
        Initialize rate limiter.
        
        Args:
            max_per_sec: Maximum requests per second (Kalshi allows ~10/sec, use 8 for safety)
        """
        self.delay = 1.0 / max_per_sec
        self.last = 0.0
    
    def wait(self):
        """Wait if needed to stay under rate limit"""
        now = time.time()
        elapsed = now - self.last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last = time.time()
    
    async def async_wait(self):
        """Async version of wait"""
        now = time.time()
        elapsed = now - self.last
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self.last = time.time()


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
    pagination_limit: int = 200  # Kalshi allows up to 1000
    use_orderbooks: bool = True  # Set False for faster scanning (no rate limits)
    series_tickers: Optional[List[str]] = None  # Fetch only from these series (e.g., ['KXNEWPOPE', 'KXG7LEADEROUT'])


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
        self.rate_limiter = RateLimiter(max_per_sec=8)  # Stay safely under Kalshi's 10/sec limit
        logger.info(f"[kalshi] KalshiClient initialized: use_orderbooks={config.use_orderbooks}, max_markets={config.max_markets}")
    
    async def fetch_markets(self) -> List[Dict[str, Any]]:
        """
        Fetch all open markets from Kalshi
        
        If series_tickers is configured, fetches only from those series.
        Otherwise fetches all open markets.
        
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
        
        # If series_tickers configured, use fetch_markets_by_series for targeted fetch
        if self.config.series_tickers:
            logger.info(f"[kalshi] Fetching markets from series: {self.config.series_tickers}")
            return await self.fetch_markets_by_series(self.config.series_tickers)
        
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

            # Fetch prices for each market if orderbooks enabled
            if self.config.use_orderbooks:
                logger.info(f"[kalshi] Fetching orderbooks for {len(markets)} markets...")
                for market in markets:
                    try:
                        await self.rate_limiter.async_wait()  # Rate limit orderbook calls
                        ticker = market['market_id']
                        yes_price = await self._get_market_price(ticker)
                        market['yes_price'] = yes_price
                        market['no_price'] = 1 - yes_price
                    except RateLimitError as e:
                        logger.warning(f"[kalshi] Rate limited fetching prices; using list price for remaining markets ({e})")
                        # Keep the price from market list (set in _parse_market_metadata)
                        break
                    except Exception as e:
                        logger.warning(f"[kalshi] Failed to fetch price for {market.get('market_id')}: {e}")
                        # Keep the price from market list
            else:
                logger.info(f"[kalshi] Skipping orderbooks (use_orderbooks=False), using list prices")
        
        logger.info(f"[kalshi] Successfully fetched {len(markets)} markets")
        return markets
    
    def _parse_market_metadata(self, market_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a single Kalshi market JSON metadata with list price.
        
        Uses yes_price from market list response (last trade price).
        If use_orderbooks=True, this will be overwritten by orderbook data later.
        
        Args:
            market_json: Raw market data from Kalshi API
        
        Returns:
            Market dict with prices from list response
        """
        try:
            # Get price from response - prefer mid of bid/ask, fallback to last_price
            yes_bid = market_json.get('yes_bid') or 0
            yes_ask = market_json.get('yes_ask') or 0
            last_price = market_json.get('last_price') or 0
            
            # Calculate mid-price if both bid and ask exist
            if yes_bid > 0 and yes_ask > 0:
                yes_price_cents = (yes_bid + yes_ask) / 2
            elif yes_ask > 0:
                yes_price_cents = yes_ask
            elif yes_bid > 0:
                yes_price_cents = yes_bid
            elif last_price > 0:
                yes_price_cents = last_price
            else:
                yes_price_cents = 50  # No price data available
            
            yes_price = yes_price_cents / 100 if yes_price_cents > 1 else yes_price_cents
            yes_price = max(0.01, min(0.99, yes_price))  # Clamp to valid range
            
            # Build a clear title that includes the specific option
            base_title = market_json.get('title', '')
            yes_option = market_json.get('yes_sub_title', '')
            
            # If yes_sub_title exists and isn't already in title, append it
            if yes_option and yes_option.lower() not in base_title.lower():
                display_title = f"{base_title} [{yes_option}]"
            else:
                display_title = base_title
            
            return {
                'platform': 'kalshi',
                'market_id': market_json.get('ticker', ''),
                'title': display_title,
                'description': market_json.get('subtitle', '') or yes_option,
                'yes_option': yes_option,
                'no_option': market_json.get('no_sub_title', ''),
                'event_ticker': market_json.get('event_ticker', ''),
                'yes_price': yes_price,
                'no_price': 1 - yes_price,
                'volume': float(market_json.get('volume', 0)),
                'liquidity': float(market_json.get('open_interest', 0)),
                'end_date': market_json.get('expiration_time', ''),
                'category': market_json.get('category', 'other'),
                'series_ticker': market_json.get('series_ticker', '')
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

    async def get_market_yes_price(self, ticker: str) -> float:
        """
        Public helper to fetch current YES price for a market ticker.

        Returns a normalized price in [0.0, 1.0].
        """
        async with self:
            return await self._get_market_price(ticker)

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
                await self.rate_limiter.async_wait()  # Rate limit between series fetches
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

    async def get_orders(self, ticker: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch orders from Kalshi portfolio.
        """
        url = f"{self.base_url}/portfolio/orders"
        headers = self._build_headers()
        params = {}
        if ticker:
            params['ticker'] = ticker
        if status:
            params['status'] = status

        async def fetch_orders():
            if not self.session:
                raise RuntimeError("Session not initialized")
            async with self.session.get(url, headers=headers, params=params) as response:
                await self._handle_response_status(response)
                return await response.json()

        try:
            async with self:
                data = await self._call_with_retry(fetch_orders, "Fetch portfolio orders")
            return data.get('orders', [])
        except Exception as e:
            logger.error(f"[kalshi] Error fetching orders: {e}")
            return []

    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Fetch current open positions from Kalshi portfolio.

        Returns:
            List of position dicts:
            {
                'market_id': str (ticker),
                'side': str ('yes' or 'no'),
                'quantity': float,
                'entry_price': float (0-1),
                'platform': 'kalshi'
            }
        """
        url = f"{self.base_url}/portfolio/positions"
        headers = self._build_headers()

        async def fetch_portfolio():
            if not self.session:
                raise RuntimeError("Session not initialized")
            async with self.session.get(url, headers=headers) as response:
                await self._handle_response_status(response)
                return await response.json()

        try:
            async with self:
                data = await self._call_with_retry(fetch_portfolio, "Fetch portfolio positions")

            positions = []
            for pos in data.get('positions', []):
                # Standardize Kalshi position format
                positions.append({
                    'market_id': pos.get('ticker'),
                    'side': pos.get('side', 'yes').lower(),
                    'quantity': float(pos.get('position', 0)),
                    'entry_price': float(pos.get('avg_cost_price', 0)) / 100,
                    'platform': 'kalshi'
                })
            return positions
        except Exception as e:
            logger.error(f"[kalshi] Error fetching positions: {e}")
            return []

    async def place_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        price_cents: int,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Place a limit order on Kalshi.

        Args:
            ticker: Market ticker
            side: 'yes' or 'no'
            action: 'buy' or 'sell'
            count: Number of contracts
            price_cents: Price in cents (1-99)
            client_order_id: Optional UUID for the order

        Returns:
            Dict containing order_id and status
        """
        url = f"{self.base_url}/portfolio/orders"
        headers = self._build_headers()

        payload = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": "limit",
            "limit_price": price_cents
        }
        if client_order_id:
            payload["client_order_id"] = client_order_id

        async def execute_order():
            if not self.session:
                raise RuntimeError("Session not initialized")
            async with self.session.post(url, headers=headers, json=payload) as response:
                await self._handle_response_status(response)
                return await response.json()

        try:
            async with self:
                data = await self._call_with_retry(execute_order, f"Place {action} {side} order on {ticker}")

            return {
                "order_id": data.get("order_id"),
                "status": data.get("status", "unknown"),
                "raw_response": data
            }
        except Exception as e:
            logger.error(f"[kalshi] Order placement failed for {ticker}: {e}")
            raise

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
            
            base_title = market_json.get('title', '')
            yes_option = market_json.get('yes_sub_title', '')
            if yes_option and yes_option.lower() not in str(base_title).lower():
                display_title = f"{base_title} [{yes_option}]"
            else:
                display_title = base_title

            return {
                'platform': 'kalshi',
                'market_id': ticker,
                'title': display_title,
                'description': market_json.get('subtitle', '') or yes_option,
                'yes_option': yes_option,
                'no_option': market_json.get('no_sub_title', ''),
                'event_ticker': market_json.get('event_ticker', '') or market_json.get('series_ticker', ''),
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
