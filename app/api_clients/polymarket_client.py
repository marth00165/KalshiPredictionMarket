"""Polymarket API client for fetching prediction market data"""

import logging
from typing import List, Dict, Optional, Any, Tuple

from dataclasses import dataclass

from .base_client import BaseAPIClient

logger = logging.getLogger(__name__)


@dataclass
class PolymarketConfig:
    """Configuration for Polymarket API"""
    api_key: Optional[str] = None
    base_url: str = "https://gamma-api.polymarket.com"
    max_markets: int = 500
    pagination_limit: int = 100


class PolymarketClient(BaseAPIClient):
    """
    Client for Polymarket prediction market API
    
    Handles:
    - Fetching markets with pagination
    - Parsing Polymarket-specific JSON structure
    - Converting to standardized MarketData format
    """
    
    def __init__(self, config: PolymarketConfig):
        """
        Initialize Polymarket client
        
        Args:
            config: PolymarketConfig with API key and settings
        """
        super().__init__(
            platform_name="polymarket",
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.config = config
    
    async def fetch_markets(self) -> List[Dict[str, Any]]:
        """
        Fetch all active markets from Polymarket
        
        Returns:
            List of standardized market data dicts with structure:
            {
                'platform': 'polymarket',
                'market_id': str,
                'title': str,
                'description': str,
                'yes_price': float (0-1),
                'no_price': float (0-1),
                'volume': float,
                'liquidity': float,
                'end_date': str,
                'category': str
            }
        
        Raises:
            APIError: If API request fails
        """
        
        url = f"{self.base_url}/markets"
        
        def params_builder(offset: int, cursor: Optional[str]) -> Dict[str, Any]:
            """Build query parameters for pagination"""
            return {
                'limit': self.config.pagination_limit,
                'offset': offset,
                'active': 'true',
                'closed': 'false'
            }
        
        def response_parser(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[int]]:
            """
            Parse Polymarket API response and extract market data
            
            Returns:
                Tuple of (parsed_markets, next_offset_or_none)
            """
            parsed_markets = []
            
            # Polymarket returns array directly
            if not isinstance(data, list):
                logger.warning(f"[polymarket] Unexpected response format: {type(data)}")
                return [], None
            
            for market in data:
                try:
                    parsed = self._parse_market(market)
                    if parsed:
                        parsed_markets.append(parsed)
                except Exception as e:
                    logger.debug(f"[polymarket] Error parsing market {market.get('id', 'unknown')}: {e}")
                    continue
            
            # Determine if there are more results
            # If we got fewer items than requested, we've reached the end
            has_more = len(data) == self.config.pagination_limit
            next_offset = None
            if has_more:
                # Get current offset from implicit position (we'll track this in _get_paginated)
                # For now, return len(data) as the number to add to offset
                next_offset = len(data)
            
            return parsed_markets, next_offset
        
        headers = self._build_headers()

        async with self:
            markets = await self._get_paginated(
                url=url,
                params_builder=params_builder,
                response_parser=response_parser,
                max_items=self.config.max_markets,
                pagination_type="offset",
                headers=headers
            )
        
        logger.info(f"[polymarket] Successfully fetched {len(markets)} markets")
        return markets
    
    def _parse_market(self, market_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a single Polymarket market JSON into standardized format
        
        Args:
            market_json: Raw market data from Polymarket API
        
        Returns:
            Standardized market dict or None if parsing fails
        """
        try:
            # Extract outcomes/prices
            outcomes = market_json.get('outcomes', [])
            if len(outcomes) < 2:
                logger.debug(f"[polymarket] Market {market_json.get('id')} has fewer than 2 outcomes")
                return None
            
            yes_price = float(outcomes[0].get('price', 0))
            no_price = float(outcomes[1].get('price', 0))
            
            # Validate price ranges
            if not (0 <= yes_price <= 1 and 0 <= no_price <= 1):
                logger.debug(f"[polymarket] Invalid prices: yes={yes_price}, no={no_price}")
                return None
            
            return {
                'platform': 'polymarket',
                'market_id': market_json.get('id', ''),
                'title': market_json.get('question', ''),
                'description': market_json.get('description', ''),
                'yes_price': yes_price,
                'no_price': no_price,
                'volume': float(market_json.get('volume', 0)),
                'liquidity': float(market_json.get('liquidity', 0)),
                'end_date': market_json.get('end_date_iso', ''),
                'category': market_json.get('category', 'other')
            }
        
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"[polymarket] Error parsing market: {e}")
            return None
