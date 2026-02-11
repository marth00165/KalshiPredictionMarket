"""Response parsing utilities for converting API responses to models"""

import json
import logging
from typing import Dict, Any, Optional

from models import MarketData, FairValueEstimate

logger = logging.getLogger(__name__)


# ============================================================================
# CLAUDE RESPONSE PARSING
# ============================================================================

class ClaudeResponseParser:
    """Parser for Claude API responses containing market analysis"""
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from Claude's response text
        
        Claude sometimes wraps JSON in markdown code blocks like:
        ```json
        {"key": "value"}
        ```
        
        This method handles both bare JSON and markdown-wrapped JSON.
        
        Args:
            text: Raw text response from Claude API
        
        Returns:
            Parsed JSON dict or None if parsing fails
        """
        if not text:
            logger.warning("Empty Claude response text")
            return None
        
        # Try markdown JSON blocks first
        if '```json' in text:
            try:
                # Extract content between ```json and ```
                json_text = text.split('```json')[1].split('```')[0].strip()
                return json.loads(json_text)
            except (IndexError, json.JSONDecodeError) as e:
                logger.debug(f"Failed to parse markdown JSON block: {e}")
        
        # Try generic markdown code blocks
        if '```' in text:
            try:
                json_text = text.split('```')[1].split('```')[0].strip()
                return json.loads(json_text)
            except (IndexError, json.JSONDecodeError) as e:
                logger.debug(f"Failed to parse generic code block: {e}")
        
        # Try bare JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Claude response as JSON: {e}\nText: {text[:200]}")
            return None
    
    @staticmethod
    def parse_fair_value_estimate(
        response_text: str,
        market_id: str,
        market_price: float
    ) -> Optional[FairValueEstimate]:
        """
        Parse Claude's market analysis into a FairValueEstimate
        
        Expected JSON structure from Claude:
        {
            "probability": <float 0-100>,
            "confidence": <float 0-100>,
            "reasoning": "<detailed explanation>",
            "key_factors": ["factor1", "factor2", ...],
            "data_sources": ["source1", "source2", ...]
        }
        
        Args:
            response_text: Raw text response from Claude API
            market_id: ID of the market being analyzed
            market_price: Current market price for YES outcome (0-1)
        
        Returns:
            FairValueEstimate object or None if parsing fails
        """
        
        # Extract JSON from Claude's response
        data = ClaudeResponseParser.extract_json_from_text(response_text)
        if not data:
            logger.error("Could not extract JSON from Claude response")
            return None
        
        try:
            # Claude returns probability as 0-100, convert to 0-1
            probability = data.get('probability', 0)
            if probability > 1:
                probability = probability / 100
            
            confidence = data.get('confidence', 0)
            if confidence > 1:
                confidence = confidence / 100
            
            # Calculate edge
            edge = probability - market_price
            
            # Create estimate object
            estimate = FairValueEstimate(
                market_id=market_id,
                estimated_probability=probability,
                confidence_level=confidence,
                reasoning=data.get('reasoning', ''),
                data_sources=data.get('data_sources', []),
                edge=edge
            )
            
            logger.debug(f"Parsed fair value estimate for {market_id}: {estimate}")
            return estimate
        
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing fair value estimate: {e}\nData: {data}")
            return None


# ============================================================================
# API RESPONSE PARSING
# ============================================================================

class MarketDataParser:
    """Convert raw API responses to standardized MarketData"""
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Optional[MarketData]:
        """
        Convert a dictionary to MarketData
        
        Assumes dictionary already has all required fields with correct types.
        This is used by platform-specific clients after they parse their API responses.
        
        Args:
            data: Dictionary with market data (from api_clients)
        
        Returns:
            MarketData object or None if validation fails
        """
        try:
            market = MarketData(
                platform=data.get('platform', ''),
                market_id=data.get('market_id', ''),
                title=data.get('title', ''),
                description=data.get('description', ''),
                yes_price=float(data.get('yes_price', 0)),
                no_price=float(data.get('no_price', 0)),
                volume=float(data.get('volume', 0)),
                liquidity=float(data.get('liquidity', 0)),
                end_date=data.get('end_date', ''),
                category=data.get('category', 'other'),
            )
            return market
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error creating MarketData from dict: {e}\nData: {data}")
            return None


# ============================================================================
# BATCH PARSING
# ============================================================================

class BatchParser:
    """Parse multiple responses in batch"""
    
    @staticmethod
    def parse_markets_batch(
        market_dicts: list,
        parser_func=None
    ) -> list:
        """
        Parse a batch of market dictionaries
        
        Args:
            market_dicts: List of market data dictionaries
            parser_func: Optional custom parser function (defaults to MarketDataParser.from_dict)
        
        Returns:
            List of parsed MarketData objects (failed parses excluded)
        """
        if parser_func is None:
            parser_func = MarketDataParser.from_dict
        
        parsed = []
        failed = 0
        
        for market_dict in market_dicts:
            try:
                market = parser_func(market_dict)
                if market:
                    parsed.append(market)
                else:
                    failed += 1
            except Exception as e:
                logger.debug(f"Error parsing market: {e}")
                failed += 1
        
        if failed > 0:
            logger.warning(f"Failed to parse {failed}/{len(market_dicts)} markets")
        
        return parsed
    
    @staticmethod
    def parse_estimates_batch(
        response_texts: dict,  # {market_id: response_text}
        market_prices: dict,   # {market_id: price}
    ) -> list:
        """
        Parse a batch of Claude analysis responses
        
        Args:
            response_texts: Dictionary mapping market_id -> Claude response text
            market_prices: Dictionary mapping market_id -> current market price
        
        Returns:
            List of parsed FairValueEstimate objects (failed parses excluded)
        """
        estimates = []
        failed = 0
        
        for market_id, response_text in response_texts.items():
            try:
                price = market_prices.get(market_id, 0.5)
                estimate = ClaudeResponseParser.parse_fair_value_estimate(
                    response_text, market_id, price
                )
                if estimate:
                    estimates.append(estimate)
                else:
                    failed += 1
            except Exception as e:
                logger.debug(f"Error parsing estimate for {market_id}: {e}")
                failed += 1
        
        if failed > 0:
            logger.warning(f"Failed to parse {failed}/{len(response_texts)} estimates")
        
        return estimates
