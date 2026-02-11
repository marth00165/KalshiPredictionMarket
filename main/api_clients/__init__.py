"""API client module for prediction market platforms"""

from .base_client import (
    BaseAPIClient,
    APIError,
    RateLimitError,
    AuthenticationError,
    ServerError,
    ClientError,
)
from .polymarket_client import PolymarketClient, PolymarketConfig
from .kalshi_client import KalshiClient, KalshiConfig

__all__ = [
    'BaseAPIClient',
    'APIError',
    'RateLimitError',
    'AuthenticationError',
    'ServerError',
    'ClientError',
    'PolymarketClient',
    'PolymarketConfig',
    'KalshiClient',
    'KalshiConfig',
]
