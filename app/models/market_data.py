"""Market data model representing a prediction market"""

from dataclasses import dataclass


@dataclass
class MarketData:
    """
    Standardized market data from any prediction market platform
    
    This dataclass unifies data from different platforms (Polymarket, Kalshi, etc.)
    into a single format for analysis and trading.
    """
    
    platform: str  # 'polymarket', 'kalshi', etc.
    market_id: str  # Platform-specific unique identifier (ID or ticker)
    title: str  # Market question/title
    description: str  # Additional market details
    yes_price: float  # Current price of YES outcome (0.0-1.0)
    no_price: float  # Current price of NO outcome (0.0-1.0)
    volume: float  # Total trading volume (in dollars or equivalent)
    liquidity: float  # Available liquidity (varies by platform)
    end_date: str  # Market expiration/resolution date (ISO format)
    category: str  # Market category (politics, sports, finance, etc.)
    event_ticker: str = ""  # Parent event ticker (for grouping related markets)
    yes_option: str = ""  # Specific option name (e.g., "Pam Bondi" for cabinet markets)
    no_option: str = ""  # Specific no option name
    
    def __post_init__(self):
        """Validate market data after initialization"""
        if not 0 <= self.yes_price <= 1:
            raise ValueError(f"yes_price must be between 0 and 1, got {self.yes_price}")
        if not 0 <= self.no_price <= 1:
            raise ValueError(f"no_price must be between 0 and 1, got {self.no_price}")
        
        # Prices should sum to approximately 1 (allowing for spreads)
        price_sum = self.yes_price + self.no_price
        if not (0.9 <= price_sum <= 1.1):
            raise ValueError(
                f"yes_price + no_price should sum to ~1.0, got {price_sum} "
                f"(yes={self.yes_price}, no={self.no_price})"
            )
    
    def __str__(self) -> str:
        """Human-readable representation"""
        return f"{self.title[:50]}... [{self.platform}] YES:{self.yes_price:.1%} NO:{self.no_price:.1%}"
    
    def get_implied_probability(self, outcome: str = 'yes') -> float:
        """
        Get implied probability from current market price
        
        Args:
            outcome: 'yes' or 'no'
        
        Returns:
            Implied probability (0.0-1.0)
        """
        if outcome.lower() == 'yes':
            return self.yes_price
        elif outcome.lower() == 'no':
            return self.no_price
        else:
            raise ValueError(f"outcome must be 'yes' or 'no', got {outcome}")
