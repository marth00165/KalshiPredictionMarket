"""Fair value estimate model from Claude AI analysis"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class FairValueEstimate:
    """
    Claude AI's fair value estimate for a prediction market
    
    This represents the result of running a market through Claude's analysis
    to get a probability estimate, confidence level, and reasoning.
    """
    
    market_id: str  # Link back to the market being analyzed
    estimated_probability: float  # Claude's estimated true probability (0.0-1.0)
    confidence_level: float  # Confidence in the estimate (0.0-1.0)
    reasoning: str  # Detailed explanation of the analysis
    data_sources: List[str] = field(default_factory=list)  # Sources used for analysis
    key_factors: List[str] = field(default_factory=list)  # Key factors in the analysis
    edge: float = 0.0  # Estimated edge vs. market price (calculated field)
    
    def __post_init__(self):
        """Validate estimate data after initialization"""
        if not 0 <= self.estimated_probability <= 1:
            raise ValueError(
                f"estimated_probability must be between 0 and 1, got {self.estimated_probability}"
            )
        if not 0 <= self.confidence_level <= 1:
            raise ValueError(
                f"confidence_level must be between 0 and 1, got {self.confidence_level}"
            )
    
    def __str__(self) -> str:
        """Human-readable representation"""
        return (
            f"Estimate: {self.estimated_probability:.1%} "
            f"(confidence: {self.confidence_level:.1%}, edge: {self.edge:+.2%})"
        )
    
    def has_significant_edge(self, min_edge: float, min_confidence: float) -> bool:
        """
        Check if this estimate represents a significant trading opportunity
        
        Args:
            min_edge: Minimum edge percentage (e.g., 0.08 for 8%)
            min_confidence: Minimum confidence level (e.g., 0.60 for 60%)
        
        Returns:
            True if both edge and confidence exceed thresholds
        """
        return abs(self.edge) >= min_edge and self.confidence_level >= min_confidence
    
    def is_buy_yes_signal(self) -> bool:
        """Returns True if estimate suggests buying YES outcome"""
        return self.edge > 0
    
    def is_buy_no_signal(self) -> bool:
        """Returns True if estimate suggests buying NO outcome"""
        return self.edge < 0
