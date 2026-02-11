"""Custom exception classes for trading operations"""

from typing import Optional, Dict, Any


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class ValidationError(ValueError):
    """Base validation error for invalid data"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.field = field
        self.value = value
        
        full_message = message
        if field and value is not None:
            full_message = f"{message} (field: {field}, value: {value})"
        elif field:
            full_message = f"{message} (field: {field})"
        
        super().__init__(full_message)


class MarketDataError(ValidationError):
    """Error in market data validation"""
    
    def __init__(self, message: str, market_id: Optional[str] = None):
        self.market_id = market_id
        super().__init__(message, field="market", value=market_id)


class EstimateError(ValidationError):
    """Error in fair value estimate"""
    
    def __init__(self, message: str, market_id: Optional[str] = None):
        self.market_id = market_id
        super().__init__(message, field="estimate", value=market_id)


class SignalError(ValidationError):
    """Error in trade signal generation"""
    
    def __init__(self, message: str, market_id: Optional[str] = None):
        self.market_id = market_id
        super().__init__(message, field="signal", value=market_id)


class ConfigError(ValidationError):
    """Error in configuration"""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, field="config", value=config_key)


# ============================================================================
# STRATEGY ERRORS
# ============================================================================

class StrategyError(Exception):
    """Base error for strategy operations"""
    pass


class InsufficientCapitalError(StrategyError):
    """Not enough capital to execute position"""
    
    def __init__(self, required: float, available: float):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient capital: need ${required:,.2f}, have ${available:,.2f}"
        )


class NoOpportunitiesError(StrategyError):
    """No trading opportunities found"""
    
    def __init__(self, reason: str = ""):
        msg = "No trading opportunities found"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class PositionLimitError(StrategyError):
    """Maximum position limit reached"""
    
    def __init__(self, current_positions: int, max_allowed: int):
        self.current_positions = current_positions
        self.max_allowed = max_allowed
        super().__init__(
            f"Position limit reached: {current_positions}/{max_allowed} positions open"
        )


# ============================================================================
# POSITION MANAGEMENT ERRORS
# ============================================================================

class PositionError(Exception):
    """Base error for position management"""
    pass


class PositionNotFoundError(PositionError):
    """Position not found"""
    
    def __init__(self, market_id: str):
        self.market_id = market_id
        super().__init__(f"Position not found for market: {market_id}")


class PositionAlreadyClosedError(PositionError):
    """Attempting to close an already closed position"""
    
    def __init__(self, market_id: str):
        self.market_id = market_id
        super().__init__(f"Position already closed for market: {market_id}")


class InsufficientBankrollError(PositionError):
    """Not enough bankroll to open position"""
    
    def __init__(self, position_size: float, bankroll: float):
        self.position_size = position_size
        self.bankroll = bankroll
        super().__init__(
            f"Insufficient bankroll: position ${position_size:,.2f} > bankroll ${bankroll:,.2f}"
        )


# ============================================================================
# EXECUTION ERRORS
# ============================================================================

class ExecutionError(Exception):
    """Base error for trade execution"""
    pass


class ExecutionFailedError(ExecutionError):
    """Trade execution failed"""
    
    def __init__(self, market_id: str, action: str, reason: str, details: Optional[Dict] = None):
        self.market_id = market_id
        self.action = action
        self.reason = reason
        self.details = details or {}
        
        super().__init__(
            f"Execution failed for {action} on {market_id}: {reason}"
        )


class OrderPlacementError(ExecutionError):
    """Error placing order with platform API"""
    
    def __init__(self, platform: str, message: str, status_code: Optional[int] = None):
        self.platform = platform
        self.status_code = status_code
        
        msg = f"[{platform}] Order placement failed: {message}"
        if status_code:
            msg += f" (HTTP {status_code})"
        
        super().__init__(msg)


class OrderConfirmationError(ExecutionError):
    """Order placed but not confirmed"""
    
    def __init__(self, order_id: str, platform: str):
        self.order_id = order_id
        self.platform = platform
        super().__init__(
            f"Order {order_id} not confirmed on {platform}"
        )


class DryRunError(ExecutionError):
    """Error in dry-run mode"""
    
    def __init__(self, message: str):
        super().__init__(f"DRY RUN: {message}")


# ============================================================================
# API ERRORS (extend from base_client errors)
# ============================================================================

class APIDataError(Exception):
    """Error parsing API response data"""
    
    def __init__(self, platform: str, operation: str, message: str, data: Optional[Any] = None):
        self.platform = platform
        self.operation = operation
        self.data = data
        
        full_msg = f"[{platform}] {operation}: {message}"
        if data:
            full_msg += f"\nData: {str(data)[:200]}"
        
        super().__init__(full_msg)


class IncompletePriceDataError(APIDataError):
    """Missing or incomplete price data from API"""
    
    def __init__(self, platform: str, market_id: str):
        super().__init__(
            platform=platform,
            operation="Fetch prices",
            message=f"Incomplete price data for {market_id}"
        )


# ============================================================================
# ERROR CONTEXT
# ============================================================================

class ErrorContext:
    """Context manager for handling and reporting errors"""
    
    def __init__(self, operation: str, critical: bool = False):
        """
        Initialize error context
        
        Args:
            operation: Description of the operation
            critical: If True, errors will be re-raised after logging
        """
        self.operation = operation
        self.critical = critical
        self.errors: list = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.errors.append((exc_type, exc_val, exc_tb))
            
            if self.critical:
                # Re-raise critical errors
                return False
            else:
                # Suppress non-critical errors
                return True
        return True
    
    def add_error(self, error: Exception) -> None:
        """Manually add error"""
        self.errors.append((type(error), error, None))
    
    def has_errors(self) -> bool:
        """Check if any errors occurred"""
        return len(self.errors) > 0
    
    def get_errors(self) -> list:
        """Get list of all errors"""
        return self.errors
    
    def get_summary(self) -> str:
        """Get summary of errors"""
        if not self.errors:
            return f"✅ {self.operation}: No errors"
        
        error_count = len(self.errors)
        error_types = list(set(str(e[0].__name__) for e in self.errors))
        
        return (
            f"❌ {self.operation}: {error_count} error(s) "
            f"({', '.join(error_types)})"
        )
