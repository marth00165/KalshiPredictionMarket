"""Base API client with shared HTTP logic, retry handling, and cost tracking"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime

import aiohttp

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTION CLASSES
# ============================================================================

class APIError(Exception):
    """Base exception for API-related errors"""
    
    def __init__(
        self,
        platform: str,
        operation: str,
        status_code: Optional[int] = None,
        message: str = "",
        details: Optional[Dict] = None
    ):
        self.platform = platform
        self.operation = operation
        self.status_code = status_code
        self.details = details or {}
        
        full_message = f"[{platform}] {operation}"
        if status_code:
            full_message += f" (HTTP {status_code})"
        if message:
            full_message += f": {message}"
        
        super().__init__(full_message)


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded (HTTP 429)"""
    
    def __init__(self, platform: str, retry_after: Optional[int] = None):
        super().__init__(
            platform=platform,
            operation="Rate limit exceeded",
            status_code=429,
            details={'retry_after': retry_after}
        )
        self.retry_after = retry_after or 60


class AuthenticationError(APIError):
    """Raised when authentication fails (HTTP 401)"""
    
    def __init__(self, platform: str, message: str = "Invalid API key"):
        super().__init__(
            platform=platform,
            operation="Authentication failed",
            status_code=401,
            message=message
        )


class ServerError(APIError):
    """Raised when server returns 5xx error"""
    
    def __init__(self, platform: str, status_code: int, response_text: str = ""):
        super().__init__(
            platform=platform,
            operation="Server error",
            status_code=status_code,
            message=response_text[:100]
        )


class ClientError(APIError):
    """Raised when request is invalid (HTTP 4xx except 401/429)"""
    
    def __init__(self, platform: str, status_code: int, message: str = ""):
        super().__init__(
            platform=platform,
            operation="Client error",
            status_code=status_code,
            message=message
        )


# ============================================================================
# COST TRACKING
# ============================================================================

@dataclass
class APIUsage:
    """Track API usage for cost calculation"""
    
    timestamp: datetime
    operation: str
    input_tokens: int = 0
    output_tokens: int = 0
    status_code: int = 200
    platform: str = "unknown"
    
    def cost(
        self,
        input_cost_per_mtok: float = 0,
        output_cost_per_mtok: float = 0
    ) -> float:
        """Calculate cost for this operation"""
        return (
            (self.input_tokens / 1_000_000) * input_cost_per_mtok +
            (self.output_tokens / 1_000_000) * output_cost_per_mtok
        )


class CostTracker:
    """Track API costs across requests"""
    
    def __init__(
        self,
        input_cost_per_mtok: float = 0,
        output_cost_per_mtok: float = 0
    ):
        self.input_cost_per_mtok = input_cost_per_mtok
        self.output_cost_per_mtok = output_cost_per_mtok
        self.usage_history: List[APIUsage] = []
    
    def record_usage(self, usage: APIUsage) -> None:
        """Record API usage"""
        self.usage_history.append(usage)
    
    def total_cost(self) -> float:
        """Calculate total cost of all recorded usage"""
        return sum(
            usage.cost(self.input_cost_per_mtok, self.output_cost_per_mtok)
            for usage in self.usage_history
        )
    
    def total_requests(self) -> int:
        """Get total number of requests"""
        return len(self.usage_history)
    
    def total_input_tokens(self) -> int:
        """Get total input tokens"""
        return sum(usage.input_tokens for usage in self.usage_history)
    
    def total_output_tokens(self) -> int:
        """Get total output tokens"""
        return sum(usage.output_tokens for usage in self.usage_history)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        total_cost = self.total_cost()
        total_requests = self.total_requests()
        
        return {
            'total_cost': total_cost,
            'total_requests': total_requests,
            'total_input_tokens': self.total_input_tokens(),
            'total_output_tokens': self.total_output_tokens(),
            'avg_cost_per_request': total_cost / max(1, total_requests),
            'usage_history': self.usage_history,
        }


# ============================================================================
# BASE API CLIENT
# ============================================================================

class BaseAPIClient:
    """
    Base client for prediction market APIs with shared functionality:
    - Consistent error handling with custom exceptions
    - Automatic retry with exponential backoff
    - Pagination support (offset-based and cursor-based)
    - Cost tracking for token-counted APIs
    - Request timeout handling
    """
    
    def __init__(
        self,
        platform_name: str,
        api_key: Optional[str] = None,
        base_url: str = "",
        timeout: int = 30,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        input_cost_per_mtok: float = 0,
        output_cost_per_mtok: float = 0,
    ):
        """
        Initialize base API client
        
        Args:
            platform_name: Name of the platform (e.g., 'polymarket', 'kalshi')
            api_key: Optional API key for authentication
            base_url: Base URL for API endpoints
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            backoff_base: Base for exponential backoff (2 = 1s, 2s, 4s, 8s...)
            input_cost_per_mtok: Cost per million input tokens (for Claude-like APIs)
            output_cost_per_mtok: Cost per million output tokens (for Claude-like APIs)
        """
        self.platform_name = platform_name
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        
        # HTTP session (created in __aenter__)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cost tracking
        self.cost_tracker = CostTracker(
            input_cost_per_mtok=input_cost_per_mtok,
            output_cost_per_mtok=output_cost_per_mtok
        )
    
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self.session = aiohttp.ClientSession(timeout=timeout)
        logger.debug(f"✅ Created session for {self.platform_name}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            logger.debug(f"✅ Closed session for {self.platform_name}")
    
    # ========================================================================
    # HEADER BUILDING
    # ========================================================================
    
    def _build_headers(
        self,
        api_key: Optional[str] = None,
        auth_type: str = "Bearer",
        additional_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Build HTTP headers with optional authentication
        
        Args:
            api_key: API key to use (defaults to self.api_key)
            auth_type: Authorization header type ("Bearer", "Basic", "Token", etc.)
            additional_headers: Additional headers to include
        
        Returns:
            Dictionary of headers
        """
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add authentication if provided
        key_to_use = api_key or self.api_key
        if key_to_use:
            headers["Authorization"] = f"{auth_type} {key_to_use}"
        
        # Merge additional headers
        if additional_headers:
            headers.update(additional_headers)
        
        return headers
    
    # ========================================================================
    # RETRY LOGIC WITH EXPONENTIAL BACKOFF
    # ========================================================================
    
    async def _call_with_retry(
        self,
        coro_fn: Callable[[], Any],
        operation_name: str = "API call",
        max_retries: Optional[int] = None,
    ) -> Any:
        """
        Execute an async operation with exponential backoff retry logic
        
        Args:
            coro_fn: Async function to execute (as callable, not coroutine)
            operation_name: Human-readable operation description for logging
            max_retries: Override default max_retries for this call
        
        Returns:
            Result from the async function
        
        Raises:
            Original exception after max retries exhausted
        """
        max_attempts = (max_retries or self.max_retries) + 1  # +1 for initial attempt
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                logger.debug(f"[{self.platform_name}] {operation_name} (attempt {attempt + 1})")
                return await coro_fn()
            
            except RateLimitError as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    wait_time = e.retry_after
                    logger.warning(
                        f"[{self.platform_name}] Rate limited. "
                        f"Waiting {wait_time}s before retry..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[{self.platform_name}] Rate limit exceeded after {attempt} attempts")
            
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    wait_time = self.backoff_base ** attempt
                    logger.warning(
                        f"[{self.platform_name}] {operation_name} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_attempts - 1})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"[{self.platform_name}] {operation_name} failed after {max_attempts} attempts"
                    )
            
            except (AuthenticationError, ServerError, ClientError) as e:
                last_exception = e
                # These are typically not retryable
                if isinstance(e, ServerError) and attempt < max_attempts - 1:
                    # Retry server errors with backoff
                    wait_time = self.backoff_base ** attempt
                    logger.warning(
                        f"[{self.platform_name}] Server error. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # Auth and client errors shouldn't be retried
                    logger.error(f"[{self.platform_name}] {operation_name} failed: {e}")
                    raise
        
        # Exhausted retries
        if last_exception:
            raise last_exception
    
    # ========================================================================
    # PAGINATION
    # ========================================================================
    
    async def _get_paginated(
        self,
        url: str,
        params_builder: Callable[[int, Optional[str]], Dict[str, Any]],
        response_parser: Callable[[Dict[str, Any]], Tuple[List[Any], Optional[str]]],
        max_items: Optional[int] = None,
        pagination_type: str = "offset",
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> List[Any]:
        """
        Fetch all paginated items from an API endpoint
        
        Args:
            url: Full API URL
            params_builder: Function that takes (offset/page, cursor) and returns params dict
            response_parser: Function that takes response JSON and returns (items_list, next_cursor_or_none)
            max_items: Stop after fetching this many items (None = fetch all)
            pagination_type: "offset" for offset-based, "cursor" for cursor-based
            headers: Additional headers to include in request
            **kwargs: Additional arguments to pass to session.get()
        
        Returns:
            Flattened list of all parsed items
        
        Example:
            # Offset-based pagination
            def params_builder(offset, cursor):
                return {'limit': 100, 'offset': offset}
            
            def response_parser(json_data):
                items = [parse_market(m) for m in json_data.get('markets', [])]
                has_more = len(items) == 100  # crude check
                return items, None if not has_more else offset + 100
            
            markets = await client._get_paginated(
                url="https://api.example.com/markets",
                params_builder=params_builder,
                response_parser=response_parser,
                max_items=500
            )
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        all_items = []
        offset = 0
        cursor = None
        request_count = 0
        
        while True:
            # Stop if we've reached max items
            if max_items and len(all_items) >= max_items:
                logger.info(
                    f"[{self.platform_name}] Reached max_items limit ({max_items}). "
                    f"Fetched {len(all_items)} items in {request_count} requests."
                )
                break
            
            # Build params for this request
            if pagination_type == "offset":
                params = params_builder(offset, None)
            else:  # cursor-based
                params = params_builder(None, cursor)
            
            # Make request with retry
            async def fetch_page():
                request_headers = headers or self._build_headers()
                async with self.session.get(url, params=params, headers=request_headers, **kwargs) as response:
                    # Check for errors
                    await self._handle_response_status(response)
                    data = await response.json()
                    request_count += 1
                    return data
            
            try:
                data = await self._call_with_retry(fetch_page, f"Fetch page (offset={offset}, cursor={cursor})")
            except Exception as e:
                logger.error(f"[{self.platform_name}] Failed to fetch paginated data: {e}")
                break
            
            # Parse response
            try:
                batch_items, next_cursor = response_parser(data)
                all_items.extend(batch_items)
                logger.debug(f"[{self.platform_name}] Fetched {len(batch_items)} items (total: {len(all_items)})")
                
                # Check if pagination is complete
                if not batch_items or next_cursor is None:
                    logger.info(
                        f"[{self.platform_name}] Pagination complete. "
                        f"Fetched {len(all_items)} items in {request_count} requests."
                    )
                    break
                
                # Update pagination state
                if pagination_type == "offset":
                    offset = next_cursor if isinstance(next_cursor, int) else offset + len(batch_items)
                else:  # cursor-based
                    cursor = next_cursor
                
                # Small delay between requests to respect rate limits
                await asyncio.sleep(0.1)
            
            except Exception as e:
                logger.error(f"[{self.platform_name}] Error parsing paginated response: {e}")
                break
        
        # Trim to max_items if needed
        if max_items and len(all_items) > max_items:
            all_items = all_items[:max_items]
        
        return all_items
    
    # ========================================================================
    # RESPONSE HANDLING
    # ========================================================================
    
    async def _handle_response_status(self, response: aiohttp.ClientResponse) -> None:
        """
        Check HTTP response status and raise appropriate exceptions
        
        Args:
            response: aiohttp response object
        
        Raises:
            RateLimitError: If status is 429
            AuthenticationError: If status is 401
            ServerError: If status is 5xx
            ClientError: If status is 4xx (except 401, 429)
        """
        if response.status == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            raise RateLimitError(self.platform_name, retry_after=retry_after)
        
        elif response.status == 401:
            raise AuthenticationError(self.platform_name)
        
        elif response.status >= 500:
            text = await response.text()
            raise ServerError(self.platform_name, response.status, text)
        
        elif response.status >= 400:
            text = await response.text()
            raise ClientError(self.platform_name, response.status, text[:200])
    
    # ========================================================================
    # COST TRACKING (for token-counted APIs like Claude)
    # ========================================================================
    
    def record_usage(
        self,
        operation: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        status_code: int = 200
    ) -> None:
        """
        Record API usage for cost calculation
        
        Args:
            operation: Description of the operation
            input_tokens: Number of input tokens consumed
            output_tokens: Number of output tokens consumed
            status_code: HTTP status code of the request
        """
        usage = APIUsage(
            timestamp=datetime.now(),
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            status_code=status_code,
            platform=self.platform_name,
        )
        self.cost_tracker.record_usage(usage)
    
    def get_cost_stats(self) -> Dict[str, Any]:
        """Get comprehensive cost and usage statistics"""
        return self.cost_tracker.get_stats()
