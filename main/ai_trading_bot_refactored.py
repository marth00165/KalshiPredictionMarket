#!/usr/bin/env python3
"""
Advanced AI-Powered Prediction Market Trading Bot (Refactored)

Features:
- Scans 1000+ markets using modular API clients
- Claude AI fair value estimation with cost tracking
- Kelly criterion position sizing via Strategy module
- Autonomous trading via TradeExecutor
- Position management and performance tracking
- Comprehensive error handling and reporting

Architecture:
- api_clients: Polymarket & Kalshi API clients with retry logic
- models: Standardized MarketData, FairValueEstimate, TradeSignal
- utils: Configuration, response parsing, error handling
- trading: Strategy, PositionManager, TradeExecutor
"""

import json
import logging
import asyncio
import time
from datetime import datetime
from typing import List, Optional
from pathlib import Path

import aiohttp

# Import configuration management
from utils.config_manager import ConfigManager

# Import data models
from models import MarketData, FairValueEstimate, TradeSignal

# Import API clients
from api_clients import (
    PolymarketClient,
    PolymarketConfig,
    KalshiClient,
    KalshiConfig,
    APIError,
    RateLimitError,
)

# Import utilities
from utils import (
    ClaudeResponseParser,
    BatchParser,
    ErrorReporter,
    get_error_reporter,
    InsufficientCapitalError,
    NoOpportunitiesError,
    ExecutionError,
)

# Import analysis providers
from analysis import OpenAIAnalyzer

# Import trading modules
from trading import PositionManager, Strategy, TradeExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClaudeAnalyzer:
    """
    Uses Claude API to estimate fair values for prediction markets
    
    This is a wrapper around the base API client that handles:
    - Market analysis prompt construction
    - Response parsing via ClaudeResponseParser
    - Cost tracking for Claude API calls
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize Claude analyzer
        
        Args:
            config: ConfigManager with Claude API settings
        """
        from api_clients.base_client import BaseAPIClient
        
        self.client = BaseAPIClient(
            platform_name="claude",
            api_key=config.claude_api_key,
            base_url="https://api.anthropic.com/v1",
            input_cost_per_mtok=config.claude.input_cost_per_mtok,
            output_cost_per_mtok=config.claude.output_cost_per_mtok,
        )
        self.config = config
    
    async def analyze_market_batch(
        self,
        markets: List[MarketData],
        session=None
    ) -> List[FairValueEstimate]:
        """
        Analyze multiple markets in parallel for efficiency
        
        Args:
            markets: List of markets to analyze
            session: Optional aiohttp session (for compatibility)
        
        Returns:
            List of FairValueEstimate objects
        """
        async with self.client:
            tasks = [
                self.analyze_single_market(market)
                for market in markets
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and None results
        valid_results = [
            r for r in results
            if isinstance(r, FairValueEstimate)
        ]
        
        logger.info(
            f"✅ Analyzed {len(valid_results)}/{len(markets)} markets successfully"
        )
        
        return valid_results
    
    async def analyze_single_market(
        self,
        market: MarketData
    ) -> Optional[FairValueEstimate]:
        """
        Use Claude to estimate fair probability for a market
        
        Args:
            market: Market to analyze
        
        Returns:
            FairValueEstimate or None if analysis fails
        """
        prompt = self._build_analysis_prompt(market)
        
        try:
            # Call Claude API
            response = await self._call_claude_api(prompt)
            
            # Parse response into FairValueEstimate
            estimate = ClaudeResponseParser.parse_fair_value_estimate(
                response_text=response.get('content_text', ''),
                market_id=market.market_id,
                market_price=market.yes_price
            )
            
            if estimate:
                logger.debug(
                    f"📊 {market.title[:50]}... → "
                    f"{estimate.estimated_probability:.1%} "
                    f"(edge: {estimate.edge:+.1%})"
                )
            
            return estimate
        
        except Exception as e:
            logger.error(f"Error analyzing market {market.market_id}: {e}")
            return None
    
    def _build_analysis_prompt(self, market: MarketData) -> str:
        """Construct prompt for Claude to analyze market"""
        
        return f"""Analyze this prediction market and estimate the TRUE probability of the outcome.

MARKET DETAILS:
Title: {market.title}
Description: {market.description}
Current Market Price: {market.yes_price:.1%} for YES
Volume: ${market.volume:,.0f}
Category: {market.category}
Closes: {market.end_date}

YOUR TASK:
1. Research what is known about this event
2. Consider base rates, historical precedents, and current data
3. Estimate the TRUE probability (0-100%)
4. Explain your reasoning
5. Rate your confidence (0-100%)

CRITICAL: Be objective. Don't just accept the market price. Use reasoning and data.

Respond in JSON format:
{{
  "probability": <float 0-100>,
  "confidence": <float 0-100>,
  "reasoning": "<detailed explanation>",
  "key_factors": ["factor1", "factor2", ...],
  "data_sources": ["source1", "source2", ...]
}}

Think step-by-step and be thorough."""
    
    async def _call_claude_api(self, prompt: str) -> dict:
        """
        Call Claude API with prompt using BaseAPIClient
        
        Args:
            prompt: Market analysis prompt for Claude
        
        Returns:
            Dictionary with 'content_text' key containing Claude's response
        
        Raises:
            APIError: If API call fails after retries
        """
        
        # Ensure session is initialized
        if not self.client.session:
            logger.error("Claude client session not initialized")
            raise RuntimeError("Cannot call Claude API without initialized session")
        
        # Build Claude API payload
        payload = {
            "model": self.config.claude.model,
            "max_tokens": self.config.claude.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.config.claude.temperature
        }
        
        # Build URL and headers
        url = f"{self.client.base_url}/messages"
        headers = self.client._build_headers(
            auth_type="",
            auth_header_name="x-api-key",
            additional_headers={
                "anthropic-version": "2023-06-01"
            }
        )
        
        logger.debug(f"Calling Claude API: {self.config.claude.model}")
        
        # Make API call with retry logic
        async def make_request():
            async with self.client.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.client.timeout_seconds)
            ) as response:
                # Check for HTTP errors
                await self.client._handle_response_status(response)
                return await response.json()
        
        try:
            response = await self.client._call_with_retry(
                make_request,
                operation_name="Claude API analysis"
            )
            
            # Track API usage/cost
            if 'usage' in response:
                self.client.record_usage(
                    operation="Market analysis",
                    input_tokens=response['usage'].get('input_tokens', 0),
                    output_tokens=response['usage'].get('output_tokens', 0)
                )
                
                logger.debug(
                    f"Claude usage: "
                    f"{response['usage'].get('input_tokens', 0)} in, "
                    f"{response['usage'].get('output_tokens', 0)} out"
                )
            
            # Extract text from Claude response format
            # Claude returns: {"content": [{"type": "text", "text": "..."}], "usage": {...}}
            content = response.get('content', [])
            if not content:
                logger.error(f"No content in Claude response: {response}")
                raise APIError(
                    platform="claude",
                    operation="Parse response",
                    message="Empty content array in response"
                )
            
            text = content[0].get('text', '')
            
            if not text:
                logger.error("No text in Claude response content")
                raise APIError(
                    platform="claude",
                    operation="Parse response",
                    message="Empty text in response"
                )
            
            logger.debug(f"Claude response received: {len(text)} characters")
            
            return {
                'content_text': text,
                'usage': response.get('usage', {})
            }
        
        except APIError as e:
            logger.error(f"Claude API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Claude API: {e}")
            raise APIError(
                platform="claude",
                operation="API call",
                message=str(e)
            )
    
    def get_api_stats(self) -> dict:
        """Get API usage statistics from cost tracker"""
        return self.client.get_cost_stats()


class MarketScanner:
    """
    Efficiently scan 1000+ markets across platforms
    
    Uses modular platform-specific clients:
    - PolymarketClient for offset-based pagination
    - KalshiClient for cursor-based pagination
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize market scanner
        
        Args:
            config: ConfigManager with platform settings
        """
        self.config = config
        
        # Initialize platform clients based on config
        self.polymarket_client = (
            PolymarketClient(
                PolymarketConfig(
                    api_key=config.platforms.polymarket.api_key,
                    base_url="https://gamma-api.polymarket.com",
                    max_markets=config.platforms.polymarket.max_markets,
                )
            )
            if config.platforms.polymarket.enabled
            else None
        )
        
        self.kalshi_client = (
            KalshiClient(
                KalshiConfig(
                    api_key=config.platforms.kalshi.api_key,
                    base_url="https://api.elections.kalshi.com/trade-api/v2",
                    max_markets=config.platforms.kalshi.max_markets,
                )
            )
            if config.platforms.kalshi.enabled
            else None
        )
    
    async def scan_all_markets(self) -> List[MarketData]:
        """
        Scan all markets from enabled platforms in parallel
        
        Returns:
            List of standardized MarketData objects
        """
        logger.info("🔍 Scanning markets across all platforms...")
        
        tasks = []
        
        if self.polymarket_client:
            tasks.append(self._fetch_platform_markets(
                'polymarket',
                self.polymarket_client.fetch_markets()
            ))
        
        if self.kalshi_client:
            tasks.append(self._fetch_platform_markets(
                'kalshi',
                self.kalshi_client.fetch_markets()
            ))
        
        if not tasks:
            logger.error("No platforms enabled!")
            return []
        
        # Fetch from all platforms in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors and flatten results
        all_markets = []
        for result in results:
            if isinstance(result, list):
                all_markets.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error fetching markets: {result}")
        
        logger.info(f"✅ Found {len(all_markets)} total markets")
        
        return all_markets
    
    async def _fetch_platform_markets(
        self,
        platform_name: str,
        fetch_coro
    ) -> List[MarketData]:
        """
        Fetch and convert markets from a platform
        
        Args:
            platform_name: Name of platform (for logging)
            fetch_coro: Coroutine that returns list of market dicts
        
        Returns:
            List of MarketData objects
        """
        try:
            market_dicts = await fetch_coro
            
            # Convert dicts to MarketData via BatchParser
            markets = BatchParser.parse_markets_batch(market_dicts)
            
            logger.info(f"   {platform_name}: {len(markets)} markets")
            return markets
        
        except APIError as e:
            logger.error(f"API error fetching {platform_name}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching {platform_name} markets: {e}")
            return []


class AdvancedTradingBot:
    """
    Main orchestrator for AI-powered trading
    
    Coordinates:
    1. Market scanning via MarketScanner
    2. Market filtering via Strategy
    3. Fair value analysis via ClaudeAnalyzer
    4. Opportunity finding via Strategy
    5. Signal generation via Strategy with Kelly sizing
    6. Trade execution via TradeExecutor
    7. Position tracking via PositionManager
    8. Error reporting via ErrorReporter
    """
    
    def __init__(self, config_file: str = 'advanced_config.json'):
        """
        Initialize trading bot
        
        Args:
            config_file: Path to configuration JSON file
        
        Raises:
            FileNotFoundError: If config file not found
            ValueError: If configuration validation fails
        """
        # Load and validate configuration
        self.config = ConfigManager(config_file)
        self.config.log_config_summary()
        
        # Initialize components
        self.scanner = MarketScanner(self.config)
        self.analyzer = self._create_analyzer(self.config.analysis_provider)
        self.strategy = Strategy(self.config)
        self.position_manager = PositionManager(
            self.config.trading.initial_bankroll
        )
        self.executor = TradeExecutor(self.config)
        
        # Error reporting
        self.error_reporter = get_error_reporter()
        
        # Cycle counter
        self.cycle_count = 0

    def _create_analyzer(self, provider: str):
        provider_norm = (provider or "").strip().lower()
        if provider_norm == "openai":
            return OpenAIAnalyzer(self.config)
        return ClaudeAnalyzer(self.config)

    def set_analysis_provider(self, provider: str) -> None:
        """Switch analysis provider before a scan begins."""
        provider_norm = (provider or "").strip().lower()
        if provider_norm not in {"claude", "openai"}:
            raise ValueError(f"Unknown analysis provider: {provider}")

        # Validate key availability for chosen provider
        if provider_norm == "openai":
            _ = self.config.openai_api_key
        else:
            _ = self.config.claude_api_key

        self.config.analysis.provider = provider_norm
        self.analyzer = self._create_analyzer(provider_norm)

    async def discover_kalshi_series(self, category: Optional[str] = None) -> List[dict]:
        """
        Discover available series on Kalshi.
        
        Useful for finding series_ticker values for targeted scanning.
        
        Args:
            category: Optional category filter (e.g., 'Economics', 'Politics')
        
        Returns:
            List of series info dicts
        """
        if not self.scanner.kalshi_client:
            logger.error("Kalshi client not configured")
            return []
        
        return await self.scanner.kalshi_client.discover_series(category)

    async def scan_series_markets(
        self,
        series_tickers: List[str],
        status: str = "open",
    ) -> List[MarketData]:
        """
        Scan markets by series ticker — lightweight, no rate limit issues.
        
        This is the recommended method for dry runs and data collection.
        Uses prices from market list response (no orderbook calls).
        
        Args:
            series_tickers: List of series tickers (e.g., ['KXFED', 'KXCPI'])
            status: Market status filter ('open', 'closed', 'settled')
        
        Returns:
            List of MarketData objects
        """
        if not self.scanner.kalshi_client:
            logger.error("Kalshi client not configured")
            return []
        
        market_dicts = await self.scanner.kalshi_client.fetch_markets_by_series(
            series_tickers=series_tickers,
            status=status,
        )
        
        # Convert to MarketData
        markets = BatchParser.parse_markets_batch(market_dicts)
        logger.info(f"✅ Scanned {len(markets)} markets from series: {series_tickers}")
        
        return markets

    async def run_series_scan(
        self,
        series_tickers: List[str],
        analyze: bool = False,
        max_analyze: int = 10,
    ) -> dict:
        """
        Run a lightweight scan on specific series — ideal for dry runs.
        
        This method:
        1. Fetches markets by series (no orderbook calls)
        2. Applies filters
        3. Optionally analyzes top N markets with AI
        4. Returns a report dict
        
        Args:
            series_tickers: List of series tickers to scan
            analyze: Whether to run AI analysis on filtered markets
            max_analyze: Max markets to analyze (to limit API costs)
        
        Returns:
            Report dict with scan results
        """
        logger.info("=" * 60)
        logger.info(f"🔍 SERIES SCAN: {series_tickers}")
        logger.info("=" * 60)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "series_tickers": series_tickers,
            "analyze": analyze,
            "counts": {
                "scanned": 0,
                "passed_filters": 0,
                "analyzed": 0,
            },
            "markets": [],
            "filtered_markets": [],
            "analyses": [],
        }
        
        # Step 1: Fetch markets
        markets = await self.scan_series_markets(series_tickers)
        report["counts"]["scanned"] = len(markets)
        
        # Store all market info
        for m in markets:
            report["markets"].append({
                "market_id": m.market_id,
                "title": m.title,
                "yes_price": m.yes_price,
                "volume": m.volume,
                "liquidity": m.liquidity,
                "end_date": m.end_date,
                "category": m.category,
            })
        
        # Step 2: Apply filters
        filtered = self.strategy.filter_markets(markets)
        report["counts"]["passed_filters"] = len(filtered)
        
        for m in filtered:
            filter_result = self.strategy.evaluate_market_filters(m)
            report["filtered_markets"].append({
                "market_id": m.market_id,
                "title": m.title,
                "yes_price": m.yes_price,
                "volume": m.volume,
                "liquidity": m.liquidity,
                "filter_checks": filter_result,
            })
        
        logger.info(f"📊 Scanned: {len(markets)}, Passed filters: {len(filtered)}")
        
        # Step 3: Optional AI analysis
        if analyze and filtered:
            to_analyze = filtered[:max_analyze]
            logger.info(f"🤖 Analyzing top {len(to_analyze)} markets...")
            
            for market in to_analyze:
                try:
                    estimate = await self.analyzer.analyze_market(market)
                    if estimate:
                        report["analyses"].append({
                            "market_id": market.market_id,
                            "title": market.title,
                            "market_price": market.yes_price,
                            "estimated_probability": estimate.probability,
                            "confidence": estimate.confidence,
                            "edge": estimate.probability - market.yes_price,
                            "reasoning": estimate.reasoning,
                        })
                        report["counts"]["analyzed"] += 1
                except Exception as e:
                    logger.warning(f"Error analyzing {market.market_id}: {e}")
        
        logger.info(f"✅ Series scan complete")
        return report
    
    async def run_trading_cycle(self):
        """
        Execute one complete trading cycle
        
        Steps:
        1. Scan all markets from enabled platforms
        2. Filter markets by volume/liquidity
        3. Analyze with Claude AI (in batches)
        4. Find mispricings above threshold
        5. Generate trade signals with Kelly sizing
        6. Execute signals
        7. Report results and errors
        """
        
        self.cycle_count += 1

        # Report container (written to JSON at end of cycle)
        cycle_started_at = datetime.now().isoformat()
        report: dict = {
            "cycle": self.cycle_count,
            "started_at": cycle_started_at,
            "finished_at": None,
            "config": {
                "analysis_provider": self.config.analysis_provider,
                "dry_run": self.config.is_dry_run,
                "api": {
                    "batch_size": self.config.api.batch_size,
                    "api_cost_limit_per_cycle": self.config.api.api_cost_limit_per_cycle,
                },
                "platforms": {
                    "polymarket": {
                        "enabled": self.config.platforms.polymarket.enabled,
                        "max_markets": self.config.platforms.polymarket.max_markets,
                    },
                    "kalshi": {
                        "enabled": self.config.platforms.kalshi.enabled,
                        "max_markets": self.config.platforms.kalshi.max_markets,
                    },
                },
                "filters": {
                    "min_volume": self.config.filters.min_volume,
                    "min_liquidity": self.config.filters.min_liquidity,
                    "price_bounds": {"min": 0.01, "max": 0.99},
                },
                "strategy": {
                    "min_edge": self.config.strategy.min_edge,
                    "min_confidence": self.config.strategy.min_confidence,
                },
                "risk": {
                    "max_kelly_fraction": self.config.risk.max_kelly_fraction,
                    "max_positions": self.config.risk.max_positions,
                    "max_position_size": self.config.risk.max_position_size,
                },
            },
            "counts": {
                "scanned": 0,
                "passed_filters": 0,
                "analyzed": 0,
                "estimates": 0,
                "opportunities": 0,
                "signals": 0,
                "executed": 0,
            },
            "api_cost": None,
            "markets": [],
            "signals": [],
            "errors": [],
        }
        
        logger.info("=" * 80)
        logger.info(f"🤖 CYCLE {self.cycle_count}: STARTING TRADING CYCLE")
        logger.info("=" * 80)
        
        cycle_start = time.time()
        cycle_report = self.error_reporter.create_report(
            f"Trading Cycle #{self.cycle_count}"
        )

        markets: List[MarketData] = []
        filtered: List[MarketData] = []
        analyzed_markets: List[MarketData] = []
        estimates: List[FairValueEstimate] = []
        opportunities = []
        signals: List[TradeSignal] = []
        execution_results: List[bool] = []
        
        try:
            # Step 1: Scan markets
            logger.info("\n📊 Step 1: Scanning markets...")
            markets = await self.scanner.scan_all_markets()

            # Initialize market rows for report
            market_rows_by_id = {}
            for m in markets:
                evaluation = self.strategy.evaluate_market_filters(m)
                row = {
                    "market_id": m.market_id,
                    "platform": m.platform,
                    "title": m.title,
                    "description": m.description,
                    "category": m.category,
                    "end_date": str(m.end_date),
                    "prices": {
                        "yes": m.yes_price,
                        "no": m.no_price,
                    },
                    "stats": {
                        "volume": m.volume,
                        "liquidity": m.liquidity,
                    },
                    "filters": evaluation,
                    "analysis": None,
                    "opportunity": None,
                    "signal": None,
                    "execution": None,
                }
                market_rows_by_id[m.market_id] = row
            report["markets"] = list(market_rows_by_id.values())
            report["counts"]["scanned"] = len(markets)
            
            if not markets:
                logger.warning("No markets found, aborting cycle")
                return
            
            # Step 2: Filter markets
            logger.info("\n🔬 Step 2: Filtering markets...")
            filtered = self.strategy.filter_markets(markets)
            logger.info(f"   {len(filtered)} markets passed filters")
            report["counts"]["passed_filters"] = len(filtered)
            
            # Step 3: Analyze with LLM (in batches)
            provider = (self.config.analysis_provider or "claude").strip().lower()
            logger.info(f"\n🧠 Step 3: Analyzing with {provider}...")

            batch_size = max(1, int(self.config.api.batch_size))
            cost_limit = float(self.config.api.api_cost_limit_per_cycle)

            total_batches = (len(filtered) + batch_size - 1) // batch_size
            for batch_index in range(total_batches):
                start = batch_index * batch_size
                batch = filtered[start:start + batch_size]
                if not batch:
                    break

                logger.info(
                    f"   Batch {batch_index + 1}/{total_batches}: analyzing {len(batch)} markets"
                )

                batch_estimates = await self.analyzer.analyze_market_batch(batch)
                estimates.extend(batch_estimates)
                analyzed_markets.extend(batch)

                report["counts"]["analyzed"] = len(analyzed_markets)
                report["counts"]["estimates"] = len(estimates)

                # Attach estimates onto market rows as they arrive
                for est in batch_estimates:
                    row = market_rows_by_id.get(est.market_id)
                    if row is not None:
                        row["analysis"] = {
                            "estimated_probability": est.estimated_probability,
                            "confidence": est.confidence_level,
                            "edge": est.edge,
                            "reasoning": est.reasoning,
                            "key_factors": est.key_factors,
                            "data_sources": est.data_sources,
                        }

                # Stop early if we hit the configured API spend limit.
                try:
                    stats = self.analyzer.get_api_stats()
                    total_cost = float(stats.get('total_cost', 0.0))
                except Exception:
                    total_cost = 0.0

                if cost_limit > 0 and total_cost >= cost_limit:
                    logger.warning(
                        f"API cost limit reached: ${total_cost:.2f} >= ${cost_limit:.2f}. "
                        "Stopping analysis early."
                    )
                    break

            # Snapshot cost stats if available
            try:
                report["api_cost"] = self.analyzer.get_api_stats()
            except Exception:
                report["api_cost"] = None
            
            if not estimates:
                logger.warning("No estimates generated")
                return
            
            # Step 4: Find mispricings
            logger.info("\n💰 Step 4: Finding mispricings...")
            opportunities = self.strategy.find_opportunities(estimates, analyzed_markets)
            report["counts"]["opportunities"] = len(opportunities)
            logger.info(
                f"   Found {len(opportunities)} opportunities with "
                f">{self.config.min_edge_percentage:.0f}% edge"
            )

            for market, est in opportunities:
                row = market_rows_by_id.get(market.market_id)
                if row is not None:
                    row["opportunity"] = {
                        "edge": est.edge,
                        "confidence": est.confidence_level,
                        "estimated_probability": est.estimated_probability,
                    }
            
            # Step 5: Generate trade signals
            logger.info("\n📐 Step 5: Calculating position sizes (Kelly)...")
            try:
                signals = self.strategy.generate_trade_signals(
                    opportunities=opportunities,
                    current_bankroll=self.position_manager.current_bankroll,
                    current_exposure=self.position_manager.get_total_exposure()
                )
            except (InsufficientCapitalError, NoOpportunitiesError) as e:
                logger.warning(f"Strategy error: {e}")
                self.error_reporter.add_error_to_report(
                    cycle_report, e, "signal generation"
                )
                signals = []

            report["counts"]["signals"] = len(signals)

            # Attach signals to markets and keep a list
            for s in signals:
                row = market_rows_by_id.get(s.market.market_id)
                signal_row = {
                    "market_id": s.market.market_id,
                    "platform": s.market.platform,
                    "action": s.action,
                    "fair_value": s.fair_value,
                    "market_price": s.market_price,
                    "edge": s.edge,
                    "kelly_fraction": s.kelly_fraction,
                    "position_size": s.position_size,
                    "expected_value": s.expected_value,
                    "reasoning": s.reasoning,
                }
                report["signals"].append(signal_row)
                if row is not None:
                    row["signal"] = signal_row
            
            # Step 6: Execute trades
            logger.info("\n⚡ Step 6: Executing trades...")
            if signals:
                # Add positions to manager
                for signal in signals:
                    try:
                        self.position_manager.add_position(signal)
                    except Exception as e:
                        logger.error(f"Error adding position: {e}")
                        self.error_reporter.add_error_to_report(
                            cycle_report, e, "position management"
                        )
                
                # Execute trades
                try:
                    execution_results = await self.executor.execute_signals(signals)
                    successful = sum(execution_results)
                    logger.info(f"Executed {successful}/{len(signals)} trades")
                    report["counts"]["executed"] = int(successful)

                    for s, ok in zip(signals, execution_results):
                        row = market_rows_by_id.get(s.market.market_id)
                        if row is not None:
                            row["execution"] = {
                                "success": bool(ok),
                                "dry_run": self.config.is_dry_run,
                            }
                except ExecutionError as e:
                    logger.error(f"Execution error: {e}")
                    self.error_reporter.add_error_to_report(
                        cycle_report, e, "trade execution"
                    )
                    report["errors"].append({
                        "type": type(e).__name__,
                        "message": str(e),
                        "stage": "trade execution",
                    })
            else:
                logger.info("No signals to execute")
            
            # Step 7: Report performance
            logger.info("\n📊 Step 7: Reporting performance...")
            self._print_cycle_summary(cycle_start)
        
        except Exception as e:
            logger.error(f"Unexpected error in trading cycle: {e}")
            self.error_reporter.add_error_to_report(
                cycle_report, e, "trading cycle"
            )

            report["errors"].append({
                "type": type(e).__name__,
                "message": str(e),
                "stage": "trading cycle",
            })
        
        finally:
            # Finish and write JSON report
            report["finished_at"] = datetime.now().isoformat()
            try:
                reports_dir = Path(__file__).resolve().parent.parent / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)

                safe_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = reports_dir / f"cycle_{self.cycle_count}_{safe_ts}.json"
                report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.info(f"🧾 Wrote cycle report: {report_path}")
            except Exception as e:
                logger.warning(f"Failed to write JSON report: {e}")

            # Log any errors from this cycle
            cycle_report.log_summary()
            if cycle_report.has_errors():
                logger.debug("Errors in this cycle:")
                for error in cycle_report.errors:
                    logger.debug(f"  - {error['type']}: {error['message']}")
    
    def _print_cycle_summary(self, cycle_start: float):
        """Print summary of trading cycle"""
        
        cycle_time = time.time() - cycle_start
        stats = self.position_manager.get_stats()
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 CYCLE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Duration: {cycle_time:.1f}s")
        logger.info(f"Current Bankroll: ${stats['current_bankroll']:,.2f} "
                    f"({stats['return_percent']:+.2f}%)")
        logger.info(f"Open Positions: {stats['open_positions']}")
        logger.info(f"Total Exposure: ${stats['total_exposure']:,.2f}")
        logger.info(f"Available Capital: ${stats['available_capital']:,.2f}")
        
        if stats['total_trades'] > 0:
            logger.info(f"Trade Stats: {stats['total_trades']} trades "
                       f"({stats['winning_trades']}W/{stats['losing_trades']}L) "
                       f"| Win Rate: {stats['win_rate_percent']:.1f}%")
        
        logger.info("=" * 80 + "\n")


async def main():
    """Main entry point for the trading bot"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           AI-POWERED PREDICTION MARKET BOT (Refactored)                      ║
║                                                                              ║
║  ✨ Claude AI Fair Value Estimation                                         ║
║  📊 1000+ Market Scanner with Modular Clients                               ║
║  📐 Kelly Criterion Position Sizing                                         ║
║  💰 Autonomous Trading with Error Handling                                  ║
║  📍 Position Management & Performance Tracking                              ║
║  🛡️  Comprehensive Error Reporting                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        bot = AdvancedTradingBot('advanced_config.json')

        # Optional: choose analysis provider before any scan
        if bot.config.allow_runtime_override:
            default_provider = bot.config.analysis_provider
            choice = input(
                "\nAnalysis provider:\n"
                f"1. Claude (default: {default_provider})\n"
                "2. OpenAI\n\n"
                "Choice (Enter to keep default): "
            ).strip()

            if choice == '1':
                bot.set_analysis_provider('claude')
            elif choice == '2':
                bot.set_analysis_provider('openai')
        
        # Run once or continuously
        mode = input(
            "\nRun mode:\n"
            "1. Single cycle\n"
            "2. Continuous (every 1 hour)\n\n"
            "Choice: "
        ).strip()
        
        if mode == '1':
            await bot.run_trading_cycle()
        
        elif mode == '2':
            logger.info("🔄 Starting continuous mode...")
            
            while True:
                try:
                    await bot.run_trading_cycle()
                    logger.info("\n💤 Sleeping for 1 hour...\n")
                    await asyncio.sleep(3600)
                
                except KeyboardInterrupt:
                    logger.info("\n🛑 Stopped by user")
                    break
                
                except Exception as e:
                    logger.error(f"Error in cycle: {e}")
                    logger.info("Retrying in 1 minute...")
                    await asyncio.sleep(60)
        
        else:
            print("Invalid choice")
        
        # Final error summary
        bot.error_reporter.log_session_summary()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
