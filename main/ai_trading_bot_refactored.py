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
        Call Claude API with prompt
        
        Note: This is a placeholder. Real implementation would use
        the actual Claude API with aiohttp session from base_client.
        """
        logger.warning("⚠️  Claude API call stubbed - implement with actual API")
        
        # Placeholder response
        return {
            'content_text': json.dumps({
                'probability': 50,
                'confidence': 50,
                'reasoning': 'Placeholder analysis',
                'data_sources': []
            })
        }
    
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
        self.analyzer = ClaudeAnalyzer(self.config)
        self.strategy = Strategy(self.config)
        self.position_manager = PositionManager(
            self.config.trading.initial_bankroll
        )
        self.executor = TradeExecutor(self.config)
        
        # Error reporting
        self.error_reporter = get_error_reporter()
        
        # Cycle counter
        self.cycle_count = 0
    
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
        
        logger.info("=" * 80)
        logger.info(f"🤖 CYCLE {self.cycle_count}: STARTING TRADING CYCLE")
        logger.info("=" * 80)
        
        cycle_start = time.time()
        cycle_report = self.error_reporter.create_report(
            f"Trading Cycle #{self.cycle_count}"
        )
        
        try:
            # Step 1: Scan markets
            logger.info("\n📊 Step 1: Scanning markets...")
            markets = await self.scanner.scan_all_markets()
            
            if not markets:
                logger.warning("No markets found, aborting cycle")
                return
            
            # Step 2: Filter markets
            logger.info("\n🔬 Step 2: Filtering markets...")
            filtered = self.strategy.filter_markets(markets)
            logger.info(f"   {len(filtered)} markets passed filters")
            
            # Step 3: Analyze with Claude (in batches)
            logger.info("\n🧠 Step 3: Analyzing with Claude AI...")
            estimates = await self.analyzer.analyze_market_batch(filtered)
            
            if not estimates:
                logger.warning("No estimates generated")
                return
            
            # Step 4: Find mispricings
            logger.info("\n💰 Step 4: Finding mispricings...")
            opportunities = self.strategy.find_opportunities(estimates, filtered)
            logger.info(
                f"   Found {len(opportunities)} opportunities with "
                f">{self.config.min_edge_percentage:.0f}% edge"
            )
            
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
                    results = await self.executor.execute_signals(signals)
                    successful = sum(results)
                    logger.info(f"Executed {successful}/{len(signals)} trades")
                except ExecutionError as e:
                    logger.error(f"Execution error: {e}")
                    self.error_reporter.add_error_to_report(
                        cycle_report, e, "trade execution"
                    )
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
        
        finally:
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
