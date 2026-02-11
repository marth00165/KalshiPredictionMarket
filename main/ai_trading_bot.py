#!/usr/bin/env python3
"""
Advanced AI-Powered Prediction Market Trading Bot
Features:
- Scans 1000+ markets
- Claude AI fair value estimation
- Kelly criterion position sizing
- Autonomous trading
- Self-pays API bills
"""

import json
import time
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncio
import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Structured market data"""
    platform: str
    market_id: str
    title: str
    description: str
    yes_price: float
    no_price: float
    volume: float
    liquidity: float
    end_date: str
    category: str


@dataclass
class FairValueEstimate:
    """Claude's fair value estimate for a market"""
    market_id: str
    estimated_probability: float
    confidence_level: float  # 0-1
    reasoning: str
    data_sources: List[str]
    edge: float  # estimated_probability - market_price
    

@dataclass
class TradeSignal:
    """Trading signal with position sizing"""
    market: MarketData
    action: str  # 'buy_yes', 'buy_no', 'sell_yes', 'sell_no'
    fair_value: float
    market_price: float
    edge: float  # as decimal
    kelly_fraction: float
    position_size: float  # in dollars
    expected_value: float
    reasoning: str


class ClaudeAnalyzer:
    """Uses Claude API to estimate fair values for prediction markets"""
    
    def __init__(self, config: Dict):
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-sonnet-4-20250514"
        self.total_api_cost = 0.0
        self.requests_made = 0
        
        # Pricing (as of Feb 2026)
        self.input_cost_per_mtok = 3.00  # $3 per million tokens
        self.output_cost_per_mtok = 15.00  # $15 per million tokens
    
    async def analyze_market_batch(
        self, 
        markets: List[MarketData],
        session: aiohttp.ClientSession
    ) -> List[FairValueEstimate]:
        """Analyze multiple markets in parallel for efficiency"""
        tasks = [
            self.analyze_single_market(market, session) 
            for market in markets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out errors
        valid_results = [r for r in results if isinstance(r, FairValueEstimate)]
        logger.info(f"✅ Analyzed {len(valid_results)}/{len(markets)} markets successfully")
        
        return valid_results
    
    async def analyze_single_market(
        self, 
        market: MarketData,
        session: aiohttp.ClientSession
    ) -> Optional[FairValueEstimate]:
        """Use Claude to estimate fair probability for a market"""
        
        prompt = self._build_analysis_prompt(market)
        
        try:
            response = await self._call_claude_api(prompt, session)
            estimate = self._parse_claude_response(response, market)
            
            if estimate:
                logger.debug(f"📊 {market.title[:50]}... → {estimate.estimated_probability:.1%} (edge: {estimate.edge:+.1%})")
            
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

    async def _call_claude_api(
        self, 
        prompt: str,
        session: aiohttp.ClientSession
    ) -> Dict:
        """Call Claude API with prompt"""
        
        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3  # Lower temperature for more consistent analysis
        }
        
        async with session.post(
            self.api_url,
            headers={"Content-Type": "application/json"},
            json=payload
        ) as response:
            data = await response.json()
            
            # Track API costs
            if 'usage' in data:
                input_tokens = data['usage'].get('input_tokens', 0)
                output_tokens = data['usage'].get('output_tokens', 0)
                
                cost = (
                    (input_tokens / 1_000_000) * self.input_cost_per_mtok +
                    (output_tokens / 1_000_000) * self.output_cost_per_mtok
                )
                
                self.total_api_cost += cost
                self.requests_made += 1
            
            return data
    
    def _parse_claude_response(
        self, 
        response: Dict, 
        market: MarketData
    ) -> Optional[FairValueEstimate]:
        """Parse Claude's response into structured estimate"""
        
        try:
            # Extract text from response
            content = response.get('content', [])
            if not content:
                return None
            
            text = content[0].get('text', '')
            
            # Parse JSON response
            # Claude might wrap JSON in markdown code blocks
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]
            
            data = json.loads(text.strip())
            
            # Convert to our format
            probability = data['probability'] / 100  # Convert to decimal
            confidence = data['confidence'] / 100
            
            # Calculate edge
            edge = probability - market.yes_price
            
            return FairValueEstimate(
                market_id=market.market_id,
                estimated_probability=probability,
                confidence_level=confidence,
                reasoning=data.get('reasoning', ''),
                data_sources=data.get('data_sources', []),
                edge=edge
            )
            
        except Exception as e:
            logger.error(f"Error parsing Claude response: {e}")
            return None
    
    def get_api_stats(self) -> Dict:
        """Get API usage statistics"""
        return {
            'total_cost': self.total_api_cost,
            'requests_made': self.requests_made,
            'avg_cost_per_request': self.total_api_cost / max(1, self.requests_made)
        }


class KellyCriterion:
    """Calculate optimal position sizes using Kelly Criterion"""
    
    @staticmethod
    def calculate_kelly_fraction(
        probability: float,
        market_price: float,
        max_fraction: float = 0.25
    ) -> float:
        """
        Calculate Kelly fraction for binary outcome
        
        Kelly = (p * (b + 1) - 1) / b
        where:
        - p = true probability of winning
        - b = odds received on bet (decimal odds - 1)
        
        For prediction markets:
        - If buying YES at price p_market, odds = (1/p_market) - 1
        - Kelly = (p_true * (1/p_market) - 1) * p_market
        """
        
        if market_price <= 0 or market_price >= 1:
            return 0
        
        # Calculate edge
        edge = probability - market_price
        
        if edge <= 0:
            return 0  # No edge, no bet
        
        # Simplified Kelly for binary markets
        # Kelly = edge / (1 - market_price)
        kelly = edge / (1 - market_price)
        
        # Cap at max_fraction to reduce risk
        kelly_fraction = min(kelly, max_fraction)
        
        # Don't bet if edge is too small
        if kelly_fraction < 0.01:  # Less than 1%
            return 0
        
        return kelly_fraction
    
    @staticmethod
    def calculate_position_size(
        kelly_fraction: float,
        bankroll: float,
        max_position: float = 1000
    ) -> float:
        """Calculate dollar amount to bet"""
        
        position = kelly_fraction * bankroll
        
        # Apply position limits
        position = min(position, max_position)
        
        return round(position, 2)


class MarketScanner:
    """Efficiently scan 1000+ markets across platforms"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.polymarket_url = "https://gamma-api.polymarket.com"
        self.kalshi_url = "https://api.elections.kalshi.com/trade-api/v2"
    
    async def scan_all_markets(self) -> List[MarketData]:
        """Scan all markets from both platforms"""
        
        logger.info("🔍 Scanning markets across all platforms...")
        
        async with aiohttp.ClientSession() as session:
            # Fetch from both platforms in parallel
            poly_task = self._fetch_polymarket_markets(session)
            kalshi_task = self._fetch_kalshi_markets(session)
            
            poly_markets, kalshi_markets = await asyncio.gather(
                poly_task, kalshi_task
            )
        
        all_markets = poly_markets + kalshi_markets
        
        logger.info(f"✅ Found {len(all_markets)} total markets")
        logger.info(f"   Polymarket: {len(poly_markets)}")
        logger.info(f"   Kalshi: {len(kalshi_markets)}")
        
        return all_markets
    
    async def _fetch_polymarket_markets(
        self, 
        session: aiohttp.ClientSession
    ) -> List[MarketData]:
        """Fetch markets from Polymarket"""
        
        markets = []
        limit = 100  # API limit per request
        offset = 0
        target = self.config.get('polymarket', {}).get('max_markets', 500)
        
        try:
            while len(markets) < target:
                url = f"{self.polymarket_url}/markets"
                params = {
                    'limit': limit,
                    'offset': offset,
                    'active': 'true',
                    'closed': 'false'
                }
                
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    if not data:
                        break
                    
                    # Convert to MarketData objects
                    for m in data:
                        try:
                            outcomes = m.get('outcomes', [])
                            if len(outcomes) >= 2:
                                markets.append(MarketData(
                                    platform='polymarket',
                                    market_id=m.get('id', ''),
                                    title=m.get('question', ''),
                                    description=m.get('description', ''),
                                    yes_price=float(outcomes[0].get('price', 0)),
                                    no_price=float(outcomes[1].get('price', 0)),
                                    volume=float(m.get('volume', 0)),
                                    liquidity=float(m.get('liquidity', 0)),
                                    end_date=m.get('end_date_iso', ''),
                                    category=m.get('category', 'other')
                                ))
                        except Exception as e:
                            logger.debug(f"Error parsing Polymarket market: {e}")
                    
                    offset += limit
                    
                    # Respect rate limits
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error fetching Polymarket markets: {e}")
        
        return markets
    
    async def _fetch_kalshi_markets(
        self, 
        session: aiohttp.ClientSession
    ) -> List[MarketData]:
        """Fetch markets from Kalshi"""
        
        markets = []
        cursor = None
        target = self.config.get('kalshi', {}).get('max_markets', 500)
        
        try:
            while len(markets) < target:
                url = f"{self.kalshi_url}/markets"
                params = {
                    'limit': 100,
                    'status': 'open'
                }
                if cursor:
                    params['cursor'] = cursor
                
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
                    market_list = data.get('markets', [])
                    if not market_list:
                        break
                    
                    for m in market_list:
                        try:
                            # Get current price from orderbook
                            yes_price = await self._get_kalshi_price(
                                m.get('ticker'), session
                            )
                            
                            markets.append(MarketData(
                                platform='kalshi',
                                market_id=m.get('ticker', ''),
                                title=m.get('title', ''),
                                description=m.get('subtitle', ''),
                                yes_price=yes_price,
                                no_price=1 - yes_price,
                                volume=float(m.get('volume', 0)),
                                liquidity=float(m.get('open_interest', 0)),
                                end_date=m.get('expiration_time', ''),
                                category=m.get('category', 'other')
                            ))
                        except Exception as e:
                            logger.debug(f"Error parsing Kalshi market: {e}")
                    
                    cursor = data.get('cursor')
                    if not cursor:
                        break
                    
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error fetching Kalshi markets: {e}")
        
        return markets
    
    async def _get_kalshi_price(
        self, 
        ticker: str, 
        session: aiohttp.ClientSession
    ) -> float:
        """Get current YES price from Kalshi orderbook"""
        
        try:
            url = f"{self.kalshi_url}/markets/{ticker}/orderbook"
            async with session.get(url) as response:
                data = await response.json()
                
                orderbook = data.get('orderbook', {})
                yes_bids = orderbook.get('yes', [])
                
                if yes_bids:
                    # Best bid price in cents
                    return yes_bids[0][0] / 100
                
                return 0.5  # Default if no orderbook
                
        except:
            return 0.5


class AdvancedTradingBot:
    """Main orchestrator for AI-powered trading"""
    
    def __init__(self, config_file: str = 'advanced_config.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.scanner = MarketScanner(self.config)
        self.analyzer = ClaudeAnalyzer(self.config)
        self.kelly = KellyCriterion()
        
        # Trading state
        self.bankroll = self.config.get('trading', {}).get('initial_bankroll', 10000)
        self.current_positions = []
        self.trade_history = []
        
        # Performance tracking
        self.total_profit = 0
        self.total_trades = 0
        self.winning_trades = 0
    
    async def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        
        logger.info("=" * 80)
        logger.info("🤖 STARTING TRADING CYCLE")
        logger.info("=" * 80)
        
        cycle_start = time.time()
        
        # Step 1: Scan markets
        logger.info("\n📊 Step 1: Scanning markets...")
        markets = await self.scanner.scan_all_markets()
        
        # Step 2: Filter markets
        logger.info("\n🔬 Step 2: Filtering markets...")
        filtered = self._filter_markets(markets)
        logger.info(f"   {len(filtered)} markets passed filters")
        
        # Step 3: Analyze with Claude (in batches)
        logger.info("\n🧠 Step 3: Analyzing with Claude AI...")
        estimates = await self._analyze_markets_in_batches(filtered)
        
        # Step 4: Find mispricings >8%
        logger.info("\n💰 Step 4: Finding mispricings...")
        min_edge = self.config.get('strategy', {}).get('min_edge', 0.08)
        opportunities = self._find_opportunities(estimates, filtered, min_edge)
        logger.info(f"   Found {len(opportunities)} opportunities with >{min_edge*100:.0f}% edge")
        
        # Step 5: Calculate position sizes (Kelly)
        logger.info("\n📐 Step 5: Calculating position sizes (Kelly)...")
        signals = self._generate_trade_signals(opportunities)
        
        # Step 6: Execute trades
        logger.info("\n⚡ Step 6: Executing trades...")
        await self._execute_trades(signals)
        
        # Step 7: Pay API bills
        logger.info("\n💳 Step 7: Paying API bills...")
        self._pay_api_bills()
        
        # Summary
        cycle_time = time.time() - cycle_start
        self._print_cycle_summary(cycle_time)
        
    def _filter_markets(self, markets: List[MarketData]) -> List[MarketData]:
        """Filter markets by volume, liquidity, etc."""
        
        min_volume = self.config.get('filters', {}).get('min_volume', 1000)
        min_liquidity = self.config.get('filters', {}).get('min_liquidity', 500)
        
        filtered = [
            m for m in markets
            if m.volume >= min_volume and m.liquidity >= min_liquidity
        ]
        
        # Additional filters
        # Remove markets with extreme prices (likely already resolved)
        filtered = [
            m for m in filtered
            if 0.01 < m.yes_price < 0.99
        ]
        
        return filtered
    
    async def _analyze_markets_in_batches(
        self, 
        markets: List[MarketData]
    ) -> List[FairValueEstimate]:
        """Analyze markets in batches to manage API usage"""
        
        batch_size = self.config.get('api', {}).get('batch_size', 50)
        all_estimates = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(markets), batch_size):
                batch = markets[i:i+batch_size]
                logger.info(f"   Analyzing batch {i//batch_size + 1} ({len(batch)} markets)...")
                
                estimates = await self.analyzer.analyze_market_batch(batch, session)
                all_estimates.extend(estimates)
                
                # Small delay between batches
                await asyncio.sleep(1)
        
        return all_estimates
    
    def _find_opportunities(
        self,
        estimates: List[FairValueEstimate],
        markets: List[MarketData],
        min_edge: float
    ) -> List[Tuple[MarketData, FairValueEstimate]]:
        """Find markets with significant mispricings"""
        
        # Create lookup dict
        market_dict = {m.market_id: m for m in markets}
        
        opportunities = []
        
        for est in estimates:
            if abs(est.edge) >= min_edge and est.confidence_level >= 0.6:
                market = market_dict.get(est.market_id)
                if market:
                    opportunities.append((market, est))
        
        # Sort by edge * confidence
        opportunities.sort(
            key=lambda x: abs(x[1].edge) * x[1].confidence_level,
            reverse=True
        )
        
        return opportunities
    
    def _generate_trade_signals(
        self,
        opportunities: List[Tuple[MarketData, FairValueEstimate]]
    ) -> List[TradeSignal]:
        """Generate trade signals with Kelly sizing"""
        
        signals = []
        max_positions = self.config.get('risk', {}).get('max_positions', 10)
        
        for market, estimate in opportunities[:max_positions]:
            # Determine action
            if estimate.edge > 0:
                action = 'buy_yes'
                market_price = market.yes_price
            else:
                action = 'buy_no'
                market_price = market.no_price
            
            # Calculate Kelly fraction
            kelly_frac = self.kelly.calculate_kelly_fraction(
                probability=estimate.estimated_probability,
                market_price=market_price,
                max_fraction=self.config.get('risk', {}).get('max_kelly_fraction', 0.25)
            )
            
            if kelly_frac > 0:
                # Calculate position size
                position_size = self.kelly.calculate_position_size(
                    kelly_fraction=kelly_frac,
                    bankroll=self.bankroll,
                    max_position=self.config.get('risk', {}).get('max_position_size', 1000)
                )
                
                # Calculate expected value
                ev = (estimate.estimated_probability * (1 - market_price) - 
                      (1 - estimate.estimated_probability) * market_price) * position_size
                
                signals.append(TradeSignal(
                    market=market,
                    action=action,
                    fair_value=estimate.estimated_probability,
                    market_price=market_price,
                    edge=estimate.edge,
                    kelly_fraction=kelly_frac,
                    position_size=position_size,
                    expected_value=ev,
                    reasoning=estimate.reasoning[:200]
                ))
        
        return signals
    
    async def _execute_trades(self, signals: List[TradeSignal]):
        """Execute trade signals"""
        
        if not signals:
            logger.info("   No trades to execute")
            return
        
        for signal in signals:
            logger.info(f"\n   🎯 TRADE SIGNAL:")
            logger.info(f"      Market: {signal.market.title[:60]}...")
            logger.info(f"      Platform: {signal.market.platform}")
            logger.info(f"      Action: {signal.action}")
            logger.info(f"      Fair Value: {signal.fair_value:.1%}")
            logger.info(f"      Market Price: {signal.market_price:.1%}")
            logger.info(f"      Edge: {signal.edge:+.1%}")
            logger.info(f"      Kelly Fraction: {signal.kelly_fraction:.2%}")
            logger.info(f"      Position Size: ${signal.position_size:,.2f}")
            logger.info(f"      Expected Value: ${signal.expected_value:+,.2f}")
            logger.info(f"      Reasoning: {signal.reasoning}...")
            
            # Execute trade (placeholder - implement actual execution)
            success = await self._execute_single_trade(signal)
            
            if success:
                self.total_trades += 1
                self.current_positions.append(signal)
                logger.info(f"      ✅ Trade executed successfully")
            else:
                logger.warning(f"      ❌ Trade execution failed")
    
    async def _execute_single_trade(self, signal: TradeSignal) -> bool:
        """Execute individual trade (implement with actual API calls)"""
        
        # This is where you'd call the actual trading APIs
        # For now, just simulate
        
        logger.warning("      ⚠️  DRY RUN MODE - Not executing real trade")
        
        # TODO: Implement actual execution
        # if signal.market.platform == 'polymarket':
        #     return await self._execute_polymarket_trade(signal)
        # elif signal.market.platform == 'kalshi':
        #     return await self._execute_kalshi_trade(signal)
        
        return True  # Simulated success
    
    def _pay_api_bills(self):
        """Track and pay Claude API costs"""
        
        stats = self.analyzer.get_api_stats()
        
        logger.info(f"   API Requests: {stats['requests_made']}")
        logger.info(f"   Total API Cost: ${stats['total_cost']:.4f}")
        logger.info(f"   Avg Cost/Request: ${stats['avg_cost_per_request']:.6f}")
        
        # In a real system, you'd:
        # 1. Deduct from profits
        # 2. Top up API credits automatically
        # 3. Track against revenue
        
        # Deduct from bankroll
        self.bankroll -= stats['total_cost']
        
        logger.info(f"   💰 Updated bankroll: ${self.bankroll:,.2f}")
    
    def _print_cycle_summary(self, cycle_time: float):
        """Print summary of trading cycle"""
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 CYCLE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Duration: {cycle_time:.1f}s")
        logger.info(f"Current Bankroll: ${self.bankroll:,.2f}")
        logger.info(f"Active Positions: {len(self.current_positions)}")
        logger.info(f"Total Trades: {self.total_trades}")
        logger.info(f"Total Profit: ${self.total_profit:+,.2f}")
        
        if self.total_trades > 0:
            win_rate = self.winning_trades / self.total_trades
            logger.info(f"Win Rate: {win_rate:.1%}")
        
        logger.info("=" * 80 + "\n")


async def main():
    """Main entry point"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   AI-POWERED PREDICTION MARKET BOT                           ║
║                                                                              ║
║  ✨ Claude AI Fair Value Estimation                                         ║
║  📊 1000+ Market Scanner                                                    ║
║  📐 Kelly Criterion Position Sizing                                         ║
║  💰 Autonomous Trading & API Bill Payment                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    bot = AdvancedTradingBot('advanced_config.json')
    
    # Run once or continuously
    mode = input("\nRun mode:\n1. Single cycle\n2. Continuous (every 1 hour)\n\nChoice: ").strip()
    
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
                await asyncio.sleep(60)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
