import logging
import asyncio
from typing import Dict, List, Optional

import aiohttp

from app.analytics import EloEngine
from app.config import ConfigManager
from app.models import MarketData, FairValueEstimate
from app.api_clients.base_client import BaseAPIClient
from app.analysis.context_loader import load_context_json_block
from app.utils import ClaudeResponseParser
from app.api_clients import APIError

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
        self.client = BaseAPIClient(
            platform_name="claude",
            api_key=config.claude_api_key,
            base_url="https://api.anthropic.com/v1",
            input_cost_per_mtok=config.claude.input_cost_per_mtok,
            output_cost_per_mtok=config.claude.output_cost_per_mtok,
        )
        self.config = config
        self._context_json_block = load_context_json_block(config)
        self._runtime_context_block = ""
        self._elo_enabled = bool(getattr(self.config.analysis, "nba_elo_enabled", True))
        self._elo_ready = False
        self._elo_engine: Optional[EloEngine] = None
        if self._elo_enabled:
            self._elo_engine = EloEngine(
                data_path=str(getattr(self.config.analysis, "nba_elo_data_path", "context/kaggleGameData.csv")),
                output_path=str(getattr(self.config.analysis, "nba_elo_output_path", "app/outputs/elo_ratings.json")),
            )
        self._llm_adjustment_max_delta = float(
            getattr(self.config.analysis, "llm_adjustment_max_delta", 0.03)
        )

    def set_runtime_context_block(self, context_text: str) -> None:
        """Set optional runtime context block (e.g., prior analysis blurbs)."""
        self._runtime_context_block = str(context_text or "").strip()

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
        if self._elo_enabled and self._elo_engine and not self._elo_ready:
            try:
                await asyncio.to_thread(self._elo_engine.load_ratings, True)
                self._elo_ready = True
            except Exception as e:
                logger.warning("NBA Elo initialization failed; disabling Elo path for this run: %s", e)
                self._elo_enabled = False

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
        nba_elo_context = self._get_nba_elo_context(market)
        if nba_elo_context:
            prompt = self._build_nba_elo_adjustment_prompt(market, nba_elo_context)
        else:
            prompt = self._build_analysis_prompt(market)

        try:
            # Call Claude API
            response = await self._call_claude_api(prompt)

            # Parse response into FairValueEstimate
            if nba_elo_context:
                estimate = ClaudeResponseParser.parse_elo_adjusted_estimate(
                    response_text=response.get('content_text', ''),
                    market_id=market.market_id,
                    market_price=market.yes_price,
                    base_probability=float(nba_elo_context["yes_probability"]),
                    max_delta=self._llm_adjustment_max_delta,
                )
                if estimate is None:
                    estimate = self._build_elo_fallback_estimate(market, nba_elo_context, "llm_parse_failed")
            else:
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
            if nba_elo_context:
                return self._build_elo_fallback_estimate(market, nba_elo_context, f"llm_api_error: {e}")
            return None

    def _build_context_section(self) -> str:
        context_blocks = []
        if self._context_json_block:
            context_blocks.append(
                f"""
ADDITIONAL USER CONTEXT (local JSON file):
{self._context_json_block}

Use this context as supplemental evidence. If it conflicts with market data or is stale, explain that in reasoning."""
            )
        if self._runtime_context_block:
            context_blocks.append(
                f"""
RECENT INTERNAL ANALYSIS CONTEXT (historical, from prior runs):
{self._runtime_context_block}

Treat this as historical signal memory. It can be stale; verify against current market state before relying on it."""
            )
        return "".join(context_blocks)

    def _build_analysis_prompt(self, market: MarketData) -> str:
        """Construct prompt for Claude to analyze market"""
        context_section = self._build_context_section()

        return f"""Analyze this prediction market and estimate the TRUE probability of the outcome.

MARKET DETAILS:
Title: {market.title}
Description: {market.description}
Current Market Price: {market.yes_price:.1%} for YES
Volume: ${market.volume:,.0f}
Category: {market.category}
Closes: {market.end_date}
{context_section}

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

    def _build_nba_elo_adjustment_prompt(self, market: MarketData, elo_ctx: Dict[str, str]) -> str:
        context_section = self._build_context_section()
        max_delta = abs(self._llm_adjustment_max_delta)
        return f"""You are assisting an NBA trading model.

IMPORTANT:
- Elo is the PRIMARY probability source.
- Do NOT generate a probability from scratch.
- You may only suggest a SMALL adjustment delta to Elo YES probability.
- Keep delta in [-{max_delta:.4f}, +{max_delta:.4f}] unless there is concrete, high-confidence evidence.

MARKET DETAILS:
Title: {market.title}
Description: {market.description}
Current Market Price (YES): {market.yes_price:.1%}
YES Selection: {elo_ctx["yes_team"]}
Matchup: {elo_ctx["away_team"]} (away) at {elo_ctx["home_team"]} (home)
Elo(away): {elo_ctx["away_elo"]}
Elo(home): {elo_ctx["home_elo"]}
Elo baseline YES probability: {elo_ctx["yes_probability"]}
Edge from Elo baseline: {elo_ctx["elo_edge"]}
{context_section}

Adjust only for concrete factors such as:
- injuries / confirmed absences
- rest days / back-to-back
- lineup or rotation changes

Respond in JSON format:
{{
  "delta": <float in [-{max_delta:.4f}, +{max_delta:.4f}]>,
  "confidence": <float 0-100>,
  "reasoning": "<detailed explanation>",
  "key_factors": ["factor1", "factor2", ...],
  "data_sources": ["source1", "source2", ...]
}}

Return ONLY valid JSON."""

    def _is_nba_game_market(self, market: MarketData) -> bool:
        if str(market.platform).strip().lower() != "kalshi":
            return False
        series = str(getattr(market, "series_ticker", "") or "").strip().upper()
        market_id = str(getattr(market, "market_id", "") or "").strip().upper()
        return series == "KXNBAGAME" or market_id.startswith("KXNBAGAME-")

    def _get_nba_elo_context(self, market: MarketData) -> Optional[Dict[str, str]]:
        if not self._elo_enabled or not self._elo_engine:
            return None
        if not self._is_nba_game_market(market):
            return None
        try:
            matchup = self._elo_engine.parse_kalshi_nba_matchup(market.market_id)
            if matchup is None:
                return None
            yes_probability = self._elo_engine.get_market_yes_probability(market.market_id)
            if yes_probability is None:
                return None
            home_elo = self._elo_engine.ratings.get(matchup.home_team)
            away_elo = self._elo_engine.ratings.get(matchup.away_team)
            if home_elo is None or away_elo is None:
                return None
            return {
                "yes_probability": f"{float(yes_probability):.6f}",
                "home_elo": f"{float(home_elo):.2f}",
                "away_elo": f"{float(away_elo):.2f}",
                "home_team": matchup.home_team,
                "away_team": matchup.away_team,
                "yes_team": matchup.yes_team,
                "elo_edge": f"{float(yes_probability - market.yes_price):+.6f}",
            }
        except Exception as e:
            logger.debug("NBA Elo context unavailable for %s: %s", market.market_id, e)
            return None

    @staticmethod
    def _build_elo_fallback_estimate(
        market: MarketData,
        elo_ctx: Dict[str, str],
        reason: str,
    ) -> FairValueEstimate:
        base_prob = float(elo_ctx["yes_probability"])
        edge = base_prob - float(market.yes_price)
        return FairValueEstimate(
            market_id=market.market_id,
            estimated_probability=base_prob,
            confidence_level=0.60,
            reasoning=(
                f"Elo-only fallback ({reason}). "
                f"base_probability={base_prob:.3f}, edge={edge:+.3f}."
            ),
            data_sources=["nba_elo"],
            key_factors=["elo_baseline_only"],
            edge=edge,
        )

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
