"""OpenAI analyzer for estimating fair probabilities for markets."""

import asyncio
import logging
from typing import Dict, List, Optional

import aiohttp

from app.analytics import EloEngine
from app.api_clients.base_client import BaseAPIClient, APIError
from app.analysis.context_loader import load_context_json_block
from app.models import MarketData, FairValueEstimate
from app.trading.engine import calculate_adjusted_yes_probability
from app.utils import ClaudeResponseParser, ConfigManager

logger = logging.getLogger(__name__)


class OpenAIAnalyzer:
    """Uses OpenAI API to estimate fair values for prediction markets."""

    def __init__(self, config: ConfigManager):
        self.client = BaseAPIClient(
            platform_name="openai",
            api_key=config.openai_api_key,
            base_url=config.openai.base_url,
            input_cost_per_mtok=config.openai.input_cost_per_mtok,
            output_cost_per_mtok=config.openai.output_cost_per_mtok,
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

    def set_runtime_context_block(self, context_text: str) -> None:
        """Set optional runtime context block (e.g., prior analysis blurbs)."""
        self._runtime_context_block = str(context_text or "").strip()

    async def analyze_market_batch(
        self,
        markets: List[MarketData],
        session=None,
    ) -> List[FairValueEstimate]:
        """Analyze multiple markets in parallel."""

        if self._elo_enabled and self._elo_engine and not self._elo_ready:
            try:
                await asyncio.to_thread(self._elo_engine.load_ratings, True)
                self._elo_ready = True
            except Exception as e:
                logger.warning("NBA Elo initialization failed; disabling Elo path for this run: %s", e)
                self._elo_enabled = False

        async with self.client:
            tasks = [self.analyze_single_market(m) for m in markets]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = [r for r in results if isinstance(r, FairValueEstimate)]
        logger.info(f"✅ OpenAI analyzed {len(valid_results)}/{len(markets)} markets successfully")
        return valid_results

    async def analyze_single_market(self, market: MarketData) -> Optional[FairValueEstimate]:
        if not self.config.api.openai_api_key:
            logger.error("OpenAI API key not configured")
            return None

        nba_elo_context = self._get_nba_elo_context(market)
        if nba_elo_context:
            prompt = self._build_nba_elo_adjustment_prompt(market, nba_elo_context)
        else:
            prompt = self._build_analysis_prompt(market)

        try:
            response = await self._call_openai_api(prompt)

            if nba_elo_context:
                estimate = ClaudeResponseParser.parse_elo_adjusted_estimate(
                    response_text=response.get("content_text", ""),
                    market_id=market.market_id,
                    market_price=market.yes_price,
                    yes_team=str(nba_elo_context["yes_team"]),
                    home_team=str(nba_elo_context["home_team"]),
                    away_team=str(nba_elo_context["away_team"]),
                    home_elo=float(nba_elo_context["home_elo"]),
                    away_elo=float(nba_elo_context["away_elo"]),
                    home_court_bonus=float(nba_elo_context["home_court_bonus"]),
                )
                if estimate is None:
                    estimate = self._build_elo_fallback_estimate(market, nba_elo_context, "llm_parse_failed")
                else:
                    self._log_elo_decision_fields(market, estimate)
            else:
                estimate = ClaudeResponseParser.parse_fair_value_estimate(
                    response_text=response.get("content_text", ""),
                    market_id=market.market_id,
                    market_price=market.yes_price,
                )

            if estimate:
                logger.debug(
                    f"📊 {market.title[:50]}... → {estimate.estimated_probability:.1%} "
                    f"(edge: {estimate.edge:+.1%})"
                )

            return estimate

        except Exception as e:
            logger.error(f"Error analyzing market {market.market_id} with OpenAI: {e}")
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
  \"probability\": <float 0-100>,
  \"confidence\": <float 0-100>,
  \"reasoning\": \"<detailed explanation>\",
  \"key_factors\": [\"factor1\", \"factor2\", ...],
  \"data_sources\": [\"source1\", \"source2\", ...]
}}

Return ONLY valid JSON."""

    def _build_nba_elo_adjustment_prompt(self, market: MarketData, elo_ctx: Dict[str, object]) -> str:
        context_section = self._build_context_section()
        max_elo_delta = 75
        return f"""You are assisting an NBA trading model.

IMPORTANT:
- Elo is the PRIMARY probability source.
- Do NOT generate a probability from scratch.
- You may only suggest an Elo adjustment for the YES team.
- Keep elo_delta in [-{max_elo_delta}, +{max_elo_delta}] unless there is concrete, high-confidence evidence.

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
  "elo_delta": <integer in [-{max_elo_delta}, +{max_elo_delta}]>,
  "confidence": <float 0-1 or 0-100>,
  "reason": "<detailed explanation>",
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

    def _get_nba_elo_context(self, market: MarketData) -> Optional[Dict[str, object]]:
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
                "yes_probability": float(yes_probability),
                "home_elo": float(home_elo),
                "away_elo": float(away_elo),
                "home_team": matchup.home_team,
                "away_team": matchup.away_team,
                "yes_team": matchup.yes_team,
                "home_court_bonus": float(self._elo_engine.home_advantage),
                "elo_edge": float(yes_probability - market.yes_price),
            }
        except Exception as e:
            logger.debug("NBA Elo context unavailable for %s: %s", market.market_id, e)
            return None

    @staticmethod
    def _build_elo_fallback_estimate(
        market: MarketData,
        elo_ctx: Dict[str, object],
        reason: str,
    ) -> FairValueEstimate:
        base_prob = float(elo_ctx["yes_probability"])
        edge = base_prob - float(market.yes_price)
        adjusted = calculate_adjusted_yes_probability(
            yes_team=str(elo_ctx["yes_team"]),
            home_team=str(elo_ctx["home_team"]),
            away_team=str(elo_ctx["away_team"]),
            home_elo=float(elo_ctx["home_elo"]),
            away_elo=float(elo_ctx["away_elo"]),
            llm_elo_delta=0,
            home_court_bonus=float(elo_ctx["home_court_bonus"]),
        )
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
            fusion_metadata={
                "elo_adjustment": {
                    "yes_team": str(elo_ctx["yes_team"]),
                    "home_team": str(elo_ctx["home_team"]),
                    "away_team": str(elo_ctx["away_team"]),
                    "home_elo": float(elo_ctx["home_elo"]),
                    "away_elo": float(elo_ctx["away_elo"]),
                    "base_probability": float(base_prob),
                    "applied_elo_delta": 0.0,
                    "yes_adjusted_elo": float(adjusted["yes_adjusted_elo"]),
                    "yes_effective_elo": float(adjusted["yes_effective_elo"]),
                    "opponent_effective_elo": float(adjusted["opponent_effective_elo"]),
                    "final_probability": float(base_prob),
                    "market_probability": float(market.yes_price),
                    "edge": float(edge),
                }
            },
        )

    def _log_elo_decision_fields(self, market: MarketData, estimate: FairValueEstimate) -> None:
        meta = (estimate.fusion_metadata or {}).get("elo_adjustment", {})
        if not isinstance(meta, dict):
            return
        try:
            logger.info(
                "ELO_DECISION | team=%s | opponent=%s | elo_base=%.2f | elo_delta=%+.0f | "
                "elo_adjusted=%.2f | probability_base=%.4f | probability_final=%.4f | "
                "market_probability=%.4f | edge=%+.4f",
                str(meta.get("yes_team") or ""),
                str(meta.get("away_team") if str(meta.get("yes_team")) == str(meta.get("home_team")) else meta.get("home_team")),
                float(meta.get("home_elo") if str(meta.get("yes_team")) == str(meta.get("home_team")) else meta.get("away_elo")),
                float(meta.get("applied_elo_delta", 0.0)),
                float(meta.get("yes_adjusted_elo", 0.0)),
                float(meta.get("base_probability", 0.0)),
                float(meta.get("final_probability", estimate.estimated_probability)),
                float(meta.get("market_probability", market.yes_price)),
                float(meta.get("edge", estimate.edge)),
            )
        except Exception:
            logger.debug("Failed logging Elo decision metadata for %s", market.market_id)

    async def _call_openai_api(self, prompt: str) -> dict:
        logger.debug("Calling OpenAI API with prompt:\n" + prompt)
        """Call OpenAI Chat Completions API."""

        if not self.client.session:
            raise RuntimeError("OpenAI client session not initialized")

        payload = {
            "model": self.config.openai.model,
            "messages": [
                {"role": "system", "content": "You are a careful forecaster. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.openai.temperature,
            "max_tokens": self.config.openai.max_tokens,
            "response_format": {"type": "json_object"},
        }

        url = f"{self.client.base_url}/chat/completions"
        headers = self.client._build_headers()

        async def make_request():
            async with self.client.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.client.timeout_seconds),
            ) as response:
                await self.client._handle_response_status(response)
                return await response.json()

        try:
            response = await self.client._call_with_retry(make_request, operation_name="OpenAI analysis")

            usage = response.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            if prompt_tokens or completion_tokens:
                self.client.record_usage(
                    operation="Market analysis",
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                )

            choices = response.get("choices") or []
            if not choices:
                raise APIError(platform="openai", operation="Parse response", message="Empty choices array")

            content = (choices[0].get("message") or {}).get("content") or ""
            if not content:
                raise APIError(platform="openai", operation="Parse response", message="Empty message content")

            return {"content_text": content, "usage": usage}

        except APIError:
            raise
        except Exception as e:
            raise APIError(platform="openai", operation="API call", message=str(e))

    def get_api_stats(self) -> dict:
        return self.client.get_cost_stats()
