"""OpenAI analyzer for estimating fair probabilities for markets."""

import asyncio
import logging
from typing import List, Optional

import aiohttp

from api_clients.base_client import BaseAPIClient, APIError
from models import MarketData, FairValueEstimate
from utils import ClaudeResponseParser, ConfigManager

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

    async def analyze_market_batch(
        self,
        markets: List[MarketData],
        session=None,
    ) -> List[FairValueEstimate]:
        """Analyze multiple markets in parallel."""

        async with self.client:
            tasks = [self.analyze_single_market(m) for m in markets]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = [r for r in results if isinstance(r, FairValueEstimate)]
        logger.info(f"✅ OpenAI analyzed {len(valid_results)}/{len(markets)} markets successfully")
        return valid_results

    async def analyze_single_market(self, market: MarketData) -> Optional[FairValueEstimate]:
        prompt = self._build_analysis_prompt(market)

        try:
            response = await self._call_openai_api(prompt)

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
            return None

    def _build_analysis_prompt(self, market: MarketData) -> str:
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
  \"probability\": <float 0-100>,
  \"confidence\": <float 0-100>,
  \"reasoning\": \"<detailed explanation>\",
  \"key_factors\": [\"factor1\", \"factor2\", ...],
  \"data_sources\": [\"source1\", \"source2\", ...]
}}

Return ONLY valid JSON."""

    async def _call_openai_api(self, prompt: str) -> dict:
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
