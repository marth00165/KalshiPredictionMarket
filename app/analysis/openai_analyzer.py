"""OpenAI analyzer for estimating fair probabilities for markets."""

import asyncio
import logging
from typing import Dict, List, Optional

import aiohttp

from app.analytics import EloEngine
from app.analytics.elo_calibration import (
    EloCalibrationConfig,
    build_calibration_table,
    load_matchups_csv,
)
from app.api_clients.base_client import BaseAPIClient, APIError
from app.analysis.context_loader import load_context_json_block
from app.analysis.injury_llm_cache import (
    InjuryLLMRefreshConfig,
    InjuryLLMRefreshService,
    MarketInjuryContext,
)
from app.analysis.nba_elo_shared import (
    apply_elo_calibration as apply_shared_elo_calibration,
    build_elo_estimate_from_delta as build_shared_elo_estimate_from_delta,
    build_elo_fallback_estimate as build_shared_elo_fallback_estimate,
    get_nba_elo_context as get_shared_nba_elo_context,
    is_nba_game_market as is_shared_nba_game_market,
    log_elo_decision_fields as log_shared_elo_decision_fields,
)
from app.models import MarketData, FairValueEstimate
from app.trading.engine import calculate_pim_elo_adjustment
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
                initial_rating=float(getattr(self.config.analysis, "nba_elo_initial_rating", 1500.0)),
                home_advantage=float(getattr(self.config.analysis, "nba_elo_home_advantage", 100.0)),
                k_factor=float(getattr(self.config.analysis, "nba_elo_k_factor", 20.0)),
                regression_factor=float(getattr(self.config.analysis, "nba_elo_regression_factor", 0.75)),
                use_mov_multiplier=bool(getattr(self.config.analysis, "nba_elo_use_mov_multiplier", True)),
                elo_round_decimals=int(getattr(self.config.analysis, "nba_elo_round_decimals", 1)),
                min_season=getattr(self.config.analysis, "nba_elo_min_season", 2004),
                allowed_seasons=getattr(self.config.analysis, "nba_elo_allowed_seasons", None),
                season_ratings_output_path=str(
                    getattr(
                        self.config.analysis,
                        "nba_elo_season_ratings_output_path",
                        "app/outputs/elo_ratings_by_season.csv",
                    )
                ),
            )
        self._elo_calibration_enabled = bool(
            self._elo_enabled and getattr(self.config.analysis, "enable_elo_calibration", True)
        )
        self._elo_calibration_table = None
        self._elo_calibration_config = EloCalibrationConfig(
            bucket_size=int(getattr(self.config.analysis, "calibration_bucket_size", 25)),
            prior=int(getattr(self.config.analysis, "calibration_prior", 100)),
            key_type="elo_diff",
            min_season=getattr(self.config.analysis, "calibration_min_season", 2004),
            recency_mode=str(getattr(self.config.analysis, "calibration_recency_mode", "exp")),
            recency_halflife_days=int(
                getattr(self.config.analysis, "calibration_recency_halflife_days", 730)
            ),
        )
        self._injury_refresh = InjuryLLMRefreshService(
            config=InjuryLLMRefreshConfig(
                enabled=bool(getattr(self.config.analysis, "enable_live_injury_news", True)),
                enable_injury_llm_cache=bool(
                    getattr(self.config.analysis, "enable_injury_llm_cache", True)
                ),
                injury_cache_file=str(
                    getattr(self.config.analysis, "injury_cache_file", "context/injury_llm_cache.json")
                ),
                llm_refresh_max_age_seconds=int(
                    getattr(self.config.analysis, "llm_refresh_max_age_seconds", 1800)
                ),
                force_llm_refresh_near_tipoff_minutes=int(
                    getattr(self.config.analysis, "force_llm_refresh_near_tipoff_minutes", 45)
                ),
                near_tipoff_llm_stale_seconds=int(
                    getattr(self.config.analysis, "near_tipoff_llm_stale_seconds", 600)
                ),
                llm_refresh_on_price_move_pct=float(
                    getattr(self.config.analysis, "llm_refresh_on_price_move_pct", 0.03)
                ),
                injury_analysis_version=str(
                    getattr(self.config.analysis, "injury_analysis_version", "injury-v2")
                ),
                injury_prompt_version=str(
                    getattr(self.config.analysis, "injury_prompt_version", "injury-prompt-v1")
                ),
                force_injury_llm_refresh=bool(
                    getattr(self.config.analysis, "force_injury_llm_refresh", False)
                ),
                injury_feed_cache_ttl_seconds=int(
                    getattr(self.config.analysis, "injury_feed_cache_ttl_seconds", 120)
                ),
            ),
            sportsradar_api_key=getattr(getattr(self.config, "api", None), "sportradar_api_key", None),
            sportsradar_base_url=str(
                getattr(getattr(self.config, "api", None), "sportradar_base_url", "https://api.sportradar.com/nba/trial/v8/en")
            ),
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

        if self._elo_enabled and self._elo_calibration_enabled and self._elo_calibration_table is None:
            await asyncio.to_thread(self._ensure_elo_calibration_loaded)
        if self._elo_enabled and self._injury_refresh.enabled:
            await self._injury_refresh.refresh_injury_feed_if_needed()

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
        injury_ctx: Optional[MarketInjuryContext] = None
        if nba_elo_context:
            injury_ctx = self._injury_refresh.build_context_for_market(
                matchup_key=market.market_id,
                home_team=str(nba_elo_context["home_team"]),
                away_team=str(nba_elo_context["away_team"]),
                end_date=market.end_date,
                market_yes_price=float(market.yes_price),
            )
            if not injury_ctx.decision.should_refresh:
                estimate = self._build_cached_delta_estimate(market, nba_elo_context, injury_ctx)
                self._attach_injury_cache_metadata(
                    estimate=estimate,
                    injury_ctx=injury_ctx,
                    llm_source_tag="cached",
                    refresh_reason=injury_ctx.decision.reason,
                )
                self._injury_refresh.touch_market_price(
                    context=injury_ctx,
                    market_yes_price=float(market.yes_price),
                )
                self._apply_elo_calibration(market, estimate, nba_elo_context)
                self._log_elo_decision_fields(market, estimate)
                return estimate
            try:
                estimate = await self._build_pim_delta_estimate(market, nba_elo_context, injury_ctx)
                self._attach_injury_cache_metadata(
                    estimate=estimate,
                    injury_ctx=injury_ctx,
                    llm_source_tag="pim_refresh",
                    refresh_reason=injury_ctx.decision.reason,
                )
                self._persist_injury_cache(
                    market=market,
                    estimate=estimate,
                    injury_ctx=injury_ctx,
                )
                self._apply_elo_calibration(market, estimate, nba_elo_context)
                self._log_elo_decision_fields(market, estimate)
                return estimate
            except Exception as e:
                logger.error("PIM injury adjustment failed for %s: %s", market.market_id, e)
                if injury_ctx is not None and isinstance(injury_ctx.cache_entry, dict):
                    estimate = self._build_cached_delta_estimate(market, nba_elo_context, injury_ctx)
                    self._attach_injury_cache_metadata(
                        estimate=estimate,
                        injury_ctx=injury_ctx,
                        llm_source_tag="cached",
                        refresh_reason="pim_error_cache_reuse",
                    )
                    self._apply_elo_calibration(market, estimate, nba_elo_context)
                    self._log_elo_decision_fields(market, estimate)
                    return estimate
                estimate = self._build_elo_fallback_estimate(market, nba_elo_context, f"pim_error: {e}")
                self._attach_injury_cache_metadata(
                    estimate=estimate,
                    injury_ctx=injury_ctx,
                    llm_source_tag="pim_refresh",
                    refresh_reason="pim_error",
                )
                self._apply_elo_calibration(market, estimate, nba_elo_context)
                self._log_elo_decision_fields(market, estimate)
                return estimate

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

    async def _build_pim_delta_estimate(
        self,
        market: MarketData,
        elo_ctx: Dict[str, object],
        injury_ctx: MarketInjuryContext,
    ) -> FairValueEstimate:
        pim_inputs = await self._injury_refresh.build_pim_inputs_for_market(
            context=injury_ctx,
            yes_team=str(elo_ctx["yes_team"]),
        )
        pim_summary = calculate_pim_elo_adjustment(
            yes_team_players=pim_inputs.get("yes_team_players", []),
            opp_team_players=pim_inputs.get("opp_team_players", []),
            injury_status_map=pim_inputs.get("injury_status_map", {}),
            k_factor=float(getattr(self.config.analysis, "pim_k_factor", 25.0)),
            max_delta=float(getattr(self.config.analysis, "pim_max_delta", 75.0)),
        )
        delta_pim = float(pim_summary.get("delta_pim", 0.0))
        considered_players = pim_inputs.get("considered_players", [])
        reason = (
            "Deterministic PIM injury adjustment from SportsRadar team/player profiles."
            f" impacted_players={len(considered_players)}."
        )
        estimate = self._build_elo_estimate_from_delta(
            market=market,
            elo_ctx=elo_ctx,
            elo_delta=delta_pim,
            confidence=0.70,
            reason=reason,
            source_tag="pim_refresh",
            llm_suggestion={
                "raw_elo_delta": delta_pim,
                "applied_elo_delta": delta_pim,
                "confidence": 0.70,
                "reason": "deterministic_pim_adjustment",
                "key_factors": ["player_impact_metric"],
                "data_sources": ["sportradar_team_profile", "sportradar_player_profile"],
                "injury_report": {
                    "status": "confirmed",
                    "impact": "deterministic",
                    "notes": f"considered_players={len(considered_players)}",
                },
            },
        )
        metadata = dict(estimate.fusion_metadata or {})
        elo_meta = metadata.get("elo_adjustment")
        if not isinstance(elo_meta, dict):
            elo_meta = {}
        elo_meta["player_impact"] = {
            "yes_team_impact": float(pim_summary.get("yes_team_impact", 0.0)),
            "opp_team_impact": float(pim_summary.get("opp_team_impact", 0.0)),
            "delta_pim": delta_pim,
        }
        elo_meta["pim_considered_players"] = considered_players
        metadata["elo_adjustment"] = elo_meta
        estimate.fusion_metadata = metadata
        return estimate

    def _build_cached_delta_estimate(
        self,
        market: MarketData,
        elo_ctx: Dict[str, object],
        injury_ctx: MarketInjuryContext,
    ) -> FairValueEstimate:
        cache_entry = injury_ctx.cache_entry or {}
        try:
            cached_delta = float(cache_entry.get("llm_delta", 0.0))
        except Exception:
            cached_delta = 0.0
        try:
            cached_confidence = float(cache_entry.get("llm_confidence", 0.55))
        except Exception:
            cached_confidence = 0.55
        estimate = self._build_elo_estimate_from_delta(
            market=market,
            elo_ctx=elo_ctx,
            elo_delta=cached_delta,
            confidence=cached_confidence,
            reason=f"Reused cached injury-adjusted Elo delta ({injury_ctx.decision.reason}).",
            source_tag="cached",
            llm_suggestion={
                "raw_elo_delta": cache_entry.get("llm_delta", cached_delta),
                "applied_elo_delta": cached_delta,
                "confidence": cached_confidence,
                "reason": "cached_injury_delta_reuse",
                "key_factors": ["injury_cache_reuse"],
                "data_sources": ["injury_delta_cache"],
                "injury_report": {
                    "status": "confirmed",
                    "impact": "unknown",
                    "notes": "Cached injury snapshot unchanged.",
                },
            },
        )
        player_impact = cache_entry.get("player_impact")
        if isinstance(player_impact, dict):
            metadata = dict(estimate.fusion_metadata or {})
            elo_meta = metadata.get("elo_adjustment")
            if isinstance(elo_meta, dict):
                elo_meta["player_impact"] = player_impact
                metadata["elo_adjustment"] = elo_meta
                estimate.fusion_metadata = metadata
        return estimate

    def _attach_injury_cache_metadata(
        self,
        *,
        estimate: FairValueEstimate,
        injury_ctx: Optional[MarketInjuryContext],
        llm_source_tag: str,
        refresh_reason: str,
    ) -> None:
        if injury_ctx is None:
            return
        metadata = dict(estimate.fusion_metadata or {})
        elo_meta = metadata.get("elo_adjustment")
        if not isinstance(elo_meta, dict):
            return
        age_seconds = injury_ctx.decision.metadata.get("cached_delta_age_seconds")
        elo_meta["injury_cache_hit"] = bool(llm_source_tag == "cached")
        elo_meta["injury_fingerprint_changed"] = bool(refresh_reason == "fingerprint_changed")
        elo_meta["llm_refresh_reason"] = str(refresh_reason)
        elo_meta["llm_source_tag"] = str(llm_source_tag)
        elo_meta["cached_delta_age_seconds"] = (
            None if age_seconds is None else float(age_seconds)
        )
        elo_meta["injury_fingerprint"] = injury_ctx.fingerprint_short
        elo_meta["analysis_version"] = self._injury_refresh.config.injury_analysis_version
        elo_meta["prompt_version"] = self._injury_refresh.config.injury_prompt_version
        metadata["elo_adjustment"] = elo_meta
        estimate.fusion_metadata = metadata

    def _persist_injury_cache(
        self,
        *,
        market: MarketData,
        estimate: FairValueEstimate,
        injury_ctx: Optional[MarketInjuryContext],
    ) -> None:
        if injury_ctx is None:
            return
        metadata = dict(estimate.fusion_metadata or {})
        elo_meta = metadata.get("elo_adjustment")
        if not isinstance(elo_meta, dict):
            return
        llm_suggestion = elo_meta.get("llm_suggestion", {})
        if not isinstance(llm_suggestion, dict):
            llm_suggestion = {}
        try:
            llm_delta = float(elo_meta.get("applied_elo_delta", 0.0))
        except Exception:
            llm_delta = 0.0
        llm_confidence = llm_suggestion.get("confidence", estimate.confidence_level)
        try:
            llm_confidence = float(llm_confidence)
        except Exception:
            llm_confidence = None
        player_impact = elo_meta.get("player_impact")
        if not isinstance(player_impact, dict):
            player_impact = None
        self._injury_refresh.persist_result(
            context=injury_ctx,
            llm_delta=llm_delta,
            llm_confidence=llm_confidence,
            llm_model=getattr(self.config.openai, "model", None),
            prompt_version=self._injury_refresh.config.injury_prompt_version,
            analysis_version=self._injury_refresh.config.injury_analysis_version,
            source_tag="pim_refresh",
            market_yes_price=float(market.yes_price),
            player_impact=player_impact,
        )

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

    def _is_nba_game_market(self, market: MarketData) -> bool:
        return is_shared_nba_game_market(market)

    def _get_nba_elo_context(self, market: MarketData) -> Optional[Dict[str, object]]:
        return get_shared_nba_elo_context(
            market=market,
            elo_enabled=self._elo_enabled,
            elo_engine=self._elo_engine,
            logger=logger,
        )

    def _ensure_elo_calibration_loaded(self) -> None:
        if not self._elo_calibration_enabled or self._elo_calibration_table is not None:
            return

        calibration_path = str(
            getattr(self.config.analysis, "calibration_csv_path", "context/historical_elo_matchups.csv")
        )
        try:
            raw_df = load_matchups_csv(calibration_path)
            self._elo_calibration_table = build_calibration_table(
                raw_df,
                self._elo_calibration_config,
            )
            logger.info(
                "Loaded Elo calibration table from %s (%d buckets)",
                calibration_path,
                0 if self._elo_calibration_table is None else len(self._elo_calibration_table),
            )
        except Exception as e:
            self._elo_calibration_enabled = False
            logger.warning("Failed to load Elo calibration table from %s: %s", calibration_path, e)

    def _apply_elo_calibration(
        self,
        market: MarketData,
        estimate: FairValueEstimate,
        elo_ctx: Dict[str, object],
    ) -> None:
        if not self._elo_calibration_enabled:
            return
        if self._elo_calibration_table is None:
            self._ensure_elo_calibration_loaded()
        apply_shared_elo_calibration(
            market=market,
            estimate=estimate,
            elo_ctx=elo_ctx,
            calibration_table=self._elo_calibration_table,
            calibration_config=self._elo_calibration_config,
        )

    @staticmethod
    def _build_elo_estimate_from_delta(
        market: MarketData,
        elo_ctx: Dict[str, object],
        *,
        elo_delta: float,
        confidence: float,
        reason: str,
        source_tag: str,
        llm_suggestion: Optional[Dict[str, object]] = None,
    ) -> FairValueEstimate:
        return build_shared_elo_estimate_from_delta(
            market=market,
            elo_ctx=elo_ctx,
            elo_delta=elo_delta,
            confidence=confidence,
            reason=reason,
            source_tag=source_tag,
            llm_suggestion=llm_suggestion,
        )

    @staticmethod
    def _build_elo_fallback_estimate(
        market: MarketData,
        elo_ctx: Dict[str, object],
        reason: str,
    ) -> FairValueEstimate:
        return build_shared_elo_fallback_estimate(
            market=market,
            elo_ctx=elo_ctx,
            reason=reason,
            source_tag="llm_refresh",
        )

    def _log_elo_decision_fields(self, market: MarketData, estimate: FairValueEstimate) -> None:
        log_shared_elo_decision_fields(
            logger=logger,
            market=market,
            estimate=estimate,
        )

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
