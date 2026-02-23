import logging
import asyncio
from typing import Dict, List, Optional

import aiohttp

from app.analytics import EloEngine
from app.analytics.elo_calibration import (
    EloCalibrationConfig,
    blend_probabilities,
    build_calibration_table,
    load_matchups_csv,
    lookup_empirical_rate,
)
from app.config import ConfigManager
from app.models import MarketData, FairValueEstimate
from app.api_clients.base_client import BaseAPIClient
from app.analysis.context_loader import load_context_json_block
from app.analysis.injury_llm_cache import (
    InjuryLLMRefreshConfig,
    InjuryLLMRefreshService,
    MarketInjuryContext,
)
from app.trading.engine import calculate_adjusted_yes_probability, calculate_pim_elo_adjustment
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
        self._elo_calibration_enabled = bool(
            self._elo_enabled and getattr(self.config.analysis, "enable_elo_calibration", True)
        )
        self._elo_calibration_table = None
        self._elo_calibration_config = EloCalibrationConfig(
            bucket_size=int(getattr(self.config.analysis, "calibration_bucket_size", 25)),
            prior=int(getattr(self.config.analysis, "calibration_prior", 100)),
            key_type="elo_diff",
            min_season=getattr(self.config.analysis, "calibration_min_season", None),
            recency_mode=str(getattr(self.config.analysis, "calibration_recency_mode", "none")),
            recency_halflife_days=int(
                getattr(self.config.analysis, "calibration_recency_halflife_days", 365)
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

        if self._elo_enabled and self._elo_calibration_enabled and self._elo_calibration_table is None:
            await asyncio.to_thread(self._ensure_elo_calibration_loaded)
        if self._elo_enabled and self._injury_refresh.enabled:
            await self._injury_refresh.refresh_injury_feed_if_needed()

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
            llm_model=getattr(self.config.claude, "model", None),
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
        if self._elo_calibration_table is None:
            return

        metadata = dict(estimate.fusion_metadata or {})
        elo_meta = metadata.get("elo_adjustment")
        if not isinstance(elo_meta, dict):
            return

        yes_team = str(elo_ctx.get("yes_team", "")).strip().upper()
        home_team = str(elo_ctx.get("home_team", "")).strip().upper()
        away_team = str(elo_ctx.get("away_team", "")).strip().upper()
        if yes_team not in {home_team, away_team}:
            return

        is_home = 1 if yes_team == home_team else 0
        home_elo = float(elo_ctx.get("home_elo", 0.0))
        away_elo = float(elo_ctx.get("away_elo", 0.0))
        yes_base_elo = home_elo if is_home else away_elo
        opponent_base_elo = away_elo if is_home else home_elo
        applied_delta = float(elo_meta.get("applied_elo_delta", 0.0))
        yes_adjusted_elo = yes_base_elo + applied_delta
        elo_difference = yes_adjusted_elo - opponent_base_elo

        p_elo = float(estimate.estimated_probability)
        p_emp, n, bucket_key = lookup_empirical_rate(
            self._elo_calibration_table,
            is_home=is_home,
            elo_difference=elo_difference,
            bucket_size=int(self._elo_calibration_config.bucket_size),
        )
        p_final, w = blend_probabilities(
            p_elo=p_elo,
            p_emp=p_emp,
            n=n,
            prior=float(self._elo_calibration_config.prior),
        )

        estimate.estimated_probability = float(p_final)
        estimate.edge = float(p_final) - float(market.yes_price)

        elo_meta["p_elo"] = float(p_elo)
        elo_meta["p_emp"] = (None if p_emp is None else float(p_emp))
        elo_meta["p_final"] = float(p_final)
        elo_meta["calibration_bucket"] = bucket_key
        elo_meta["calibration_n"] = float(n)
        elo_meta["calibration_weight_w"] = float(w)
        elo_meta["calibration_min_season"] = self._elo_calibration_config.min_season
        elo_meta["calibration_recency_mode"] = self._elo_calibration_config.recency_mode
        elo_meta["calibration_recency_halflife_days"] = (
            self._elo_calibration_config.recency_halflife_days
        )
        elo_meta["final_probability"] = float(p_final)
        elo_meta["edge"] = float(estimate.edge)

        metadata["elo_adjustment"] = elo_meta
        estimate.fusion_metadata = metadata

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
        adjusted = calculate_adjusted_yes_probability(
            yes_team=str(elo_ctx["yes_team"]),
            home_team=str(elo_ctx["home_team"]),
            away_team=str(elo_ctx["away_team"]),
            home_elo=float(elo_ctx["home_elo"]),
            away_elo=float(elo_ctx["away_elo"]),
            llm_elo_delta=elo_delta,
            home_court_bonus=float(elo_ctx["home_court_bonus"]),
        )
        base = calculate_adjusted_yes_probability(
            yes_team=str(elo_ctx["yes_team"]),
            home_team=str(elo_ctx["home_team"]),
            away_team=str(elo_ctx["away_team"]),
            home_elo=float(elo_ctx["home_elo"]),
            away_elo=float(elo_ctx["away_elo"]),
            llm_elo_delta=0,
            home_court_bonus=float(elo_ctx["home_court_bonus"]),
        )
        base_prob = float(base["yes_probability"])
        final_prob = float(adjusted["yes_probability"])
        edge = final_prob - float(market.yes_price)
        confidence_clamped = float(max(0.0, min(1.0, float(confidence))))
        suggestion_payload = llm_suggestion if isinstance(llm_suggestion, dict) else {}
        if not suggestion_payload:
            suggestion_payload = {
                "raw_elo_delta": float(elo_delta),
                "applied_elo_delta": float(elo_delta),
                "confidence": confidence_clamped,
                "reason": str(reason or ""),
                "key_factors": [],
                "data_sources": [],
                "injury_report": {
                    "status": "unknown",
                    "impact": "unknown",
                    "notes": "",
                },
            }
        return FairValueEstimate(
            market_id=market.market_id,
            estimated_probability=final_prob,
            confidence_level=confidence_clamped,
            reasoning=(
                f"Elo base={base_prob:.3f}, elo_delta={float(elo_delta):+0.0f}. "
                f"{reason}"
            ).strip(),
            data_sources=["nba_elo", "sportradar_injuries"],
            key_factors=["elo_with_injury_delta"],
            edge=edge,
            fusion_metadata={
                "elo_adjustment": {
                    "yes_team": str(elo_ctx["yes_team"]),
                    "home_team": str(elo_ctx["home_team"]),
                    "away_team": str(elo_ctx["away_team"]),
                    "home_elo": float(elo_ctx["home_elo"]),
                    "away_elo": float(elo_ctx["away_elo"]),
                    "base_probability": float(base_prob),
                    "applied_elo_delta": float(adjusted["applied_elo_delta"]),
                    "yes_adjusted_elo": float(adjusted["yes_adjusted_elo"]),
                    "yes_effective_elo": float(adjusted["yes_effective_elo"]),
                    "opponent_effective_elo": float(adjusted["opponent_effective_elo"]),
                    "final_probability": float(final_prob),
                    "market_probability": float(market.yes_price),
                    "edge": float(edge),
                    "llm_suggestion": suggestion_payload,
                    "llm_source_tag": str(source_tag),
                }
            },
        )

    @staticmethod
    def _build_elo_fallback_estimate(
        market: MarketData,
        elo_ctx: Dict[str, object],
        reason: str,
    ) -> FairValueEstimate:
        return ClaudeAnalyzer._build_elo_estimate_from_delta(
            market=market,
            elo_ctx=elo_ctx,
            elo_delta=0.0,
            confidence=0.60,
            reason=f"Elo-only fallback ({reason}).",
            source_tag="llm_refresh",
            llm_suggestion={
                "raw_elo_delta": 0.0,
                "applied_elo_delta": 0.0,
                "confidence": 0.60,
                "reason": f"Elo-only fallback ({reason}).",
                "key_factors": ["elo_baseline_only"],
                "data_sources": ["nba_elo"],
                "injury_report": {
                    "status": "unknown",
                    "impact": "neutral",
                    "notes": "",
                },
            },
        )

    def _log_elo_decision_fields(self, market: MarketData, estimate: FairValueEstimate) -> None:
        meta = (estimate.fusion_metadata or {}).get("elo_adjustment", {})
        if not isinstance(meta, dict):
            return
        try:
            p_elo = float(meta.get("p_elo", meta.get("final_probability", estimate.estimated_probability)))
            p_final = float(meta.get("p_final", meta.get("final_probability", estimate.estimated_probability)))
            p_emp_raw = meta.get("p_emp")
            p_emp = "na" if p_emp_raw is None else f"{float(p_emp_raw):.4f}"
            logger.info(
                "ELO_DECISION | team=%s | opponent=%s | elo_base=%.2f | elo_delta=%+.0f | "
                "elo_adjusted=%.2f | probability_base=%.4f | probability_p_elo=%.4f | "
                "probability_final=%.4f | p_emp=%s | cal_n=%.1f | cal_w=%.4f | "
                "market_probability=%.4f | edge=%+.4f | llm_source=%s | refresh_reason=%s",
                str(meta.get("yes_team") or ""),
                str(meta.get("away_team") if str(meta.get("yes_team")) == str(meta.get("home_team")) else meta.get("home_team")),
                float(meta.get("home_elo") if str(meta.get("yes_team")) == str(meta.get("home_team")) else meta.get("away_elo")),
                float(meta.get("applied_elo_delta", 0.0)),
                float(meta.get("yes_adjusted_elo", 0.0)),
                float(meta.get("base_probability", 0.0)),
                p_elo,
                p_final,
                p_emp,
                float(meta.get("calibration_n", 0.0)),
                float(meta.get("calibration_weight_w", 0.0)),
                float(meta.get("market_probability", market.yes_price)),
                float(meta.get("edge", estimate.edge)),
                str(meta.get("llm_source_tag", "llm_refresh")),
                str(meta.get("llm_refresh_reason", "")),
            )
            suggestion = meta.get("llm_suggestion", {})
            if isinstance(suggestion, dict):
                injury = suggestion.get("injury_report", {})
                if not isinstance(injury, dict):
                    injury = {}
                logger.info(
                    "ELO_SUGGESTION | market_id=%s | raw_delta=%s | applied_delta=%+.0f | confidence=%.2f | "
                    "injury_status=%s | injury_impact=%s | factors=%s | sources=%s",
                    market.market_id,
                    str(suggestion.get("raw_elo_delta")),
                    float(suggestion.get("applied_elo_delta", meta.get("applied_elo_delta", 0.0))),
                    float(suggestion.get("confidence", estimate.confidence_level)),
                    str(injury.get("status", "unknown")),
                    str(injury.get("impact", "unknown")),
                    ",".join(str(x) for x in (suggestion.get("key_factors") or [])[:5]),
                    ",".join(str(x) for x in (suggestion.get("data_sources") or [])[:5]),
                )
        except Exception:
            logger.debug("Failed logging Elo decision metadata for %s", market.market_id)

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
