import logging
import asyncio
import os
import sys
import time
import json
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from app.config import ConfigManager
from app.storage.db import DatabaseManager
from app.models import MarketData, FairValueEstimate, TradeSignal
from app.api_clients.scanner import MarketScanner
from app.analysis.claude_analyzer import ClaudeAnalyzer
from app.analysis import OpenAIAnalyzer
from app.trading import PositionManager, Strategy, TradeExecutor
from app.trading.engine import log_model_divergence_warning
from app.trading.decision_engine import append_decision_jsonl, evaluate_opportunity
from app.trading.bankroll_manager import BankrollManager
from app.trading.reconciliation import ReconciliationManager
from app.signals import SignalFusionService
from app.utils.notifier import Notifier
from app.utils import (
    get_error_reporter,
    BatchParser,
    InsufficientCapitalError,
    NoOpportunitiesError,
    PositionLimitError,
    ExecutionError,
)

logger = logging.getLogger(__name__)

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
        
        # Database
        self.db = DatabaseManager(self.config.db_path)

        # Initialize components
        self.scanner = MarketScanner(self.config)
        self._analyzer = None # Created lazily
        self.strategy = Strategy(self.config)
        self.signal_fusion_service = SignalFusionService(self.config)

        # New persistent managers
        self.bankroll_manager = BankrollManager(
            self.db,
            self.config.trading.initial_bankroll
        )
        self.position_manager = PositionManager(self.db, self.bankroll_manager)
        self.reconciliation_manager = None # Created after DB init

        self.executor = TradeExecutor(
            self.config,
            self.db,
            self.scanner.kalshi_client
        )

        # Error reporting
        self.error_reporter = get_error_reporter()
        self.notifier = Notifier(self.config.risk.critical_webhook_url)

        # Cycle counter
        self.cycle_count = 0
        self._initialized = False
        self.interactive_market_pick = False
        self.runtime_scan_series_tickers: Optional[List[str]] = None
        self.runtime_scan_scope_description: Optional[str] = None
        self.single_run_dry_output_json = False

    async def initialize(self):
        """Initialize database and managers."""
        if self._initialized:
            return

        # Live mode scope validation
        if not self.config.is_dry_run and self.config.trading.require_scope_in_live:
            has_scope = (
                bool(self.config.trading.allowed_market_ids) or
                bool(self.config.trading.allowed_event_tickers) or
                (self.config.kalshi_enabled and (
                    bool(self.config.platforms.kalshi.series_tickers) or
                    bool(self.config.platforms.kalshi.allowed_market_ids) or
                    bool(self.config.platforms.kalshi.allowed_event_tickers)
                ))
            )
            if not has_scope:
                raise ValueError(
                    "Live trading requires an explicit market scope for safety. "
                    "Configure allowed_market_ids, allowed_event_tickers, or series_tickers."
                )

        # Live mode safety: verify configured bankroll does not exceed current Kalshi cash.
        await self._verify_live_cash_balance()

        await self.db.initialize()

        # Load cycle count from DB for stable idempotency across restarts
        status = await self.db.get_last_status()
        self.cycle_count = int(status.get('cycle_count', 0))
        logger.info(f"Loaded cycle_count: {self.cycle_count}")

        if self.config.is_dry_run:
            # Fresh paper session per app startup; continuous loops keep evolving in-process.
            await self.position_manager.reset_for_new_dry_run_session()
            await self.bankroll_manager.reset_for_new_dry_run_session()
        else:
            await self.bankroll_manager.initialize()

        await self.position_manager.load_positions()

        self.reconciliation_manager = ReconciliationManager(
            self.scanner.kalshi_client,
            self.position_manager,
            db=self.db,
            config=self.config,
        )

        # Run reconciliation on startup if not in dry-run
        if not self.config.is_dry_run and self.scanner.kalshi_client:
            await self.reconciliation_manager.reconcile()

        self._initialized = True

    async def _verify_live_cash_balance(self) -> None:
        """
        In live mode, ensure Kalshi available cash can cover configured initial bankroll.
        """
        if self.config.is_dry_run:
            return
        if not bool(getattr(self.config.trading, "enforce_live_cash_check", True)):
            logger.info("Live cash check disabled by trading.enforce_live_cash_check=false")
            return
        if not self.config.kalshi_enabled:
            return

        required = float(getattr(self.config.trading, "initial_bankroll", 0.0) or 0.0)
        if required <= 0:
            return

        kalshi_client = getattr(self.scanner, "kalshi_client", None)
        if not kalshi_client:
            raise ValueError("Kalshi client is not available for live bankroll verification")

        try:
            available_cash = float(await kalshi_client.get_cash_balance())
        except Exception as e:
            raise ValueError(
                "Unable to verify Kalshi available cash before live run. "
                f"Error: {e}"
            ) from e

        if available_cash < required:
            raise ValueError(
                "Live mode bankroll check failed: "
                f"Kalshi available cash (${available_cash:,.2f}) is below "
                f"configured trading.initial_bankroll (${required:,.2f}). "
                "Lower initial_bankroll or fund the account."
            )

        logger.info(
            "✅ Live bankroll verified against Kalshi cash: "
            f"required=${required:,.2f}, available=${available_cash:,.2f}"
        )

    @property
    def analyzer(self):
        """Lazy initializer for analyzer"""
        if self._analyzer is None:
            self._analyzer = self._create_analyzer(self.config.analysis_provider)
        return self._analyzer

    def _create_analyzer(self, provider: str):
        provider_norm = (provider or "").strip().lower()
        if provider_norm == "openai":
            return OpenAIAnalyzer(self.config)
        return ClaudeAnalyzer(self.config)

    def _is_market_in_allowed_scope(self, market: MarketData) -> bool:
        """Check if a market is within the explicitly allowed scope."""
        # Collect all allowed identifiers
        allowed_market_ids = set(self.config.trading.allowed_market_ids)
        allowed_event_tickers = set(self.config.trading.allowed_event_tickers)
        allowed_series_tickers = set()

        if market.platform == 'kalshi':
            if self.config.platforms.kalshi.series_tickers:
                allowed_series_tickers.update(self.config.platforms.kalshi.series_tickers)
            if self.config.platforms.kalshi.allowed_market_ids:
                allowed_market_ids.update(self.config.platforms.kalshi.allowed_market_ids)
            if self.config.platforms.kalshi.allowed_event_tickers:
                allowed_event_tickers.update(self.config.platforms.kalshi.allowed_event_tickers)

        # If no scope defined anywhere, everything is allowed
        # (startup check ensures we have scope if required in live)
        if not allowed_market_ids and not allowed_event_tickers and not allowed_series_tickers:
            return True

        # 1. market_id match
        if market.market_id in allowed_market_ids:
            return True

        # 2. event_ticker match
        if market.event_ticker and market.event_ticker in allowed_event_tickers:
            return True

        # 3. series_ticker match
        if market.series_ticker and market.series_ticker in allowed_series_tickers:
            return True

        # 4. Kalshi fallback: market_id starts with series-
        if market.platform == 'kalshi' and market.market_id:
            for series in allowed_series_tickers:
                if market.market_id.startswith(f"{series}-"):
                    return True

        return False

    async def _update_heartbeat(self, status: str = "active"):
        """Write current bot status to reports/heartbeat.json."""
        stats = self.position_manager.get_stats()
        bankroll = self.bankroll_manager.get_balance()

        heartbeat = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "cycle_count": self.cycle_count,
            "bankroll": bankroll,
            "open_positions": stats['open_positions'],
            "total_exposure": stats['total_exposure'],
            "total_trades": stats['total_trades'],
            "win_rate": stats['win_rate_percent'],
            "dry_run": self.config.is_dry_run,
            "status": status
        }

        report_dir = Path("reports")
        try:
            report_dir.mkdir(exist_ok=True)
            with open(report_dir / "heartbeat.json", "w") as f:
                json.dump(heartbeat, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update heartbeat file: {e}")

    def _trade_journal_path_for_today(self) -> Path:
        """Path to UTC day trade journal JSON."""
        date_key = datetime.utcnow().strftime("%Y-%m-%d")
        journal_dir = Path("reports") / "trade_journal"
        journal_dir.mkdir(parents=True, exist_ok=True)
        return journal_dir / f"{date_key}.json"

    def _ensure_daily_trade_journal(self) -> None:
        """
        Ensure today's trade journal exists, even if no trades are placed.
        """
        path = self._trade_journal_path_for_today()
        now = datetime.utcnow().isoformat() + "Z"
        base_payload = {
            "date_utc": datetime.utcnow().strftime("%Y-%m-%d"),
            "created_at_utc": now,
            "updated_at_utc": now,
            "trades": [],
        }
        try:
            if path.exists():
                with open(path, "r") as f:
                    existing = json.load(f)
                if isinstance(existing, dict) and isinstance(existing.get("trades"), list):
                    existing["updated_at_utc"] = now
                    with open(path, "w") as f:
                        json.dump(existing, f, indent=2)
                    return
            with open(path, "w") as f:
                json.dump(base_payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to initialize daily trade journal at {path}: {e}")

    def _append_trade_journal_entry(self, entry: Dict[str, Any]) -> None:
        """Append one trade entry to today's trade journal JSON."""
        path = self._trade_journal_path_for_today()
        now = datetime.utcnow().isoformat() + "Z"
        try:
            payload: Dict[str, Any]
            if path.exists():
                with open(path, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and isinstance(loaded.get("trades"), list):
                    payload = loaded
                else:
                    payload = {
                        "date_utc": datetime.utcnow().strftime("%Y-%m-%d"),
                        "created_at_utc": now,
                        "trades": [],
                    }
            else:
                payload = {
                    "date_utc": datetime.utcnow().strftime("%Y-%m-%d"),
                    "created_at_utc": now,
                    "trades": [],
                }

            payload["trades"].append(entry)
            payload["updated_at_utc"] = now
            payload["trade_count"] = len(payload["trades"])
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to append trade journal entry: {e}")

    def _write_single_run_dry_analysis_json(
        self,
        report: Dict[str, Any],
        analysis_rows: List[Dict[str, Any]],
    ) -> Optional[Path]:
        """
        Persist single-run dry analysis results to a JSON artifact.
        """
        out_dir = Path("reports") / "dry_run_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        cycle_id = int(report.get("cycle") or self.cycle_count or 0)
        out_path = out_dir / f"cycle_{cycle_id:04d}_{timestamp}.json"
        payload = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "cycle": cycle_id,
            "dry_run": True,
            "analysis_provider": ((report.get("config") or {}).get("analysis_provider")),
            "results_count": len(analysis_rows),
            "results": analysis_rows,
        }
        try:
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            return out_path
        except Exception as e:
            logger.warning(f"Failed to write dry-run analysis JSON: {e}")
            return None

    async def _set_recent_reasoning_prompt_context(self, markets: List[MarketData]) -> None:
        """
        Load recent persisted LLM blurbs and pass them into analyzer runtime context.
        """
        analyzer = self.analyzer
        setter = getattr(analyzer, "set_runtime_context_block", None)
        if not callable(setter):
            return

        if not bool(getattr(self.config.analysis, "use_recent_reasoning_context", True)):
            setter("")
            return

        try:
            max_entries = max(1, int(getattr(self.config.analysis, "recent_reasoning_entries", 12) or 12))
            max_chars = max(1, int(getattr(self.config.analysis, "recent_reasoning_max_chars", 4000) or 4000))
        except Exception:
            max_entries = 12
            max_chars = 4000

        if not markets:
            setter("")
            return

        unique_platforms = {m.platform for m in markets if m.platform}
        platform = next(iter(unique_platforms)) if len(unique_platforms) == 1 else None
        series_tickers = sorted({m.series_ticker for m in markets if m.series_ticker})

        rows_result = self.db.get_recent_analysis_reasoning(
            limit=max_entries * 3,
            platform=platform,
            series_tickers=series_tickers or None,
        )
        if asyncio.iscoroutine(rows_result):
            rows = await rows_result
        else:
            rows = rows_result
        if not rows:
            setter("")
            return

        lines = [
            "Recent prior-run analysis blurbs (historical; verify against current market data):"
        ]
        added = 0
        seen_keys = set()
        for row in rows:
            if added >= max_entries:
                break
            market_id = str(row.get("market_id") or "").strip()
            reasoning = str(row.get("reasoning") or "").strip()
            if not market_id or not reasoning:
                continue
            key = (market_id, reasoning)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            analyzed_at = str(row.get("analyzed_at_utc") or "")
            edge = row.get("edge")
            conf = row.get("confidence")
            edge_txt = f"{float(edge):+.3f}" if isinstance(edge, (int, float)) else "-"
            conf_txt = f"{float(conf):.3f}" if isinstance(conf, (int, float)) else "-"
            lines.append(
                f"- {analyzed_at} | {market_id} | edge={edge_txt} conf={conf_txt} | "
                f"{self._truncate(reasoning, 240)}"
            )
            added += 1

        if added == 0:
            setter("")
            return

        block = "\n".join(lines)
        if len(block) > max_chars:
            block = self._truncate(block, max_chars)
        setter(block)
        logger.info("Loaded %d recent analysis blurbs into runtime prompt context", added)

    async def _persist_analysis_reasoning_to_db(self, report: Dict[str, Any]) -> None:
        """
        Persist LLM reasoning blurbs from this cycle for future prompt context.
        """
        if not bool(getattr(self.config.analysis, "persist_reasoning_to_db", True)):
            return

        markets = report.get("markets")
        if not isinstance(markets, list) or not markets:
            return

        config_block = report.get("config") if isinstance(report.get("config"), dict) else {}
        provider = (
            (config_block.get("analysis_provider") if isinstance(config_block, dict) else None)
            or self.config.analysis_provider
            or "unknown"
        )
        cycle_id = int(report.get("cycle") or self.cycle_count or 0)
        analyzed_at = str(report.get("finished_at") or datetime.utcnow().isoformat() + "Z")

        entries = []
        for row in markets:
            if not isinstance(row, dict):
                continue
            analysis = row.get("analysis")
            if not isinstance(analysis, dict):
                continue
            reasoning = str(analysis.get("reasoning") or "").strip()
            if not reasoning:
                continue
            signal = row.get("signal") if isinstance(row.get("signal"), dict) else {}
            entries.append({
                "cycle_id": cycle_id,
                "market_id": row.get("market_id"),
                "event_ticker": row.get("event_ticker"),
                "series_ticker": row.get("series_ticker"),
                "platform": row.get("platform"),
                "title": row.get("title"),
                "selection": row.get("yes_option"),
                "yes_price": (row.get("prices") or {}).get("yes"),
                "fair_probability": analysis.get(
                    "effective_probability",
                    analysis.get("estimated_probability"),
                ),
                "edge": analysis.get("effective_edge", analysis.get("edge")),
                "confidence": analysis.get(
                    "effective_confidence",
                    analysis.get("confidence"),
                ),
                "action": signal.get("action"),
                "position_size": signal.get("position_size"),
                "provider": provider,
                "reasoning": reasoning,
                "analyzed_at_utc": analyzed_at,
            })

        if not entries:
            return

        write_result = self.db.save_analysis_reasoning_entries(entries)
        if asyncio.iscoroutine(write_result):
            written = await write_result
        else:
            written = int(write_result or 0)
        logger.info("Persisted %d analysis blurbs to DB context memory", written)

    async def _compute_current_equity(self) -> float:
        """
        Compute current equity for risk guards.

        In live mode we prefer mark-to-market equity from reconciliation manager.
        In dry-run (or on any live-data failure), fallback to balance + local exposure.
        """
        current_balance = self.bankroll_manager.get_balance()
        fallback_equity = current_balance + self.position_manager.get_total_exposure()

        if self.config.is_dry_run or not self.reconciliation_manager:
            return fallback_equity

        try:
            return await self.reconciliation_manager.compute_true_equity(current_balance)
        except Exception as e:
            logger.warning(f"Falling back to local equity approximation after live equity error: {e}")
            return fallback_equity

    async def _get_market_trade_count_today(self, market_id: str) -> int:
        """
        Count executions for market_id in current UTC day.

        Includes submitted/open/filled statuses as "attempted trades" for cap enforcement.
        """
        statuses = (
            "submitted",
            "pending_submit",
            "pending_fill",
            "resting",
            "open",
            "filled",
            "partially_filled",
            "executed",
            "dry_run",
        )
        today = datetime.utcnow().strftime("%Y-%m-%d")
        placeholders = ",".join(["?"] * len(statuses))

        query = f"""
            SELECT COUNT(*) AS n
            FROM executions
            WHERE market_id = ?
              AND status IN ({placeholders})
              AND substr(executed_at_utc, 1, 10) = ?
        """
        params = (market_id, *statuses, today)

        async with self.db.connect() as db:
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                return int(row[0] or 0) if row else 0

    async def _update_execution_failure_state(self, results: List[Dict[str, Any]]) -> None:
        """
        Update consecutive execution failure tracking and optional cooldown state.
        """
        if not results:
            return

        status_map = await self.db.get_last_status()
        current_streak = int(status_map.get("execution_failure_streak", 0) or 0)

        hard_failures = [
            r for r in results
            if (not r.get("success")) and (not r.get("skipped"))
        ]
        successes = [r for r in results if r.get("success")]

        if hard_failures and not successes:
            current_streak += 1
        elif successes:
            current_streak = 0

        await self.db.update_status("execution_failure_streak", current_streak)

        threshold = int(self.config.risk.failure_streak_cooldown_threshold or 0)
        cooldown_cycles = int(self.config.risk.failure_cooldown_cycles or 0)
        if threshold <= 0 or cooldown_cycles <= 0:
            return

        if current_streak >= threshold:
            cooldown_until = self.cycle_count + cooldown_cycles
            await self.db.update_status("execution_cooldown_until_cycle", cooldown_until)
            msg = (
                f"🛑 Execution cooldown triggered: streak={current_streak} >= {threshold}. "
                f"New orders paused until cycle {cooldown_until}."
            )
            logger.warning(msg)
            await self.notifier.send_notification(msg, level="CRITICAL")

    async def _run_risk_guards(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """
        Apply hard risk rails before execution.
        """
        if not signals:
            return []

        # 1. Kill switch env var
        kill_switch_var = self.config.risk.kill_switch_env_var
        if kill_switch_var and os.getenv(kill_switch_var):
            msg = f"🛑 KILL SWITCH DETECTED ({kill_switch_var}). Skipping execution."
            logger.warning(msg)
            await self.notifier.send_notification(msg, level="CRITICAL")
            return []

        # 2. Cooldown after consecutive failures
        threshold = int(self.config.risk.failure_streak_cooldown_threshold or 0)
        cooldown_cycles = int(self.config.risk.failure_cooldown_cycles or 0)
        if threshold > 0 and cooldown_cycles > 0:
            status_map = await self.db.get_last_status()
            cooldown_until = int(status_map.get("execution_cooldown_until_cycle", 0) or 0)
            if self.cycle_count < cooldown_until:
                logger.warning(
                    f"🧊 Execution cooldown active until cycle {cooldown_until}; "
                    f"current cycle={self.cycle_count}. Skipping execution."
                )
                return []

        # 3. Daily Loss Limit (true equity in live mode)
        current_equity = await self._compute_current_equity()

        starting_equity = await self.bankroll_manager.get_daily_starting_balance(
            dry_run_session_anchored=bool(self.config.is_dry_run)
        )

        loss_limit_fraction = self.config.risk.daily_loss_limit_fraction
        max_allowed_loss = starting_equity * loss_limit_fraction
        current_loss = starting_equity - current_equity

        if current_loss >= max_allowed_loss:
            msg = (
                f"🛑 DAILY LOSS LIMIT BREACHED: loss=${current_loss:.2f} >= limit=${max_allowed_loss:.2f}. "
                "Skipping new orders."
            )
            logger.warning(msg)
            await self.notifier.send_notification(msg, level="CRITICAL")
            return []

        # 4. Per-cycle caps + per-market/day cap
        max_orders = self.config.risk.max_orders_per_cycle
        max_notional = self.config.risk.max_notional_per_cycle
        max_market_trades = int(self.config.risk.max_trades_per_market_per_day or 0)

        allowed_signals = []
        cumulative_notional = 0.0
        per_market_counts: Dict[str, int] = {}

        for s in signals:
            if len(allowed_signals) >= max_orders:
                logger.info(f"⏭️ Skipping signal {s.market.market_id}: max_orders_per_cycle reached ({max_orders})")
                continue
            if cumulative_notional + s.position_size > max_notional:
                logger.info(f"⏭️ Skipping signal {s.market.market_id}: max_notional_per_cycle reached ({max_notional})")
                continue
            if max_market_trades > 0:
                existing = per_market_counts.get(s.market.market_id)
                if existing is None:
                    existing = await self._get_market_trade_count_today(s.market.market_id)
                if existing >= max_market_trades:
                    logger.info(
                        f"⏭️ Skipping signal {s.market.market_id}: "
                        f"max_trades_per_market_per_day reached ({max_market_trades})"
                    )
                    continue
                per_market_counts[s.market.market_id] = existing + 1

            allowed_signals.append(s)
            cumulative_notional += s.position_size

        return allowed_signals

    def set_analysis_provider(self, provider: str) -> None:
        """Switch analysis provider before a scan begins."""
        provider_norm = (provider or "").strip().lower()
        if provider_norm not in {"claude", "openai"}:
            raise ValueError(f"Unknown analysis provider: {provider}")

        # Validate key availability for chosen provider
        if provider_norm == "openai":
            if not self.config.api.openai_api_key:
                raise ValueError("OpenAI API key not configured")
        else:
            if not self.config.api.claude_api_key:
                raise ValueError("Claude API key not configured")

        self.config.analysis.provider = provider_norm
        self._analyzer = self._create_analyzer(provider_norm)

    @staticmethod
    def _truncate(text: object, max_len: int) -> str:
        raw = str(text) if text is not None else ""
        if max_len <= 0:
            return ""
        if len(raw) <= max_len:
            return raw
        if max_len <= 3:
            return raw[:max_len]
        return raw[: max_len - 3] + "..."

    @staticmethod
    def _extract_elo_decision_fields(signal: TradeSignal, market_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build normalized structured decision fields for logging.
        """
        analysis = market_row.get("analysis") if isinstance(market_row, dict) else {}
        if not isinstance(analysis, dict):
            analysis = {}
        elo_meta = analysis.get("elo_adjustment")
        if not isinstance(elo_meta, dict):
            elo_meta = {}

        team = str(elo_meta.get("yes_team") or "").strip() or (
            signal.market.yes_option if signal.action in ("buy_yes", "sell_no") else signal.market.no_option
        ) or "unknown"

        home_team = str(elo_meta.get("home_team") or "").strip()
        away_team = str(elo_meta.get("away_team") or "").strip()
        if team and home_team and away_team:
            opponent = away_team if team == home_team else home_team
        else:
            opponent = "unknown"

        probability_base = elo_meta.get("base_probability")
        probability_final = elo_meta.get("final_probability", signal.fair_value)
        market_probability = elo_meta.get("market_probability", signal.market_price)
        edge = elo_meta.get("edge", signal.edge)

        return {
            "team": team,
            "opponent": opponent,
            "elo_base": elo_meta.get("yes_base_elo", elo_meta.get("home_elo") if team == home_team else elo_meta.get("away_elo")),
            "elo_delta": elo_meta.get("applied_elo_delta", 0.0),
            "elo_adjusted": elo_meta.get("yes_adjusted_elo"),
            "probability_base": probability_base,
            "probability_final": probability_final,
            "market_probability": market_probability,
            "edge": edge,
        }

    def _log_structured_trade_decision(self, signal: TradeSignal, market_row: Optional[Dict[str, Any]]) -> None:
        """
        Structured pre-execution decision log for auditing and debugging.
        """
        fields = self._extract_elo_decision_fields(signal, market_row)
        payload = {
            "market_id": signal.market.market_id,
            "team": fields["team"],
            "opponent": fields["opponent"],
            "elo_base": fields["elo_base"],
            "elo_delta": fields["elo_delta"],
            "elo_adjusted": fields["elo_adjusted"],
            "probability_base": fields["probability_base"],
            "probability_final": fields["probability_final"],
            "market_probability": fields["market_probability"],
            "edge": fields["edge"],
            "decision": "execute",
            "proposed_bet_size": float(signal.position_size),
        }
        logger.info("TRADE_DECISION | %s", json.dumps(payload, sort_keys=True, default=str))

    def _run_shadow_decision_logging(
        self,
        opportunities: List[tuple[MarketData, FairValueEstimate]],
        market_rows_by_id: Dict[str, Dict[str, Any]],
        report: Dict[str, Any],
    ) -> None:
        if not bool(getattr(self.config.strategy, "enable_shadow_decision_logging", True)):
            return

        decisions_path = str(
            getattr(
                self.config.trading,
                "shadow_decision_jsonl_path",
                "reports/trade_opportunities/opportunities.jsonl",
            )
        )
        log_all = bool(getattr(self.config.strategy, "log_all_opportunities", True))

        considered = 0
        take_count = 0
        skip_count = 0
        reason_counts: Dict[str, int] = {}
        persisted = 0

        for market, est in opportunities:
            record = evaluate_opportunity(market, est, self.config)
            considered += 1
            reason_counts[record.reason] = int(reason_counts.get(record.reason, 0)) + 1
            if record.decision == "take":
                take_count += 1
            else:
                skip_count += 1

            row = market_rows_by_id.get(market.market_id)
            if isinstance(row, dict):
                row["shadow_decision"] = asdict(record)

            if log_all or record.decision == "take":
                append_decision_jsonl(record, decisions_path)
                persisted += 1

        report_counts = report.setdefault("counts", {})
        report_counts["opportunities_considered"] = considered
        report_counts["shadow_take_count"] = take_count
        report_counts["shadow_skip_count"] = skip_count
        report.setdefault("shadow_decisions", {})["skip_reasons"] = reason_counts
        report.setdefault("shadow_decisions", {})["log_path"] = decisions_path
        report.setdefault("shadow_decisions", {})["persisted_records"] = persisted

        logger.info(
            "Shadow decisions evaluated: considered=%d take=%d skip=%d persisted=%d",
            considered,
            take_count,
            skip_count,
            persisted,
        )

        try:
            log_model_divergence_warning(
                team=str(fields["team"]),
                opponent=str(fields["opponent"]),
                probability_final=float(fields["probability_final"]),
                market_probability=float(fields["market_probability"]),
                edge=float(fields["edge"]),
            )
        except Exception:
            # Divergence warning is informational and must never break execution.
            logger.debug("Failed to evaluate divergence warning for %s", signal.market.market_id)

    @staticmethod
    def _format_text_table(headers: List[str], rows: List[List[str]]) -> str:
        """Render a simple ASCII table for terminal output."""
        if not headers:
            return ""

        normalized_rows = []
        widths = [len(str(h)) for h in headers]
        for row in rows:
            normalized = [str(cell) for cell in row]
            if len(normalized) < len(headers):
                normalized.extend([""] * (len(headers) - len(normalized)))
            normalized_rows.append(normalized[:len(headers)])
            for idx, cell in enumerate(normalized[:len(headers)]):
                widths[idx] = max(widths[idx], len(cell))

        sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
        header_line = "| " + " | ".join(
            str(headers[i]).ljust(widths[i]) for i in range(len(headers))
        ) + " |"

        lines = [sep, header_line, sep]
        for row in normalized_rows:
            lines.append(
                "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
            )
        lines.append(sep)
        return "\n".join(lines)

    @staticmethod
    def _parse_index_selection(raw: str, max_index: int) -> List[int]:
        selected = set()
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if not tokens:
            raise ValueError("selection cannot be empty")

        for token in tokens:
            if "-" in token:
                left, right = token.split("-", 1)
                start = int(left)
                end = int(right)
                if start > end:
                    raise ValueError(f"invalid range '{token}'")
                if start < 1 or end > max_index:
                    raise ValueError(f"range '{token}' is out of bounds 1-{max_index}")
                for idx in range(start, end + 1):
                    selected.add(idx)
                continue

            idx = int(token)
            if idx < 1 or idx > max_index:
                raise ValueError(f"index '{idx}' is out of bounds 1-{max_index}")
            selected.add(idx)

        return sorted(selected)

    def _prompt_market_selection(self, markets: List[MarketData]) -> List[MarketData]:
        if not markets:
            return markets
        if not self.interactive_market_pick:
            return markets
        if not self.config.is_dry_run:
            logger.info("Interactive market picker is only enabled in dry-run mode; using all markets.")
            return markets
        if not sys.stdin.isatty():
            logger.info("Interactive market picker requested, but no TTY is attached; using all markets.")
            return markets

        headers = ["#", "Market", "Question", "Selection", "YES", "Volume"]
        rows = []
        for idx, market in enumerate(markets, start=1):
            rows.append([
                str(idx),
                self._truncate(market.market_id, 34),
                self._truncate(market.title, 52),
                self._truncate(market.yes_option or "-", 16),
                f"{market.yes_price:.3f}",
                f"{market.volume:,.0f}",
            ])

        print("\nDRY RUN: Choose markets to analyze")
        print(self._format_text_table(headers, rows))
        print("Enter 'all', 'none', or comma/range list like: 1,3,5-8")

        while True:
            raw = input("Selection [all]: ").strip().lower()
            if raw in {"", "all", "a"}:
                logger.info(f"Selected all {len(markets)} filtered markets for analysis")
                return markets
            if raw in {"none", "n"}:
                logger.info("No markets selected for analysis")
                return []
            try:
                indices = self._parse_index_selection(raw, len(markets))
                selected = [markets[i - 1] for i in indices]
                logger.info(f"Selected {len(selected)}/{len(markets)} filtered markets for analysis")
                return selected
            except ValueError as e:
                print(f"Invalid selection: {e}")
            except Exception:
                print("Invalid selection format. Example: 1,3,5-8")

    def _print_dry_run_analysis_table(self, report: dict) -> None:
        if not self.config.is_dry_run:
            return

        market_rows = report.get("markets") or []
        analysis_rows = []

        for market in market_rows:
            analysis = market.get("analysis")
            if not isinstance(analysis, dict):
                continue

            signal = market.get("signal") or {}
            fair = analysis.get("effective_probability", analysis.get("estimated_probability"))
            signal_edge = signal.get("edge")
            edge = signal_edge if isinstance(signal_edge, (int, float)) else analysis.get("effective_edge", analysis.get("edge"))
            conf = analysis.get("effective_confidence", analysis.get("confidence"))
            yes_price = (market.get("prices") or {}).get("yes")
            size = signal.get("position_size")
            reasoning = signal.get("reasoning") or analysis.get("reasoning") or "-"
            elo_adjustment = analysis.get("elo_adjustment") if isinstance(analysis, dict) else None
            if not isinstance(elo_adjustment, dict):
                elo_adjustment = {}
            llm_suggestion = elo_adjustment.get("llm_suggestion")
            if not isinstance(llm_suggestion, dict):
                llm_suggestion = {}
            injury_report = llm_suggestion.get("injury_report")
            if not isinstance(injury_report, dict):
                injury_report = {}

            analysis_rows.append({
                "market_id": market.get("market_id", ""),
                "question": market.get("title", ""),
                "selection": market.get("yes_option") or "-",
                "yes_price": yes_price,
                "fair": fair,
                "edge": edge,
                "conf": conf,
                "signal": signal.get("action", "-"),
                "size": size,
                "reasoning": reasoning,
                "elo_adjustment": elo_adjustment,
                "player_impact": elo_adjustment.get("player_impact"),
                "llm_suggestion": llm_suggestion,
                "injury_report": injury_report,
            })

        if not analysis_rows:
            return

        analysis_rows.sort(
            key=lambda row: float(row["edge"]) if isinstance(row["edge"], (int, float)) else float("-inf"),
            reverse=True,
        )

        if bool(getattr(self, "single_run_dry_output_json", False)):
            out_path = self._write_single_run_dry_analysis_json(report, analysis_rows)
            if out_path:
                logger.info(f"Saved dry-run analysis JSON: {out_path}")
            return

        headers = ["Market", "Question", "Selection", "YES", "Fair", "Edge", "Conf", "Signal", "Size", "Why"]
        rows = []
        for row in analysis_rows:
            yes_val = row["yes_price"]
            fair_val = row["fair"]
            edge_val = row["edge"]
            conf_val = row["conf"]
            size_val = row["size"]
            rows.append([
                self._truncate(row["market_id"], 34),
                self._truncate(row["question"], 46),
                self._truncate(row["selection"], 14),
                f"{yes_val:.3f}" if isinstance(yes_val, (int, float)) else "-",
                f"{fair_val:.3f}" if isinstance(fair_val, (int, float)) else "-",
                f"{edge_val:+.3f}" if isinstance(edge_val, (int, float)) else "-",
                f"{conf_val:.3f}" if isinstance(conf_val, (int, float)) else "-",
                self._truncate(row["signal"], 10),
                f"${size_val:,.2f}" if isinstance(size_val, (int, float)) else "-",
                self._truncate(row["reasoning"], 84),
            ])

        print("\nDRY RUN ANALYSIS RESULTS")
        print(self._format_text_table(headers, rows))

    @staticmethod
    def _build_analysis_payload(est: FairValueEstimate) -> Dict[str, Any]:
        """Build a consistent analysis payload for dry-run/report rows."""
        metadata = est.fusion_metadata or {}
        if not isinstance(metadata, dict):
            metadata = {}
        elo_adjustment = metadata.get("elo_adjustment")
        if not isinstance(elo_adjustment, dict):
            elo_adjustment = None
        feature_meta = metadata.get("feature")
        if not isinstance(feature_meta, dict):
            feature_meta = {}
        return {
            "estimated_probability": est.estimated_probability,
            "confidence": est.confidence_level,
            "edge": est.edge,
            "effective_probability": est.effective_probability,
            "effective_confidence": est.effective_confidence,
            "effective_edge": est.effective_edge,
            "reasoning": est.reasoning,
            "key_factors": est.key_factors,
            "data_sources": est.data_sources,
            "feature_confidence": est.feature_confidence,
            "feature_signal_score": est.feature_signal_score,
            "feature_anomaly": est.feature_anomaly,
            "feature_recommendation": est.feature_recommendation,
            "feature_regime": est.feature_regime,
            "feature_provider": est.feature_provider,
            "feature_reason": feature_meta.get("reason"),
            "feature_rules": metadata.get("applied_rules", []),
            "elo_adjustment": elo_adjustment,
            "player_impact": (elo_adjustment or {}).get("player_impact"),
        }

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
        4. Returns a report dict (same format as run_trading_cycle)
        
        Args:
            series_tickers: List of series tickers to scan
            analyze: Whether to run AI analysis on filtered markets
            max_analyze: Max markets to analyze (to limit API costs)
        
        Returns:
            Report dict with scan results
        """
        await self.initialize()
        self.cycle_count += 1
        await self.db.update_status('cycle_count', self.cycle_count)
        cycle_started_at = datetime.now().isoformat() + "Z"
        
        logger.info("=" * 60)
        logger.info(f"🔍 SERIES SCAN (Cycle {self.cycle_count}): {series_tickers}")
        logger.info("=" * 60)
        
        # Report structure matches run_trading_cycle for consistency
        report: dict = {
            "cycle": self.cycle_count,
            "scan_type": "series",
            "started_at": cycle_started_at,
            "finished_at": None,
            "config": {
                "series_tickers": series_tickers,
                "analyze": analyze,
                "max_analyze": max_analyze,
                "analysis_provider": self.config.analysis_provider if analyze else None,
                "dry_run": self.config.is_dry_run,
                "filters": {
                    "min_volume": self.config.filters.min_volume,
                    "min_liquidity": self.config.filters.min_liquidity,
                    "price_bounds": {"min": 0.01, "max": 0.99},
                },
                "signal_fusion": self.signal_fusion_service.report_status(),
                "strategy": {
                    "min_edge": self.config.strategy.min_edge,
                    "min_edge_net": self.config.strategy.min_edge_net,
                    "min_confidence": self.config.strategy.min_confidence,
                    "enable_shadow_decision_logging": self.config.strategy.enable_shadow_decision_logging,
                    "log_all_opportunities": self.config.strategy.log_all_opportunities,
                },
                "execution": {
                    "kalshi_taker_fee_pct": self.config.execution.kalshi_taker_fee_pct,
                    "estimated_slippage_pct": self.config.execution.estimated_slippage_pct,
                    "use_dynamic_slippage": self.config.execution.use_dynamic_slippage,
                },
                "trading": {
                    "shadow_decision_jsonl_path": self.config.trading.shadow_decision_jsonl_path,
                },
            },
            "counts": {
                "scanned": 0,
                "passed_filters": 0,
                "analyzed": 0,
                "estimates": 0,
                "opportunities": 0,
                "opportunities_considered": 0,
                "shadow_take_count": 0,
                "shadow_skip_count": 0,
                "signals": 0,
                "skipped_duplicates": 0,
                "executed": 0,
            },
            "api_cost": None,
            "markets": [],
            "signals": [],
            "shadow_decisions": {
                "skip_reasons": {},
                "log_path": self.config.trading.shadow_decision_jsonl_path,
                "persisted_records": 0,
            },
            "errors": [],
        }
        
        try:
            # Step 1: Fetch markets by series
            logger.info("\n📊 Step 1: Fetching markets by series...")
            markets = await self.scan_series_markets(series_tickers)
            report["counts"]["scanned"] = len(markets)
            
            # Build market rows with full detail (same format as trading cycle)
            market_rows_by_id = {}
            for m in markets:
                evaluation = self.strategy.evaluate_market_filters(m)
                volume_tier = self.strategy.classify_volume_tier(m)
                row = {
                    "market_id": m.market_id,
                    "platform": m.platform,
                    "title": m.title,
                    "description": m.description,
                    "category": m.category,
                    "end_date": str(m.end_date),
                    "event_ticker": m.event_ticker,
                    "series_ticker": m.series_ticker,
                    "yes_option": m.yes_option,
                    "no_option": m.no_option,
                    "prices": {
                        "yes": m.yes_price,
                        "no": m.no_price,
                    },
                    "stats": {
                        "volume": m.volume,
                        "liquidity": m.liquidity,
                    },
                    "volume_tier": volume_tier,
                    "filters": evaluation,
                    "analysis": None,
                    "opportunity": None,
                    "shadow_decision": None,
                    "signal": None,
                    "execution": None,
                }
                market_rows_by_id[m.market_id] = row
            
            report["markets"] = list(market_rows_by_id.values())
            logger.info(f"   Found {len(markets)} markets")
            
            if not markets:
                logger.warning("No markets found")
                report["finished_at"] = datetime.now().isoformat()
                return report
            
            # Step 2: Apply filters and scope guard
            logger.info("\n🔬 Step 2: Filtering markets and applying scope guard...")
            all_filtered = self.strategy.filter_markets(markets)
            filtered = [m for m in all_filtered if self._is_market_in_allowed_scope(m)]

            if len(filtered) < len(all_filtered):
                logger.info(f"   Scope guard excluded {len(all_filtered) - len(filtered)} markets")

            report["counts"]["passed_filters"] = len(filtered)
            logger.info(f"   {len(filtered)} markets passed filters and scope guard")
            
            # Step 3: Optional AI analysis
            if analyze and filtered:
                to_analyze = filtered[:max_analyze]
                provider = (self.config.analysis_provider or "claude").strip().lower()
                logger.info(f"\n🧠 Step 3: Analyzing {len(to_analyze)} markets with {provider}...")
                analyzed_estimates: List[FairValueEstimate] = []
                await self._set_recent_reasoning_prompt_context(to_analyze)
                
                try:
                    estimates = await self.analyzer.analyze_market_batch(to_analyze)
                    analyzed_estimates = list(estimates) if estimates else []
                    report["counts"]["analyzed"] = len(to_analyze)
                    report["counts"]["estimates"] = len(analyzed_estimates)
                except Exception as e:
                    logger.warning(f"Error during batch analysis: {e}")
                    report["errors"].append({"error": str(e)})
                    analyzed_estimates = []
                
                # Apply optional external feature fusion.
                if analyzed_estimates and self.signal_fusion_service.enabled:
                    analyzed_by_id = {m.market_id: m for m in to_analyze}
                    analyzed_estimates, fused_features = self.signal_fusion_service.apply(
                        analyzed_estimates,
                        analyzed_by_id,
                    )
                    report.setdefault("signal_fusion", {})["applied"] = len(fused_features)

                # Recompute opportunities from effective values (including any fusion-adjusted estimates).
                report["counts"]["opportunities"] = 0
                min_edge = self.config.strategy.min_edge
                min_confidence = self.config.strategy.min_confidence
                for est in analyzed_estimates:
                    row = market_rows_by_id.get(est.market_id)
                    if not row:
                        continue
                    row["analysis"] = self._build_analysis_payload(est)
                    if est.has_significant_edge(min_edge, min_confidence):
                        report["counts"]["opportunities"] += 1
                        row["opportunity"] = {
                            "edge": est.effective_edge,
                            "confidence": est.effective_confidence,
                            "estimated_probability": est.effective_probability,
                            "meets_threshold": True,
                        }
                
                # Update markets list with analysis results
                report["markets"] = list(market_rows_by_id.values())
                
                # Get API cost if available
                if hasattr(self.analyzer, 'get_api_stats'):
                    report["api_cost"] = self.analyzer.get_api_stats()
            
            logger.info(f"\n✅ Series scan complete")
            
        except Exception as e:
            logger.error(f"Series scan error: {e}")
            report["errors"].append({"error": str(e)})

        report["finished_at"] = datetime.now().isoformat()
        try:
            await self._persist_analysis_reasoning_to_db(report)
        except Exception as e:
            logger.warning(f"Failed persisting analysis blurbs from series scan: {e}")
        await self._update_heartbeat()
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
        await self.initialize()
        self.cycle_count += 1
        await self.db.update_status('cycle_count', self.cycle_count)

        # Report container (written to JSON at end of cycle)
        cycle_started_at = datetime.now().isoformat() + "Z"
        report: dict = {
            "cycle": self.cycle_count,
            "started_at": cycle_started_at,
            "finished_at": None,
            "config": {
                "analysis_provider": self.config.analysis_provider,
                "dry_run": self.config.is_dry_run,
                "scan_scope": self.runtime_scan_scope_description or "all_markets",
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
                    "min_edge_net": self.config.strategy.min_edge_net,
                    "min_confidence": self.config.strategy.min_confidence,
                    "enable_shadow_decision_logging": self.config.strategy.enable_shadow_decision_logging,
                    "log_all_opportunities": self.config.strategy.log_all_opportunities,
                },
                "execution": {
                    "kalshi_taker_fee_pct": self.config.execution.kalshi_taker_fee_pct,
                    "estimated_slippage_pct": self.config.execution.estimated_slippage_pct,
                    "use_dynamic_slippage": self.config.execution.use_dynamic_slippage,
                },
                "trading": {
                    "shadow_decision_jsonl_path": self.config.trading.shadow_decision_jsonl_path,
                },
                "signal_fusion": self.signal_fusion_service.report_status(),
                "risk": {
                    "max_kelly_fraction": self.config.risk.max_kelly_fraction,
                    "max_positions": self.config.risk.max_positions,
                    "max_position_size": self.config.risk.max_position_size,
                    "max_total_exposure_fraction": self.config.risk.max_total_exposure_fraction,
                    "max_new_exposure_per_day_fraction": self.config.risk.max_new_exposure_per_day_fraction,
                },
            },
            "counts": {
                "scanned": 0,
                "passed_filters": 0,
                "analyzed": 0,
                "estimates": 0,
                "opportunities": 0,
                "opportunities_considered": 0,
                "shadow_take_count": 0,
                "shadow_skip_count": 0,
                "signals": 0,
                "skipped_duplicates": 0,
                "executed": 0,
            },
            "api_cost": None,
            "markets": [],
            "signals": [],
            "shadow_decisions": {
                "skip_reasons": {},
                "log_path": self.config.trading.shadow_decision_jsonl_path,
                "persisted_records": 0,
            },
            "errors": [],
        }
        
        logger.info("=" * 80)
        logger.info(f"🤖 CYCLE {self.cycle_count}: STARTING TRADING CYCLE")
        logger.info("=" * 80)
        
        cycle_start = time.time()
        cycle_report = self.error_reporter.create_report(
            f"Trading Cycle #{self.cycle_count}"
        )
        self._ensure_daily_trade_journal()

        markets: List[MarketData] = []
        filtered: List[MarketData] = []
        analyzed_markets: List[MarketData] = []
        estimates: List[FairValueEstimate] = []
        opportunities = []
        signals: List[TradeSignal] = []
        execution_results: List[bool] = []
        
        try:
            # Step 0: Reconcile local vs remote live portfolio each cycle.
            if not self.config.is_dry_run and self.reconciliation_manager:
                logger.info("\n🔄 Step 0: Reconciling live portfolio...")
                try:
                    await self.reconciliation_manager.reconcile()
                except Exception as e:
                    logger.error(f"Reconciliation error: {e}")
                    report["errors"].append({
                        "type": type(e).__name__,
                        "message": str(e),
                        "stage": "reconciliation",
                    })

            # Step 1: Scan markets
            logger.info("\n📊 Step 1: Scanning markets...")
            if self.runtime_scan_series_tickers:
                logger.info(
                    f"Using scoped Kalshi series scan ({self.runtime_scan_scope_description or 'custom scope'}): "
                    f"{self.runtime_scan_series_tickers}"
                )
                markets = await self.scan_series_markets(self.runtime_scan_series_tickers)
            else:
                markets = await self.scanner.scan_all_markets()

            # Initialize market rows for report
            market_rows_by_id = {}
            for m in markets:
                evaluation = self.strategy.evaluate_market_filters(m)
                volume_tier = self.strategy.classify_volume_tier(m)
                row = {
                    "market_id": m.market_id,
                    "platform": m.platform,
                    "title": m.title,
                    "description": m.description,
                    "category": m.category,
                    "end_date": str(m.end_date),
                    "event_ticker": m.event_ticker,
                    "yes_option": m.yes_option,
                    "no_option": m.no_option,
                    "prices": {
                        "yes": m.yes_price,
                        "no": m.no_price,
                    },
                    "stats": {
                        "volume": m.volume,
                        "liquidity": m.liquidity,
                    },
                    "volume_tier": volume_tier,
                    "filters": evaluation,
                    "analysis": None,
                    "opportunity": None,
                    "shadow_decision": None,
                    "signal": None,
                    "execution": None,
                }
                market_rows_by_id[m.market_id] = row
            report["markets"] = list(market_rows_by_id.values())
            
            # Create grouped view by event_ticker
            events_grouped = {}
            for m in markets:
                event_key = m.event_ticker or m.market_id
                if event_key not in events_grouped:
                    # Extract base event title (remove option-specific part)
                    base_title = m.title
                    if m.yes_option and m.yes_option in base_title:
                        # Try to get cleaner event title
                        base_title = m.title.replace(f" [{m.yes_option}]", "")
                    events_grouped[event_key] = {
                        "event_ticker": event_key,
                        "event_title": base_title,
                        "platform": m.platform,
                        "category": m.category,
                        "end_date": str(m.end_date),
                        "options": []
                    }
                events_grouped[event_key]["options"].append({
                    "market_id": m.market_id,
                    "option_name": m.yes_option or m.title[:50],
                    "yes_price": m.yes_price,
                    "volume": m.volume,
                })
            report["events"] = list(events_grouped.values())
            
            report["counts"]["scanned"] = len(markets)
            
            # Add tier breakdown to report
            tier_counts = self.strategy.categorize_markets_by_tier(markets)
            report["tier_summary"] = {
                "high": len(tier_counts['high']),
                "medium": len(tier_counts['medium']),
                "low": len(tier_counts['low']),
                "skip": len(tier_counts['skip']),
            }
            logger.info(f"   Volume tiers: high={report['tier_summary']['high']}, medium={report['tier_summary']['medium']}, low={report['tier_summary']['low']}")
            
            if not markets:
                logger.warning("No markets found, aborting cycle")
                return
            
            # Step 2: Filter markets and apply scope guard
            logger.info("\n🔬 Step 2: Filtering markets and applying scope guard...")
            all_filtered = self.strategy.filter_markets(markets)
            filtered = [m for m in all_filtered if self._is_market_in_allowed_scope(m)]

            if len(filtered) < len(all_filtered):
                logger.info(f"   Scope guard excluded {len(all_filtered) - len(filtered)} markets")

            logger.info(f"   {len(filtered)} markets passed filters and scope guard")
            report["counts"]["passed_filters"] = len(filtered)

            # Optional interactive picker for dry-run sessions.
            if self.interactive_market_pick and not filtered and markets:
                logger.info(
                    "No markets passed filters; showing scanned markets for manual dry-run selection."
                )
                filtered = self._prompt_market_selection(markets)
            else:
                filtered = self._prompt_market_selection(filtered)
            if not filtered:
                logger.warning("No markets selected for analysis, aborting cycle")
                return
            
            # Step 3: Analyze with LLM (in batches)
            provider = (self.config.analysis_provider or "claude").strip().lower()
            logger.info(f"\n🧠 Step 3: Analyzing with {provider}...")
            await self._set_recent_reasoning_prompt_context(filtered)

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
                        row["analysis"] = self._build_analysis_payload(est)

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

            if self.signal_fusion_service.enabled:
                estimates, fused_features = self.signal_fusion_service.apply(
                    estimates,
                    {m.market_id: m for m in analyzed_markets},
                )
                report.setdefault("signal_fusion", {})["applied"] = len(fused_features)
                for est in estimates:
                    row = market_rows_by_id.get(est.market_id)
                    if row is not None:
                        row["analysis"] = self._build_analysis_payload(est)

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

            # Step 4.5: Mark and filter duplicate opportunities for reporting
            open_market_keys = self.position_manager.get_open_market_keys()
            open_event_keys = {
                self.strategy.get_event_key_for_platform_market(p.platform, p.market_id)
                for p in self.position_manager.get_open_positions()
            }
            filtered_opportunities = []
            seen_market_in_cycle = set()
            seen_event_in_cycle = set()

            # Keep one best opportunity per event using deterministic score ranking.
            best_opportunity_by_event = {}
            for market, est in opportunities:
                event_key = self.strategy.get_event_key_for_market(market)
                score = self.strategy.get_opportunity_score_tuple(market, est)
                existing = best_opportunity_by_event.get(event_key)
                if existing is None:
                    best_opportunity_by_event[event_key] = (market, est, score)
                else:
                    existing_market, _, existing_score = existing
                    if self.strategy.is_better_score(score, existing_score):
                        logger.info(
                            f"Skipping opportunity for {existing_market.market_id} "
                            f"(duplicate_event_guard: replaced_by_higher_score {market.market_id})"
                        )
                        report["counts"]["skipped_duplicates"] += 1
                        existing_row = market_rows_by_id.get(existing_market.market_id)
                        if existing_row is not None:
                            existing_row["execution"] = {
                                "success": False,
                                "skipped": True,
                                "reason": "duplicate_event_guard",
                            }
                        best_opportunity_by_event[event_key] = (market, est, score)
                    else:
                        logger.info(
                            f"Skipping opportunity for {market.market_id} "
                            f"(duplicate_event_guard: lower_score_same_event)"
                        )
                        report["counts"]["skipped_duplicates"] += 1
                        row = market_rows_by_id.get(market.market_id)
                        if row is not None:
                            row["execution"] = {
                                "success": False,
                                "skipped": True,
                                "reason": "duplicate_event_guard",
                            }

            deduped_opportunities = [
                (market, est) for market, est, _ in best_opportunity_by_event.values()
            ]
            deduped_opportunities.sort(
                key=lambda item: (
                    -self.strategy.get_opportunity_score_tuple(item[0], item[1])[0],
                    -self.strategy.get_opportunity_score_tuple(item[0], item[1])[1],
                    self.strategy.get_opportunity_score_tuple(item[0], item[1])[2],
                )
            )

            for market, est in deduped_opportunities:
                row = market_rows_by_id.get(market.market_id)
                if row is not None:
                    row["opportunity"] = {
                        "edge": est.effective_edge,
                        "confidence": est.effective_confidence,
                        "estimated_probability": est.effective_probability,
                    }

                market_key = f"{market.platform}:{market.market_id}"
                event_key = self.strategy.get_event_key_for_market(market)
                if market_key in open_market_keys or market_key in seen_market_in_cycle:
                    reason = "already open" if market_key in open_market_keys else "duplicate in cycle"
                    logger.info(f"Skipping opportunity for {market_key} (duplicate_market_guard: {reason})")
                    report["counts"]["skipped_duplicates"] += 1
                    if row is not None:
                        row["execution"] = {
                            "success": False,
                            "skipped": True,
                            "reason": "duplicate_market_guard",
                        }
                    continue
                if event_key in open_event_keys or event_key in seen_event_in_cycle:
                    reason = "already open event" if event_key in open_event_keys else "duplicate event in cycle"
                    logger.info(f"Skipping opportunity for {market_key} (duplicate_event_guard: {reason} {event_key})")
                    report["counts"]["skipped_duplicates"] += 1
                    if row is not None:
                        row["execution"] = {
                            "success": False,
                            "skipped": True,
                            "reason": "duplicate_event_guard",
                        }
                    continue

                filtered_opportunities.append((market, est))
                seen_market_in_cycle.add(market_key)
                seen_event_in_cycle.add(event_key)

            self._run_shadow_decision_logging(
                opportunities=filtered_opportunities,
                market_rows_by_id=market_rows_by_id,
                report=report,
            )
            
            # Step 5: Generate trade signals
            logger.info("\n📐 Step 5: Calculating position sizes (Kelly)...")

            # Check bankroll before generating signals
            current_bankroll = self.bankroll_manager.get_balance()
            current_exposure = self.position_manager.get_total_exposure()
            opened_today = self.position_manager.get_opened_exposure_today_utc()
            capital_base = max(0.0, current_bankroll + current_exposure)

            max_total_exposure = capital_base * self.config.risk.max_total_exposure_fraction
            remaining_total_exposure_room = max(0.0, max_total_exposure - current_exposure)

            max_new_exposure_today = capital_base * self.config.risk.max_new_exposure_per_day_fraction
            remaining_daily_exposure_room = max(0.0, max_new_exposure_today - opened_today)

            max_new_allocation = min(
                current_bankroll,
                remaining_total_exposure_room,
                remaining_daily_exposure_room,
            )

            if current_bankroll <= 0:
                logger.error("🛑 BANKROLL EXHAUSTED (<= 0). Trading execution disabled.")
                signals = []
            elif max_new_allocation <= 0:
                logger.info(
                    "🛑 Exposure cap reached. No new signals this cycle "
                    f"(cash=${current_bankroll:.2f}, exposure=${current_exposure:.2f}, "
                    f"opened_today=${opened_today:.2f}, total_room=${remaining_total_exposure_room:.2f}, "
                    f"daily_room=${remaining_daily_exposure_room:.2f})"
                )
                signals = []
            else:
                try:
                    signals = self.strategy.generate_trade_signals(
                        opportunities=filtered_opportunities,
                        current_bankroll=current_bankroll,
                        current_exposure=current_exposure,
                        current_open_positions=self.position_manager.get_position_count(),
                        current_open_market_keys=open_market_keys,
                        current_open_event_keys=open_event_keys,
                        max_new_allocation=max_new_allocation,
                    )
                except (InsufficientCapitalError, NoOpportunitiesError, PositionLimitError) as e:
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
                    "gross_edge": s.gross_edge,
                    "net_edge": s.net_edge,
                    "estimated_fee": s.estimated_fee,
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
            if signals and current_bankroll > 0:
                # Defense-in-depth: Filter out any duplicates that might have slipped through
                executable_signals = []
                seen_market_in_cycle = set()
                seen_event_in_cycle = set()

                # Keep one best signal per event.
                best_signal_by_event = {}
                for s in signals:
                    event_key = self.strategy.get_event_key_for_market(s.market)
                    row = market_rows_by_id.get(s.market.market_id)
                    analysis = row.get("analysis") if isinstance(row, dict) else {}
                    confidence = 0.0
                    if isinstance(analysis, dict):
                        confidence = float(
                            analysis.get("effective_confidence", analysis.get("confidence", 0.0))
                            or 0.0
                        )
                    score = (abs(float(s.edge)) * confidence, abs(float(s.edge)), str(s.market.market_id))
                    existing = best_signal_by_event.get(event_key)
                    if existing is None:
                        best_signal_by_event[event_key] = (s, score)
                    else:
                        existing_signal, existing_score = existing
                        if self.strategy.is_better_score(score, existing_score):
                            logger.warning(
                                f"Skipping signal for {existing_signal.market.market_id} "
                                f"(duplicate_event_guard: replaced_by_higher_score {s.market.market_id})"
                            )
                            report["counts"]["skipped_duplicates"] += 1
                            existing_row = market_rows_by_id.get(existing_signal.market.market_id)
                            if existing_row is not None:
                                existing_row["execution"] = {
                                    "success": False,
                                    "skipped": True,
                                    "reason": "duplicate_event_guard",
                                }
                            best_signal_by_event[event_key] = (s, score)
                        else:
                            logger.warning(
                                f"Skipping signal for {s.market.market_id} "
                                f"(duplicate_event_guard: lower_score_same_event)"
                            )
                            report["counts"]["skipped_duplicates"] += 1
                            row = market_rows_by_id.get(s.market.market_id)
                            if row is not None:
                                row["execution"] = {
                                    "success": False,
                                    "skipped": True,
                                    "reason": "duplicate_event_guard",
                                }

                deduped_signal_entries = list(best_signal_by_event.values())
                deduped_signal_entries.sort(
                    key=lambda item: (
                        -item[1][0],
                        -item[1][1],
                        item[0].market.market_id,
                    ),
                )
                deduped_signals = [item[0] for item in deduped_signal_entries]

                for s in deduped_signals:
                    market_key = f"{s.market.platform}:{s.market.market_id}"
                    event_key = self.strategy.get_event_key_for_market(s.market)

                    if market_key in open_market_keys or market_key in seen_market_in_cycle:
                        reason = "already open" if market_key in open_market_keys else "duplicate in cycle"
                        logger.warning(f"Skipping signal for {market_key} (duplicate_market_guard: {reason})")
                        report["counts"]["skipped_duplicates"] += 1

                        row = market_rows_by_id.get(s.market.market_id)
                        if row is not None:
                            row["execution"] = {
                                "success": False,
                                "skipped": True,
                                "reason": "duplicate_market_guard",
                            }
                        continue

                    if event_key in open_event_keys or event_key in seen_event_in_cycle:
                        reason = "already open event" if event_key in open_event_keys else "duplicate event in cycle"
                        logger.warning(f"Skipping signal for {market_key} (duplicate_event_guard: {reason} {event_key})")
                        report["counts"]["skipped_duplicates"] += 1

                        row = market_rows_by_id.get(s.market.market_id)
                        if row is not None:
                            row["execution"] = {
                                "success": False,
                                "skipped": True,
                                "reason": "duplicate_event_guard",
                            }
                        continue

                    executable_signals.append(s)
                    seen_market_in_cycle.add(market_key)
                    seen_event_in_cycle.add(event_key)

                if not executable_signals:
                    logger.info("No executable signals after duplicate guard filtering")
                else:
                    # Apply risk guards (kill switch, daily loss, per-cycle caps)
                    guarded_signals = await self._run_risk_guards(executable_signals)

                    if not guarded_signals:
                        logger.info("No signals remaining after risk guards")
                        return

                    # Defense-in-depth: final scope check before execution
                    execution_ready_signals = [
                        s for s in guarded_signals
                        if self._is_market_in_allowed_scope(s.market)
                    ]

                    if len(execution_ready_signals) < len(guarded_signals):
                        logger.warning(
                            f"Final scope guard excluded {len(guarded_signals) - len(execution_ready_signals)} "
                            "signals right before execution!"
                        )
                        for s in guarded_signals:
                            if s not in execution_ready_signals:
                                row = market_rows_by_id.get(s.market.market_id)
                                if row:
                                    row["execution"] = {
                                        "success": False,
                                        "skipped": True,
                                        "reason": "final_scope_guard_violation",
                                    }

                    if not execution_ready_signals:
                        logger.info("No signals remaining after final scope guard")
                        return

                    # Structured decision logs immediately before execution.
                    for s in execution_ready_signals:
                        self._log_structured_trade_decision(
                            s,
                            market_rows_by_id.get(s.market.market_id),
                        )

                    # Execute trades
                    try:
                        results = await self.executor.execute_signals(execution_ready_signals, cycle_id=self.cycle_count)
                        successful = sum(1 for r in results if r.get("success"))
                        logger.info(f"Executed {successful}/{len(execution_ready_signals)} trades")
                        report["counts"]["executed"] = int(successful)

                        for s, res in zip(execution_ready_signals, results):
                            row = market_rows_by_id.get(s.market.market_id)
                            status = res.get("status")
                            filled_qty = res.get("filled_quantity", 0.0)
                            avg_price = res.get("avg_fill_price", 0.0)
                            submitted_qty = float(res.get("submitted_quantity", 0.0) or 0.0)
                            submitted_price = float(res.get("submitted_price", 0.0) or 0.0)
                            submitted_notional = float(res.get("submitted_notional", 0.0) or 0.0)

                            if row is not None:
                                row["execution"] = {
                                    "success": res.get("success"),
                                    "order_id": res.get("order_id"),
                                    "status": status,
                                    "filled_quantity": filled_qty,
                                    "avg_fill_price": avg_price,
                                    "dry_run": self.config.is_dry_run,
                                }

                            # Persist a daily trade journal entry with reasoning (live and dry-run).
                            selection = (
                                s.market.yes_option
                                if s.action in ("buy_yes", "sell_no")
                                else s.market.no_option
                            ) or ("YES" if s.action in ("buy_yes", "sell_no") else "NO")
                            self._append_trade_journal_entry({
                                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                                "cycle_id": self.cycle_count,
                                "dry_run": self.config.is_dry_run,
                                "platform": s.market.platform,
                                "market_id": s.market.market_id,
                                "event_ticker": s.market.event_ticker,
                                "series_ticker": s.market.series_ticker,
                                "question": s.market.title,
                                "selection": selection,
                                "action": s.action,
                                "status": status,
                                "success": bool(res.get("success")),
                                "order_id": res.get("order_id"),
                                "position_size_requested": float(s.position_size),
                                "submitted_price": submitted_price,
                                "submitted_quantity": submitted_qty,
                                "submitted_notional": submitted_notional,
                                "filled_quantity": float(filled_qty or 0.0),
                                "avg_fill_price": float(avg_price or 0.0),
                                "edge": float(s.edge),
                                "fair_value": float(s.fair_value),
                                "market_price": float(s.market_price),
                                "kelly_fraction": float(s.kelly_fraction),
                                "expected_value": float(s.expected_value),
                                "reasoning": s.reasoning,
                                "error": res.get("error"),
                            })

                            # Explicit live logs for order placement/opening.
                            if (not self.config.is_dry_run) and res.get("success"):
                                logger.info(
                                    "💸 LIVE ORDER PLACED | %s | action=%s | order_id=%s | status=%s | "
                                    "submitted_qty=%.2f | submitted_price=%.4f | submitted_notional=$%.2f",
                                    s.market.market_id,
                                    s.action,
                                    res.get("order_id"),
                                    status,
                                    submitted_qty,
                                    submitted_price,
                                    submitted_notional,
                                )

                            # Only update local state if we have a confirmed fill or it's a dry run
                            # dry_run status is used for simulation success
                            if status in ("filled", "partially_filled", "dry_run") and filled_qty > 0:
                                logger.info(f"📈 Order {res.get('order_id')} had immediate fill: {filled_qty} contracts")
                                # Update local positions
                                await self.position_manager.add_position(
                                    s,
                                    external_order_id=res.get("order_id"),
                                    quantity=filled_qty,
                                    price=avg_price
                                )
                                # Update bankroll
                                cost = filled_qty * avg_price
                                await self.bankroll_manager.adjust_balance(
                                    -cost,
                                    reason="trade_execution",
                                    reference_id=res.get("order_id")
                                )
                                if not self.config.is_dry_run:
                                    logger.info(
                                        "📌 LIVE POSITION OPENED | %s | side=%s | qty=%.2f | price=%.4f | cost=$%.2f",
                                        s.market.market_id,
                                        "yes" if s.action in ("buy_yes", "sell_no") else "no",
                                        float(filled_qty),
                                        float(avg_price),
                                        float(cost),
                                    )
                            elif status in ("submitted", "pending_fill", "resting", "open"):
                                logger.info(f"⏳ Order {res.get('order_id')} accepted but not yet filled (status={status}). Will reconcile later.")
                            elif status == "pending_submit":
                                logger.warning(f"⚠️ Order for {s.market.market_id} stuck in pending_submit.")

                        # Update failure/cooldown tracking from execution outcomes.
                        await self._update_execution_failure_state(results)

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
            
            # Step 7: Deduct API costs
            logger.info("\n💸 Step 7: Deducting API costs...")
            try:
                stats = self.analyzer.get_api_stats()
                total_cost = float(stats.get('total_cost', 0.0))
                if total_cost > 0:
                    await self.bankroll_manager.adjust_balance(
                        -total_cost,
                        reason="api_cost",
                        reference_id=f"cycle_{self.cycle_count}"
                    )
            except Exception as e:
                logger.warning(f"Failed to deduct API costs: {e}")

            # Step 8: Report performance
            logger.info("\n📊 Step 8: Reporting performance...")
            self._print_dry_run_analysis_table(report)
            self._print_cycle_summary(cycle_start)
        
        except Exception as e:
            msg = f"Unexpected error in trading cycle: {e}"
            logger.error(msg)
            await self.notifier.send_notification(msg, level="ERROR")
            self.error_reporter.add_error_to_report(
                cycle_report, e, "trading cycle"
            )

            report["errors"].append({
                "type": type(e).__name__,
                "message": str(e),
                "stage": "trading cycle",
            })
        
        finally:
            # Finish cycle report in memory only.
            # JSON artifacts are intentionally limited to collect mode.
            report["finished_at"] = datetime.now().isoformat() + "Z"

            try:
                await self._persist_analysis_reasoning_to_db(report)
            except Exception as e:
                logger.warning(f"Failed persisting analysis blurbs from trading cycle: {e}")

            # Update heartbeat at end of cycle
            await self._update_heartbeat()

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
        bankroll = self.bankroll_manager.get_balance()
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 CYCLE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Duration: {cycle_time:.1f}s")
        logger.info(f"Current Bankroll: ${bankroll:,.2f}")
        logger.info(f"Open Positions: {stats['open_positions']}")
        logger.info(f"Total Exposure: ${stats['total_exposure']:,.2f}")
        
        if stats['total_trades'] > 0:
            logger.info(f"Trade Stats: {stats['total_trades']} trades "
                       f"({stats['winning_trades']}W/{stats['losing_trades']}L) "
                       f"| Win Rate: {stats['win_rate_percent']:.1f}%")
        
        logger.info("=" * 80 + "\n")
