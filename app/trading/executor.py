"""Trade executor for executing signals (placeholder for implementation)"""

import logging
import uuid
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.models import TradeSignal
from app.storage.db import DatabaseManager
from app.api_clients.kalshi_client import KalshiClient
from app.utils import (
    ConfigManager,
    ExecutionFailedError,
    OrderPlacementError,
    DryRunError,
    ErrorContext,
)

logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    Executes trade signals on actual trading platforms with persistence.
    """
    
    def __init__(self, config: ConfigManager, db: DatabaseManager, kalshi_client: Optional[KalshiClient] = None):
        """
        Initialize trade executor
        
        Args:
            config: ConfigManager with trading settings
            db: DatabaseManager for persistence
            kalshi_client: Optional KalshiClient for execution
        """
        self.config = config
        self.db = db
        self.kalshi_client = kalshi_client
        self.dry_run = config.is_dry_run
        self.executed_trades = []

    @staticmethod
    def _is_terminal_idempotent_status(status: Optional[str]) -> bool:
        """
        Return True if an execution status should short-circuit duplicate attempts.

        Retryable statuses (e.g. skipped_drift/skipped_edge/failed) are intentionally excluded.
        """
        status_norm = str(status or "").strip().lower()
        return status_norm in {
            "executed",
            "submitted",
            "dry_run",
            "pending_submit",
            "pending_fill",
            "resting",
            "open",
            "filled",
            "partially_filled",
        }
    
    async def execute_signals(self, signals: List[TradeSignal], cycle_id: int = 0) -> List[Dict[str, Any]]:
        """
        Execute a batch of trade signals
        
        Args:
            signals: List of TradeSignal objects to execute
            cycle_id: Current trading cycle ID
        
        Returns:
            List of result dicts for each signal
        """
        
        if not signals:
            logger.warning("No signals to execute")
            return []
        
        results = []
        error_context = ErrorContext("Batch execution", critical=False)
        
        for signal in signals:
            try:
                result = await self.execute_signal(signal, cycle_id=cycle_id)
                results.append(result)
            except Exception as e:
                error_context.add_error(e)
                logger.error(f"Error executing signal for {signal.market.market_id}: {e}")
                results.append({"success": False, "error": str(e)})
        
        successful = sum(1 for r in results if r.get("success"))
        logger.info(f"Executed {successful}/{len(signals)} signals successfully")
        
        return results
    
    async def execute_signal(self, signal: TradeSignal, cycle_id: int = 0) -> Dict[str, Any]:
        """
        Execute a single trade signal
        
        Args:
            signal: TradeSignal to execute
            cycle_id: Current trading cycle ID
        
        Returns:
            Dict containing success status and order_id
        """
        
        # Log the signal
        self._log_signal(signal)
        
        client_order_id = self._generate_client_order_id(signal)

        # Idempotency short-circuit: Check if this signal was already processed
        async with self.db.connect() as db:
            import aiosqlite
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT status, external_order_id FROM executions WHERE client_order_id = ?",
                (client_order_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    status = row['status']
                    # Only terminal/non-retryable statuses should short-circuit.
                    if self._is_terminal_idempotent_status(status):
                        logger.info(f"⏭️ Signal already handled (status={status}), short-circuiting: {signal}")
                        return {
                            "success": status in ('executed', 'submitted', 'dry_run', 'filled', 'partially_filled'),
                            "order_id": row['external_order_id'],
                            "status": status,
                            "idempotent": True
                        }

        if self.dry_run:
            # Simulate success
            logger.info(
                f"DRY RUN: would execute {signal.action} on {signal.market.platform} "
                f"for {signal.market.market_id} (${signal.position_size:,.2f})"
            )

            # Determine price/quantity for dry run record
            if signal.action in ('buy_yes', 'sell_no'):
                price = signal.market.yes_price
            else:
                price = signal.market.no_price
            quantity = signal.position_size / max(0.01, price)

            result = {
                "success": True,
                "order_id": f"dry_run_{uuid.uuid4().hex[:8]}",
                "status": "dry_run",
                "filled_quantity": quantity,
                "avg_fill_price": price,
                "submitted_price": price,
                "submitted_quantity": quantity,
                "submitted_notional": quantity * price,
            }
            await self._record_execution(signal, result, cycle_id=cycle_id, client_order_id=client_order_id)
            return result
        
        # Record pending execution for crash idempotency
        await self._record_execution(
            signal,
            {"success": False, "status": "pending_submit"},
            cycle_id=cycle_id,
            client_order_id=client_order_id
        )

        try:
            result = await self._execute_on_platform(signal, client_order_id=client_order_id)
            if result.get("success"):
                logger.info(f"✅ Trade executed: {signal} (Order ID: {result.get('order_id')})")
            else:
                if result.get("skipped"):
                    logger.warning(f"⏭️ Execution skipped for {signal.market.market_id}: {result.get('error')}")
                else:
                    logger.error(f"❌ Execution failed for {signal.market.market_id}: {result.get('error')}")

            await self._record_execution(signal, result, cycle_id=cycle_id, client_order_id=client_order_id)
            return result
        
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            await self._record_execution(signal, error_result, cycle_id=cycle_id, client_order_id=client_order_id)
            return error_result

    def _generate_client_order_id(self, signal: TradeSignal) -> str:
        """Generate a deterministic client order ID for idempotency."""
        # Derived from stable signal identity only: platform, market, action, rounded price/size
        raw = f"{signal.market.platform}:{signal.market.market_id}:{signal.action}:{signal.market_price:.2f}:{signal.position_size:.2f}"
        return hashlib.sha256(raw.encode()).hexdigest()
    
    async def _record_execution(self, signal: TradeSignal, result: Dict[str, Any], cycle_id: int = 0, client_order_id: Optional[str] = None):
        """Persist execution record to database."""
        now = datetime.utcnow().isoformat() + "Z"

        # Persist actual execution-time values when available.
        submitted_price = result.get("submitted_price")
        submitted_quantity = result.get("submitted_quantity")
        submitted_notional = result.get("submitted_notional")

        # Fall back to signal-time values if execution-time values are unavailable.
        if submitted_price is None:
            if signal.action in ('buy_yes', 'sell_no'):
                submitted_price = signal.market.yes_price
            else:
                submitted_price = signal.market.no_price

        if submitted_quantity is None:
            submitted_quantity = signal.position_size / max(0.01, float(submitted_price))
        if submitted_notional is None:
            submitted_notional = float(submitted_quantity) * float(submitted_price)

        price = float(submitted_price)
        quantity = float(submitted_quantity)
        cost = float(submitted_notional)
        status = result.get("status", "failed")
        filled_quantity = result.get("filled_quantity", 0.0)
        avg_fill_price = result.get("avg_fill_price", 0.0)
        remaining_quantity = max(0.0, quantity - filled_quantity)

        async with self.db.connect() as db:
            if client_order_id:
                # Check if record already exists for this client_order_id
                async with db.execute("SELECT id FROM executions WHERE client_order_id = ?", (client_order_id,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        # Update existing record
                        await db.execute("""
                            UPDATE executions SET
                                quantity = ?,
                                price = ?,
                                cost = ?,
                                status = ?,
                                external_order_id = ?,
                                filled_quantity = ?,
                                avg_fill_price = ?,
                                remaining_quantity = ?,
                                reconcile_attempts = 0,
                                executed_at_utc = ?
                            WHERE client_order_id = ?
                        """, (
                            quantity, price, cost,
                            status, result.get("order_id"),
                            filled_quantity, avg_fill_price,
                            remaining_quantity,
                            now, client_order_id
                        ))
                        await db.commit()
                        return

            await db.execute("""
                INSERT INTO executions
                (market_id, platform, action, quantity, price, cost, external_order_id, status, executed_at_utc, created_at_utc, cycle_id, client_order_id, filled_quantity, avg_fill_price, remaining_quantity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.market.market_id, signal.market.platform, signal.action,
                quantity, price, cost, result.get("order_id"),
                status, now, now, cycle_id, client_order_id,
                filled_quantity, avg_fill_price, remaining_quantity
            ))
            await db.commit()

    async def _execute_on_platform(self, signal: TradeSignal, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute trade on the appropriate platform
        """
        
        if signal.market.platform == 'polymarket':
            return await self._execute_polymarket(signal)
        elif signal.market.platform == 'kalshi':
            return await self._execute_kalshi(signal, client_order_id=client_order_id)
        else:
            return {"success": False, "error": f"Unknown platform: {signal.market.platform}"}
    
    async def _execute_polymarket(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute trade on Polymarket (stub)"""
        # TODO: Implement Polymarket order placement
        logger.info(f"Polymarket execution not implemented: {signal}")
        return {"success": False, "error": "Polymarket execution not implemented"}
    
    async def _execute_kalshi(self, signal: TradeSignal, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute trade on Kalshi with price revalidation"""
        if not self.kalshi_client:
            return {"success": False, "error": "Kalshi client not configured for execution"}

        try:
            # 1. Fetch latest price for revalidation
            latest_yes_price = await self.kalshi_client.get_market_yes_price(signal.market.market_id)

            # 2. Standardize action and side for Kalshi
            # action: 'buy' or 'sell'
            # side: 'yes' or 'no'
            if signal.action.startswith('buy_'):
                action = 'buy'
                side = signal.action.replace('buy_', '')
            elif signal.action.startswith('sell_'):
                action = 'sell'
                side = signal.action.replace('sell_', '')
            else:
                return {"success": False, "error": f"Unsupported action: {signal.action}"}

            # Price in cents
            latest_price = latest_yes_price if side == 'yes' else (1.0 - latest_yes_price)
            original_price = signal.market_price

            # 3. Revalidate price drift
            drift = abs(latest_price - original_price)
            if drift > self.config.max_price_drift:
                return {
                    "success": False,
                    "error": f"Price drift too large: {drift:.3f} > {self.config.max_price_drift}",
                    "skipped": True,
                    "status": "skipped_drift"
                }

            # 4. Revalidate edge
            # edge = fair_value - latest_price (for buy_yes)
            # edge = (1 - fair_value) - latest_price (for buy_no)
            if signal.action == 'buy_yes':
                new_edge = signal.fair_value - latest_price
            elif signal.action == 'buy_no':
                new_edge = (1.0 - signal.fair_value) - latest_price
            else:
                new_edge = signal.edge # fallback for sells

            if new_edge < self.config.min_edge_at_execution:
                return {
                    "success": False,
                    "error": f"Edge decayed below minimum: {new_edge:.3f} < {self.config.min_edge_at_execution}",
                    "skipped": True,
                    "status": "skipped_edge"
                }

            # Price in cents
            price_cents = int(latest_price * 100)
            submitted_price = price_cents / 100.0

            # 4b. Slippage guard at submit-time (independent from drift guard)
            execution_cfg = getattr(self.config, "execution", None)
            max_submit_slippage = getattr(execution_cfg, "max_submit_slippage", None)
            if max_submit_slippage is None:
                max_submit_slippage = getattr(self.config, "max_submit_slippage", 0.10)
            submit_slippage = abs(submitted_price - original_price)
            if submit_slippage > float(max_submit_slippage):
                return {
                    "success": False,
                    "error": f"Submit slippage too large: {submit_slippage:.3f} > {float(max_submit_slippage):.3f}",
                    "skipped": True,
                    "status": "skipped_slippage",
                    "submitted_price": submitted_price,
                }

            # Count
            count = int(signal.position_size / max(0.01, latest_price))

            if count <= 0:
                return {"success": False, "error": f"Calculated count is zero: size=${signal.position_size}, price={price_cents}c"}

            result = await self.kalshi_client.place_order(
                ticker=signal.market.market_id,
                side=side,
                action=action,
                count=count,
                price_cents=price_cents,
                client_order_id=client_order_id or uuid.uuid4().hex
            )

            # Map Kalshi response to our execution lifecycle
            # result contains: order_id, status, raw_response
            raw = result.get("raw_response", {})
            status = result.get("status", "unknown")

            # Kalshi V2 sometimes returns filled info in order placement response
            # if it was matched immediately.
            filled_quantity = float(raw.get("filled_count", 0.0) or 0.0)
            avg_fill_price = float(raw.get("avg_fill_price", 0.0)) / 100.0 if raw.get("avg_fill_price") else 0.0

            # Safety fallback: some responses mark an order as filled/executed
            # without including fill_count/avg_fill_price.
            status_norm = str(status).lower()
            if status_norm in ("filled", "executed") and filled_quantity <= 0:
                filled_quantity = float(count)
            if filled_quantity > 0 and avg_fill_price <= 0:
                avg_fill_price = latest_price

            submitted_notional = float(count) * float(submitted_price)

            return {
                "success": result.get("order_id") is not None,
                "order_id": result.get("order_id"),
                "status": status,
                "filled_quantity": filled_quantity,
                "avg_fill_price": avg_fill_price,
                "submitted_price": submitted_price,
                "submitted_quantity": float(count),
                "submitted_notional": submitted_notional,
                "raw": raw
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _log_signal(self, signal: TradeSignal) -> None:
        """Log detailed signal information"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 TRADE SIGNAL")
        logger.info(f"{'='*80}")
        logger.info(f"Market: {signal.market.title[:60]}...")
        logger.info(f"Platform: {signal.market.platform}")
        logger.info(f"Action: {signal.action}")
        logger.info(f"Fair Value: {signal.fair_value:.1%}")
        logger.info(f"Market Price: {signal.market_price:.1%}")
        logger.info(f"Edge: {signal.edge:+.1%}")
        logger.info(f"Kelly Fraction: {signal.kelly_fraction:.2%}")
        logger.info(f"Position Size: ${signal.position_size:,.2f}")
        logger.info(f"Expected Value: ${signal.expected_value:+,.2f}")
        logger.info(f"Reasoning: {signal.reasoning}...")
        logger.info(f"{'='*80}")
    
    def get_execution_stats(self) -> dict:
        """Get statistics about executed trades"""
        
        return {
            'total_executed': len(self.executed_trades),
            'dry_run_mode': self.dry_run,
            'executed_trades': self.executed_trades,
        }
