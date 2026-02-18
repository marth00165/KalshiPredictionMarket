"""Portfolio reconciliation manager for syncing local state with Kalshi"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Any, Dict

import aiosqlite

from app.api_clients.kalshi_client import KalshiClient
from app.trading.position_manager import PositionManager
from app.storage.db import DatabaseManager

logger = logging.getLogger(__name__)

class ReconciliationManager:
    """
    Handles reconciliation between local database and platform portfolio.

    Ensures that PositionManager accurately reflects actual holdings on Kalshi.
    """

    def __init__(
        self,
        kalshi_client: KalshiClient,
        position_manager: PositionManager,
        db: Optional[DatabaseManager] = None,
        config: Optional[Any] = None,
    ):
        """
        Initialize ReconciliationManager.

        Args:
            kalshi_client: KalshiClient for fetching remote portfolio
            position_manager: PositionManager for managing local state
            db: Optional DatabaseManager for execution reconciliation
        """
        self.kalshi_client = kalshi_client
        self.position_manager = position_manager
        self.db = db
        self.config = config

    @staticmethod
    def _parse_iso_utc(value: Optional[str]) -> datetime:
        raw = str(value or "").strip()
        if not raw:
            return datetime.now(timezone.utc)
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _pending_not_found_retries(self) -> int:
        if self.config and getattr(self.config, "execution", None):
            return int(getattr(self.config.execution, "pending_not_found_retries", 3))
        return 3

    def _pending_timeout_minutes(self) -> int:
        if self.config and getattr(self.config, "execution", None):
            return int(getattr(self.config.execution, "pending_timeout_minutes", 30))
        return 30

    def _orders_reconcile_max_pages(self) -> int:
        if self.config and getattr(self.config, "execution", None):
            return int(getattr(self.config.execution, "order_reconciliation_max_pages", 5))
        return 5

    def _orders_reconcile_page_limit(self) -> int:
        if self.config and getattr(self.config, "execution", None):
            return int(getattr(self.config.execution, "order_reconciliation_page_limit", 200))
        return 200

    @staticmethod
    def _extract_settlement_outcome(market: Dict[str, Any]) -> Optional[str]:
        """
        Normalize settlement outcome to 'yes' or 'no' when determinable.
        """
        # Direct textual fields first
        direct_fields = (
            market.get("result"),
            market.get("outcome"),
            market.get("settlement_result"),
            market.get("winner"),
            market.get("settlement"),
        )
        for raw in direct_fields:
            norm = str(raw or "").strip().lower()
            if norm in {"yes", "y", "true", "1"}:
                return "yes"
            if norm in {"no", "n", "false", "0"}:
                return "no"

        # Numeric settlement fields as fallback.
        for key in ("settlement_price", "settle_price", "yes_settlement_price"):
            val = market.get(key)
            if val is None:
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
            if num > 1:
                num = num / 100.0
            if 0.0 <= num <= 1.0:
                return "yes" if num >= 0.5 else "no"
        return None

    async def _resolve_exit_price(self, ticker: str, side: str, fallback: float) -> float:
        """
        Resolve exit price for a local position that disappeared remotely.

        close_position expects price of the side held:
        - side=yes -> yes price
        - side=no  -> no price (= 1 - yes price)
        """
        try:
            yes_price = await self.kalshi_client.get_market_yes_price(ticker)
            if side == "no":
                return max(0.0, min(1.0, 1.0 - yes_price))
            return max(0.0, min(1.0, yes_price))
        except Exception as e:
            logger.warning(
                f"⚠️ Could not fetch live exit price for {ticker} (side={side}). "
                f"Falling back to entry price. Error: {e}"
            )
            return fallback

    async def compute_true_equity(self, cash_balance: float) -> float:
        """
        Compute mark-to-market equity as cash + current value of open positions.
        """
        remote_positions = await self.kalshi_client.get_positions()
        mtm_value = 0.0
        for pos in remote_positions:
            ticker = pos.get("market_id")
            if not ticker:
                continue
            qty = float(pos.get("quantity", 0.0) or 0.0)
            side = str(pos.get("side", "yes")).lower()
            try:
                yes_price = await self.kalshi_client.get_market_yes_price(ticker)
                side_price = yes_price if side == "yes" else (1.0 - yes_price)
                mtm_value += qty * side_price
            except Exception as e:
                logger.warning(f"⚠️ Equity MTM pricing failed for {ticker}, skipping position: {e}")
        return float(cash_balance) + float(mtm_value)

    async def reconcile_settled_positions(self):
        """
        Close local positions with deterministic payout when market settlement outcome is known.
        """
        local_positions = list(self.position_manager.get_open_positions())
        if not local_positions:
            return

        # One closure operation handles all open fills for the market.
        market_to_side: Dict[str, str] = {}
        for local_pos in local_positions:
            market_to_side.setdefault(local_pos.market_id, local_pos.side)

        for market_id, local_side in market_to_side.items():
            market = await self.kalshi_client.get_market(market_id)
            if not market:
                continue

            outcome = self._extract_settlement_outcome(market)
            if outcome not in {"yes", "no"}:
                continue

            # Deterministic payout at settlement:
            # winning side pays 1.0, losing side 0.0.
            exit_price = 1.0 if local_side == outcome else 0.0
            logger.info(
                f"🏁 Settled market detected for {market_id}: "
                f"outcome={outcome}, local_side={local_side}, exit_price={exit_price:.2f}"
            )
            await self.position_manager.close_position(market_id, exit_price=exit_price)

    async def reconcile_pending_executions(self):
        """
        Check for any executions that are not in terminal states and resolve them.
        """
        if not self.db:
            return

        # Statuses that are non-terminal and need checking.
        # Also backfill legacy rows that are marked filled/executed but
        # still have no recorded fills.
        check_statuses = (
            'pending_submit', 'submitted', 'pending_fill',
            'resting', 'open', 'partially_filled'
        )

        async with self.db.connect() as db:
            db.row_factory = aiosqlite.Row
            query = f"""
                SELECT * FROM executions
                WHERE platform = 'kalshi'
                  AND (
                        status IN ({','.join(['?']*len(check_statuses))})
                        OR (
                            status IN ('filled', 'executed')
                            AND COALESCE(filled_quantity, 0) <= 0
                            AND COALESCE(remaining_quantity, 0) > 0
                        )
                  )
            """
            async with db.execute(query, check_statuses) as cursor:
                pending = await cursor.fetchall()

        if not pending:
            return

        logger.info(f"🔄 Found {len(pending)} pending/non-terminal executions to reconcile.")

        # Fetch recent orders from Kalshi with pagination support for robustness.
        recent_orders = await self.kalshi_client.get_orders(
            max_pages=self._orders_reconcile_max_pages(),
            limit=self._orders_reconcile_page_limit(),
        )
        orders_by_client_id = {o.get('client_order_id'): o for o in recent_orders if o.get('client_order_id')}
        orders_by_external_id = {o.get('order_id'): o for o in recent_orders if o.get('order_id')}

        max_retries = self._pending_not_found_retries()
        timeout_delta = timedelta(minutes=self._pending_timeout_minutes())
        now = datetime.now(timezone.utc)

        async with self.db.connect() as db:
            for row in pending:
                coid = row['client_order_id']
                eoid = row['external_order_id']

                order = None
                if coid and coid in orders_by_client_id:
                    order = orders_by_client_id[coid]
                elif eoid and eoid in orders_by_external_id:
                    order = orders_by_external_id[eoid]

                if order:
                    status = str(order.get('status', 'unknown')).lower()
                    total_count = float(row['quantity'] or 0.0)

                    new_filled = float(order.get('filled_count', 0) or 0.0)
                    old_filled = float(row['filled_quantity'] or 0.0)
                    old_avg_price = float(row['avg_fill_price'] or 0.0)

                    # Safety fallback: some order payloads omit fill_count even
                    # when the status is already terminal.
                    if status in ('filled', 'executed') and new_filled <= 0:
                        new_filled = total_count

                    delta_filled = new_filled - old_filled
                    remaining_quantity = max(0.0, total_count - new_filled)

                    # Update execution record
                    if order.get('avg_fill_price'):
                        avg_fill_price = float(order.get('avg_fill_price', 0)) / 100.0
                    elif old_avg_price > 0:
                        avg_fill_price = old_avg_price
                    else:
                        avg_fill_price = float(row['price'] or 0.0)

                    logger.info(f"✅ Found order {coid or eoid} on exchange. Status: {status}, New Fill: {delta_filled}")

                    await db.execute("""
                        UPDATE executions SET
                            status = ?,
                            external_order_id = ?,
                            filled_quantity = ?,
                            avg_fill_price = ?,
                            remaining_quantity = ?,
                            reconcile_attempts = 0,
                            last_reconciled_at_utc = ?
                        WHERE id = ?
                    """, (
                        status,
                        order.get('order_id'),
                        new_filled,
                        avg_fill_price,
                        remaining_quantity,
                        now.isoformat().replace("+00:00", "Z"),
                        row['id'],
                    ))

                    # If new fills detected, update PositionManager and BankrollManager
                    if delta_filled > 0:
                        # For bankroll, we must use the correct incremental cost basis:
                        # (new_total * new_avg) - (old_total * old_avg)
                        new_total_cost = new_filled * avg_fill_price
                        old_total_cost = old_filled * old_avg_price
                        incremental_cost = new_total_cost - old_total_cost

                        # When adding to PositionManager, we add the delta quantity.
                        # To keep cost basis consistent, the "price" for this delta fill
                        # should be the incremental cost divided by the delta quantity.
                        effective_incremental_price = incremental_cost / delta_filled if delta_filled > 0 else avg_fill_price

                        await self.position_manager.add_fill(
                            market_id=row['market_id'],
                            platform=row['platform'],
                            action=row['action'],
                            quantity=delta_filled,
                            price=effective_incremental_price,
                            external_order_id=order.get('order_id')
                        )

                        if self.position_manager.bankroll_manager:
                            await self.position_manager.bankroll_manager.adjust_balance(
                                -incremental_cost,
                                reason="trade_execution_reconciled",
                                reference_id=order.get('order_id')
                            )
                else:
                    attempts = int(row["reconcile_attempts"] or 0) + 1
                    pending_since = self._parse_iso_utc(row["executed_at_utc"] or row["created_at_utc"])
                    is_stale = (now - pending_since) >= timeout_delta
                    should_fail = attempts >= max_retries or is_stale

                    if should_fail:
                        logger.warning(
                            f"❌ Pending order {coid or eoid} NOT found on exchange "
                            f"(attempts={attempts}/{max_retries}, stale={is_stale}). Marking as failed_not_found."
                        )
                        await db.execute("""
                            UPDATE executions
                            SET status = 'failed_not_found',
                                reconcile_attempts = ?,
                                last_reconciled_at_utc = ?
                            WHERE id = ?
                        """, (
                            attempts,
                            now.isoformat().replace("+00:00", "Z"),
                            row["id"],
                        ))
                    else:
                        logger.info(
                            f"⏳ Pending order {coid or eoid} not visible yet "
                            f"(attempts={attempts}/{max_retries}). Retaining status={row['status']}."
                        )
                        await db.execute("""
                            UPDATE executions
                            SET reconcile_attempts = ?,
                                last_reconciled_at_utc = ?
                            WHERE id = ?
                        """, (
                            attempts,
                            now.isoformat().replace("+00:00", "Z"),
                            row["id"],
                        ))
            await db.commit()

    async def reconcile(self):
        """
        Run full reconciliation cycle.
        """
        # 0. Reconcile pending executions first
        await self.reconcile_pending_executions()
        await self.reconcile_settled_positions()

        logger.info("🔄 Starting portfolio reconciliation...")

        # 1. Fetch remote positions from Kalshi
        remote_positions = await self.kalshi_client.get_positions()
        remote_map = {p['market_id']: p for p in remote_positions}

        # 2. Get local open positions
        local_positions = self.position_manager.get_open_positions()
        local_map = {p.market_id: p for p in local_positions}

        # 3. Find positions in Kalshi but not in local state (Recover)
        for ticker, remote_pos in remote_map.items():
            if ticker not in local_map:
                logger.warning(f"⚠️ Found unknown position on Kalshi: {ticker}. Recovering...")
                await self.position_manager.add_recovered_position(remote_pos)
            else:
                # Check for significant mismatch (e.g., quantity)
                local_pos = local_map[ticker]
                if abs(local_pos.quantity - remote_pos['quantity']) > 0.001:
                    logger.warning(f"⚠️ Quantity mismatch for {ticker}: local={local_pos.quantity}, remote={remote_pos['quantity']}")
                    # For now we just log it, but we could update the local quantity

        # 4. Find positions in local state but not on Kalshi (Close)
        for ticker, local_pos in local_map.items():
            if ticker not in remote_map:
                logger.info(f"ℹ️ Position {ticker} no longer on Kalshi. Marking as closed.")
                exit_price = await self._resolve_exit_price(
                    ticker=ticker,
                    side=local_pos.side,
                    fallback=local_pos.entry_price,
                )
                await self.position_manager.close_position(ticker, exit_price=exit_price)

        logger.info("✅ Portfolio reconciliation complete.")
