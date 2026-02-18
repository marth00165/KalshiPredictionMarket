"""Portfolio reconciliation manager for syncing local state with Kalshi"""

import logging
from typing import Optional

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

    def __init__(self, kalshi_client: KalshiClient, position_manager: PositionManager, db: Optional[DatabaseManager] = None):
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

    async def reconcile_pending_executions(self):
        """
        Check for any executions that are not in terminal states and resolve them.
        """
        if not self.db:
            return

        # Statuses that are non-terminal and need checking
        check_statuses = (
            'pending_submit', 'submitted', 'pending_fill',
            'resting', 'open', 'partially_filled'
        )

        async with self.db.connect() as db:
            db.row_factory = aiosqlite.Row
            query = f"SELECT * FROM executions WHERE status IN ({','.join(['?']*len(check_statuses))}) AND platform = 'kalshi'"
            async with db.execute(query, check_statuses) as cursor:
                pending = await cursor.fetchall()

        if not pending:
            return

        logger.info(f"🔄 Found {len(pending)} pending/non-terminal executions to reconcile.")

        # Fetch recent orders from Kalshi
        recent_orders = await self.kalshi_client.get_orders()
        orders_by_client_id = {o.get('client_order_id'): o for o in recent_orders if o.get('client_order_id')}
        orders_by_external_id = {o.get('order_id'): o for o in recent_orders if o.get('order_id')}

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
                    status = order.get('status', 'unknown')
                    new_filled = float(order.get('filled_count', 0))
                    old_filled = float(row['filled_quantity'] or 0.0)
                    old_avg_price = float(row['avg_fill_price'] or 0.0)
                    delta_filled = new_filled - old_filled

                    # Total contracts requested (from original execution record)
                    total_count = float(row['quantity'] or 0.0)
                    remaining_quantity = max(0.0, total_count - new_filled)

                    # Update execution record
                    avg_fill_price = float(order.get('avg_fill_price', 0)) / 100.0 if order.get('avg_fill_price') else 0.0

                    logger.info(f"✅ Found order {coid or eoid} on exchange. Status: {status}, New Fill: {delta_filled}")

                    await db.execute("""
                        UPDATE executions SET
                            status = ?,
                            external_order_id = ?,
                            filled_quantity = ?,
                            avg_fill_price = ?,
                            remaining_quantity = ?
                        WHERE id = ?
                    """, (status, order.get('order_id'), new_filled, avg_fill_price, remaining_quantity, row['id']))

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
                    # If not found in recent orders, it might be older or failed to submit
                    # For now, if it's 'pending_submit', we mark it failed after reconciliation if not found.
                    if row['status'] == 'pending_submit':
                        logger.warning(f"❌ Pending order {coid} NOT found on exchange. Marking as failed.")
                        await db.execute(
                            "UPDATE executions SET status = 'failed_not_found' WHERE id = ?",
                            (row['id'],)
                        )
            await db.commit()

    async def reconcile(self):
        """
        Run full reconciliation cycle.
        """
        # 0. Reconcile pending executions first
        await self.reconcile_pending_executions()

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
