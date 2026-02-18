"""Portfolio reconciliation manager for syncing local state with Kalshi"""

import logging

from app.api_clients.kalshi_client import KalshiClient
from app.trading.position_manager import PositionManager

logger = logging.getLogger(__name__)

class ReconciliationManager:
    """
    Handles reconciliation between local database and platform portfolio.

    Ensures that PositionManager accurately reflects actual holdings on Kalshi.
    """

    def __init__(self, kalshi_client: KalshiClient, position_manager: PositionManager):
        """
        Initialize ReconciliationManager.

        Args:
            kalshi_client: KalshiClient for fetching remote portfolio
            position_manager: PositionManager for managing local state
        """
        self.kalshi_client = kalshi_client
        self.position_manager = position_manager

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

    async def reconcile(self):
        """
        Run full reconciliation cycle.
        """
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
