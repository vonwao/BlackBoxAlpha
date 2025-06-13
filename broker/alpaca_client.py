import logging
import time
from typing import Optional

import alpaca_trade_api as tradeapi

logger = logging.getLogger(__name__)


class AlpacaClient:
    """Lightweight wrapper around alpaca-trade-api."""

    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version="v2",
        )
        logger.info("Alpaca client initialized")

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
    ):
        """Submit an order and return the order object."""
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side.lower(),
                type=order_type,
                time_in_force=time_in_force,
            )
            logger.info("Submitted %s order for %s (%s)", side, symbol, order.id)
            return order
        except Exception as exc:
            logger.error("Error submitting order for %s: %s", symbol, exc)
            raise

    def close_position(self, symbol: str):
        """Close an open position."""
        try:
            order = self.api.close_position(symbol)
            logger.info("Close order submitted for %s (%s)", symbol, order.id)
            return order
        except Exception as exc:
            logger.error("Error closing position %s: %s", symbol, exc)
            raise

    def wait_for_fill(self, order_id: str, timeout: int = 30) -> Optional[tradeapi.entity.Order]:
        """Wait until the given order is filled or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            order = self.api.get_order(order_id)
            if order.status == "filled":
                return order
            time.sleep(1)
        logger.warning("Order %s not filled within %s seconds", order_id, timeout)
        return None

    def get_account(self):
        """Return account information."""
        try:
            return self.api.get_account()
        except Exception as exc:
            logger.error("Error fetching account info: %s", exc)
            raise
