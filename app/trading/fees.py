"""Kalshi fee helpers used for net edge and EV calculations."""

from __future__ import annotations

import math


def estimate_kalshi_fee_from_count(
    count: int,
    side_price: float,
    fee_rate: float,
) -> float:
    """
    Estimate Kalshi order fee in dollars using a rate-based model.

    The fee model uses:
        fee = ceil(100 * fee_rate * count * price * (1 - price)) / 100
    """
    contracts = int(count)
    price = float(side_price)
    rate = float(fee_rate)

    if contracts <= 0 or rate <= 0:
        return 0.0

    price = min(max(price, 0.0), 1.0)
    gross_fee = rate * contracts * price * (1.0 - price)
    return math.ceil(gross_fee * 100.0) / 100.0


def estimate_kalshi_fee_edge(side_price: float, fee_rate: float) -> float:
    """
    Approximate fee impact as an edge penalty in probability points.

    This is the per-dollar-notional approximation:
        fee_edge ~= fee_rate * (1 - side_price)
    """
    price = min(max(float(side_price), 0.0), 1.0)
    rate = max(float(fee_rate), 0.0)
    return rate * (1.0 - price)
