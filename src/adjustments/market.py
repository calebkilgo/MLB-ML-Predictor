"""Market-aware adjustment: blend the model's probability with the
sportsbook-implied probability.

Weight is read from src.adjustments.weights.W_MARKET so the analyzer
can tune it.
"""
from __future__ import annotations

import math
from typing import Any

from src.adjustments.weights import W_MARKET


def american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def devig_two_way(p_a: float, p_b: float) -> tuple[float, float]:
    total = p_a + p_b
    if total <= 0:
        return 0.5, 0.5
    return p_a / total, p_b / total


def _logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def market_delta(base_p: float, ctx: dict[str, Any]) -> float:
    h = ctx.get("market_home_american")
    a = ctx.get("market_away_american")
    if h is None or a is None:
        return 0.0
    try:
        p_h_raw = american_to_implied(int(h))
        p_a_raw = american_to_implied(int(a))
        p_h_fair, _ = devig_two_way(p_h_raw, p_a_raw)
    except Exception:
        return 0.0
    target = (1 - W_MARKET) * base_p + W_MARKET * p_h_fair
    return _logit(target) - _logit(base_p)