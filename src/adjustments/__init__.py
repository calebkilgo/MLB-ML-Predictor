"""Runtime adjustment layer applied on top of the base model probability.

v8: all constants centralized in src/adjustments/weights.py. The
analyzer script can update that file based on logged outcomes.
"""
from __future__ import annotations

import math
from typing import Any

from src.adjustments.bullpen import bullpen_delta
from src.adjustments.lineup import lineup_delta
from src.adjustments.market import market_delta
from src.adjustments.park_wind import park_wind_delta
from src.adjustments.statcast_pitcher import statcast_pitcher_delta
from src.adjustments.umpire import umpire_total_delta
from src.adjustments.weights import CONFIDENCE_SHRINK


def _p_to_logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def _logit_to_p(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def _shrink(p: float, factor: float = CONFIDENCE_SHRINK) -> float:
    return 0.5 + (p - 0.5) * factor


def apply_adjustments(base_p: float, ctx: dict[str, Any]) -> tuple[float, dict]:
    z = _p_to_logit(base_p)
    breakdown: dict[str, float] = {"base_logit": z, "base_p": base_p}

    d_market = market_delta(base_p, ctx)
    d_statcast = statcast_pitcher_delta(ctx)
    d_lineup = lineup_delta(ctx)
    d_bullpen = bullpen_delta(ctx)
    d_wind = park_wind_delta(ctx)

    z_adj = z + d_market + d_statcast + d_lineup + d_bullpen + d_wind
    p_raw = _logit_to_p(z_adj)
    p_adj = _shrink(p_raw)

    breakdown.update({
        "market_logit_delta": d_market,
        "statcast_pitcher_logit_delta": d_statcast,
        "lineup_logit_delta": d_lineup,
        "bullpen_logit_delta": d_bullpen,
        "wind_logit_delta": d_wind,
        "total_logit_delta": z_adj - z,
        "pre_shrink_p": p_raw,
        "confidence_shrink_factor": CONFIDENCE_SHRINK,
        "adjusted_p": p_adj,
    })
    return p_adj, breakdown


def total_runs_adjustment(base_total: float, ctx: dict[str, Any]) -> tuple[float, dict]:
    ump = umpire_total_delta(ctx)
    wind = park_wind_delta(ctx, for_total=True)
    adj = base_total + ump + wind
    return adj, {
        "base_total": base_total,
        "umpire_runs_delta": ump,
        "wind_runs_delta": wind,
        "adjusted_total": adj,
    }