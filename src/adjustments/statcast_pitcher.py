"""Statcast expected-stats adjustment for starting pitchers.

Constant ERA_GAP_TO_LOGIT is read from weights.py.
"""
from __future__ import annotations

from typing import Any

from src.adjustments.weights import ERA_GAP_TO_LOGIT


def statcast_pitcher_delta(ctx: dict[str, Any]) -> float:
    h_x = ctx.get("home_sp_xera")
    a_x = ctx.get("away_sp_xera")
    h_e = ctx.get("home_sp_era")
    a_e = ctx.get("away_sp_era")
    if None in (h_x, a_x, h_e, a_e):
        return 0.0
    home_luck = float(h_e) - float(h_x)
    away_luck = float(a_e) - float(a_x)
    return (away_luck - home_luck) * ERA_GAP_TO_LOGIT