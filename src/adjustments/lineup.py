"""Lineup-based wOBA-vs-handedness adjustment with Bayesian shrinkage.

Constants WOBA_PER_LOGIT and SHRINKAGE_K are read from weights.py.
"""
from __future__ import annotations

from typing import Any

from src.adjustments.weights import SHRINKAGE_K, WOBA_PER_LOGIT


def _shrink(woba: float | None, pa: float | None, league: float) -> float | None:
    if woba is None:
        return None
    if pa is None or pa <= 0:
        return league
    return (woba * pa + league * SHRINKAGE_K) / (pa + SHRINKAGE_K)


def lineup_delta(ctx: dict[str, Any]) -> float:
    league = ctx.get("league_avg_woba", 0.315)
    h = _shrink(ctx.get("home_lineup_woba_vs_hand"),
                ctx.get("home_lineup_pa_vs_hand"), league)
    a = _shrink(ctx.get("away_lineup_woba_vs_hand"),
                ctx.get("away_lineup_pa_vs_hand"), league)
    if h is None or a is None:
        return 0.0
    net = (h - league) - (a - league)
    return net / WOBA_PER_LOGIT