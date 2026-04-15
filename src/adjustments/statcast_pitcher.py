"""Statcast expected-stats and pitch-quality adjustment for starting pitchers.

Three complementary signals are combined:

1. xERA gap (ERA - xERA): positive = pitcher has been lucky (ERA < xERA
   means the ball has gone his way; he should regress). Negative = unlucky.
   Weight: ERA_GAP_TO_LOGIT per run-gap.

2. Whiff rate: strikeout-generating stuff.  League average ≈ 24%.
   A pitcher at 30% whiff vs 24% avg has ~+0.06 logit edge.
   Weight: WHIFF_PER_LOGIT per percentage-point above/below average.

3. Hard-hit rate: quality-of-contact allowed.  League average ≈ 38%.
   A pitcher at 32% hard-hit (good) vs 38% avg has ~+0.04 logit edge.
   Weight: HARD_HIT_PER_LOGIT per pp below average (lower = better).

All constants are read from weights.py so auto-calibration can tune them.
"""
from __future__ import annotations

from typing import Any

from src.adjustments.weights import ERA_GAP_TO_LOGIT

# League-average baselines (2023-2024 season)
_LEAGUE_WHIFF_PCT = 24.0   # %
_LEAGUE_HARD_HIT_PCT = 38.0  # %

# Logit per 1 percentage-point above league average whiff rate.
# Empirical: 1 pp whiff ≈ 0.006 logit (~0.15% win-prob at 50%).
_WHIFF_PER_LOGIT_PP = 0.006

# Logit per 1 pp below league average hard-hit rate (lower hard-hit = better).
_HARD_HIT_PER_LOGIT_PP = 0.004


def statcast_pitcher_delta(ctx: dict[str, Any]) -> float:
    delta = 0.0

    # --- 1. xERA luck gap ---
    h_x = ctx.get("home_sp_xera")
    a_x = ctx.get("away_sp_xera")
    h_e = ctx.get("home_sp_era")
    a_e = ctx.get("away_sp_era")
    if None not in (h_x, a_x, h_e, a_e):
        home_luck = float(h_e) - float(h_x)   # positive = home pitcher got lucky
        away_luck = float(a_e) - float(a_x)
        # away pitcher lucky = bad for home; home pitcher lucky = bad for home
        delta += (away_luck - home_luck) * ERA_GAP_TO_LOGIT

    # --- 2. Whiff rate advantage ---
    h_w = ctx.get("home_sp_whiff")
    a_w = ctx.get("away_sp_whiff")
    if h_w is not None and a_w is not None:
        # Each pp above league average adds to win prob for the *pitching* team.
        home_whiff_edge = (float(h_w) - _LEAGUE_WHIFF_PCT) * _WHIFF_PER_LOGIT_PP
        away_whiff_edge = (float(a_w) - _LEAGUE_WHIFF_PCT) * _WHIFF_PER_LOGIT_PP
        delta += home_whiff_edge - away_whiff_edge

    # --- 3. Hard-hit rate (lower = better for pitcher) ---
    h_hh = ctx.get("home_sp_hard_hit")
    a_hh = ctx.get("away_sp_hard_hit")
    if h_hh is not None and a_hh is not None:
        # Each pp *below* league average adds to win prob.
        home_hh_edge = (_LEAGUE_HARD_HIT_PCT - float(h_hh)) * _HARD_HIT_PER_LOGIT_PP
        away_hh_edge = (_LEAGUE_HARD_HIT_PCT - float(a_hh)) * _HARD_HIT_PER_LOGIT_PP
        delta += home_hh_edge - away_hh_edge

    return delta
