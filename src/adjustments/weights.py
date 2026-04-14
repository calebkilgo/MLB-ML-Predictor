"""Centralized weights for the adjustment layer.

Every tunable constant in src/adjustments/ lives here. Values can be
updated by the analyzer (`src/analysis/analyze_adjustments.py --apply`)
after enough games have been logged, or edited by hand.

Do NOT import other adjustment modules from this file — it must stay a
leaf dependency so the analyzer can import it cleanly without pulling
in the full model stack.
"""
from __future__ import annotations

# ---- market adjustment ---------------------------------------------------

# How much weight the de-vigged market-implied probability gets in the
# log-odds blend with the model. 0.0 = pure model, 1.0 = pure market.
W_MARKET: float = 0.50


# ---- statcast pitcher adjustment -----------------------------------------

# Logits of home WP added per 1.0 ERA-run gap between traditional ERA
# and xERA. Half-strength because ERA is already in the base model.
ERA_GAP_TO_LOGIT: float = 0.06


# ---- lineup adjustment ---------------------------------------------------

# Points of wOBA that correspond to 1.0 logit of home WP.
WOBA_PER_LOGIT: float = 160.0

# Pseudo-count (in PAs) for Bayesian shrinkage of small-sample lineups
# toward league average. Higher = more shrinkage.
SHRINKAGE_K: float = 100.0


# ---- bullpen adjustment --------------------------------------------------

# FIP runs of bullpen degradation that correspond to 1.0 logit of WP.
FIP_PER_LOGIT: float = 5.0


# ---- wind / park adjustment ----------------------------------------------

# Runs added per kph of wind blowing straight out to CF.
KPH_TO_RUNS: float = 0.025

# Logits of WP per 1.0 run of expected total-runs shift (small).
WIND_RUN_TO_LOGIT: float = 0.02


# ---- umpire adjustment (totals only) -------------------------------------

# Runs/game bias for known umpires. Starter table — replace with scraped
# umpscorecards.com data for better fidelity.
UMPIRE_BIAS: dict[str, float] = {
    "Angel Hernandez":  +0.25,
    "Laz Diaz":         +0.20,
    "CB Bucknor":       +0.15,
    "Doug Eddings":     +0.10,
    "Phil Cuzzi":       +0.10,
    "Pat Hoberg":       -0.20,
    "Tripp Gibson":     -0.10,
    "Will Little":      -0.10,
    "Jordan Baker":     -0.05,
}


# ---- final confidence regularization -------------------------------------

# How much of the model's "distance from 0.5" to keep after adjustment.
# 1.00 = no shrinkage, 0.70 = current default, 0.0 = always 50%.
CONFIDENCE_SHRINK: float = 0.70