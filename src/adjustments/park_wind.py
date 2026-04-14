"""Wind-adjusted park factor.

Constants KPH_TO_RUNS and WIND_RUN_TO_LOGIT are read from weights.py.
"""
from __future__ import annotations

import math
from typing import Any

from src.adjustments.weights import KPH_TO_RUNS, WIND_RUN_TO_LOGIT

PARK_CF_BEARING: dict[str, float] = {
    "ARI":  0.0,   "ATL":  60.0,  "BAL":  32.0,  "BOS":  45.0,
    "CHC":  30.0,  "CHW":  45.0,  "CIN":  30.0,  "CLE":  0.0,
    "COL":  0.0,   "DET":  145.0, "HOU":  345.0, "KCR":  45.0,
    "LAA":  45.0,  "LAD":  20.0,  "MIA":  40.0,  "MIL":  135.0,
    "MIN":  90.0,  "NYM":  25.0,  "NYY":  75.0,  "OAK":  60.0,
    "PHI":  15.0,  "PIT":  60.0,  "SDP":  0.0,   "SEA":  45.0,
    "SFG":  90.0,  "STL":  60.0,  "TBR":  45.0,  "TEX":  20.0,
    "TOR":  0.0,   "WSN":  30.0,
}

DOMES = {"ARI", "HOU", "MIA", "MIL", "SEA", "TEX", "TOR", "TBR"}


def _wind_out_component(bearing_cf: float, wind_from_deg: float,
                        wind_kph: float) -> float:
    wind_to_deg = (wind_from_deg + 180.0) % 360.0
    theta = math.radians(wind_to_deg - bearing_cf)
    return wind_kph * math.cos(theta)


def park_wind_delta(ctx: dict[str, Any], for_total: bool = False) -> float:
    team = ctx.get("home_team")
    kph = ctx.get("wind_kph")
    direction = ctx.get("wind_dir_deg")
    if not team or kph is None or direction is None:
        return 0.0
    if team in DOMES or team not in PARK_CF_BEARING:
        return 0.0
    out_component = _wind_out_component(PARK_CF_BEARING[team], direction, kph)
    runs_delta = out_component * KPH_TO_RUNS
    if for_total:
        return runs_delta
    return WIND_RUN_TO_LOGIT * runs_delta