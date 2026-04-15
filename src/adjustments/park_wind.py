"""Wind, temperature, humidity, and pressure park factor adjustment.

Constants KPH_TO_RUNS and WIND_RUN_TO_LOGIT are read from weights.py.

Physics model for ball carry:
- Wind blowing out to CF adds runs (positive component).
- Hot air (>15°C baseline) expands, reducing drag: ~0.7% carry per °C above 15°C.
- Low humidity (<50%) is drier/less dense: ~0.3% carry per 10pp below 50%.
- Low pressure (Coors ~840 hPa vs sea level ~1013 hPa): ~0.5% carry per
  10 hPa below 1013.  Coors alone adds ~0.4-0.5 runs/game vs neutral.

All effects are small individually but compound on hot, dry, windy days.
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

# Approximate stadium elevations in meters; sea level = 0.
# Used as a cross-check / fallback when pressure_hpa is unavailable.
PARK_ELEVATION_M: dict[str, float] = {
    "COL": 1580.0,  # Coors Field — highest in MLB
    "ARI":  331.0,  "TEX":  183.0,  "KCR":  274.0,
    "MIN":  264.0,  "DEN":  1580.0,
}
SEA_LEVEL_PRESSURE = 1013.25  # hPa

DOMES = {"ARI", "HOU", "MIA", "MIL", "SEA", "TEX", "TOR", "TBR"}

# Temperature baseline: ~15°C (59°F) is typical neutral MLB game temp.
TEMP_BASELINE_C = 15.0
# Runs added per °C above baseline (ball carry increases ~1% per 10°C,
# translating to roughly 0.03 runs/game per °C for a typical game).
RUNS_PER_TEMP_C = 0.030

# Runs added per 10pp below 50% humidity (dry air = less drag).
RUNS_PER_10PP_LOW_HUMIDITY = 0.015

# Runs added per 10 hPa below sea-level pressure.
RUNS_PER_10HPA_LOW_PRESSURE = 0.018


def _wind_out_component(bearing_cf: float, wind_from_deg: float,
                        wind_kph: float) -> float:
    wind_to_deg = (wind_from_deg + 180.0) % 360.0
    theta = math.radians(wind_to_deg - bearing_cf)
    return wind_kph * math.cos(theta)


def _weather_runs_delta(ctx: dict[str, Any], team: str) -> float:
    """Extra runs/game from temperature, humidity, and pressure."""
    if team in DOMES:
        return 0.0

    total = 0.0

    # Temperature effect
    temp = ctx.get("temp_c")
    if temp is not None:
        total += (float(temp) - TEMP_BASELINE_C) * RUNS_PER_TEMP_C

    # Humidity effect (low humidity = more carry)
    hum = ctx.get("humidity_pct")
    if hum is not None:
        deficit = max(0.0, 50.0 - float(hum))  # pp below 50%
        total += (deficit / 10.0) * RUNS_PER_10PP_LOW_HUMIDITY

    # Pressure effect (low pressure = less air resistance)
    pres = ctx.get("pressure_hpa")
    if pres is not None:
        deficit = max(0.0, SEA_LEVEL_PRESSURE - float(pres))
        total += (deficit / 10.0) * RUNS_PER_10HPA_LOW_PRESSURE
    elif team in PARK_ELEVATION_M:
        # Approximate pressure from elevation if API didn't return it
        elev = PARK_ELEVATION_M[team]
        approx_pres = SEA_LEVEL_PRESSURE * math.exp(-elev / 8500.0)
        deficit = max(0.0, SEA_LEVEL_PRESSURE - approx_pres)
        total += (deficit / 10.0) * RUNS_PER_10HPA_LOW_PRESSURE

    return total


def park_wind_delta(ctx: dict[str, Any], for_total: bool = False) -> float:
    team = ctx.get("home_team")
    kph = ctx.get("wind_kph")
    direction = ctx.get("wind_dir_deg")
    if not team or team not in PARK_CF_BEARING:
        return 0.0

    # Wind component (zero if dome or no wind data)
    wind_runs = 0.0
    if kph is not None and direction is not None and team not in DOMES:
        out_component = _wind_out_component(PARK_CF_BEARING[team], direction, float(kph))
        wind_runs = out_component * KPH_TO_RUNS

    # Temperature / humidity / pressure component
    weather_runs = _weather_runs_delta(ctx, team)

    total_runs_delta = wind_runs + weather_runs

    if for_total:
        return total_runs_delta
    return WIND_RUN_TO_LOGIT * total_runs_delta