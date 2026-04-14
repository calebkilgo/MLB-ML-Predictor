"""Bullpen availability adjustment.

Constant FIP_PER_LOGIT is read from weights.py.
"""
from __future__ import annotations

from typing import Any

from src.adjustments.weights import FIP_PER_LOGIT


def bullpen_delta(ctx: dict[str, Any]) -> float:
    h_avail = ctx.get("home_bullpen_available_fip")
    a_avail = ctx.get("away_bullpen_available_fip")
    h_full = ctx.get("home_bullpen_full_fip")
    a_full = ctx.get("away_bullpen_full_fip")
    if None in (h_avail, a_avail, h_full, a_full):
        return 0.0
    h_deg = h_avail - h_full
    a_deg = a_avail - a_full
    net = a_deg - h_deg
    return net / FIP_PER_LOGIT