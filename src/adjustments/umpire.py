"""Umpire strike-zone bias adjustment (totals only).

UMPIRE_BIAS table is read from weights.py.
"""
from __future__ import annotations

from typing import Any

from src.adjustments.weights import UMPIRE_BIAS


def umpire_total_delta(ctx: dict[str, Any]) -> float:
    explicit = ctx.get("umpire_runs_bias")
    if explicit is not None:
        return float(explicit)
    name = ctx.get("umpire_name")
    if not name:
        return 0.0
    return UMPIRE_BIAS.get(name, 0.0)