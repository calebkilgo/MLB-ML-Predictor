"""Umpire strike-zone bias adjustment (totals only).

Uses live-scraped data from umpscorecards.com (cached to
data/raw/umpire_bias.json, refreshed daily).  Falls back to the static
UMPIRE_BIAS table in weights.py if the file is missing or stale.
"""
from __future__ import annotations

from typing import Any

from src.adjustments.weights import UMPIRE_BIAS


def _live_bias() -> dict[str, float]:
    """Return the freshest available umpire bias dict."""
    try:
        from src.etl.umpire_scraper import get_umpire_bias
        return get_umpire_bias()
    except Exception:
        return UMPIRE_BIAS


def umpire_total_delta(ctx: dict[str, Any]) -> float:
    explicit = ctx.get("umpire_runs_bias")
    if explicit is not None:
        return float(explicit)
    name = ctx.get("umpire_name")
    if not name:
        return 0.0
    # Prefer live-scraped bias; fall back to static table
    bias = _live_bias()
    return bias.get(name, 0.0)
