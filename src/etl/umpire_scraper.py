"""Daily umpire bias scraper from umpscorecards.com.

Fetches the current-season strike-zone data for each home-plate umpire and
computes a per-umpire run bias (positive = generous zone = more runs, negative
= tight zone = fewer runs).  Results are cached to
data/raw/umpire_bias.json and reloaded by src/adjustments/umpire.py.

Schedule: call `refresh_umpire_bias()` once per day (e.g., from the
nightly maintenance loop in app/main.py).

Data source: umpscorecards.com/api/umpires?year=YYYY — free JSON API.

Run bias formula:
  run_bias = -sign(accuracy_above_x_wmean) * total_run_impact_mean * 0.10

  - total_run_impact_mean: unsigned magnitude of run impact per game
  - accuracy_above_x_wmean: signed accuracy vs expectation; negative =
    looser zone (more balls called, more walks) = more runs; positive =
    tighter / more accurate zone = fewer runs.
  - 0.10 scale maps the impact range (~1.0–2.0) to our ±0.25 target range.
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import date
from pathlib import Path

import httpx

from src.config import CFG

logger = logging.getLogger(__name__)

_BIAS_PATH = CFG.raw_dir / "umpire_bias.json"
_TTL_SECONDS = 18 * 3600  # refresh at most once every 18 hours

_API_URL = "https://umpscorecards.com/api/umpires"

# Minimum games threshold — ignore umpires with very few tracked games.
_MIN_GAMES = 10

# Fallback hardcoded values used when the API is unavailable.
_FALLBACK: dict[str, float] = {
    "Angel Hernandez": +0.25,
    "Laz Diaz":        +0.20,
    "CB Bucknor":      +0.15,
    "Doug Eddings":    +0.10,
    "Phil Cuzzi":      +0.10,
    "Pat Hoberg":      -0.20,
    "Tripp Gibson":    -0.10,
    "Will Little":     -0.10,
    "Jordan Baker":    -0.05,
}


def _parse_api_response(data: dict) -> dict[str, float]:
    """Parse the /api/umpires JSON response into {name: run_bias}.

    The API returns {"rows": [...]} where each record has:
      - umpire: str
      - n: int (games umpired)
      - accuracy_above_x_wmean: float (signed; negative = looser zone)
      - total_run_impact_mean: float (unsigned magnitude of run impact/game)
    """
    bias: dict[str, float] = {}
    rows = data.get("rows") or (data if isinstance(data, list) else [])
    for rec in rows:
        name = (rec.get("umpire") or "").strip()
        if not name:
            continue
        n = int(rec.get("n") or 0)
        if n < _MIN_GAMES:
            continue
        acc = rec.get("accuracy_above_x_wmean")
        impact = rec.get("total_run_impact_mean")
        if acc is None or impact is None:
            continue
        try:
            acc_f = float(acc)
            impact_f = float(impact)
        except (TypeError, ValueError):
            continue
        # direction: negative accuracy → more runs (positive bias)
        direction = -1.0 if acc_f < 0 else (1.0 if acc_f > 0 else 0.0)
        run_bias = direction * impact_f * 0.10
        bias[name] = round(run_bias, 3)
    return bias


def refresh_umpire_bias(force: bool = False) -> dict[str, float]:
    """Fetch latest umpire bias data and cache to umpire_bias.json.

    Returns the bias dict {name: run_bias}.  Uses the cached file if it
    was written within _TTL_SECONDS and force=False.
    """
    _BIAS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Check cache freshness.
    if not force and _BIAS_PATH.exists():
        age = time.time() - _BIAS_PATH.stat().st_mtime
        if age < _TTL_SECONDS:
            try:
                return json.loads(_BIAS_PATH.read_text())
            except Exception:
                pass

    year = date.today().year
    url = f"{_API_URL}?year={year}"

    try:
        resp = httpx.get(url, timeout=15.0,
                         headers={"User-Agent": "mlb-predict/1.0"})
        resp.raise_for_status()
        data = resp.json()
        bias = _parse_api_response(data)
        if bias:
            logger.info("[umpire_scraper] fetched %d umpire biases for %d",
                        len(bias), year)
            _BIAS_PATH.write_text(json.dumps(bias, indent=2))
            return bias
        else:
            logger.warning("[umpire_scraper] parsed 0 umpires for %d "
                           "— using fallback", year)
    except Exception as e:
        logger.warning("[umpire_scraper] fetch failed (%s); using fallback", e)

    # Write fallback so we don't hammer the site on every board build.
    _BIAS_PATH.write_text(json.dumps(_FALLBACK, indent=2))
    return dict(_FALLBACK)


def get_umpire_bias() -> dict[str, float]:
    """Return cached umpire bias dict, refreshing if stale."""
    if _BIAS_PATH.exists():
        age = time.time() - _BIAS_PATH.stat().st_mtime
        if age < _TTL_SECONDS:
            try:
                return json.loads(_BIAS_PATH.read_text())
            except Exception:
                pass
    return refresh_umpire_bias()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = refresh_umpire_bias(force=True)
    print(f"Fetched {len(result)} umpires:")
    for name, bias in sorted(result.items(), key=lambda x: -abs(x[1])):
        sign = "+" if bias >= 0 else ""
        print(f"  {name:<28} {sign}{bias:.3f} runs/game")
