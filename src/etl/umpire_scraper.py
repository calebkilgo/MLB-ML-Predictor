"""Daily umpire bias scraper from umpscorecards.com.

Fetches the current-season strike-zone data for each home-plate umpire and
computes a per-umpire run bias (positive = generous zone = more runs, negative
= tight zone = fewer runs).  Results are cached to
data/raw/umpire_bias.json and reloaded by src/adjustments/umpire.py.

Schedule: call `refresh_umpire_bias()` once per day (e.g., from the
nightly maintenance loop in app/main.py).

Data source: umpscorecards.com/umpires — free, no auth required.
We scrape the HTML table of season totals (called strikes above average,
runs per game etc.) rather than relying on the hardcoded table in weights.py.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import httpx

from src.config import CFG

logger = logging.getLogger(__name__)

_BIAS_PATH = CFG.raw_dir / "umpire_bias.json"
_TTL_SECONDS = 18 * 3600  # refresh at most once every 18 hours

# Base URL for the umpire season-stats page.
_BASE_URL = "https://umpscorecards.com/umpires/"

# Fallback hardcoded values (from weights.py) used when scraping fails.
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


def _parse_umpire_table(html: str) -> dict[str, float]:
    """Parse the umpire stats table from the HTML.

    Returns {umpire_name: run_bias_per_game} where positive = more runs.
    We approximate run_bias from the 'favor_home' or 'csaa' (called strikes
    above average) columns.  CSAA of +10 ≈ +0.1 runs/game.
    """
    # Look for JSON data embedded in the page (the site uses a JS table).
    # Pattern: umpire name + csaa or raa (runs above average) values.
    bias: dict[str, float] = {}

    # Try to find embedded JSON data array
    json_match = re.search(
        r'var\s+(?:umpireData|tableData|data)\s*=\s*(\[.*?\]);',
        html, re.DOTALL
    )
    if json_match:
        try:
            records = json.loads(json_match.group(1))
            for rec in records:
                name = rec.get("name") or rec.get("umpire") or ""
                # csaa: called strikes above average (positive = tight zone)
                # Tight zone = fewer called strikes = pitchers get less help
                # = more walks/HBPs = more runs allowed. So flip sign.
                csaa = rec.get("csaa") or rec.get("called_strikes_above_avg") or 0
                # Some tables have 'favor' as runs/game directly
                raa = rec.get("raa") or rec.get("runs_above_avg") or 0
                if name:
                    # Prefer raa if available; otherwise estimate from csaa.
                    # Empirical: 10 CSAA ≈ 0.1 run/game (based on historical data)
                    run_bias = float(raa) if raa else -float(csaa) * 0.010
                    bias[name.strip()] = round(run_bias, 3)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Fallback: parse simple HTML table rows
    if not bias:
        # Match rows like: <td>Ángel Hernández</td>...<td>+0.23</td>
        rows = re.findall(
            r'<tr[^>]*>.*?<td[^>]*>([A-Za-záéíóúÁÉÍÓÚñÑ\s\.\-]+)</td>'
            r'.*?<td[^>]*>([-+]?\d+\.?\d*)</td>',
            html, re.DOTALL
        )
        for name, val in rows:
            try:
                bias[name.strip()] = float(val)
            except ValueError:
                pass

    return bias


def refresh_umpire_bias(force: bool = False) -> dict[str, float]:
    """Fetch latest umpire bias data and cache to umpire_bias.json.

    Returns the bias dict {name: run_bias}.  Uses the cached file if it
    was written within _TTL_SECONDS and force=False.
    """
    _BIAS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Check cache freshness
    if not force and _BIAS_PATH.exists():
        age = time.time() - _BIAS_PATH.stat().st_mtime
        if age < _TTL_SECONDS:
            try:
                return json.loads(_BIAS_PATH.read_text())
            except Exception:
                pass

    try:
        resp = httpx.get(_BASE_URL, timeout=15.0,
                         headers={"User-Agent": "mlb-predict/1.0"})
        resp.raise_for_status()
        bias = _parse_umpire_table(resp.text)
        if bias:
            logger.info("[umpire_scraper] fetched %d umpire biases", len(bias))
            _BIAS_PATH.write_text(json.dumps(bias, indent=2))
            return bias
        else:
            logger.warning("[umpire_scraper] parsed 0 umpires — page format may "
                           "have changed; using fallback")
    except Exception as e:
        logger.warning("[umpire_scraper] fetch failed (%s); using fallback", e)

    # Write fallback so we don't hammer the site on every game
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
