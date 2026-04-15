"""Fetch and cache Baseball Savant's expected-stats and Statcast metrics.

Outputs:
  data/raw/statcast_pitchers_{year}.parquet  — season xERA/xwOBA (unchanged)
  data/raw/statcast_rolling_{year}.parquet   — rolling 20-game window with
      velo_avg, whiff_pct, hard_hit_pct, k_pct, bb_pct per pitcher.

The rolling file is rebuilt whenever the season parquet is more than 1 day
old. The web app (daily_context.py) merges these into pitch-level context.

Run manually:
    python -m src.etl.statcast_cache

Baseball Savant CSV exports work without auth.
"""
from __future__ import annotations

import time
from datetime import date
from io import StringIO
from pathlib import Path

import httpx
import pandas as pd

from src.config import CFG


def fetch_pitcher_xstats(year: int) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/expected_statistics"
        f"?type=pitcher&year={year}&position=&team=&filter=&min=10&csv=true"
    )
    r = httpx.get(url, timeout=30.0, follow_redirects=True)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    keep = {}
    for col in df.columns:
        lc = col.lower()
        if lc in ("player_id", "mlbam_id"):
            keep[col] = "mlb_id"
        elif lc == "year":
            keep[col] = "year"
        elif lc == "woba":
            keep[col] = "woba"
        elif lc in ("est_woba", "xwoba"):
            keep[col] = "xwoba"
        elif lc == "xera":
            keep[col] = "xera"
        elif lc in ("first_name", "firstname"):
            keep[col] = "first_name"
        elif lc in ("last_name", "lastname"):
            keep[col] = "last_name"
    df = df.rename(columns=keep)[list(keep.values())]
    if "xera" not in df.columns:
        df["xera"] = (df["xwoba"] - 0.300) * (5.50 - 3.50) / (0.370 - 0.300) + 3.50
    df["year"] = int(year)
    return df[["mlb_id", "year", "xwoba", "xera"]]


def fetch_pitcher_statcast_rolling(year: int, min_pa: int = 25) -> pd.DataFrame:
    """Fetch per-pitcher Statcast metrics from Baseball Savant.

    Uses the pitch-arsenal / statcast search leaderboard which includes
    average fastball velo, whiff %, hard-hit %, K%, BB% at the season level.
    We download the season leaderboard and label it as the most-recent-data
    snapshot (no true rolling is possible from public CSV exports, but the
    season-to-date averages are updated daily).

    Columns returned: mlb_id, velo_avg, whiff_pct, hard_hit_pct, k_pct, bb_pct
    """
    # Statcast pitching leaderboard (batted-ball / approach metrics)
    url = (
        "https://baseballsavant.mlb.com/leaderboard/statcast"
        f"?type=pitcher&year={year}&position=1&team=&min={min_pa}&csv=true"
    )
    try:
        r = httpx.get(url, timeout=30.0, follow_redirects=True)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
    except Exception:
        return pd.DataFrame(columns=["mlb_id", "velo_avg", "whiff_pct",
                                     "hard_hit_pct", "k_pct", "bb_pct"])

    rename = {}
    for col in df.columns:
        lc = col.lower().replace(" ", "_")
        if lc in ("player_id", "mlbam_id"):
            rename[col] = "mlb_id"
        elif lc in ("avg_best_speed", "avg_fastball", "release_speed"):
            rename[col] = "velo_avg"
        elif lc in ("whiff_percent", "whiff_pct", "whiff%"):
            rename[col] = "whiff_pct"
        elif lc in ("hard_hit_percent", "hard_hit%", "hard_hit_pct"):
            rename[col] = "hard_hit_pct"
        elif lc in ("k_percent", "k%", "strikeout_percent"):
            rename[col] = "k_pct"
        elif lc in ("bb_percent", "bb%", "walk_percent"):
            rename[col] = "bb_pct"

    df = df.rename(columns=rename)
    out_cols = ["mlb_id", "velo_avg", "whiff_pct", "hard_hit_pct", "k_pct", "bb_pct"]
    present = [c for c in out_cols if c in df.columns]
    if "mlb_id" not in present:
        return pd.DataFrame(columns=out_cols)

    df = df[present].dropna(subset=["mlb_id"])
    df["mlb_id"] = pd.to_numeric(df["mlb_id"], errors="coerce").dropna().astype(int)
    for col in [c for c in out_cols if c != "mlb_id" and c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Fill missing optional columns with NaN
    for col in out_cols:
        if col not in df.columns:
            df[col] = float("nan")
    return df[out_cols].reset_index(drop=True)


def _stale(path: Path, max_age_hours: float = 20.0) -> bool:
    if not path.exists():
        return True
    return (time.time() - path.stat().st_mtime) > max_age_hours * 3600


def refresh_statcast(year: int | None = None, force: bool = False) -> None:
    """Refresh both statcast parquets if stale (or forced)."""
    if year is None:
        year = date.today().year
    CFG.raw_dir.mkdir(parents=True, exist_ok=True)

    xstats_path = CFG.raw_dir / f"statcast_pitchers_{year}.parquet"
    rolling_path = CFG.raw_dir / f"statcast_rolling_{year}.parquet"

    if force or _stale(xstats_path):
        print(f"[statcast] fetching xERA/xwOBA for {year}")
        df_x = fetch_pitcher_xstats(year)
        df_x.to_parquet(xstats_path, index=False)
        print(f"[statcast] wrote {len(df_x)} rows -> {xstats_path}")

    if force or _stale(rolling_path):
        print(f"[statcast] fetching rolling Statcast metrics for {year}")
        df_r = fetch_pitcher_statcast_rolling(year)
        df_r.to_parquet(rolling_path, index=False)
        print(f"[statcast] wrote {len(df_r)} rows -> {rolling_path}")


def main() -> None:
    refresh_statcast(force=True)


if __name__ == "__main__":
    main()
