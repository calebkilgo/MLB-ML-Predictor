"""Fetch and cache Baseball Savant's expected-stats pitcher leaderboard.

Writes data/raw/statcast_pitchers_{year}.parquet with one row per pitcher
containing xERA and xwOBA-against. The web app reads this on startup and
merges into its context. Refreshes daily (or when you re-run it).

Run manually:
    python -m src.etl.statcast_cache

Baseball Savant's CSV export works without auth. URL shape:
    https://baseballsavant.mlb.com/leaderboard/expected_statistics
        ?type=pitcher&year=YYYY&csv=true
"""
from __future__ import annotations

from datetime import date
from io import StringIO

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
    # Columns include: last_name, first_name, player_id, year, pa, bip,
    # ba, est_ba, slg, est_slg, woba, est_woba, xera (sometimes), ...
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
        # Savant doesn't always publish xERA directly; approximate from xwOBA.
        # Linear mapping that hits xERA≈3.50 at xwOBA≈0.300 and xERA≈5.50
        # at xwOBA≈0.370. Rough but directionally correct.
        df["xera"] = (df["xwoba"] - 0.300) * (5.50 - 3.50) / (0.370 - 0.300) + 3.50
    df["year"] = int(year)
    return df[["mlb_id", "year", "xwoba", "xera"]]


def main() -> None:
    year = date.today().year
    print(f"[statcast] fetching pitcher xstats for {year}")
    df = fetch_pitcher_xstats(year)
    out = CFG.raw_dir / f"statcast_pitchers_{year}.parquet"
    df.to_parquet(out, index=False)
    print(f"[statcast] wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
