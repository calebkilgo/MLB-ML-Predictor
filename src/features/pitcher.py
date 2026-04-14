"""Starting-pitcher feature aggregation (season-level, shifted for safety)."""
from __future__ import annotations

import pandas as pd


def pitcher_features(pitchers: pd.DataFrame) -> pd.DataFrame:
    """Return season-aggregated pitcher features keyed by (season, Team)."""
    df = pitchers.copy()
    agg = df.groupby(["season", "Team"]).agg(
        sp_era=("ERA", "min"),
        sp_fip=("FIP", "min"),
        sp_k9=("K/9", "max"),
        sp_bb9=("BB/9", "min"),
        sp_whip=("WHIP", "min"),
    ).reset_index()
    return agg
