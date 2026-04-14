"""Static park factors (approximate, 3-year averages)."""
from __future__ import annotations

import pandas as pd

PARK_FACTORS: dict[str, float] = {
    "COL": 1.12, "CIN": 1.05, "BOS": 1.04, "TEX": 1.04, "KCR": 1.03,
    "TOR": 1.02, "PHI": 1.02, "ARI": 1.02, "BAL": 1.01, "CHC": 1.01,
    "MIN": 1.00, "HOU": 1.00, "ATL": 1.00, "WSN": 1.00, "NYY": 1.00,
    "MIL": 0.99, "LAA": 0.99, "STL": 0.99, "CHW": 0.99, "LAD": 0.98,
    "CLE": 0.98, "TBR": 0.97, "DET": 0.97, "NYM": 0.96, "SEA": 0.96,
    "PIT": 0.96, "OAK": 0.95, "SFG": 0.94, "MIA": 0.93, "SDP": 0.93,
}


def park_factor(team: str) -> float:
    return PARK_FACTORS.get(team, 1.00)


def attach_park(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["park_factor"] = df["home_team"].map(PARK_FACTORS).fillna(1.0)
    return df
