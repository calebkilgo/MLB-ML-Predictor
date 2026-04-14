"""Season-based train/val/test splitting to prevent leakage."""
from __future__ import annotations

import pandas as pd


def season_split(
    df: pd.DataFrame,
    date_col: str = "game_date",
    val_seasons: int = 1,
    test_seasons: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically by season. Last N seasons = test, prior N = val."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_season"] = df[date_col].dt.year
    seasons = sorted(df["_season"].unique())
    if len(seasons) < val_seasons + test_seasons + 1:
        raise ValueError("Not enough seasons for requested split.")
    test_s = seasons[-test_seasons:]
    val_s = seasons[-(val_seasons + test_seasons) : -test_seasons]
    train = df[~df["_season"].isin(test_s + val_s)].drop(columns="_season")
    val = df[df["_season"].isin(val_s)].drop(columns="_season")
    test = df[df["_season"].isin(test_s)].drop(columns="_season")
    return train, val, test
