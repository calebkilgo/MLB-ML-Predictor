"""Leakage-safe rolling team statistics."""
from __future__ import annotations

import pandas as pd


def rolling_team_form(games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """For each (team, date), compute rolling run diff and win% over last N games.

    Uses shift(1) to ensure the row's own game is not included.
    """
    # Long format: one row per team-game.
    home = games[["game_pk", "game_date", "home_team", "home_runs",
                  "away_runs", "home_win"]].rename(
        columns={"home_team": "team", "home_runs": "rs",
                 "away_runs": "ra", "home_win": "win"})
    away = games[["game_pk", "game_date", "away_team", "away_runs",
                  "home_runs", "home_win"]].rename(
        columns={"away_team": "team", "away_runs": "rs",
                 "home_runs": "ra"})
    away["win"] = 1 - away["home_win"]
    away = away.drop(columns="home_win")
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "game_date"])

    g = long.groupby("team", group_keys=False)
    long["form_rs"] = g["rs"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["form_ra"] = g["ra"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["form_win_pct"] = g["win"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    long["rest_days"] = g["game_date"].diff().dt.days.fillna(4).clip(0, 10)
    return long[["game_pk", "team", "form_rs", "form_ra", "form_win_pct", "rest_days"]]
