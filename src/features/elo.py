"""Simple 538-style ELO ratings as a strong baseline feature."""
from __future__ import annotations

import pandas as pd

K = 4.0
HOME_ADV = 24.0
INIT = 1500.0


def compute_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Return per-game pre-match ELO for home and away teams."""
    games = games.sort_values("game_date").reset_index(drop=True)
    ratings: dict[str, float] = {}
    out = []
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        rh = ratings.get(h, INIT)
        ra = ratings.get(a, INIT)
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + HOME_ADV) - ra) / 400))
        out.append({"game_pk": g["game_pk"], "elo_home": rh,
                    "elo_away": ra, "elo_exp_home": exp_h})
        result_h = 1.0 if g["home_win"] == 1 else 0.0
        ratings[h] = rh + K * (result_h - exp_h)
        ratings[a] = ra + K * ((1 - result_h) - (1 - exp_h))
    return pd.DataFrame(out)
