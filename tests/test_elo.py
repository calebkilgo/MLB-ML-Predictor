import pandas as pd
from src.features.elo import compute_elo


def test_elo_monotonic():
    games = pd.DataFrame([
        {"game_pk":"1","game_date":pd.Timestamp("2024-04-01"),
         "home_team":"A","away_team":"B","home_win":1},
        {"game_pk":"2","game_date":pd.Timestamp("2024-04-02"),
         "home_team":"A","away_team":"B","home_win":1},
    ])
    elo = compute_elo(games)
    assert elo.iloc[1]["elo_home"] > 1500
    assert elo.iloc[1]["elo_away"] < 1500
