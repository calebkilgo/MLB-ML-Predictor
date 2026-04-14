import pandas as pd
from src.features.rolling import rolling_team_form


def test_rolling_no_leakage():
    games = pd.DataFrame([
        {"game_pk":"1","game_date":pd.Timestamp("2024-04-01"),
         "home_team":"NYY","away_team":"BOS",
         "home_runs":5,"away_runs":3,"home_win":1},
        {"game_pk":"2","game_date":pd.Timestamp("2024-04-02"),
         "home_team":"NYY","away_team":"BOS",
         "home_runs":2,"away_runs":6,"home_win":0},
    ])
    r = rolling_team_form(games, window=5)
    nyy_first = r[(r["team"]=="NYY")].iloc[0]
    assert nyy_first["form_rs"] != 5
