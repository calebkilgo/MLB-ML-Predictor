"""Assemble the final feature table and write features.parquet."""
from __future__ import annotations

import pandas as pd

from src.config import CFG
from src.features.elo import compute_elo
from src.features.park import attach_park
from src.features.pitcher import pitcher_features
from src.features.rolling import rolling_team_form
from src.utils.io import read_parquet, write_parquet


def main() -> None:
    games = read_parquet(CFG.raw_dir / "games.parquet")
    team_stats = read_parquet(CFG.raw_dir / "team_stats.parquet")
    pitchers = read_parquet(CFG.raw_dir / "pitchers.parquet")
    weather = read_parquet(CFG.raw_dir / "weather.parquet")

    games = attach_park(games)
    elo = compute_elo(games)
    form = rolling_team_form(games, window=10)

    home_form = form.rename(columns={
        "team": "home_team", "form_rs": "home_form_rs",
        "form_ra": "home_form_ra", "form_win_pct": "home_form_win",
        "rest_days": "home_rest",
    })
    away_form = form.rename(columns={
        "team": "away_team", "form_rs": "away_form_rs",
        "form_ra": "away_form_ra", "form_win_pct": "away_form_win",
        "rest_days": "away_rest",
    })

    df = games.merge(elo, on="game_pk")
    df = df.merge(home_form, on=["game_pk", "home_team"], how="left")
    df = df.merge(away_form, on=["game_pk", "away_team"], how="left")

    # Team season stats: shift to prior season for leakage safety.
    ts = team_stats.copy()
    ts["season"] = ts["season"] + 1

    home_ts = ts.rename(columns={
        "Team": "home_team", "OPS": "home_team_ops", "R": "home_team_r",
        "team_ERA": "home_team_era", "team_K9": "home_team_k9",
        "team_BB9": "home_team_bb9",
    })
    df = df.merge(home_ts, on=["season", "home_team"], how="left")

    away_ts = ts.rename(columns={
        "Team": "away_team", "OPS": "away_team_ops", "R": "away_team_r",
        "team_ERA": "away_team_era", "team_K9": "away_team_k9",
        "team_BB9": "away_team_bb9",
    })
    df = df.merge(away_ts, on=["season", "away_team"], how="left")

    # Pitcher aggregates (prior season).
    pf = pitcher_features(pitchers)
    pf["season"] = pf["season"] + 1

    home_pf = pf.rename(columns={
        "Team": "home_team",
        "sp_era": "home_sp_era", "sp_fip": "home_sp_fip",
        "sp_k9": "home_sp_k9",   "sp_bb9": "home_sp_bb9",
        "sp_whip": "home_sp_whip",
    })
    df = df.merge(home_pf, on=["season", "home_team"], how="left")

    away_pf = pf.rename(columns={
        "Team": "away_team",
        "sp_era": "away_sp_era", "sp_fip": "away_sp_fip",
        "sp_k9": "away_sp_k9",   "sp_bb9": "away_sp_bb9",
        "sp_whip": "away_sp_whip",
    })
    df = df.merge(away_pf, on=["season", "away_team"], how="left")

    df = df.merge(weather, on=["home_team", "game_date"], how="left")

    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(df[num].median(numeric_only=True))

    write_parquet(df, CFG.features_path)
    print(f"[FEATURES] wrote {len(df)} rows to {CFG.features_path}")


if __name__ == "__main__":
    main()