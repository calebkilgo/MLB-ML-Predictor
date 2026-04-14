"""Top-level ETL: builds data/raw/games.parquet and supporting tables.

Skips any step whose output parquet already exists. Delete files in
data/raw/ to force a refresh.
"""
from __future__ import annotations

import pandas as pd

from src.config import CFG
from src.etl import pybaseball_loader as pbl
from src.etl import retrosheet, weather
from src.utils.io import read_parquet, write_parquet


def _build_games_and_stats() -> pd.DataFrame:
    games_path = CFG.raw_dir / "games.parquet"
    team_stats_path = CFG.raw_dir / "team_stats.parquet"
    pitchers_path = CFG.raw_dir / "pitchers.parquet"

    if games_path.exists() and team_stats_path.exists() and pitchers_path.exists():
        print("[ETL] games/team_stats/pitchers already cached, skipping.")
        return read_parquet(games_path)

    all_team_games: list[pd.DataFrame] = []
    all_team_stats: list[pd.DataFrame] = []
    all_pitchers: list[pd.DataFrame] = []

    for year in range(CFG.seasons_start, CFG.seasons_end + 1):
        print(f"[ETL] season {year}")
        all_team_games.append(retrosheet.load_season(year))
        all_team_stats.append(pbl.team_season_stats(year))
        all_pitchers.append(pbl.pitcher_season_stats(year))

    team_games = pd.concat(all_team_games, ignore_index=True)
    games = retrosheet.to_game_level(team_games)
    write_parquet(games, games_path)
    write_parquet(pd.concat(all_team_stats, ignore_index=True), team_stats_path)
    write_parquet(pd.concat(all_pitchers, ignore_index=True), pitchers_path)
    return games


def main() -> None:
    games = _build_games_and_stats()

    weather_path = CFG.raw_dir / "weather.parquet"
    if weather_path.exists():
        print("[ETL] weather already cached, skipping.")
    else:
        print("[ETL] fetching weather...")
        wx = weather.bulk_weather(games)
        write_parquet(wx, weather_path)

    print("[ETL] done.")


if __name__ == "__main__":
    main()