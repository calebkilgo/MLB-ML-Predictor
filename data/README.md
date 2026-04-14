# Data Schema

## raw/games.parquet
One row per MLB game (home-team perspective).

| column       | type     | notes |
|--------------|----------|-------|
| game_pk      | str      | `YYYYMMDD_AWAY_HOME` |
| season       | int      | |
| game_date    | datetime | |
| home_team    | str      | 3-letter abbr |
| away_team    | str      | |
| home_runs    | int      | |
| away_runs    | int      | |
| home_win     | int 0/1  | target |
| total_runs   | int      | secondary target |

## raw/team_stats.parquet
Season-level team batting/pitching from FanGraphs (via pybaseball).

## raw/pitchers.parquet
Season-level pitcher stats (ERA, FIP, K/9, BB/9, WHIP, WAR).

## raw/weather.parquet
Daily weather at home stadium from Open-Meteo archive API.

## processed/features.parquet
Joined, leakage-safe feature matrix consumed by training and prediction.
Key: `game_pk`. See `src/models/train.py::FEATURE_COLS`.
