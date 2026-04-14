"""Per-game rolling features computed as-of the morning of each game.

v7: adds rolling starter features (last 5 starts FIP/K9/BB9/ERA) joined
via data/raw/starters.parquet and data/raw/starter_logs.parquet.

Every feature is protected from leakage by a strict shift(1) before the
rolling window — the row for game G never sees anything from game G
itself or later.

Run:
    python -m src.features.rolling_v2
"""
from __future__ import annotations

import pandas as pd


def _team_long(games: pd.DataFrame) -> pd.DataFrame:
    home = games[[
        "game_pk", "game_date", "season",
        "home_team", "away_team", "home_runs", "away_runs", "home_win",
    ]].rename(columns={
        "home_team": "team", "away_team": "opp",
        "home_runs": "rs", "away_runs": "ra",
    }).copy()
    home["is_home"] = 1
    home["win"] = home["home_win"]

    away = games[[
        "game_pk", "game_date", "season",
        "home_team", "away_team", "home_runs", "away_runs", "home_win",
    ]].rename(columns={
        "away_team": "team", "home_team": "opp",
        "away_runs": "rs", "home_runs": "ra",
    }).copy()
    away["is_home"] = 0
    away["win"] = 1 - away["home_win"]

    long = pd.concat([
        home[["game_pk", "game_date", "season", "team", "opp",
              "is_home", "rs", "ra", "win"]],
        away[["game_pk", "game_date", "season", "team", "opp",
              "is_home", "rs", "ra", "win"]],
    ], ignore_index=True)
    long["game_date"] = pd.to_datetime(long["game_date"])
    return long.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)


def _roll_shifted(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


def _expanding_shifted(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=1).mean()


def compute_team_rolling(games: pd.DataFrame) -> pd.DataFrame:
    long = _team_long(games)
    g = long.groupby("team", group_keys=False)

    long["rs_10"] = g["rs"].apply(lambda s: _roll_shifted(s, 10))
    long["ra_10"] = g["ra"].apply(lambda s: _roll_shifted(s, 10))
    long["wp_10"] = g["win"].apply(lambda s: _roll_shifted(s, 10))

    long["rs_30"] = g["rs"].apply(lambda s: _roll_shifted(s, 30))
    long["ra_30"] = g["ra"].apply(lambda s: _roll_shifted(s, 30))
    long["wp_30"] = g["win"].apply(lambda s: _roll_shifted(s, 30))

    gs = long.groupby(["team", "season"], group_keys=False)
    long["rs_std"] = gs["rs"].apply(_expanding_shifted)
    long["ra_std"] = gs["ra"].apply(_expanding_shifted)
    long["wp_std"] = gs["win"].apply(_expanding_shifted)

    long["rd_10"] = long["rs_10"] - long["ra_10"]
    long["rd_30"] = long["rs_30"] - long["ra_30"]
    long["rd_std"] = long["rs_std"] - long["ra_std"]

    long["rest"] = g["game_date"].diff().dt.days.fillna(4).clip(0, 10)

    keep = [
        "game_pk", "team", "is_home",
        "rs_10", "ra_10", "wp_10", "rd_10",
        "rs_30", "ra_30", "wp_30", "rd_30",
        "rs_std", "ra_std", "wp_std", "rd_std",
        "rest",
    ]
    return long[keep]


def _elo_walk(games: pd.DataFrame,
              k: float = 4.0, home_adv: float = 24.0, init: float = 1500.0,
              season_regress: float = 0.33) -> pd.DataFrame:
    games = games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    rows = []
    for _, g in games.iterrows():
        h, a, season = g["home_team"], g["away_team"], int(g["season"])
        for team in (h, a):
            if team in last_season and last_season[team] != season:
                ratings[team] = init + (ratings[team] - init) * (1 - season_regress)
            last_season[team] = season
        rh = ratings.get(h, init)
        ra = ratings.get(a, init)
        exp_h = 1.0 / (1.0 + 10 ** (-((rh + home_adv) - ra) / 400))
        rows.append({
            "game_pk": g["game_pk"],
            "elo_home": rh, "elo_away": ra,
            "elo_exp_home": exp_h,
            "elo_diff": (rh + home_adv) - ra,
        })
        result_h = 1.0 if int(g["home_win"]) == 1 else 0.0
        ratings[h] = rh + k * (result_h - exp_h)
        ratings[a] = ra + k * ((1 - result_h) - (1 - exp_h))
    return pd.DataFrame(rows)


def compute_starter_rolling(logs: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rolling last-N-start stats for every pitcher, strictly shift(1)."""
    logs = logs.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)
    g = logs.groupby("pitcher_id", group_keys=False)

    def _rsum(col: str) -> pd.Series:
        return g[col].apply(lambda s: s.shift(1).rolling(window, min_periods=1).sum())

    ip = _rsum("ip")
    er = _rsum("er")
    k = _rsum("k")
    bb = _rsum("bb")
    hbp = _rsum("hbp")
    hr = _rsum("hr")

    ip_safe = ip.where(ip > 0, other=pd.NA)
    logs["sp_era_5"] = (er * 9) / ip_safe
    logs["sp_k9_5"] = (k * 9) / ip_safe
    logs["sp_bb9_5"] = (bb * 9) / ip_safe
    logs["sp_fip_5"] = ((13 * hr + 3 * (bb + hbp) - 2 * k) / ip_safe) + 3.10

    return logs[["pitcher_id", "game_date",
                 "sp_era_5", "sp_k9_5", "sp_bb9_5", "sp_fip_5"]]


def _attach_starter_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.config import CFG
    starters_path = CFG.raw_dir / "starters.parquet"
    logs_path = CFG.raw_dir / "starter_logs.parquet"

    if not starters_path.exists() or not logs_path.exists():
        print("[rolling_v2] WARNING starter files missing. Run "
              "`python -m src.etl.starter_logs` first for best results.")
        for col in ["home_sp_era_5", "home_sp_k9_5", "home_sp_bb9_5",
                    "home_sp_fip_5", "away_sp_era_5", "away_sp_k9_5",
                    "away_sp_bb9_5", "away_sp_fip_5",
                    "diff_sp_era_5", "diff_sp_k9_5", "diff_sp_bb9_5",
                    "diff_sp_fip_5"]:
            df[col] = pd.NA
        return df

    starters = pd.read_parquet(starters_path)
    starters["game_date"] = pd.to_datetime(starters["game_date"])
    logs = pd.read_parquet(logs_path)
    logs["game_date"] = pd.to_datetime(logs["game_date"])

    sp_roll = compute_starter_rolling(logs, window=5)

    starters_dates = starters[["game_pk", "game_date"]]
    home_map = starters[["game_pk", "home_sp_id"]].rename(
        columns={"home_sp_id": "pitcher_id"})
    away_map = starters[["game_pk", "away_sp_id"]].rename(
        columns={"away_sp_id": "pitcher_id"})

    home_join = home_map.merge(starters_dates, on="game_pk").merge(
        sp_roll, on=["pitcher_id", "game_date"], how="left"
    ).drop(columns=["pitcher_id", "game_date"])
    away_join = away_map.merge(starters_dates, on="game_pk").merge(
        sp_roll, on=["pitcher_id", "game_date"], how="left"
    ).drop(columns=["pitcher_id", "game_date"])

    home_join = home_join.rename(columns={
        "sp_era_5": "home_sp_era_5", "sp_k9_5": "home_sp_k9_5",
        "sp_bb9_5": "home_sp_bb9_5", "sp_fip_5": "home_sp_fip_5",
    })
    away_join = away_join.rename(columns={
        "sp_era_5": "away_sp_era_5", "sp_k9_5": "away_sp_k9_5",
        "sp_bb9_5": "away_sp_bb9_5", "sp_fip_5": "away_sp_fip_5",
    })

    df = df.merge(home_join, on="game_pk", how="left")
    df = df.merge(away_join, on="game_pk", how="left")

    for metric in ("sp_era_5", "sp_k9_5", "sp_bb9_5", "sp_fip_5"):
        df[f"diff_{metric}"] = df[f"home_{metric}"] - df[f"away_{metric}"]

    return df


def build_rolling_matrix(games: pd.DataFrame) -> pd.DataFrame:
    games = games.sort_values("game_date").reset_index(drop=True)
    games["game_date"] = pd.to_datetime(games["game_date"])

    team_roll = compute_team_rolling(games)

    home_roll = team_roll[team_roll["is_home"] == 1].drop(columns="is_home")
    home_roll = home_roll.rename(columns={
        c: f"home_{c}" for c in home_roll.columns
        if c not in ("game_pk", "team")
    }).rename(columns={"team": "home_team"})

    away_roll = team_roll[team_roll["is_home"] == 0].drop(columns="is_home")
    away_roll = away_roll.rename(columns={
        c: f"away_{c}" for c in away_roll.columns
        if c not in ("game_pk", "team")
    }).rename(columns={"team": "away_team"})

    df = games[["game_pk", "season", "game_date", "home_team", "away_team",
                "home_runs", "away_runs", "home_win", "total_runs"]].copy()
    df = df.merge(home_roll, on=["game_pk", "home_team"], how="left")
    df = df.merge(away_roll, on=["game_pk", "away_team"], how="left")

    elo = _elo_walk(games)
    df = df.merge(elo, on="game_pk", how="left")

    for base in ("rs_10", "ra_10", "wp_10", "rd_10",
                 "rs_30", "ra_30", "wp_30", "rd_30",
                 "rs_std", "ra_std", "wp_std", "rd_std"):
        df[f"diff_{base}"] = df[f"home_{base}"] - df[f"away_{base}"]

    df = _attach_starter_features(df)

    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(df[num].median(numeric_only=True))
    return df


FEATURE_COLS_V2 = [
    "elo_home", "elo_away", "elo_exp_home", "elo_diff",
    "home_rs_10", "home_ra_10", "home_wp_10", "home_rd_10",
    "away_rs_10", "away_ra_10", "away_wp_10", "away_rd_10",
    "home_rs_30", "home_ra_30", "home_wp_30", "home_rd_30",
    "away_rs_30", "away_ra_30", "away_wp_30", "away_rd_30",
    "home_rs_std", "home_ra_std", "home_wp_std", "home_rd_std",
    "away_rs_std", "away_ra_std", "away_wp_std", "away_rd_std",
    "home_rest", "away_rest",
    "diff_rs_10", "diff_ra_10", "diff_wp_10", "diff_rd_10",
    "diff_rs_30", "diff_ra_30", "diff_wp_30", "diff_rd_30",
    "diff_rs_std", "diff_ra_std", "diff_wp_std", "diff_rd_std",
    "home_sp_era_5", "home_sp_k9_5", "home_sp_bb9_5", "home_sp_fip_5",
    "away_sp_era_5", "away_sp_k9_5", "away_sp_bb9_5", "away_sp_fip_5",
    "diff_sp_era_5", "diff_sp_k9_5", "diff_sp_bb9_5", "diff_sp_fip_5",
]


def main() -> None:
    from src.config import CFG
    from src.utils.io import read_parquet, write_parquet

    games = read_parquet(CFG.raw_dir / "games.parquet")
    df = build_rolling_matrix(games)
    out_path = CFG.processed_dir / "features_v2.parquet"
    write_parquet(df, out_path)
    print(f"[rolling_v2] wrote {len(df)} rows, {len(df.columns)} cols -> {out_path}")
    missing = [c for c in FEATURE_COLS_V2 if c not in df.columns]
    if missing:
        print(f"[rolling_v2] WARNING missing cols: {missing}")


if __name__ == "__main__":
    main()