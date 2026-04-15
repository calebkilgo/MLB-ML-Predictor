"""Runtime prediction helpers for the v2 rolling-feature model (v7).

Loads on import:
  - models/clf_v2.pkl
  - models/runs_reg_v2.pkl
  - data/processed/features_v2.parquet
  - data/raw/starter_logs.parquet (for live starter rolling lookup)

Live prediction path:
  1. Look up each team's most recent feature row (team-level rolling).
  2. If home_pitcher_id and away_pitcher_id are supplied in GameInput,
     look up each starter's most recent rolling FIP/ERA/K9/BB9 as of
     today from starter_logs.parquet.
  3. If starter IDs are missing or have no logs, fall back to
     league-average placeholders.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import joblib
import pandas as pd

from src.config import CFG
from src.features.rolling_v2 import compute_starter_rolling
from src.utils.io import read_parquet


@dataclass
class GameInput:
    home_team: str
    away_team: str
    # v7: live pitcher IDs for rolling-stat lookup. Optional; fall back
    # to league-average placeholders if missing or not in starter_logs.
    home_pitcher_id: int | None = None
    away_pitcher_id: int | None = None
    home_rest: float = 1.0
    away_rest: float = 1.0
    # Back-compat placeholders, not consumed by the v2 model.
    home_sp_era: float = 4.00
    away_sp_era: float = 4.00
    home_sp_fip: float = 4.00
    away_sp_fip: float = 4.00
    home_sp_k9: float = 8.5
    away_sp_k9: float = 8.5
    home_sp_bb9: float = 3.0
    away_sp_bb9: float = 3.0
    temp_c: float = 22.0
    wind_kph: float = 12.0
    precip_mm: float = 0.0


LEAGUE_SP = {
    "sp_era_5": 4.20, "sp_fip_5": 4.20, "sp_k9_5": 8.8, "sp_bb9_5": 3.1,
}


@lru_cache(maxsize=1)
def _load():
    clf = joblib.load(CFG.model_dir / "clf_v2.pkl")
    reg = joblib.load(CFG.model_dir / "runs_reg_v2.pkl")
    feats = read_parquet(CFG.processed_dir / "features_v2.parquet")
    feats["game_date"] = pd.to_datetime(feats["game_date"])

    logs_path = CFG.raw_dir / "starter_logs.parquet"
    if logs_path.exists():
        logs = pd.read_parquet(logs_path)
        logs["game_date"] = pd.to_datetime(logs["game_date"])
        sp_roll = compute_starter_rolling(logs, window=5)
        sp_roll = sp_roll.sort_values("game_date")
        sp_latest = sp_roll.groupby("pitcher_id", as_index=False).tail(1)
        sp_latest = sp_latest.set_index("pitcher_id")
    else:
        sp_latest = pd.DataFrame(
            columns=["sp_era_5", "sp_k9_5", "sp_bb9_5", "sp_fip_5"]
        ).set_index(pd.Index([], name="pitcher_id"))
    return clf, reg, feats, sp_latest


def _latest_team_row(team: str) -> dict[str, float]:
    _, _, feats, _ = _load()
    h = feats[feats["home_team"] == team].sort_values("game_date").tail(1)
    a = feats[feats["away_team"] == team].sort_values("game_date").tail(1)

    def _extract(row: pd.Series, side: str) -> dict[str, float]:
        out = {}
        metrics = (
            "rs_5",  "ra_5",  "wp_5",  "rd_5",
            "rs_10", "ra_10", "wp_10", "rd_10",
            "rs_30", "ra_30", "wp_30", "rd_30",
            "rs_std", "ra_std", "wp_std", "rd_std",
            "rs_ewm", "ra_ewm", "wp_ewm", "rd_ewm",
            "streak",
        )
        for metric in metrics:
            col = f"{side}_{metric}"
            out[metric] = float(row[col]) if col in row.index else 0.0
        out["elo"] = float(row[f"elo_{'home' if side == 'home' else 'away'}"])
        return out

    if len(h) and (not len(a) or
                   h["game_date"].iloc[0] >= a["game_date"].iloc[0]):
        return _extract(h.iloc[0], "home")
    if len(a):
        return _extract(a.iloc[0], "away")
    return {
        "rs_5": 4.5, "ra_5": 4.5, "wp_5": 0.5, "rd_5": 0.0,
        "rs_10": 4.5, "ra_10": 4.5, "wp_10": 0.5, "rd_10": 0.0,
        "rs_30": 4.5, "ra_30": 4.5, "wp_30": 0.5, "rd_30": 0.0,
        "rs_std": 4.5, "ra_std": 4.5, "wp_std": 0.5, "rd_std": 0.0,
        "rs_ewm": 4.5, "ra_ewm": 4.5, "wp_ewm": 0.5, "rd_ewm": 0.0,
        "streak": 0.0, "elo": 1500.0,
    }


def _latest_starter_stats(pitcher_id: int | None) -> dict[str, float]:
    if pitcher_id is None:
        return dict(LEAGUE_SP)
    _, _, _, sp_latest = _load()
    try:
        row = sp_latest.loc[int(pitcher_id)]
    except (KeyError, ValueError):
        return dict(LEAGUE_SP)
    out = {}
    for k, league in LEAGUE_SP.items():
        val = row.get(k)
        try:
            fval = float(val) if val is not None else league
            if pd.isna(fval):
                fval = league
        except (TypeError, ValueError):
            fval = league
        out[k] = fval
    return out


def build_row(g: GameInput) -> pd.DataFrame:
    h = _latest_team_row(g.home_team)
    a = _latest_team_row(g.away_team)

    home_sp = _latest_starter_stats(g.home_pitcher_id)
    away_sp = _latest_starter_stats(g.away_pitcher_id)

    home_adv = 24.0
    elo_diff = (h["elo"] + home_adv) - a["elo"]
    elo_exp_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400))

    row: dict[str, float] = {
        "elo_home": h["elo"],
        "elo_away": a["elo"],
        "elo_exp_home": elo_exp_home,
        "elo_diff": elo_diff,
        "home_rest": g.home_rest,
        "away_rest": g.away_rest,
    }
    for metric in (
        "rs_5",  "ra_5",  "wp_5",  "rd_5",
        "rs_10", "ra_10", "wp_10", "rd_10",
        "rs_30", "ra_30", "wp_30", "rd_30",
        "rs_std", "ra_std", "wp_std", "rd_std",
        "rs_ewm", "ra_ewm", "wp_ewm", "rd_ewm",
        "streak",
    ):
        row[f"home_{metric}"] = h[metric]
        row[f"away_{metric}"] = a[metric]
        row[f"diff_{metric}"] = h[metric] - a[metric]

    for metric in ("sp_era_5", "sp_k9_5", "sp_bb9_5", "sp_fip_5"):
        row[f"home_{metric}"] = home_sp[metric]
        row[f"away_{metric}"] = away_sp[metric]
        row[f"diff_{metric}"] = home_sp[metric] - away_sp[metric]

    return pd.DataFrame([row])


def predict(g: GameInput) -> dict:
    clf_b, reg_b, _, _ = _load()
    X = build_row(g)
    X_clf = X[[c for c in clf_b["features"] if c in X.columns]].astype(float)
    X_reg = X[[c for c in reg_b["features"] if c in X.columns]].astype(float)

    # Use stacked ensemble if available, fall back to single model.
    lgb_m = clf_b.get("lgb_model")
    xgb_m = clf_b.get("xgb_model")
    meta_m = clf_b.get("meta_model")
    if lgb_m is not None and xgb_m is not None and meta_m is not None:
        import numpy as np
        p_lgb = float(lgb_m.predict_proba(X_clf)[0, 1])
        p_xgb = float(xgb_m.predict_proba(X_clf)[0, 1])
        X_meta = np.array([[p_lgb, p_xgb]])
        p_home = float(meta_m.predict_proba(X_meta)[0, 1])
    else:
        p_home = float(clf_b["model"].predict_proba(X_clf)[0, 1])
    runs = float(reg_b["model"].predict(X_reg)[0])

    # Confidence = how far the model is from a coin-flip, expressed on [0, 1].
    # Using 2*(p - 0.5) gives 0% at p=0.5 and 100% at p=1.0 / p=0.0.
    # We also factor in signal strength from ELO diff and team rolling stats
    # so confidence reflects both the probability estimate AND the consistency
    # of the underlying signals.
    row = X.iloc[0]
    elo_diff = float(row.get("elo_diff", 0.0))
    diff_wp_10 = float(row.get("diff_wp_10", 0.0))
    diff_rd_10 = float(row.get("diff_rd_10", 0.0))

    # Normalize signal strength contributions (clamp to reasonable range)
    elo_signal = min(1.0, abs(elo_diff) / 200.0)      # 200 Elo pts ≈ strong
    form_signal = min(1.0, abs(diff_wp_10) / 0.20)    # 20 pp win-rate gap ≈ strong
    run_diff_signal = min(1.0, abs(diff_rd_10) / 1.5)  # 1.5 run/game ≈ strong

    # Blend: probability distance (primary) + signal consistency (secondary)
    p_dist = abs(p_home - 0.5) * 2.0  # [0, 1]
    signal_avg = (elo_signal + form_signal + run_diff_signal) / 3.0
    conf = float(min(1.0, 0.70 * p_dist + 0.30 * signal_avg))

    # Run distribution: weight home runs slightly more if home team favored
    weight = 0.45 + 0.1 * p_home
    home_runs = runs * weight
    away_runs = runs - home_runs

    return {
        "p_home_win": p_home,
        "p_away_win": 1 - p_home,
        "proj_total_runs": runs,
        "proj_home_runs": home_runs,
        "proj_away_runs": away_runs,
        "confidence": conf,
    }