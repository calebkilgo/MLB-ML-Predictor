"""Train calibrated classifier (home win) + run-total regressor."""
from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.config import CFG
from src.models.calibration import calibration_report
from src.utils.io import read_parquet
from src.utils.time_splits import season_split

FEATURE_COLS = [
    "elo_home", "elo_away", "elo_exp_home",
    "home_form_rs", "home_form_ra", "home_form_win", "home_rest",
    "away_form_rs", "away_form_ra", "away_form_win", "away_rest",
    "park_factor", "temp_c", "wind_kph", "precip_mm",
    "home_sp_era", "home_sp_fip", "home_sp_k9", "home_sp_bb9",
    "away_sp_era", "away_sp_fip", "away_sp_k9", "away_sp_bb9",
]


def _xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols].astype(float), df[target]


def main() -> None:
    df = read_parquet(CFG.features_path)
    train, val, test = season_split(df, val_seasons=1, test_seasons=1)

    X_tr, y_tr = _xy(train, "home_win")
    X_val, y_val = _xy(val, "home_win")
    X_te, y_te = _xy(test, "home_win")

    base = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=42,
    )
    base.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    clf = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    clf.fit(X_val, y_val)

    p_test = clf.predict_proba(X_te)[:, 1]
    metrics = calibration_report(
        y_te.values, p_test, CFG.report_dir / "calibration_test.png",
        title="Test calibration",
    )
    print("[TRAIN] test metrics:", metrics)

    # Run-total regressor (Tweedie handles skewed non-negative totals well).
    X_tr_r, y_tr_r = _xy(train, "total_runs")
    reg = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.2,
        n_estimators=600, learning_rate=0.03, num_leaves=31, random_state=42,
    )
    reg.fit(X_tr_r, y_tr_r, eval_set=[(_xy(val, "total_runs"))])

    joblib.dump({"model": clf, "features": list(X_tr.columns)},
                CFG.model_dir / "clf_calibrated.pkl")
    joblib.dump({"model": reg, "features": list(X_tr.columns)},
                CFG.model_dir / "runs_reg.pkl")
    (CFG.report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[TRAIN] saved models to", CFG.model_dir)


if __name__ == "__main__":
    main()
