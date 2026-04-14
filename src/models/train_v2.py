"""Walk-forward cross-validated training on the v2 rolling feature set.

v7: switches calibration from isotonic (overfit on small slices) to Platt
scaling (sigmoid) trained on the full last season in the training
window. Also fixes the int64 JSON serialization bug.

For each test season S:
  1. Train LightGBM on all seasons strictly before S.
  2. Platt-scale calibrate on the entirety of season S-1 (held out
     from the base-model fit).
  3. Score season S as pure holdout.

Outputs:
  models/clf_v2.pkl
  models/runs_reg_v2.pkl
  reports/backtest_v2.csv
  reports/calibration_v2_*.png
  reports/metrics_v2.json

Run:
    python -m src.models.train_v2
"""
from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.config import CFG
from src.features.rolling_v2 import FEATURE_COLS_V2
from src.models.calibration import calibration_report
from src.utils.io import read_parquet

FEATURES_PATH = CFG.processed_dir / "features_v2.parquet"

LGB_PARAMS = dict(
    n_estimators=700,
    learning_rate=0.025,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)


def _xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    cols = [c for c in FEATURE_COLS_V2 if c in df.columns]
    return df[cols].astype(float), df[target]


def _split_core_and_cal(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the ENTIRE most recent training season for calibration.

    Platt scaling needs a reasonably large, representative sample; 20%
    of one season was too small and unstable in v6.
    """
    last = train_df["season"].max()
    calibrate = train_df[train_df["season"] == last].copy()
    core = train_df[train_df["season"] < last].copy()
    return core, calibrate


def _train_one_season(df: pd.DataFrame, test_season: int) -> dict:
    train = df[df["season"] < test_season].copy()
    test = df[df["season"] == test_season].copy()
    if len(train) < 2000 or len(test) < 100:
        return {}

    core_train, calibrate = _split_core_and_cal(train)
    if len(core_train) < 1000 or len(calibrate) < 500:
        return {}

    X_tr, y_tr = _xy(core_train, "home_win")
    X_cal, y_cal = _xy(calibrate, "home_win")
    X_te, y_te = _xy(test, "home_win")

    base = lgb.LGBMClassifier(**LGB_PARAMS)
    base.fit(X_tr, y_tr)
    clf = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    clf.fit(X_cal, y_cal)

    p = clf.predict_proba(X_te)[:, 1]
    m = calibration_report(
        y_te.values, p,
        CFG.report_dir / f"calibration_v2_{test_season}.png",
        title=f"v2 walk-forward · season {test_season}",
    )
    m.update({
        "season": int(test_season), "n": int(len(test)),
        "n_train": int(len(core_train)), "n_cal": int(len(calibrate)),
    })
    m = {k: (float(v) if hasattr(v, "item") else v) for k, v in m.items()}
    return m


def _fit_final(df: pd.DataFrame) -> tuple:
    core_train, calibrate = _split_core_and_cal(df)

    X_tr, y_tr = _xy(core_train, "home_win")
    X_cal, y_cal = _xy(calibrate, "home_win")

    base = lgb.LGBMClassifier(**LGB_PARAMS)
    base.fit(X_tr, y_tr)
    clf = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    clf.fit(X_cal, y_cal)

    X_tr_r, y_tr_r = _xy(core_train, "total_runs")
    reg = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.2,
        n_estimators=700, learning_rate=0.025, num_leaves=31,
        random_state=42, verbose=-1,
    )
    reg.fit(X_tr_r, y_tr_r)

    features = list(X_tr.columns)
    return clf, reg, features, len(core_train), len(calibrate)


def main() -> None:
    df = read_parquet(FEATURES_PATH)
    seasons = sorted(df["season"].unique())
    print(f"[train_v2] {len(df)} games across seasons "
          f"{seasons[0]}-{seasons[-1]}")

    test_seasons = seasons[2:]
    rows: list[dict] = []
    for s in test_seasons:
        print(f"[train_v2] walk-forward test season {s}")
        m = _train_one_season(df, s)
        if m:
            rows.append(m)
            print(f"           brier={m['brier']:.4f} "
                  f"logloss={m['log_loss']:.4f} "
                  f"auc={m['auc']:.4f} n={m['n']}")

    if rows:
        bdf = pd.DataFrame(rows)
        bdf.to_csv(CFG.report_dir / "backtest_v2.csv", index=False)
        print("[train_v2] walk-forward summary:")
        print(bdf[["season", "n", "brier", "log_loss", "auc"]]
              .to_string(index=False))

    clf, reg, features, n_train, n_cal = _fit_final(df)
    joblib.dump({"model": clf, "features": features},
                CFG.model_dir / "clf_v2.pkl")
    joblib.dump({"model": reg, "features": features},
                CFG.model_dir / "runs_reg_v2.pkl")

    metrics = {
        "n_features": int(len(features)),
        "final_n_train": int(n_train),
        "final_n_cal": int(n_cal),
        "walk_forward": rows,
    }
    (CFG.report_dir / "metrics_v2.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )
    print(f"[train_v2] saved production models to {CFG.model_dir}")


if __name__ == "__main__":
    main()