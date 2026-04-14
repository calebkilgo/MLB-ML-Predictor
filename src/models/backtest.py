"""Walk-forward backtest: predict each test season, aggregate metrics."""
from __future__ import annotations

import joblib
import pandas as pd

from src.config import CFG
from src.models.calibration import calibration_report
from src.utils.io import read_parquet


def main() -> None:
    df = read_parquet(CFG.features_path)
    bundle = joblib.load(CFG.model_dir / "clf_calibrated.pkl")
    model, feats = bundle["model"], bundle["features"]

    seasons = sorted(df["season"].unique())[-3:]
    rows = []
    for s in seasons:
        sub = df[df["season"] == s]
        X = sub[feats].astype(float)
        p = model.predict_proba(X)[:, 1]
        m = calibration_report(
            sub["home_win"].values, p,
            CFG.report_dir / f"calibration_{s}.png",
            title=f"Season {s}",
        )
        rows.append({"season": s, **m, "n": len(sub)})

    out = pd.DataFrame(rows)
    out.to_csv(CFG.report_dir / "backtest.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
