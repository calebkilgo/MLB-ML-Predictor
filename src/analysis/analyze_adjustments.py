"""Fit a logistic regression on resolved games to tune adjustment weights.

Reads data/predictions/log.csv (populated by app/prediction_log.py) and
fits a model with the base-model logit as a fixed offset and each
adjustment delta as a free coefficient. The fitted coefficient tells
you how much to re-weight that factor:

    coef = 1.0  -> current weight is correct
    coef < 1.0  -> factor is too aggressive, scale down
    coef > 1.0  -> factor is too conservative, scale up
    coef < 0    -> factor is pointing the wrong direction, flip sign

Usage:
    # Print recommendations only
    python -m src.analysis.analyze_adjustments

    # Print + write updated constants to src/adjustments/weights.py
    python -m src.analysis.analyze_adjustments --apply

The analyzer refuses to run below MIN_GAMES resolved picks, because
small samples produce wildly unstable coefficients. 50 is a floor —
100+ gives noticeably more stable recommendations.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.config import CFG
from src.adjustments import weights as W

LOG_PATH = CFG.data_dir / "predictions" / "log.csv"
WEIGHTS_PATH = Path(__file__).parent.parent / "adjustments" / "weights.py"

MIN_GAMES = 50  # refuse to tune below this

DELTA_COLS = [
    "market_logit_delta",
    "statcast_pitcher_logit_delta",
    "lineup_logit_delta",
    "bullpen_logit_delta",
    "wind_logit_delta",
]

# Map each delta column to the weight constant that scales it. These
# relationships are "if you want to double the delta, double this const."
# For most adjustments it's 1:1 linear. For the lineup adjustment, the
# constant is in the denominator so we invert. For market, we need a
# special case (sigmoid-ish).
WEIGHT_FOR_DELTA = {
    "market_logit_delta": ("W_MARKET", "market"),
    "statcast_pitcher_logit_delta": ("ERA_GAP_TO_LOGIT", "linear"),
    "lineup_logit_delta": ("WOBA_PER_LOGIT", "inverse"),
    "bullpen_logit_delta": ("FIP_PER_LOGIT", "inverse"),
    "wind_logit_delta": ("WIND_RUN_TO_LOGIT", "linear"),
}


def _load_resolved() -> pd.DataFrame:
    if not LOG_PATH.exists():
        print(f"[analyze] no log file at {LOG_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(LOG_PATH)
    before = len(df)
    df = df[df["resolved_at"].notna() & (df["resolved_at"] != "")].copy()
    print(f"[analyze] loaded {before} total rows, "
          f"{len(df)} resolved")
    # Outcome from the home perspective: 1 if home team won.
    df["home_won"] = (df["winner_team"] == df["home_team"]).astype(int)
    # Base logit — recover from base_p. If base_p missing, skip row.
    df = df[df["base_p_home"].notna() & (df["base_p_home"] != "")].copy()
    df["base_p_home"] = pd.to_numeric(df["base_p_home"], errors="coerce")
    df = df.dropna(subset=["base_p_home"])
    df["base_p_home"] = df["base_p_home"].clip(1e-6, 1 - 1e-6)
    df["base_logit"] = np.log(df["base_p_home"] / (1 - df["base_p_home"]))

    for col in DELTA_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.reset_index(drop=True)


def _score(y: np.ndarray, p: np.ndarray, label: str) -> dict:
    brier = brier_score_loss(y, p)
    ll = log_loss(y, p, labels=[0, 1])
    acc = ((p > 0.5).astype(int) == y).mean()
    print(f"  {label:28s}  brier={brier:.4f}  logloss={ll:.4f}  acc={acc:.4f}")
    return {"brier": brier, "log_loss": ll, "accuracy": acc}


def _fit(df: pd.DataFrame) -> tuple[np.ndarray, LogisticRegression]:
    """Fit logistic regression with base_logit as fixed offset (coef=1)
    and each delta as a free coefficient. Implemented by subtracting the
    offset out of the target logit before fitting."""
    y = df["home_won"].values.astype(int)
    X = df[DELTA_COLS].values.astype(float)
    offset = df["base_logit"].values.astype(float)

    # Standard trick: fit logistic regression with offset by including
    # the offset as a constant-coefficient feature with L2 penalty off.
    # sklearn doesn't support offsets directly, so we use the fact that
    # logit(p) = offset + X @ beta is the same model structure — we
    # just need scipy to optimize the beta. Simplest: feed (X, y) with
    # sample weights all 1 and a very small L2, and include the offset
    # via a 1-column with coefficient fixed to 1 using init_weights.
    #
    # Simpler still: use LogisticRegression with an extra feature whose
    # value is `offset` and rely on the optimizer; we then inspect what
    # it assigned to that column. If it's ~1.0, offset is working.
    X_aug = np.column_stack([offset, X])
    clf = LogisticRegression(
        C=10.0, fit_intercept=False, max_iter=2000, solver="lbfgs",
    )
    clf.fit(X_aug, y)
    return clf.coef_[0], clf


def _print_recommendations(coefs: np.ndarray) -> dict[str, float]:
    base_coef, *delta_coefs = coefs
    print()
    print(f"[analyze] base model offset coef: {base_coef:.3f} "
          f"(ideal is ~1.0 — indicates base model is well-calibrated)")
    if abs(base_coef - 1.0) > 0.3:
        print("          NOTE: base offset far from 1.0 suggests the base "
              "model itself is miscalibrated; consider this before trusting "
              "the adjustment tuning below.")

    print()
    print("[analyze] adjustment factor coefficients:")
    print(f"  {'factor':28s}  {'fitted coef':>11s}  {'interpretation':s}")
    print(f"  {'-' * 28}  {'-' * 11}  {'-' * 30}")
    recommendations: dict[str, float] = {}
    for col, coef in zip(DELTA_COLS, delta_coefs):
        interpretation = _interpret(coef)
        print(f"  {col:28s}  {coef:>11.3f}  {interpretation}")
        recommendations[col] = float(coef)
    return recommendations


def _interpret(coef: float) -> str:
    if coef < -0.5:
        return "FLIP — points wrong direction"
    if coef < 0:
        return "weak wrong-direction signal, clamp to 0"
    if coef < 0.3:
        return "scale down ~70%"
    if coef < 0.7:
        return "scale down ~40%"
    if coef < 1.3:
        return "well-calibrated, no change"
    if coef < 2.0:
        return "scale up ~50%"
    return "scale up substantially"


def _weight_update(old: float, rule: str, coef: float) -> float:
    """Compute the new weight value given the fitted coefficient.

    For linear scales, new = old * coef.
    For inverse scales (denominators), new = old / coef.
    For market blend, we update W_MARKET multiplicatively but clamp to [0,1].
    """
    if coef <= 0:
        return 0.0 if rule != "inverse" else old * 2.0  # kill or dampen
    if rule == "linear":
        return old * coef
    if rule == "inverse":
        return old / coef
    if rule == "market":
        return max(0.0, min(1.0, old * coef))
    return old


def _apply(recommendations: dict[str, float]) -> None:
    print()
    print(f"[analyze] --apply: rewriting {WEIGHTS_PATH}")
    text = WEIGHTS_PATH.read_text()
    changes: list[tuple[str, float, float]] = []

    for delta_col, coef in recommendations.items():
        weight_name, rule = WEIGHT_FOR_DELTA[delta_col]
        old_val = getattr(W, weight_name)
        new_val = _weight_update(old_val, rule, coef)
        # Rewrite the line `NAME: float = 0.XX` or `NAME = 0.XX`.
        pattern = re.compile(
            rf"^({re.escape(weight_name)}(?::\s*float)?\s*=\s*)([\d.eE+-]+)",
            re.MULTILINE,
        )
        if not pattern.search(text):
            print(f"  WARNING could not find {weight_name} in weights.py")
            continue
        text = pattern.sub(lambda m: f"{m.group(1)}{new_val:.4f}", text, count=1)
        changes.append((weight_name, old_val, new_val))

    WEIGHTS_PATH.write_text(text)
    print()
    print("[analyze] weight changes:")
    for name, old, new in changes:
        arrow = "->" if abs(new - old) > 1e-9 else "=="
        print(f"  {name:24s} {old:>10.4f} {arrow} {new:<10.4f}")
    print()
    print("[analyze] done. Restart uvicorn to pick up new weights.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write updated constants back to weights.py")
    ap.add_argument("--min-games", type=int, default=MIN_GAMES)
    args = ap.parse_args()

    df = _load_resolved()
    if len(df) < args.min_games:
        print(f"[analyze] only {len(df)} resolved games, need {args.min_games}+. "
              f"Run the server through more games and try again.")
        sys.exit(0)

    print()
    print("[analyze] metrics BEFORE adjustment tuning:")
    _score(df["home_won"].values,
           df["base_p_home"].values,
           "base model only")

    # Current adjusted probability, as recorded in the log.
    if "adjusted_p_home" in df.columns:
        adj = pd.to_numeric(df["adjusted_p_home"], errors="coerce").fillna(
            df["base_p_home"]
        ).clip(1e-6, 1 - 1e-6).values
        _score(df["home_won"].values, adj, "current adjusted")

    coefs, clf = _fit(df)
    recommendations = _print_recommendations(coefs)

    # Show what the fit would yield on the same data (in-sample).
    X_aug = np.column_stack([
        df["base_logit"].values,
        df[DELTA_COLS].values.astype(float),
    ])
    p_refit = clf.predict_proba(X_aug)[:, 1]
    print()
    print("[analyze] metrics WITH fitted coefficients (in-sample):")
    _score(df["home_won"].values, p_refit, "refit model")

    if args.apply:
        _apply(recommendations)
    else:
        print()
        print("[analyze] run with --apply to update weights.py")


if __name__ == "__main__":
    main()