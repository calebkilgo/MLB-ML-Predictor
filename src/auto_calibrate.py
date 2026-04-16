"""Continuous auto-calibration: analyze resolved predictions and update
adjustment weights when enough data has accumulated.

This module is called automatically by the board-builder background thread
after each resolution pass. It runs the logistic-regression fitting from
src.analysis.analyze_adjustments and applies the recommended weight updates
to src/adjustments/weights.py, then hot-reloads the affected modules so
the new weights take effect immediately without a server restart.

Calibration is gated by:
  - Minimum resolved games (MIN_GAMES, default 50)
  - Minimum games since last calibration run (MIN_NEW_SINCE_CAL, default 20)
  - Minimum time between calibration runs (MIN_HOURS_BETWEEN, default 12)

Calibration history is stored in data/calibration/history.json.
"""
from __future__ import annotations

import importlib
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.config import CFG

logger = logging.getLogger(__name__)

_LOCK = Lock()

LOG_PATH = CFG.data_dir / "predictions" / "log.csv"
WEIGHTS_PATH = Path(__file__).parent / "adjustments" / "weights.py"
CAL_DIR = CFG.data_dir / "calibration"
HISTORY_PATH = CAL_DIR / "history.json"

MIN_GAMES = 20           # won't run below this many resolved games
MIN_NEW_SINCE_CAL = 10   # won't re-run unless this many new games since last run
MIN_HOURS_BETWEEN = 12   # won't run more frequently than this

DELTA_COLS = [
    "market_logit_delta",
    "statcast_pitcher_logit_delta",
    "lineup_logit_delta",
    "bullpen_logit_delta",
    "wind_logit_delta",
]

WEIGHT_FOR_DELTA = {
    "market_logit_delta":            ("W_MARKET",          "market"),
    "statcast_pitcher_logit_delta":  ("ERA_GAP_TO_LOGIT",  "linear"),
    "lineup_logit_delta":            ("WOBA_PER_LOGIT",     "inverse"),
    "bullpen_logit_delta":           ("FIP_PER_LOGIT",      "inverse"),
    "wind_logit_delta":              ("WIND_RUN_TO_LOGIT",  "linear"),
}

# Clamp factors to avoid runaway updates in small-sample fits.
_MAX_SCALE_UP   = 2.5
_MAX_SCALE_DOWN = 0.20   # floor at 20% of old value


def _load_history() -> dict:
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text())
        except Exception:
            pass
    return {"runs": [], "n_resolved_at_last_run": 0}


def _save_history(h: dict) -> None:
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(h, indent=2, default=str))


def _load_resolved() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(LOG_PATH)
    df = df[df["resolved_at"].notna() & (df["resolved_at"] != "")].copy()
    df["home_won"] = (df["winner_team"] == df["home_team"]).astype(int)
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


def _fit(df: pd.DataFrame):
    y = df["home_won"].values.astype(int)
    X = df[DELTA_COLS].values.astype(float)
    offset = df["base_logit"].values.astype(float)
    X_aug = np.column_stack([offset, X])
    clf = LogisticRegression(C=10.0, fit_intercept=False, max_iter=2000, solver="lbfgs")
    clf.fit(X_aug, y)
    return clf.coef_[0]


def _weight_update(old: float, rule: str, coef: float) -> float:
    if coef <= 0:
        return max(0.0, old * _MAX_SCALE_DOWN)
    if rule == "linear":
        scaled = coef
        scaled = max(_MAX_SCALE_DOWN, min(_MAX_SCALE_UP, scaled))
        return old * scaled
    if rule == "inverse":
        scaled = 1.0 / coef
        scaled = max(_MAX_SCALE_DOWN, min(_MAX_SCALE_UP, scaled))
        return old * scaled
    if rule == "market":
        # W_MARKET is a blend weight [0,1]; scale multiplicatively, clamp
        new = old * max(_MAX_SCALE_DOWN, min(_MAX_SCALE_UP, coef))
        return max(0.0, min(1.0, new))
    return old


def _apply_weights(coefs: np.ndarray) -> list[tuple[str, float, float]]:
    """Rewrite weights.py with fitted coefficients. Returns list of (name, old, new)."""
    _base_coef, *delta_coefs = coefs

    # Import current weight values
    import importlib
    import src.adjustments.weights as W
    importlib.reload(W)  # ensure fresh values

    text = WEIGHTS_PATH.read_text()
    changes = []
    for col, coef in zip(DELTA_COLS, delta_coefs):
        weight_name, rule = WEIGHT_FOR_DELTA[col]
        old_val = getattr(W, weight_name)
        new_val = _weight_update(old_val, rule, coef)
        # Round to 4 decimal places to keep the file tidy
        new_val = round(new_val, 4)
        pattern = re.compile(
            rf"^({re.escape(weight_name)}(?::\s*float)?\s*=\s*)([\d.eE+-]+)",
            re.MULTILINE,
        )
        if not pattern.search(text):
            logger.warning("[auto_calibrate] could not find %s in weights.py", weight_name)
            continue
        text = pattern.sub(lambda m, nv=new_val: f"{m.group(1)}{nv}", text, count=1)
        changes.append((weight_name, old_val, new_val))

    WEIGHTS_PATH.write_text(text)
    return changes


def _hot_reload_adjustments() -> None:
    """Reload all adjustment modules so new weights take effect immediately."""
    mods_to_reload = [
        "src.adjustments.weights",
        "src.adjustments.market",
        "src.adjustments.statcast_pitcher",
        "src.adjustments.lineup",
        "src.adjustments.bullpen",
        "src.adjustments.park_wind",
        "src.adjustments.umpire",
        "src.adjustments",
    ]
    for mod_name in mods_to_reload:
        if mod_name in sys.modules:
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception as e:
                logger.warning("[auto_calibrate] reload %s failed: %s", mod_name, e)

    # Clear lru_cache on predict._load so the model bundle is re-examined
    try:
        from src.models.predict import _load
        _load.cache_clear()
    except Exception:
        pass


def _fit_temperature_scale(y: np.ndarray, p: np.ndarray,
                           n_iter: int = 200) -> float:
    """Find optimal temperature T such that p_scaled = sigmoid(logit(p) / T).

    T < 1.0 → model is over-confident (probabilities too extreme), shrink more.
    T > 1.0 → model is under-confident (probabilities too close to 0.5), shrink less.

    Minimises cross-entropy via grid search over T ∈ [0.5, 2.0].
    """
    from scipy.optimize import minimize_scalar

    def nll(T: float) -> float:
        T = max(0.1, T)
        logits = np.log(np.clip(p, 1e-7, 1 - 1e-7) /
                        (1 - np.clip(p, 1e-7, 1 - 1e-7)))
        p_t = 1.0 / (1.0 + np.exp(-logits / T))
        p_t = np.clip(p_t, 1e-7, 1 - 1e-7)
        return -float(np.mean(y * np.log(p_t) + (1 - y) * np.log(1 - p_t)))

    result = minimize_scalar(nll, bounds=(0.5, 2.0), method="bounded")
    return float(result.x)


def _update_confidence_shrink(T: float) -> None:
    """Rewrite CONFIDENCE_SHRINK in weights.py based on temperature T.

    Current shrink factor S maps to new S' via:
        S' = clamp(S / T, 0.40, 0.95)

    Intuition: T=1.2 means the model is under-confident, so we reduce
    shrinkage (allow the model to be more decisive). T=0.85 means it's
    over-confident, so we increase shrinkage.
    """
    import src.adjustments.weights as W
    importlib.reload(W)
    old_shrink = W.CONFIDENCE_SHRINK
    new_shrink = round(max(0.40, min(0.95, old_shrink / T)), 4)
    if abs(new_shrink - old_shrink) < 0.005:
        return  # change too small to bother writing
    text = WEIGHTS_PATH.read_text()
    pattern = re.compile(
        r"^(CONFIDENCE_SHRINK(?::\s*float)?\s*=\s*)([\d.eE+-]+)",
        re.MULTILINE,
    )
    text = pattern.sub(lambda m: f"{m.group(1)}{new_shrink}", text, count=1)
    WEIGHTS_PATH.write_text(text)
    logger.info(
        "[auto_calibrate] CONFIDENCE_SHRINK %s -> %s (T=%.3f)",
        old_shrink, new_shrink, T,
    )


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    brier = float(brier_score_loss(y, p))
    ll = float(log_loss(y, p, labels=[0, 1]))
    acc = float(((p > 0.5).astype(int) == y).mean())
    return {"brier": brier, "log_loss": ll, "accuracy": acc}


def check_and_calibrate(min_games: int = MIN_GAMES,
                        min_new: int = MIN_NEW_SINCE_CAL,
                        min_hours: float = MIN_HOURS_BETWEEN,
                        force: bool = False) -> dict:
    """Run calibration if conditions are met. Thread-safe.

    Returns a status dict describing what was done.
    """
    with _LOCK:
        history = _load_history()
        df = _load_resolved()
        n = len(df)

        if n < min_games and not force:
            return {
                "ran": False,
                "reason": f"only {n} resolved games, need {min_games}",
                "n_resolved": n,
            }

        n_at_last = history.get("n_resolved_at_last_run", 0)
        new_since = n - n_at_last
        if new_since < min_new and not force:
            return {
                "ran": False,
                "reason": f"only {new_since} new games since last run ({min_new} required)",
                "n_resolved": n,
                "new_since_last": new_since,
            }

        last_runs = history.get("runs", [])
        if last_runs and not force:
            last_ts_str = last_runs[-1].get("timestamp", "")
            if last_ts_str:
                try:
                    last_ts = datetime.fromisoformat(last_ts_str).replace(
                        tzinfo=timezone.utc
                    )
                    hours_ago = (
                        datetime.now(tz=timezone.utc) - last_ts
                    ).total_seconds() / 3600
                    if hours_ago < min_hours:
                        return {
                            "ran": False,
                            "reason": f"last run {hours_ago:.1f}h ago, need {min_hours}h gap",
                            "n_resolved": n,
                        }
                except Exception:
                    pass

        # --- Run calibration ---
        logger.info("[auto_calibrate] running with %d resolved games", n)
        try:
            coefs = _fit(df)
        except Exception as e:
            logger.error("[auto_calibrate] fit failed: %s", e)
            return {"ran": False, "reason": f"fit failed: {e}", "n_resolved": n}

        # Compute metrics before and after
        base_metrics = _metrics(
            df["home_won"].values,
            df["base_p_home"].values,
        )
        if "adjusted_p_home" in df.columns:
            adj = pd.to_numeric(df["adjusted_p_home"], errors="coerce").fillna(
                df["base_p_home"]
            ).clip(1e-6, 1 - 1e-6).values
            adj_metrics = _metrics(df["home_won"].values, adj)
        else:
            adj_metrics = base_metrics

        # Apply updated adjustment weights
        changes = _apply_weights(coefs)

        # Temperature scaling: dynamically tune CONFIDENCE_SHRINK so the
        # model stays well-calibrated as the season progresses.
        if "adjusted_p_home" in df.columns:
            adj_p = pd.to_numeric(df["adjusted_p_home"], errors="coerce").fillna(
                df["base_p_home"]
            ).clip(1e-7, 1 - 1e-7).values
        else:
            adj_p = df["base_p_home"].values
        try:
            T = _fit_temperature_scale(df["home_won"].values, adj_p)
            _update_confidence_shrink(T)
        except Exception as e:
            logger.warning("[auto_calibrate] temperature scaling failed: %s", e)
            T = 1.0

        _hot_reload_adjustments()

        now_str = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        base_coef = float(coefs[0])
        delta_coefs = {col: float(c) for col, c in zip(DELTA_COLS, coefs[1:])}

        run_record = {
            "timestamp": now_str,
            "n_resolved": n,
            "new_since_last": int(new_since),
            "base_coef": base_coef,
            "delta_coefs": delta_coefs,
            "before_metrics": base_metrics,
            "adj_metrics_before": adj_metrics,
            "weight_changes": [
                {"name": name, "old": old, "new": new}
                for name, old, new in changes
            ],
        }
        history["runs"].append(run_record)
        # Keep only the last 30 calibration run records
        history["runs"] = history["runs"][-30:]
        history["n_resolved_at_last_run"] = n
        _save_history(history)

        logger.info(
            "[auto_calibrate] done. base_coef=%.3f acc_before=%.4f acc_adj=%.4f "
            "changes=%s",
            base_coef,
            base_metrics["accuracy"],
            adj_metrics["accuracy"],
            [(c[0], f"{c[1]:.4f}->{c[2]:.4f}") for c in changes],
        )

        return {
            "ran": True,
            "timestamp": now_str,
            "n_resolved": n,
            "new_since_last": new_since,
            "base_coef": base_coef,
            "delta_coefs": delta_coefs,
            "weight_changes": run_record["weight_changes"],
            "base_metrics": base_metrics,
            "adj_metrics_before": adj_metrics,
        }


def history_summary() -> dict:
    """Return the calibration history for the /api/calibration endpoint."""
    h = _load_history()
    runs = h.get("runs", [])
    last = runs[-1] if runs else None
    return {
        "n_calibration_runs": len(runs),
        "last_run": last,
        "n_resolved_at_last_run": h.get("n_resolved_at_last_run", 0),
    }
