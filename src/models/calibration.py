"""Reliability diagram + Brier/log-loss summary."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def calibration_report(y_true: np.ndarray, p: np.ndarray,
                       out_path: Path, title: str = "Reliability") -> dict:
    brier = brier_score_loss(y_true, p)
    ll = log_loss(y_true, p, labels=[0, 1])
    auc = roc_auc_score(y_true, p)
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(f"{title}\nBrier={brier:.4f}  LogLoss={ll:.4f}  AUC={auc:.4f}")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return {"brier": brier, "log_loss": ll, "auc": auc}
