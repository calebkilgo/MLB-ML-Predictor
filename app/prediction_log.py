"""Prediction logger: tracks every prediction through game resolution.

Writes one row per game to data/predictions/log.csv. The row is CREATED
the first time we see a game in 'Preview' state (locking in the pregame
prediction so we can't accidentally overwrite it with a live-state one),
and UPDATED once when the game reaches 'Final' with the actual outcome.

The schema is stable: never edit existing columns, only append new ones
at the end of FIELDS to keep old rows readable.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from threading import Lock

import httpx

from src.config import CFG

_LOG_PATH = CFG.data_dir / "predictions" / "log.csv"
_LOCK = Lock()

FIELDS = [
    "game_pk",
    "logged_at",
    "game_date",
    "home_team",
    "away_team",
    "home_pitcher",
    "away_pitcher",
    "base_p_home",
    "pre_shrink_p_home",
    "adjusted_p_home",
    "pick_team",
    "pick_prob",
    "ev",
    "market_home_american",
    "market_away_american",
    "market_logit_delta",
    "statcast_pitcher_logit_delta",
    "lineup_logit_delta",
    "bullpen_logit_delta",
    "wind_logit_delta",
    "total_logit_delta",
    "resolved_at",
    "final_home_score",
    "final_away_score",
    "winner_team",
    "pick_correct",
]


def _load() -> dict[str, dict]:
    if not _LOG_PATH.exists():
        return {}
    out: dict[str, dict] = {}
    with open(_LOG_PATH, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pk = row.get("game_pk")
            if pk:
                out[pk] = row
    return out


def _save(rows: dict[str, dict]) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for pk in sorted(rows.keys()):
            writer.writerow(rows[pk])


def _ev_at_110(p: float) -> float:
    return p * (100 / 110) - (1 - p)


def _compute_row(g: dict) -> dict | None:
    m = g.get("model") or {}
    if "p_home_win" not in m or m.get("error"):
        return None
    p_home = float(m["p_home_win"])
    p_away = 1 - p_home
    if p_home >= p_away:
        pick_team, pick_prob = g["home_team"], p_home
    else:
        pick_team, pick_prob = g["away_team"], p_away

    adj = g.get("adjustments", {}) or {}
    prob_adj = adj.get("probability", {}) or {}
    ctx = adj.get("context", {}) or {}

    return {
        "game_pk": str(g.get("game_pk", "")),
        "logged_at": datetime.utcnow().isoformat(timespec="seconds"),
        "game_date": g.get("game_date", "") or "",
        "home_team": g.get("home_team", "") or "",
        "away_team": g.get("away_team", "") or "",
        "home_pitcher": g.get("home_pitcher") or "",
        "away_pitcher": g.get("away_pitcher") or "",
        "base_p_home": prob_adj.get("base_p", ""),
        "pre_shrink_p_home": prob_adj.get("pre_shrink_p", ""),
        "adjusted_p_home": p_home,
        "pick_team": pick_team,
        "pick_prob": pick_prob,
        "ev": _ev_at_110(pick_prob),
        "market_home_american": ctx.get("market_home_american", "") or "",
        "market_away_american": ctx.get("market_away_american", "") or "",
        "market_logit_delta": prob_adj.get("market_logit_delta", ""),
        "statcast_pitcher_logit_delta": prob_adj.get("statcast_pitcher_logit_delta", ""),
        "lineup_logit_delta": prob_adj.get("lineup_logit_delta", ""),
        "bullpen_logit_delta": prob_adj.get("bullpen_logit_delta", ""),
        "wind_logit_delta": prob_adj.get("wind_logit_delta", ""),
        "total_logit_delta": prob_adj.get("total_logit_delta", ""),
        "resolved_at": "",
        "final_home_score": "",
        "final_away_score": "",
        "winner_team": "",
        "pick_correct": "",
    }


def record_games(games: list[dict]) -> dict:
    """Upsert rows for each game. Creates Preview rows, resolves Final rows.

    Retroactive mode: if a Final game has model data but no existing log entry
    (e.g. Preview was missed due to a restart), the entry is created and
    immediately resolved in the same pass.  Live/in-progress games are skipped
    for new entries to avoid using mid-game pitcher states.
    """
    with _LOCK:
        existing = _load()
        n_new = 0
        n_resolved = 0

        for g in games:
            pk = str(g.get("game_pk", ""))
            if not pk:
                continue
            state = g.get("state", "") or ""

            if pk not in existing:
                # Allow Preview (prospective) and Final (retroactive — missed
                # Preview window, e.g. after a redeploy).  Skip Live to avoid
                # using in-game pitcher changes as pre-game inputs.
                if state not in ("Preview", "Final"):
                    continue
                row = _compute_row(g)
                if row is None:
                    continue
                existing[pk] = row
                n_new += 1
                if state == "Preview":
                    continue
                # Final: fall through to resolve immediately below.

            # Existing row: resolve once the game is Final.
            row = existing[pk]
            if row.get("resolved_at"):
                continue
            if state != "Final":
                continue
            hs, as_ = g.get("home_score"), g.get("away_score")
            if hs is None or as_ is None:
                continue
            try:
                hs_i, as_i = int(hs), int(as_)
            except (TypeError, ValueError):
                continue
            winner = g["home_team"] if hs_i > as_i else g["away_team"]
            pick_correct = 1 if winner == row.get("pick_team") else 0
            row["resolved_at"] = datetime.utcnow().isoformat(timespec="seconds")
            row["final_home_score"] = hs_i
            row["final_away_score"] = as_i
            row["winner_team"] = winner
            row["pick_correct"] = pick_correct
            n_resolved += 1

        _save(existing)
        return {"new": n_new, "resolved": n_resolved, "total": len(existing)}


def resolve_past_games() -> dict:
    """Retroactive resolution pass: fetch Final scores for any unresolved
    games whose game_date is strictly before today. Useful after a server
    restart or downtime that caused us to miss the Final state transition.

    Returns counts of how many games were newly resolved.
    """
    today_str = date.today().isoformat()
    with _LOCK:
        existing = _load()
        n_resolved = 0
        n_checked = 0

        for pk, row in existing.items():
            if row.get("resolved_at"):
                continue  # already resolved
            gd = row.get("game_date", "")
            if not gd or gd > today_str:
                continue  # future only — also retroactively resolve today

            # Query the MLB API for the game's final state.
            n_checked += 1
            try:
                url = (
                    f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
                    "?fields=gameData,status,linescore,teams"
                )
                r = httpx.get(url, timeout=10.0)
                r.raise_for_status()
                data = r.json()
            except Exception:
                continue

            gd_data = data.get("gameData", {})
            status = gd_data.get("status", {})
            abstract = status.get("abstractGameState", "")
            if abstract != "Final":
                continue

            teams = gd_data.get("teams", {})
            home_name = (teams.get("home") or {}).get("abbreviation", "")
            away_name = (teams.get("away") or {}).get("abbreviation", "")

            # Get linescore runs
            ls = data.get("liveData", {}).get("linescore", {}) or {}
            home_runs = (ls.get("teams") or {}).get("home", {}).get("runs")
            away_runs = (ls.get("teams") or {}).get("away", {}).get("runs")
            if home_runs is None or away_runs is None:
                continue

            try:
                hs_i, as_i = int(home_runs), int(away_runs)
            except (TypeError, ValueError):
                continue

            # Use abbreviations from the row if the live data abbreviation is empty
            home_team = row.get("home_team", "") or home_name
            away_team = row.get("away_team", "") or away_name
            winner = home_team if hs_i > as_i else away_team
            pick_correct = 1 if winner == row.get("pick_team") else 0
            row["resolved_at"] = datetime.utcnow().isoformat(timespec="seconds")
            row["final_home_score"] = hs_i
            row["final_away_score"] = as_i
            row["winner_team"] = winner
            row["pick_correct"] = pick_correct
            n_resolved += 1

        if n_resolved > 0:
            _save(existing)
        return {"checked": n_checked, "resolved": n_resolved, "total": len(existing)}


def summary() -> dict:
    """Bucketed calibration summary: win rate vs predicted probability."""
    rows = _load()
    resolved = [r for r in rows.values() if r.get("resolved_at")]
    if not resolved:
        return {"n_resolved": 0, "message": "No resolved games yet."}

    n = len(resolved)
    correct = 0
    probs: list[float] = []
    for r in resolved:
        try:
            correct += int(r.get("pick_correct") or 0)
            probs.append(float(r["pick_prob"]))
        except (TypeError, ValueError, KeyError):
            pass

    mean_pred = sum(probs) / len(probs) if probs else 0.0

    # Calibration buckets
    buckets: dict[str, list[int]] = {
        "50-54%": [], "54-58%": [], "58-62%": [], "62%+": [],
    }
    ev_bucket: list[tuple[float, int]] = []
    for r in resolved:
        try:
            p = float(r["pick_prob"])
            c = int(r["pick_correct"])
        except (TypeError, ValueError, KeyError):
            continue
        if p < 0.54:
            buckets["50-54%"].append(c)
        elif p < 0.58:
            buckets["54-58%"].append(c)
        elif p < 0.62:
            buckets["58-62%"].append(c)
        else:
            buckets["62%+"].append(c)
        ev_bucket.append((float(r.get("ev", 0) or 0), c))

    bucket_stats = {}
    for k, vals in buckets.items():
        if vals:
            wr = sum(vals) / len(vals)
            bucket_stats[k] = {"n": len(vals), "win_rate": round(wr, 4)}
        else:
            bucket_stats[k] = {"n": 0, "win_rate": None}

    # Realized ROI at -110 on positive-EV picks only
    pos_ev = [(ev, c) for ev, c in ev_bucket if ev > 0]
    roi = None
    if pos_ev:
        # $1 bets at -110: win pays $0.9091, loss costs $1.
        pnl = sum((100/110) if c else -1.0 for _, c in pos_ev)
        roi = round(pnl / len(pos_ev), 4)

    return {
        "n_resolved": n,
        "overall_win_rate": round(correct / n, 4),
        "mean_predicted_prob": round(mean_pred, 4),
        "buckets": bucket_stats,
        "pos_ev_count": len(pos_ev),
        "pos_ev_roi_at_110": roi,
    }
