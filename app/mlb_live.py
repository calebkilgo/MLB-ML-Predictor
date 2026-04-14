"""MLB Stats API helpers: schedule, live scores, probable pitchers.

All functions hit https://statsapi.mlb.com — free, no auth, no rate limits
in practice. Results are lightly cached (30s) to avoid hammering the API
on page refresh.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

import httpx

from src.etl.retrosheet import TEAMS  # noqa: F401  (re-export for callers)

# MLB team id -> our 3-letter abbreviation. Mirrors src/etl/retrosheet.py.
TEAM_ID_TO_ABBR: dict[int, str] = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC",
    145: "CHW", 113: "CIN", 114: "CLE", 115: "COL", 116: "DET",
    117: "HOU", 118: "KCR", 108: "LAA", 119: "LAD", 146: "MIA",
    158: "MIL", 142: "MIN", 121: "NYM", 147: "NYY", 133: "OAK",
    143: "PHI", 134: "PIT", 135: "SDP", 136: "SEA", 137: "SFG",
    138: "STL", 139: "TBR", 140: "TEX", 141: "TOR", 120: "WSN",
}

_CACHE: dict[str, tuple[float, Any]] = {}
_TTL = 30.0  # seconds


def _cached_get(url: str) -> dict:
    now = time.time()
    hit = _CACHE.get(url)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    r = httpx.get(url, timeout=15.0)
    r.raise_for_status()
    data = r.json()
    _CACHE[url] = (now, data)
    return data


def _pitcher_stats(person_id: int, season: int) -> dict[str, float]:
    """Return season ERA/FIP-ish/K9/BB9 for a pitcher. Fallback to league avg."""
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{person_id}/stats"
        f"?stats=season&group=pitching&season={season}"
    )
    try:
        data = _cached_get(url)
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return _league_avg_pitcher()
        s = splits[0].get("stat", {})
        ip_str = str(s.get("inningsPitched", "0"))
        if "." in ip_str:
            whole, part = ip_str.split(".")
            ip = float(whole) + int(part) / 3.0
        else:
            ip = float(ip_str or 0)
        if ip < 1:
            return _league_avg_pitcher()
        k = int(s.get("strikeOuts", 0) or 0)
        bb = int(s.get("baseOnBalls", 0) or 0)
        hbp = int(s.get("hitBatsmen", 0) or 0)
        hr = int(s.get("homeRuns", 0) or 0)
        era = float(s.get("era", 4.0) or 4.0)
        fip = ((13 * hr + 3 * (bb + hbp) - 2 * k) / ip) + 3.10
        return {
            "era": era,
            "fip": fip,
            "k9": (k * 9) / ip,
            "bb9": (bb * 9) / ip,
        }
    except Exception:
        return _league_avg_pitcher()


def _league_avg_pitcher() -> dict[str, float]:
    return {"era": 4.20, "fip": 4.20, "k9": 8.8, "bb9": 3.1}


def get_schedule(start: date, end: date) -> list[dict]:
    """Return a list of games between two dates, enriched for the UI."""
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&startDate={start.isoformat()}&endDate={end.isoformat()}"
        "&hydrate=probablePitcher,linescore,team"
    )
    data = _cached_get(url)

    games: list[dict] = []
    season = start.year
    for date_block in data.get("dates", []):
        game_date_str = date_block.get("date", "")
        for g in date_block.get("games", []):
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            home_abbr = TEAM_ID_TO_ABBR.get(home["team"].get("id"))
            away_abbr = TEAM_ID_TO_ABBR.get(away["team"].get("id"))
            if not home_abbr or not away_abbr:
                continue

            status = g.get("status", {})
            abstract = status.get("abstractGameState", "")  # Preview/Live/Final
            detailed = status.get("detailedState", "")

            home_pp = home.get("probablePitcher") or {}
            away_pp = away.get("probablePitcher") or {}
            home_pp_id = home_pp.get("id")
            away_pp_id = away_pp.get("id")

            home_pp_stats = _pitcher_stats(home_pp_id, season) if home_pp_id else _league_avg_pitcher()
            away_pp_stats = _pitcher_stats(away_pp_id, season) if away_pp_id else _league_avg_pitcher()

            linescore = g.get("linescore") or {}
            inning = linescore.get("currentInning")
            inning_half = linescore.get("inningHalf")  # Top / Bottom
            home_score = home.get("score")
            away_score = away.get("score")

            # MLB-provided live win probability (pregame absent).
            live_wp = None
            wp = linescore.get("winProbability")
            if isinstance(wp, dict):
                live_wp = wp.get("home")

            games.append({
                "game_pk": g.get("gamePk"),
                "game_date": game_date_str,
                "start_time_utc": g.get("gameDate"),
                "state": abstract,  # Preview / Live / Final
                "detailed_state": detailed,
                "home_team": home_abbr,
                "away_team": away_abbr,
                "home_full": home["team"].get("name", home_abbr),
                "away_full": away["team"].get("name", away_abbr),
                "home_score": home_score,
                "away_score": away_score,
                "inning": inning,
                "inning_half": inning_half,
                "home_pitcher": home_pp.get("fullName"),
                "away_pitcher": away_pp.get("fullName"),
                "home_pitcher_id": home_pp_id,
                "away_pitcher_id": away_pp_id,
                "home_pitcher_stats": home_pp_stats,
                "away_pitcher_stats": away_pp_stats,
                "live_home_win_prob": live_wp,
            })
    return games


def get_default_window() -> tuple[date, date]:
    """Yesterday through tomorrow, in US Eastern-ish terms (local machine)."""
    today = date.today()
    return today - timedelta(days=1), today + timedelta(days=1)
