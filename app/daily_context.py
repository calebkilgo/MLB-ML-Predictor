"""Fetch today-specific context for each game.

Sources:
  - MLB Stats API: lineups, bullpen, umpire, starter splits
  - Open-Meteo: wind speed + direction
  - Baseball Savant cache: xERA/xwOBA for each starter
  - The Odds API: moneylines (optional, requires ODDS_API_KEY)

In-memory caching with TTL. The app/main.py board builder calls
build_context() for every game in parallel; total cold time is bounded
by the slowest game's internal fetch chain.
"""
from __future__ import annotations

import os
import time
from datetime import date, timedelta
from typing import Any

import httpx
import pandas as pd

from app.mlb_live import TEAM_ID_TO_ABBR
from src.config import CFG
from src.etl.weather import STADIUMS

_CACHE: dict[str, tuple[float, Any]] = {}
_TTL = 300.0

LEAGUE_AVG_WOBA = 0.315
ABBR_TO_TEAM_ID = {v: k for k, v in TEAM_ID_TO_ABBR.items()}

_XSTATS: pd.DataFrame | None = None
_ROLLING_STATS: pd.DataFrame | None = None


def _cached_get(url: str, headers: dict | None = None) -> dict:
    now = time.time()
    hit = _CACHE.get(url)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    try:
        r = httpx.get(url, timeout=10.0, headers=headers or {})
        r.raise_for_status()
        data = r.json()
    except Exception:
        data = {}
    _CACHE[url] = (now, data)
    return data


def _xstats() -> pd.DataFrame:
    global _XSTATS
    if _XSTATS is not None:
        return _XSTATS
    year = date.today().year
    for y in (year, year - 1):
        path = CFG.raw_dir / f"statcast_pitchers_{y}.parquet"
        if path.exists():
            _XSTATS = pd.read_parquet(path).set_index("mlb_id")
            return _XSTATS
    _XSTATS = pd.DataFrame(columns=["xwoba", "xera"]).set_index(
        pd.Index([], name="mlb_id"))
    return _XSTATS


def _rolling_stats() -> pd.DataFrame:
    global _ROLLING_STATS
    if _ROLLING_STATS is not None:
        return _ROLLING_STATS
    year = date.today().year
    for y in (year, year - 1):
        path = CFG.raw_dir / f"statcast_rolling_{y}.parquet"
        if path.exists():
            _ROLLING_STATS = pd.read_parquet(path).set_index("mlb_id")
            return _ROLLING_STATS
    _ROLLING_STATS = pd.DataFrame(
        columns=["velo_avg", "whiff_pct", "hard_hit_pct", "k_pct", "bb_pct"]
    ).set_index(pd.Index([], name="mlb_id"))
    return _ROLLING_STATS


def pitcher_xstats(mlb_id: int | None) -> tuple[float | None, float | None]:
    if mlb_id is None:
        return None, None
    df = _xstats()
    if mlb_id in df.index:
        row = df.loc[mlb_id]
        xera = float(row.get("xera", None) or 0) or None
        xwoba = float(row.get("xwoba", None) or 0) or None
        return xera, xwoba
    return None, None


def pitcher_rolling_statcast(mlb_id: int | None) -> dict[str, float | None]:
    """Return season-to-date Statcast metrics for a pitcher (velo, whiff, etc.)."""
    empty: dict[str, float | None] = {
        "velo_avg": None, "whiff_pct": None,
        "hard_hit_pct": None, "k_pct": None, "bb_pct": None,
    }
    if mlb_id is None:
        return empty
    df = _rolling_stats()
    if mlb_id not in df.index:
        return empty
    row = df.loc[mlb_id]
    out: dict[str, float | None] = {}
    for k in empty:
        v = row.get(k)
        try:
            out[k] = float(v) if v is not None and not pd.isna(v) else None
        except (TypeError, ValueError):
            out[k] = None
    return out


def _player_woba_vs_hand(person_id: int, season: int, hand: str
                         ) -> tuple[float | None, float | None]:
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{person_id}/stats"
        f"?stats=statSplits&sitCodes=v{hand.lower()}hp&group=hitting&season={season}"
    )
    data = _cached_get(url)
    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return None, None
    stat = splits[0].get("stat", {})
    woba = None
    if stat.get("woba") not in (None, ""):
        try:
            woba = float(stat["woba"])
        except (TypeError, ValueError):
            pass
    if woba is None and stat.get("ops") not in (None, ""):
        try:
            woba = float(stat["ops"]) * 0.42
        except (TypeError, ValueError):
            pass
    pa = None
    try:
        pa = float(stat.get("plateAppearances", 0) or 0)
    except (TypeError, ValueError):
        pass
    return woba, pa


def _starter_hand(person_id: int | None) -> str | None:
    if not person_id:
        return None
    data = _cached_get(f"https://statsapi.mlb.com/api/v1/people/{person_id}")
    people = data.get("people", [])
    if not people:
        return None
    code = (people[0].get("pitchHand") or {}).get("code")
    return code if code in ("L", "R") else None


def _game_lineups(game_pk: int) -> tuple[list[int], list[int]]:
    data = _cached_get(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live")
    box = data.get("liveData", {}).get("boxscore", {}).get("teams", {})
    away = box.get("away", {}).get("battingOrder", []) or []
    home = box.get("home", {}).get("battingOrder", []) or []
    return away, home


# Batting-order position weights: cleanup slots (3-5) matter most;
# leadoff (1) matters less; bottom of order (7-9) least.
# Weights sum to len(ids) so the weighted average stays in the same
# scale as the unweighted one (no rescaling needed downstream).
_ORDER_WEIGHTS = [0.80, 0.95, 1.20, 1.30, 1.25, 1.10, 0.90, 0.75, 0.70]


def lineup_woba(game_pk: int, home_sp_id: int | None, away_sp_id: int | None,
                season: int) -> tuple[tuple[float | None, float | None],
                                      tuple[float | None, float | None]]:
    away_ids, home_ids = _game_lineups(game_pk)
    if not away_ids and not home_ids:
        return (None, None), (None, None)
    away_sp_hand = _starter_hand(away_sp_id)
    home_sp_hand = _starter_hand(home_sp_id)

    def _agg(ids: list[int], hand: str | None) -> tuple[float | None, float | None]:
        if not ids or not hand:
            return None, None
        total_woba_x_w = 0.0
        total_w = 0.0
        n_with_data = 0
        for i, pid in enumerate(ids[:9]):
            order_w = _ORDER_WEIGHTS[i] if i < len(_ORDER_WEIGHTS) else 0.70
            w, pa = _player_woba_vs_hand(pid, season, hand)
            if w is None or pa is None or pa <= 0:
                continue
            # Weight by both PA-based Bayesian reliability AND batting-order position.
            combined_w = pa * order_w
            total_woba_x_w += w * combined_w
            total_w += combined_w
            n_with_data += 1
        if total_w <= 0:
            return None, None
        return total_woba_x_w / total_w, total_w / max(n_with_data, 1)

    return _agg(home_ids, away_sp_hand), _agg(away_ids, home_sp_hand)


def _ip_to_float(ip) -> float:
    s = str(ip or "0")
    if "." in s:
        whole, part = s.split(".")
        return float(whole) + int(part) / 3.0
    return float(s) if s else 0.0


def _bullpen_stress_penalty(ip_3day: float) -> float:
    """FIP degradation from recent workload.

    Pitchers who threw >3 IP over the prior 3 days are considered stressed.
    Each additional inning above 3 adds 0.15 to expected FIP.
    Cap at +0.75 (5 extra IP ≈ fully taxed bullpen arm).
    """
    excess = max(0.0, ip_3day - 3.0)
    return min(0.75, excess * 0.15)


def bullpen_fips(team_id: int, game_date: date, season: int
                 ) -> tuple[float | None, float | None]:
    """Return (full_bullpen_fip, stress_adjusted_available_fip).

    Improvement over prior version: instead of a binary available/unavailable
    flag, each arm now gets a continuous FIP penalty based on how many IP
    they threw in the last 3 days.  A pitcher who threw 1 IP yesterday is
    slightly degraded; one who threw 3+ IP is fully unavailable.
    """
    roster = _cached_get(
        f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
    ).get("roster", [])
    pitcher_ids = [p["person"]["id"] for p in roster
                   if p.get("position", {}).get("code") == "1"]

    fips: list[tuple[int, float]] = []
    for pid in pitcher_ids:
        data = _cached_get(
            f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
            f"?stats=season&group=pitching&season={season}"
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            continue
        s = splits[0].get("stat", {})
        ip = _ip_to_float(s.get("inningsPitched"))
        gs = int(s.get("gamesStarted", 0) or 0)
        gp = int(s.get("gamesPlayed", 0) or 0)
        if gp == 0 or gs / gp > 0.4 or ip < 5:
            continue
        k = int(s.get("strikeOuts", 0) or 0)
        bb = int(s.get("baseOnBalls", 0) or 0)
        hbp = int(s.get("hitBatsmen", 0) or 0)
        hr = int(s.get("homeRuns", 0) or 0)
        fip = ((13 * hr + 3 * (bb + hbp) - 2 * k) / ip) + 3.10
        fips.append((pid, fip))

    if not fips:
        return None, None
    full_fip = sum(f for _, f in fips) / len(fips)

    # 3-day workload window: day-before weighted 1.0, 2-ago 0.6, 3-ago 0.3
    day_weights = {
        (game_date - timedelta(days=1)).isoformat(): 1.0,
        (game_date - timedelta(days=2)).isoformat(): 0.6,
        (game_date - timedelta(days=3)).isoformat(): 0.3,
    }
    stressed_fips: list[float] = []
    for pid, base_fip in fips:
        data = _cached_get(
            f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
            f"?stats=gameLog&group=pitching&season={season}"
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        ip_weighted = 0.0
        for sp in splits:
            w = day_weights.get(sp.get("date", ""), 0.0)
            if w > 0:
                ip_weighted += _ip_to_float(
                    (sp.get("stat") or {}).get("inningsPitched", 0)
                ) * w
        penalty = _bullpen_stress_penalty(ip_weighted)
        # Pitchers who threw 3+ weighted IP are effectively unavailable
        if ip_weighted < 3.0:
            stressed_fips.append(base_fip + penalty)

    if not stressed_fips:
        return full_fip, full_fip
    return full_fip, sum(stressed_fips) / len(stressed_fips)


def home_plate_umpire(game_pk: int) -> str | None:
    data = _cached_get(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live")
    officials = data.get("liveData", {}).get("boxscore", {}).get("officials", [])
    for o in officials:
        if o.get("officialType") == "Home Plate":
            return (o.get("official") or {}).get("fullName")
    return None


def wind_for_game(home_team: str, game_date: date
                  ) -> tuple[float | None, float | None, float | None,
                             float | None, float | None]:
    """Return (wind_kph, wind_dir_deg, temp_c, humidity_pct, pressure_hpa).

    Uses Open-Meteo forecast API for future games and archive API for past
    games. All five values are None if the stadium is not in STADIUMS.

    Why these fields matter:
    - temp_c: Ball carries ~1% further per 10°C above 15°C; hot day at Coors
      can add 0.3–0.5 runs/game.
    - humidity_pct: Low humidity (dry air, <40%) adds carry; high humidity
      (>80%) reduces it. Effect is smaller than temp but measurable.
    - pressure_hpa: Sea-level pressure ~1013 hPa; Denver ~840 hPa at Coors.
      Lower pressure = less air resistance = more carry on fly balls.
    """
    if home_team not in STADIUMS:
        return None, None, None, None, None
    lat, lon = STADIUMS[home_team]
    today = date.today()

    if game_date <= today:
        # Use archive endpoint for past / today's games
        base_url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        # Use forecast endpoint for future games
        base_url = "https://api.open-meteo.com/v1/forecast"

    url = (
        f"{base_url}"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={game_date.isoformat()}&end_date={game_date.isoformat()}"
        "&daily=wind_speed_10m_max,wind_direction_10m_dominant,"
        "temperature_2m_max,relative_humidity_2m_mean,pressure_msl_mean"
        "&timezone=America%2FNew_York"
    )
    d = _cached_get(url).get("daily", {})

    def _first(key: str) -> float | None:
        try:
            return float((d.get(key) or [None])[0])
        except (TypeError, ValueError):
            return None

    return (
        _first("wind_speed_10m_max"),
        _first("wind_direction_10m_dominant"),
        _first("temperature_2m_max"),
        _first("relative_humidity_2m_mean"),
        _first("pressure_msl_mean"),
    )


_ODDS_SNAPSHOT: dict | None = None
_ODDS_FETCHED_AT: float = 0.0


def _fetch_odds_snapshot() -> dict:
    global _ODDS_SNAPSHOT, _ODDS_FETCHED_AT
    now = time.time()
    if _ODDS_SNAPSHOT is not None and now - _ODDS_FETCHED_AT < 300:
        return _ODDS_SNAPSHOT
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        _ODDS_SNAPSHOT = {}
        _ODDS_FETCHED_AT = now
        return {}
    url = (
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        f"?regions=us&markets=h2h&oddsFormat=american&apiKey={key}"
    )
    try:
        r = httpx.get(url, timeout=15.0)
        r.raise_for_status()
        games = r.json() or []
    except Exception:
        _ODDS_SNAPSHOT = {}
        _ODDS_FETCHED_AT = now
        return {}

    out: dict[frozenset, dict] = {}
    for g in games:
        home = g.get("home_team", "")
        away = g.get("away_team", "")
        if not home or not away:
            continue
        h_sum, a_sum, n = 0, 0, 0
        for bk in g.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                for oc in mkt.get("outcomes", []):
                    if oc.get("name") == home and oc.get("price") is not None:
                        h_sum += int(oc["price"])
                    elif oc.get("name") == away and oc.get("price") is not None:
                        a_sum += int(oc["price"])
            n += 1
        if n == 0:
            continue
        out[frozenset((home, away))] = {
            "home_ml": int(round(h_sum / n)),
            "away_ml": int(round(a_sum / n)),
        }
    _ODDS_SNAPSHOT = out
    _ODDS_FETCHED_AT = now
    return out


def odds_for_game(home_full: str, away_full: str) -> tuple[int | None, int | None]:
    snap = _fetch_odds_snapshot()
    rec = snap.get(frozenset((home_full, away_full)))
    if not rec:
        return None, None
    return rec["home_ml"], rec["away_ml"]


def build_context(game: dict) -> dict[str, Any]:
    game_pk = game.get("game_pk")
    season = int((game.get("game_date") or "")[:4] or date.today().year)
    gd = date.fromisoformat(game["game_date"]) if game.get("game_date") else date.today()

    (home_w, home_pa), (away_w, away_pa) = lineup_woba(
        game_pk,
        home_sp_id=game.get("home_pitcher_id"),
        away_sp_id=game.get("away_pitcher_id"),
        season=season,
    )

    home_team_id = ABBR_TO_TEAM_ID.get(game["home_team"])
    away_team_id = ABBR_TO_TEAM_ID.get(game["away_team"])
    h_full, h_avail = bullpen_fips(home_team_id, gd, season) if home_team_id else (None, None)
    a_full, a_avail = bullpen_fips(away_team_id, gd, season) if away_team_id else (None, None)

    ump_name = home_plate_umpire(game_pk)
    wind_kph, wind_dir, temp_c, humidity_pct, pressure_hpa = wind_for_game(
        game["home_team"], gd
    )

    home_xera, _ = pitcher_xstats(game.get("home_pitcher_id"))
    away_xera, _ = pitcher_xstats(game.get("away_pitcher_id"))

    # Statcast rolling metrics (velo, whiff, hard-hit, K%, BB%)
    home_sc = pitcher_rolling_statcast(game.get("home_pitcher_id"))
    away_sc = pitcher_rolling_statcast(game.get("away_pitcher_id"))

    # ERA comes from the live schedule fetch, not GameInput anymore.
    home_sp_stats = game.get("home_pitcher_stats") or {}
    away_sp_stats = game.get("away_pitcher_stats") or {}

    home_ml, away_ml = odds_for_game(
        game.get("home_full", ""), game.get("away_full", "")
    )

    return {
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "league_avg_woba": LEAGUE_AVG_WOBA,
        "home_lineup_woba_vs_hand": home_w,
        "home_lineup_pa_vs_hand": home_pa,
        "away_lineup_woba_vs_hand": away_w,
        "away_lineup_pa_vs_hand": away_pa,
        "home_bullpen_full_fip": h_full,
        "home_bullpen_available_fip": h_avail,
        "away_bullpen_full_fip": a_full,
        "away_bullpen_available_fip": a_avail,
        "umpire_name": ump_name,
        "wind_kph": wind_kph,
        "wind_dir_deg": wind_dir,
        "temp_c": temp_c,
        "humidity_pct": humidity_pct,
        "pressure_hpa": pressure_hpa,
        "home_sp_xera": home_xera,
        "away_sp_xera": away_xera,
        "home_sp_era": home_sp_stats.get("era"),
        "away_sp_era": away_sp_stats.get("era"),
        # Statcast rolling metrics (velo, whiff, hard-hit, K%, BB%)
        "home_sp_velo": home_sc.get("velo_avg"),
        "away_sp_velo": away_sc.get("velo_avg"),
        "home_sp_whiff": home_sc.get("whiff_pct"),
        "away_sp_whiff": away_sc.get("whiff_pct"),
        "home_sp_hard_hit": home_sc.get("hard_hit_pct"),
        "away_sp_hard_hit": away_sc.get("hard_hit_pct"),
        "home_sp_k_pct": home_sc.get("k_pct"),
        "away_sp_k_pct": away_sc.get("k_pct"),
        "market_home_american": home_ml,
        "market_away_american": away_ml,
    }