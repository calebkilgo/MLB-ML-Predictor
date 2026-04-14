"""Team and pitcher season stats via the MLB Stats API (no scraping, no auth)."""
from __future__ import annotations

import httpx
import pandas as pd

_TEAM_ID: dict[int, str] = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC",
    145: "CHW", 113: "CIN", 114: "CLE", 115: "COL", 116: "DET",
    117: "HOU", 118: "KCR", 108: "LAA", 119: "LAD", 146: "MIA",
    158: "MIL", 142: "MIN", 121: "NYM", 147: "NYY", 133: "OAK",
    143: "PHI", 134: "PIT", 135: "SDP", 136: "SEA", 137: "SFG",
    138: "STL", 139: "TBR", 140: "TEX", 141: "TOR", 120: "WSN",
}


def _ip_to_float(ip) -> float:
    """Convert MLB-style innings pitched string '200.1' -> 200.333."""
    if ip is None:
        return 0.0
    s = str(ip)
    if "." in s:
        whole, part = s.split(".")
        return float(whole) + (int(part) / 3.0)
    return float(s)


def pitcher_season_stats(year: int) -> pd.DataFrame:
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=pitching&season={year}&sportIds=1&limit=2000"
    )
    payload = httpx.get(url, timeout=60.0).raise_for_status().json() if False else \
        httpx.get(url, timeout=60.0).json()

    rows = []
    stats_blocks = payload.get("stats", [])
    if not stats_blocks:
        return pd.DataFrame()
    for split in stats_blocks[0].get("splits", []):
        team_abbr = _TEAM_ID.get(split.get("team", {}).get("id"))
        if not team_abbr:
            continue
        s = split.get("stat", {})
        ip = _ip_to_float(s.get("inningsPitched", 0))
        if ip < 1:
            continue
        k = int(s.get("strikeOuts", 0) or 0)
        bb = int(s.get("baseOnBalls", 0) or 0)
        hbp = int(s.get("hitBatsmen", 0) or 0)
        hr = int(s.get("homeRuns", 0) or 0)
        era = float(s.get("era", 0) or 0)
        whip = float(s.get("whip", 0) or 0)
        # FIP approximation with constant ~3.10
        fip = ((13 * hr + 3 * (bb + hbp) - 2 * k) / ip) + 3.10
        rows.append({
            "Name": split.get("player", {}).get("fullName", ""),
            "Team": team_abbr,
            "IP": ip,
            "ERA": era,
            "FIP": fip,
            "K/9": (k * 9) / ip,
            "BB/9": (bb * 9) / ip,
            "HR/9": (hr * 9) / ip,
            "WHIP": whip,
            "season": year,
        })
    return pd.DataFrame(rows)


def team_season_stats(year: int) -> pd.DataFrame:
    bat_url = (
        "https://statsapi.mlb.com/api/v1/teams/stats"
        f"?season={year}&group=hitting&sportIds=1&stats=season"
    )
    pit_url = (
        "https://statsapi.mlb.com/api/v1/teams/stats"
        f"?season={year}&group=pitching&sportIds=1&stats=season"
    )

    rows: dict[str, dict] = {}
    bat = httpx.get(bat_url, timeout=60.0).json()
    for split in bat.get("stats", [{}])[0].get("splits", []):
        abbr = _TEAM_ID.get(split.get("team", {}).get("id"))
        if not abbr:
            continue
        s = split.get("stat", {})
        rows[abbr] = {
            "Team": abbr,
            "OPS": float(s.get("ops", 0) or 0),
            "R": int(s.get("runs", 0) or 0),
        }

    pit = httpx.get(pit_url, timeout=60.0).json()
    for split in pit.get("stats", [{}])[0].get("splits", []):
        abbr = _TEAM_ID.get(split.get("team", {}).get("id"))
        if not abbr or abbr not in rows:
            continue
        s = split.get("stat", {})
        ip = _ip_to_float(s.get("inningsPitched", 0))
        k = int(s.get("strikeOuts", 0) or 0)
        bb = int(s.get("baseOnBalls", 0) or 0)
        rows[abbr]["team_ERA"] = float(s.get("era", 0) or 0)
        rows[abbr]["team_K9"] = (k * 9) / ip if ip > 0 else 0.0
        rows[abbr]["team_BB9"] = (bb * 9) / ip if ip > 0 else 0.0

    df = pd.DataFrame(list(rows.values()))
    df["season"] = year
    return df