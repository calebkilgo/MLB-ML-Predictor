"""Load MLB game logs via the official MLB Stats API (no scraping, no auth).

v7b: game_pk now includes the MLB Stats API `gameNumber` field to
disambiguate doubleheaders. Without this, game 1 and game 2 of a
doubleheader collide and downstream merges explode row counts.
"""
from __future__ import annotations

import httpx
import pandas as pd

TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]

_TEAM_ID: dict[int, str] = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC",
    145: "CHW", 113: "CIN", 114: "CLE", 115: "COL", 116: "DET",
    117: "HOU", 118: "KCR", 108: "LAA", 119: "LAD", 146: "MIA",
    158: "MIL", 142: "MIN", 121: "NYM", 147: "NYY", 133: "OAK",
    143: "PHI", 134: "PIT", 135: "SDP", 136: "SEA", 137: "SFG",
    138: "STL", 139: "TBR", 140: "TEX", 141: "TOR", 120: "WSN",
}


def load_season(year: int) -> pd.DataFrame:
    """Return one row per team-game for a given season via MLB Stats API."""
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&startDate={year}-01-01&endDate={year}-12-31"
        "&gameType=R"
    )
    r = httpx.get(url, timeout=60.0)
    r.raise_for_status()
    payload = r.json()

    rows = []
    skipped_no_team = 0
    skipped_not_final = 0
    skipped_no_score = 0
    for date_block in payload.get("dates", []):
        for g in date_block.get("games", []):
            if g.get("status", {}).get("abstractGameState", "") != "Final":
                skipped_not_final += 1
                continue
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            home_id = home.get("team", {}).get("id")
            away_id = away.get("team", {}).get("id")
            home_abbr = _TEAM_ID.get(home_id)
            away_abbr = _TEAM_ID.get(away_id)
            if not home_abbr or not away_abbr:
                skipped_no_team += 1
                continue
            hr, ar = home.get("score"), away.get("score")
            if hr is None or ar is None:
                skipped_no_score += 1
                continue
            game_date = pd.to_datetime(g["gameDate"], utc=True).tz_convert(None).normalize()
            # gameNumber disambiguates doubleheaders (1, 2, sometimes 3 for
            # continuation of suspended games).
            game_num = int(g.get("gameNumber", 1) or 1)
            common = {
                "season": year, "game_date": game_date,
                "game_num": game_num,
            }
            rows.append({
                **common,
                "team": home_abbr, "opp": away_abbr, "is_home": 1,
                "runs": int(hr), "runs_allowed": int(ar),
                "W/L": "W" if hr > ar else "L",
            })
            rows.append({
                **common,
                "team": away_abbr, "opp": home_abbr, "is_home": 0,
                "runs": int(ar), "runs_allowed": int(hr),
                "W/L": "W" if ar > hr else "L",
            })

    print(f"  [{year}] kept {len(rows)//2} games "
          f"(skipped: not_final={skipped_not_final}, "
          f"no_team={skipped_no_team}, no_score={skipped_no_score})")
    if not rows:
        raise RuntimeError(f"No games returned from MLB Stats API for {year}")
    return pd.DataFrame(rows)


def to_game_level(team_games: pd.DataFrame) -> pd.DataFrame:
    """Collapse two team-game rows into one home-perspective game row."""
    home = team_games[team_games["is_home"] == 1].copy()
    home = home.rename(columns={
        "team": "home_team", "opp": "away_team",
        "runs": "home_runs", "runs_allowed": "away_runs",
    })
    home["home_win"] = (home["home_runs"] > home["away_runs"]).astype(int)
    home["total_runs"] = home["home_runs"] + home["away_runs"]
    # game_pk now includes the game number suffix for doubleheaders.
    home["game_pk"] = (
        home["game_date"].dt.strftime("%Y%m%d") + "_" +
        home["away_team"] + "_" + home["home_team"] + "_" +
        home["game_num"].astype(str)
    )
    # De-dup just in case the API ever returns the same (date, teams, num)
    # tuple twice due to a feed hiccup.
    home = home.drop_duplicates(subset=["game_pk"], keep="first")
    return home[["game_pk", "season", "game_date", "home_team", "away_team",
                 "home_runs", "away_runs", "home_win", "total_runs"]]