"""Fetch historical starter assignments and per-start pitching logs.

v7b: game_pk now includes gameNumber to match retrosheet.py and avoid
doubleheader collisions. Also fixes the empty-stats crash.

Outputs:
  data/raw/starters.parquet      — one row per game with starter IDs
  data/raw/starter_logs.parquet  — one row per (pitcher, date) with
                                    ip/er/k/bb/hr/bf

Cached. Delete either file to force a refresh.

Run:
    python -m src.etl.starter_logs
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd
from tqdm import tqdm

from src.config import CFG
from src.utils.io import write_parquet

_STARTERS_PATH = CFG.raw_dir / "starters.parquet"
_LOGS_PATH = CFG.raw_dir / "starter_logs.parquet"

_TEAM_ID: dict[int, str] = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC",
    145: "CHW", 113: "CIN", 114: "CLE", 115: "COL", 116: "DET",
    117: "HOU", 118: "KCR", 108: "LAA", 119: "LAD", 146: "MIA",
    158: "MIL", 142: "MIN", 121: "NYM", 147: "NYY", 133: "OAK",
    143: "PHI", 134: "PIT", 135: "SDP", 136: "SEA", 137: "SFG",
    138: "STL", 139: "TBR", 140: "TEX", 141: "TOR", 120: "WSN",
}


def _ip_to_float(ip) -> float:
    s = str(ip or "0")
    if "." in s:
        whole, part = s.split(".")
        try:
            return float(whole) + int(part) / 3.0
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(s) if s else 0.0
    except (TypeError, ValueError):
        return 0.0


def _make_client() -> httpx.Client:
    return httpx.Client(
        timeout=20.0,
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=32),
    )


def fetch_starters_for_season(year: int, client: httpx.Client) -> pd.DataFrame:
    """Return one row per completed regular-season game with starter IDs."""
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&startDate={year}-01-01&endDate={year}-12-31"
        "&gameType=R&hydrate=probablePitcher,decisions"
    )
    r = client.get(url)
    r.raise_for_status()
    payload = r.json()

    rows = []
    for date_block in payload.get("dates", []):
        for g in date_block.get("games", []):
            if g.get("status", {}).get("abstractGameState", "") != "Final":
                continue
            home = g["teams"]["home"]
            away = g["teams"]["away"]
            home_abbr = _TEAM_ID.get(home["team"].get("id"))
            away_abbr = _TEAM_ID.get(away["team"].get("id"))
            if not home_abbr or not away_abbr:
                continue
            hr, ar = home.get("score"), away.get("score")
            if hr is None or ar is None:
                continue

            gd = pd.to_datetime(g["gameDate"], utc=True).tz_convert(None).normalize()
            game_num = int(g.get("gameNumber", 1) or 1)
            game_pk = (
                gd.strftime("%Y%m%d") + "_" + away_abbr + "_" +
                home_abbr + "_" + str(game_num)
            )

            home_sp = (home.get("probablePitcher") or {}).get("id")
            away_sp = (away.get("probablePitcher") or {}).get("id")

            rows.append({
                "game_pk": game_pk,
                "game_date": gd,
                "season": year,
                "game_num": game_num,
                "home_team": home_abbr,
                "away_team": away_abbr,
                "home_sp_id": int(home_sp) if home_sp else None,
                "away_sp_id": int(away_sp) if away_sp else None,
            })
    df = pd.DataFrame(rows)
    # Defensive de-dup on game_pk.
    if not df.empty:
        df = df.drop_duplicates(subset=["game_pk"], keep="first")
    return df


def build_starters() -> pd.DataFrame:
    if _STARTERS_PATH.exists():
        print(f"[starters] using cached {_STARTERS_PATH}")
        return pd.read_parquet(_STARTERS_PATH)

    print(f"[starters] fetching seasons {CFG.seasons_start}-{CFG.seasons_end}")
    frames = []
    with _make_client() as client:
        for year in range(CFG.seasons_start, CFG.seasons_end + 1):
            print(f"  [starters] season {year}")
            frames.append(fetch_starters_for_season(year, client))
    df = pd.concat(frames, ignore_index=True)
    write_parquet(df, _STARTERS_PATH)
    print(f"[starters] wrote {len(df)} rows -> {_STARTERS_PATH}")
    return df


def fetch_game_log(pitcher_id: int, season: int, client: httpx.Client) -> list[dict]:
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
        f"?stats=gameLog&group=pitching&season={season}"
    )
    try:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    # Defensive: some pitchers return {"stats": []} or are missing entirely.
    stats_list = data.get("stats") or []
    if not stats_list:
        return []
    splits = stats_list[0].get("splits", []) if stats_list else []

    rows = []
    for sp in splits:
        stat = sp.get("stat", {}) or {}
        d = sp.get("date")
        if not d:
            continue
        ip = _ip_to_float(stat.get("inningsPitched"))
        if ip < 0.5:
            continue
        gs = int(stat.get("gamesStarted", 0) or 0)
        if gs == 0:
            continue
        rows.append({
            "pitcher_id": pitcher_id,
            "game_date": pd.to_datetime(d),
            "season": season,
            "ip": ip,
            "er": int(stat.get("earnedRuns", 0) or 0),
            "k": int(stat.get("strikeOuts", 0) or 0),
            "bb": int(stat.get("baseOnBalls", 0) or 0),
            "hbp": int(stat.get("hitBatsmen", 0) or 0),
            "hr": int(stat.get("homeRuns", 0) or 0),
            "bf": int(stat.get("battersFaced", 0) or 0),
        })
    return rows


def build_game_logs(starters: pd.DataFrame) -> pd.DataFrame:
    if _LOGS_PATH.exists():
        print(f"[logs] using cached {_LOGS_PATH}")
        return pd.read_parquet(_LOGS_PATH)

    pairs: set[tuple[int, int]] = set()
    for col in ("home_sp_id", "away_sp_id"):
        sub = starters.dropna(subset=[col])[[col, "season"]]
        for _, row in sub.iterrows():
            pairs.add((int(row[col]), int(row["season"])))
    pairs_list = sorted(pairs)
    print(f"[logs] fetching {len(pairs_list)} (pitcher, season) pairs")

    rows: list[dict] = []
    with _make_client() as client:
        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = [ex.submit(fetch_game_log, pid, season, client)
                       for pid, season in pairs_list]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="starter logs", unit="pair", ncols=90):
                try:
                    rows.extend(fut.result())
                except Exception as e:
                    print(f"[logs] skipping one: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("[logs] WARNING empty dataframe")
        return df
    df = df.drop_duplicates(subset=["pitcher_id", "game_date"])
    write_parquet(df, _LOGS_PATH)
    print(f"[logs] wrote {len(df)} rows -> {_LOGS_PATH}")
    return df


def main() -> None:
    starters = build_starters()
    build_game_logs(starters)
    print("[starter_logs] done.")


if __name__ == "__main__":
    main()