"""Historical weather via Open-Meteo (free, no key). Concurrent + progress."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd

STADIUMS: dict[str, tuple[float, float]] = {
    "ARI": (33.4455, -112.0667), "ATL": (33.8908, -84.4678),
    "BAL": (39.2839, -76.6217),  "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),  "CHW": (41.8300, -87.6339),
    "CIN": (39.0974, -84.5071),  "CLE": (41.4962, -81.6852),
    "COL": (39.7559, -104.9942), "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),  "KCR": (39.0516, -94.4803),
    "LAA": (33.8003, -117.8827), "LAD": (34.0739, -118.2400),
    "MIA": (25.7780, -80.2197),  "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2776),  "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262),  "OAK": (37.7516, -122.2005),
    "PHI": (39.9061, -75.1665),  "PIT": (40.4469, -80.0057),
    "SDP": (32.7073, -117.1566), "SEA": (47.5914, -122.3325),
    "SFG": (37.7786, -122.3893), "STL": (38.6226, -90.1928),
    "TBR": (27.7683, -82.6534),  "TEX": (32.7473, -97.0817),
    "TOR": (43.6414, -79.3894),  "WSN": (38.8730, -77.0074),
}

_NAN = {"temp_c": float("nan"), "wind_kph": float("nan"), "precip_mm": float("nan")}


def _fetch_one(client: httpx.Client, team: str, date: str) -> dict:
    if team not in STADIUMS:
        return {"home_team": team, "game_date": date, **_NAN}
    lat, lon = STADIUMS[team]
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
        "&daily=temperature_2m_mean,wind_speed_10m_max,precipitation_sum"
        "&timezone=America%2FNew_York"
    )
    try:
        r = client.get(url, timeout=15.0)
        r.raise_for_status()
        d = r.json()["daily"]
        return {
            "home_team": team, "game_date": date,
            "temp_c": float(d["temperature_2m_mean"][0]),
            "wind_kph": float(d["wind_speed_10m_max"][0]),
            "precip_mm": float(d["precipitation_sum"][0]),
        }
    except Exception:
        return {"home_team": team, "game_date": date, **_NAN}


def fetch_weather(team: str, date: str) -> dict[str, float]:
    """Single-call helper (kept for compatibility)."""
    with httpx.Client() as client:
        out = _fetch_one(client, team, date)
    return {k: out[k] for k in ("temp_c", "wind_kph", "precip_mm")}


def bulk_weather(games: pd.DataFrame, max_workers: int = 16) -> pd.DataFrame:
    """Concurrent fetch with a tqdm progress bar."""
    from tqdm import tqdm

    uniq = games[["home_team", "game_date"]].drop_duplicates().reset_index(drop=True)
    total = len(uniq)

    results: list[dict] = []
    with httpx.Client(http2=False) as client:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(_fetch_one, client, row["home_team"],
                          row["game_date"].strftime("%Y-%m-%d"))
                for _, row in uniq.iterrows()
            ]
            for fut in tqdm(as_completed(futures), total=total,
                            desc="weather", unit="req", ncols=90):
                results.append(fut.result())

    df = pd.DataFrame(results)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df