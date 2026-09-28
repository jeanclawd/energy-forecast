"""Hourly weather per zone from Open-Meteo (free, no key).

History comes from the ERA5 archive, which lags a few days; the most recent
days and the forecast horizon come from the forecast API (`past_days`).
"""
import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests

log = logging.getLogger("energy")

VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
        "wind_speed_10m", "shortwave_radiation", "cloud_cover"]
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"


def _get(url: str, zones, **params) -> list[dict]:
    """One request for all zones (Open-Meteo accepts coordinate lists), with retries:
    shared CI runners regularly hit timeouts or 429s."""
    q = {"hourly": ",".join(VARS), "timezone": "UTC",
         "latitude": ",".join(str(z.lat) for z in zones), "longitude": ",".join(str(z.lon) for z in zones), **params}
    for attempt in range(6):
        try:
            r = requests.get(url, params=q, timeout=90)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            js = r.json()
            return js if isinstance(js, list) else [js]
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            if attempt == 5 or (isinstance(e, requests.HTTPError) and e.response is not None
                                and e.response.status_code < 429):
                raise
            wait = 10 * 2 ** attempt
            log.warning("open-meteo %s (attempt %d), retrying in %ds", e, attempt + 1, wait)
            time.sleep(wait)


def _frames(results: list[dict], zones) -> pd.DataFrame:
    frames = []
    for js, z in zip(results, zones):
        df = pd.DataFrame(js["hourly"])
        df["ts"] = pd.to_datetime(df.pop("time"))
        df["zone"] = z.code
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fetch_history(zones, start: date, end: date) -> pd.DataFrame:
    df = _frames(_get(ARCHIVE, zones, start_date=start.isoformat(), end_date=end.isoformat()), zones)
    return df.dropna(subset=["temperature_2m"])


def fetch_recent_and_forecast(zones, past_days: int = 14, forecast_days: int = 3) -> pd.DataFrame:
    return _frames(_get(FORECAST, zones, past_days=past_days, forecast_days=forecast_days), zones)


def fetch_all(zones, start: date, forecast_days: int = 3) -> pd.DataFrame:
    """Archive from `start`, overlaid with the forecast API for the last 14 days + horizon."""
    today = date.today()
    recent = fetch_recent_and_forecast(zones, past_days=14, forecast_days=forecast_days)
    parts = [recent]
    archive_end = today - timedelta(days=13)
    if start < archive_end:
        parts.insert(0, fetch_history(zones, start, archive_end))
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(["ts", "zone"], keep="last").sort_values(["zone", "ts"]).reset_index(drop=True)
