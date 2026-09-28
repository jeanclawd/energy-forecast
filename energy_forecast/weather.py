"""Hourly weather per zone from Open-Meteo (free, no key).

History comes from the ERA5 archive, which lags a few days; the most recent
days and the forecast horizon come from the forecast API (`past_days`).
"""
from datetime import date, timedelta

import pandas as pd
import requests

VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
        "wind_speed_10m", "shortwave_radiation", "cloud_cover"]
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"


def _get(url: str, **params) -> dict:
    r = requests.get(url, params={"hourly": ",".join(VARS), "timezone": "UTC", **params}, timeout=120)
    r.raise_for_status()
    return r.json()


def _frame(js: dict, zone: str) -> pd.DataFrame:
    df = pd.DataFrame(js["hourly"])
    df["ts"] = pd.to_datetime(df.pop("time"))
    df["zone"] = zone
    return df


def fetch_history(zones, start: date, end: date) -> pd.DataFrame:
    frames = []
    for z in zones:
        js = _get(ARCHIVE, latitude=z.lat, longitude=z.lon, start_date=start.isoformat(), end_date=end.isoformat())
        frames.append(_frame(js, z.code))
    return pd.concat(frames, ignore_index=True).dropna(subset=["temperature_2m"])


def fetch_recent_and_forecast(zones, past_days: int = 14, forecast_days: int = 3) -> pd.DataFrame:
    frames = []
    for z in zones:
        js = _get(FORECAST, latitude=z.lat, longitude=z.lon, past_days=past_days, forecast_days=forecast_days)
        frames.append(_frame(js, z.code))
    return pd.concat(frames, ignore_index=True)


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
