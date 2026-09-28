"""Feature engineering shared by training and forecasting.

Only information available at forecast time is used: weather (observed for
the past, forecast for the future) and load/PV lagged by at least 48 hours,
which covers the whole horizon (run time → end of the next local day).
"""
import holidays as hol
import numpy as np
import pandas as pd

from .weather import VARS

MIN_LAG = 48
COMMON = ["zone_id", "hour", "dow", "doy_sin", "doy_cos"] + VARS
FEATURES = {
    "load": COMMON + ["month", "is_holiday", "is_weekend", "bridge",
                      "load_lag48", "load_lag168", "load_mean168", "temp_mean24", "temp_mean72"],
    "pv": COMMON + ["pv_lag48", "pv_lag168", "pv_cap", "clear_ratio"],
}


def build(actuals: pd.DataFrame, weather: pd.DataFrame, market) -> pd.DataFrame:
    zone_ids = {z.code: i for i, z in enumerate(market.zones)}
    frames = []
    for code in zone_ids:
        w = weather[weather["zone"] == code].set_index("ts").sort_index()
        a = actuals[actuals["zone"] == code].set_index("ts").sort_index()
        if w.empty:
            continue
        idx = pd.date_range(min(w.index.min(), a.index.min() if len(a) else w.index.min()), w.index.max(), freq="h")
        f = w.reindex(idx)[VARS].interpolate(limit=3)
        f["load"] = a["load"].reindex(idx)
        f["pv"] = a["pv"].reindex(idx)
        for t in ("load", "pv"):
            f[f"{t}_lag48"] = f[t].shift(MIN_LAG)
            f[f"{t}_lag168"] = f[t].shift(168)
        f["load_mean168"] = f["load"].shift(MIN_LAG).rolling(168, min_periods=100).mean()
        # installed-capacity proxy: best PV hour of the last two weeks
        f["pv_cap"] = f["pv"].shift(MIN_LAG).rolling(24 * 14, min_periods=24).max()
        f["temp_mean24"] = f["temperature_2m"].rolling(24, min_periods=1).mean()
        f["temp_mean72"] = f["temperature_2m"].rolling(72, min_periods=1).mean()
        f["zone"] = code
        f["zone_id"] = zone_ids[code]
        frames.append(f)
    df = pd.concat(frames)
    df.index.name = "ts"
    df = df.reset_index()

    local = df["ts"].dt.tz_localize("UTC").dt.tz_convert(market.timezone)
    df["hour"] = local.dt.hour
    df["dow"] = local.dt.dayofweek
    df["month"] = local.dt.month
    doy = local.dt.dayofyear
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    days = local.dt.date
    cal = hol.country_holidays(market.holidays_country, years=range(df["ts"].dt.year.min() - 1, df["ts"].dt.year.max() + 2))
    df["is_holiday"] = days.map(lambda d: d in cal).astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    # "pont": a working day squeezed between a holiday and the weekend
    prev_h = days.map(lambda d: (d - pd.Timedelta(days=1)) in cal)
    next_h = days.map(lambda d: (d + pd.Timedelta(days=1)) in cal)
    df["bridge"] = (((df["dow"] == 0) & next_h) | ((df["dow"] == 4) & prev_h)).astype(int)
    # how clear the sky is relative to the zone's best radiation for that hour/season
    ref = df.groupby(["zone", "month", "hour"])["shortwave_radiation"].transform(lambda s: s.quantile(0.95))
    df["clear_ratio"] = (df["shortwave_radiation"] / ref.replace(0, np.nan)).fillna(0).clip(0, 1.5)
    return df
