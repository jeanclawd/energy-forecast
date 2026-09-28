import numpy as np
import pandas as pd

from energy_forecast.features import FEATURES, MIN_LAG, build
from energy_forecast.markets.base import Market, Zone
from energy_forecast.model import Ensemble, metrics
from energy_forecast.weather import VARS


class Toy(Market):
    name = "toy"
    timezone = "Europe/Paris"
    holidays_country = "FR"
    zones = [Zone("A", "Alpha", 48.0, 2.0), Zone("B", "Beta", 45.0, 5.0)]


def toy_data(days=40, future_hours=40):
    ts = pd.date_range("2026-05-01", periods=days * 24, freq="h")
    fut = pd.date_range(ts[-1] + pd.Timedelta(hours=1), periods=future_hours, freq="h")
    rng = np.random.default_rng(0)
    act, w = [], []
    for z, base in (("A", 5000), ("B", 3000)):
        hour = ts.hour.to_numpy()
        sun = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
        act.append(pd.DataFrame({"ts": ts, "zone": z, "load": base + 800 * np.sin(hour / 24 * 2 * np.pi) + rng.normal(0, 50, len(ts)),
                                 "pv": 400 * sun}))
        allts = ts.append(fut)
        h2 = allts.hour.to_numpy()
        wd = pd.DataFrame({"ts": allts, "zone": z})
        for v in VARS:
            wd[v] = rng.normal(15, 5, len(allts))
        wd["shortwave_radiation"] = 800 * np.clip(np.sin((h2 - 6) / 12 * np.pi), 0, None)
        w.append(wd)
    return pd.concat(act), pd.concat(w)


def test_lags_never_peek_into_the_horizon():
    act, w = toy_data()
    df = build(act, w, Toy())
    a = df[df.zone == "A"].set_index("ts")
    t = a.index[500]
    assert a.loc[t, "load_lag48"] == a.loc[t - pd.Timedelta(hours=MIN_LAG), "load"]
    # future rows (no actuals yet) still get lag features for the first 48 hours
    future = a[a["load"].isna()]
    assert len(future) == 40
    assert future["load_lag48"].notna().all()


def test_ensemble_beats_naive_on_toy_data():
    act, w = toy_data()
    df = build(act, w, Toy())
    df = df[df["load_lag168"].notna()]
    train, test = df[df.ts < "2026-06-03"], df[(df.ts >= "2026-06-03") & df["load"].notna()]
    m = Ensemble("load")
    m.lgb.set_params(n_estimators=100)
    m.xgb.set_params(n_estimators=100)
    m.fit(train)
    p = m.predict_all(test)
    assert set(p) == {"lgbm", "xgb", "ensemble", "naive"}
    assert metrics(test["load"], p["ensemble"])["mae"] < 150
    assert all(c in df for c in FEATURES["load"] + FEATURES["pv"])


def test_pv_is_zero_at_night():
    act, w = toy_data()
    df = build(act, w, Toy())
    df = df[df["pv_lag168"].notna()]
    m = Ensemble("pv")
    m.lgb.set_params(n_estimators=50)
    m.xgb.set_params(n_estimators=50)
    m.fit(df[df["pv"].notna()])
    p = m.predict_all(df)
    night = df["shortwave_radiation"] <= 0
    assert (p["ensemble"][night.to_numpy()] == 0).all()
    assert (p["ensemble"] >= 0).all()
