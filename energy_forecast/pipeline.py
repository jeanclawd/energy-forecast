"""Pipeline steps. Run with `python -m energy_forecast <step>`.

  update    fetch new actuals + weather into the DB (backfills on first run)
  train     (re)train the load & PV ensembles, report a 28-day holdout score
  forecast  issue forecasts from now to the end of the next local day
  daily     update → train if needed (no model yet, or Monday) → forecast
"""
import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from . import db
from .config import DB_PATH, HISTORY_DAYS, MARKET, MODEL_DIR, TARGETS
from .features import build
from .markets import get_market
from .model import Ensemble, metrics
from .weather import fetch_all

log = logging.getLogger("energy")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)


def update(con, market) -> None:
    now = _now()
    last = db.last_ts(con, "actuals")
    start = (last - timedelta(days=3)) if last is not None else now - timedelta(days=HISTORY_DAYS)
    act = market.fetch_actuals(start.to_pydatetime() if hasattr(start, "to_pydatetime") else start, now + timedelta(hours=1))
    n = db.upsert(con, "actuals", act)
    log.info("actuals: %d rows from %s", n, start)
    wlast = con.execute("SELECT max(ts) FROM actuals").fetchone()[0]
    have_weather = con.execute("SELECT count(*) FROM weather").fetchone()[0] > 0
    wstart = (date.today() - timedelta(days=16)) if have_weather else (now - timedelta(days=HISTORY_DAYS + 10)).date()
    w = fetch_all(market.zones, wstart)
    n = db.upsert(con, "weather", w)
    log.info("weather: %d rows from %s (actuals up to %s)", n, wstart, wlast)


def _frame(con, market) -> pd.DataFrame:
    act = db.read(con, "SELECT ts, zone, load, pv FROM actuals")
    w = db.read(con, "SELECT * FROM weather")
    return build(act, w, market)


def train(con, market) -> dict:
    df = _frame(con, market)
    df = df[df["load_lag168"].notna()]
    cutoff = df.dropna(subset=["load"])["ts"].max() - pd.Timedelta(days=28)
    report = {}
    for target in TARGETS:
        hold = Ensemble(target).fit(df[df["ts"] <= cutoff])
        test = df[(df["ts"] > cutoff) & df[target].notna()]
        preds = hold.predict_all(test)
        report[target] = {k: metrics(test[target], v) for k, v in preds.items()}
        # Keep the holdout predictions as a backtest so the analytics have history
        # from day one. Pretend each day was forecast the previous morning.
        run_ts = test["ts"].dt.normalize() - pd.Timedelta(hours=19)
        bt = pd.concat([pd.DataFrame({"run_ts": run_ts.values, "ts": test["ts"].values, "zone": test["zone"].values,
                                      "target": target, "model": k, "value": v}) for k, v in preds.items()])
        con.execute("DELETE FROM backtests WHERE target = ?", (target,))
        db.upsert(con, "backtests", bt.dropna(subset=["value"]))
        log.info("%s holdout WAPE: %s", target, {k: round(v["wape"] * 100, 2) for k, v in report[target].items()})
        full = Ensemble(target).fit(df)
        full.meta["holdout"] = report[target]
        full.save(MODEL_DIR)
    con.execute("INSERT OR REPLACE INTO runs VALUES (?, 'train', ?)", (_now().strftime("%Y-%m-%d %H:%M"), json.dumps(report)))
    con.commit()
    return report


def forecast(con, market) -> int:
    run = _now()
    df = _frame(con, market)
    # horizon: now → end of the next local day
    local_now = pd.Timestamp(run).tz_localize("UTC").tz_convert(market.timezone)
    end_local = (local_now + pd.Timedelta(days=2)).normalize()
    end = end_local.tz_convert("UTC").tz_localize(None)
    fut = df[(df["ts"] > run) & (df["ts"] < end)]
    rows = []
    for target in TARGETS:
        m = Ensemble.load(MODEL_DIR, target)
        for model, vals in m.predict_all(fut).items():
            rows.append(pd.DataFrame({"run_ts": run, "ts": fut["ts"].values, "zone": fut["zone"].values,
                                      "target": target, "model": model, "value": vals}))
    out = pd.concat(rows, ignore_index=True).dropna(subset=["value"])
    n = db.upsert(con, "forecasts", out)
    con.execute("INSERT OR REPLACE INTO runs VALUES (?, 'forecast', ?)",
                (run.strftime("%Y-%m-%d %H:%M"), json.dumps({"rows": n, "until": str(end)})))
    con.commit()
    log.info("forecast: %d rows, %s → %s", n, run, end)
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="energy_forecast")
    ap.add_argument("step", choices=["update", "train", "forecast", "daily"])
    ap.add_argument("--retrain", action="store_true", help="force retraining in `daily`")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    market = get_market(MARKET)
    con = db.connect(DB_PATH)
    if a.step in ("update", "daily"):
        update(con, market)
    if a.step == "train" or (a.step == "daily" and (a.retrain or date.today().weekday() == 0
                                                    or not (MODEL_DIR / "load_lgbm.txt").exists())):
        train(con, market)
    if a.step in ("forecast", "daily"):
        forecast(con, market)
