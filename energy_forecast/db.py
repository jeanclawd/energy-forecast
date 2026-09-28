"""SQLite storage. One file, committed to the repo's `data` branch by the pipeline.

Tables
  actuals(ts, zone, load, pv)                       observed hourly MW
  weather(ts, zone, <vars>)                         observed + forecast weather
  forecasts(run_ts, ts, zone, target, model, value) every forecast ever issued
  backtests(...)                                    same shape: holdout hindcasts from the last training
"""
import sqlite3
from pathlib import Path

import pandas as pd

from .weather import VARS

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS actuals (ts TEXT, zone TEXT, load REAL, pv REAL, PRIMARY KEY (ts, zone));
CREATE TABLE IF NOT EXISTS weather (ts TEXT, zone TEXT, {", ".join(v + " REAL" for v in VARS)}, PRIMARY KEY (ts, zone));
CREATE TABLE IF NOT EXISTS forecasts (run_ts TEXT, ts TEXT, zone TEXT, target TEXT, model TEXT, value REAL,
  PRIMARY KEY (run_ts, ts, zone, target, model));
CREATE TABLE IF NOT EXISTS backtests (run_ts TEXT, ts TEXT, zone TEXT, target TEXT, model TEXT, value REAL,
  PRIMARY KEY (run_ts, ts, zone, target, model));
CREATE INDEX IF NOT EXISTS forecasts_ts ON forecasts (ts, zone, target);
CREATE TABLE IF NOT EXISTS runs (run_ts TEXT PRIMARY KEY, kind TEXT, info TEXT);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def _ts(df: pd.DataFrame, cols=("ts", "run_ts")) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d %H:%M")
    return df


def upsert(con: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = _ts(df)
    cols = list(df.columns)
    q = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    con.executemany(q, df.astype(object).where(df.notna(), None).itertuples(index=False, name=None))
    con.commit()
    return len(df)


def read(con: sqlite3.Connection, sql: str, params=()) -> pd.DataFrame:
    df = pd.read_sql_query(sql, con, params=params)
    for c in ("ts", "run_ts"):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


def last_ts(con: sqlite3.Connection, table: str):
    v = con.execute(f"SELECT max(ts) FROM {table}").fetchone()[0]
    return pd.Timestamp(v) if v else None
