"""Public dashboard: reads the forecasts DB, never touches the models."""
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from energy_forecast.markets import get_market  # noqa: E402
from energy_forecast.weather import VARS  # noqa: E402

st.set_page_config(page_title="Energy forecast", page_icon="⚡", layout="wide")
MARKET = get_market(os.environ.get("ENERGY_MARKET", "france"))
ZONES = {z.code: z.name for z in MARKET.zones}
TOTAL = "__all__"
COLORS = {"actual": "#1f2937", "ensemble": "#e8590c", "lgbm": "#2f9e44", "xgb": "#1971c2", "naive": "#adb5bd"}
WEATHER_LABELS = {"temperature_2m": "Temperature (°C)", "apparent_temperature": "Feels-like (°C)",
                  "relative_humidity_2m": "Humidity (%)", "wind_speed_10m": "Wind speed (km/h)",
                  "shortwave_radiation": "Solar radiation (W/m²)", "cloud_cover": "Cloud cover (%)"}


def setting(name: str) -> str:
    """Environment variable, else Streamlit secret (Streamlit Community Cloud), else empty."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, "")
    except Exception:  # no secrets.toml
        return ""


def db_path() -> Path:
    """Local DB file, or download it (e.g. from the repo's data branch) when ENERGY_DB_URL is set."""
    local = Path(os.environ.get("ENERGY_DB", Path(__file__).resolve().parent.parent / "data" / "energy.db"))
    url = setting("ENERGY_DB_URL")
    if not url:
        return local
    dest = Path(tempfile.gettempdir()) / "energy.db"
    if not dest.exists() or time.time() - dest.stat().st_mtime > 900:
        token = setting("GITHUB_TOKEN")
        headers = {"Accept": "application/vnd.github.raw"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        r = requests.get(url, headers=headers, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


@st.cache_data(ttl=600)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True) as con:
        df = pd.read_sql_query(sql, con, params=params)
    for c in ("ts", "run_ts"):
        if c in df:
            df[c] = pd.to_datetime(df[c]).dt.tz_localize("UTC").dt.tz_convert(MARKET.timezone)
    return df


def zone_filter(df: pd.DataFrame, zone: str, value_cols) -> pd.DataFrame:
    if zone == TOTAL:
        # national total only where every zone has reported, so the last hour isn't a partial sum
        keys = [c for c in df.columns if c not in value_cols and c != "zone"]
        g = df.dropna(subset=value_cols).groupby(keys)
        out = g[value_cols].sum()
        return out[g.size() == len(ZONES)].reset_index()
    return df[df["zone"] == zone]


st.title("⚡ Energy forecast")
st.caption(f"Day-ahead electricity demand and solar generation for **{MARKET.name.title()}** · LightGBM + XGBoost ensemble · "
           "open data (RTE eco2mix, Open-Meteo) · refreshed daily by GitHub Actions")

last_run = q("SELECT max(run_ts) AS run_ts FROM forecasts")
if last_run["run_ts"].isna().all():
    st.warning("No forecasts yet — run `python -m energy_forecast daily`.")
    st.stop()
run_ts = last_run["run_ts"].iloc[0]
run_key = run_ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M")

with st.sidebar:
    st.header("Settings")
    zone = st.selectbox("Zone", [TOTAL] + list(ZONES), format_func=lambda z: "🇫🇷 Whole country" if z == TOTAL else ZONES[z])
    days_back = st.slider("History shown (days)", 2, 21, 7)
    models = st.multiselect("Models", ["ensemble", "lgbm", "xgb", "naive"], default=["ensemble"])
    st.caption(f"Latest run: **{run_ts:%a %d %b %H:%M}** ({MARKET.timezone})")

since = (run_ts - pd.Timedelta(days=days_back)).tz_convert("UTC").strftime("%Y-%m-%d %H:%M")


def target_tab(target: str, unit_label: str):
    act = q("SELECT ts, zone, load, pv FROM actuals WHERE ts >= ?", (since,))
    fc = q("SELECT ts, zone, model, value FROM forecasts WHERE run_ts = ? AND target = ?", (run_key, target))
    # the forecast that was issued the morning before each past day (what we'd have known)
    past = q("""SELECT f.ts, f.zone, f.model, f.value FROM forecasts f
                JOIN (SELECT ts, zone, max(run_ts) r FROM forecasts WHERE target = ? AND run_ts <= datetime(ts, '-12 hours')
                      AND ts >= ? GROUP BY ts, zone) m ON f.ts = m.ts AND f.zone = m.zone AND f.run_ts = m.r
                WHERE f.target = ?""", (target, since, target))
    a = zone_filter(act[["ts", "zone", target]], zone, [target])
    f = zone_filter(fc, zone, ["value"])
    p = zone_filter(past, zone, ["value"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a["ts"], y=a[target], name="Actual", line=dict(color=COLORS["actual"], width=2)))
    for m in models:
        pm = p[p["model"] == m]
        if len(pm):
            fig.add_trace(go.Scatter(x=pm["ts"], y=pm["value"], name=f"{m} (issued day before)",
                                     line=dict(color=COLORS[m], width=1, dash="dot"), opacity=0.8))
        fm = f[f["model"] == m]
        fig.add_trace(go.Scatter(x=fm["ts"], y=fm["value"], name=f"{m} forecast", line=dict(color=COLORS[m], width=2.5)))
    fig.add_vline(x=run_ts, line=dict(color="#868e96", dash="dash"))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=unit_label,
                      legend=dict(orientation="h", y=1.08), hovermode="x unified")
    fe = f[f["model"] == "ensemble"]
    tomorrow = (run_ts + pd.Timedelta(days=1)).date()
    ft = fe[fe["ts"].dt.date == tomorrow]
    c1, c2, c3 = st.columns(3)
    if len(ft):
        pk = ft.loc[ft["value"].idxmax()]
        c1.metric(f"Tomorrow's peak", f"{pk['value']:,.0f} MW", f"at {pk['ts']:%H:%M}", delta_color="off")
        c2.metric("Tomorrow's energy", f"{ft['value'].sum() / 1000:,.1f} GWh")
    if len(a.dropna()):
        la = a.dropna().iloc[-1]
        c3.metric("Latest actual", f"{la[target]:,.0f} MW", f"{la['ts']:%a %H:%M}", delta_color="off")
    st.plotly_chart(fig, width="stretch")


tab_load, tab_pv, tab_w, tab_an = st.tabs(["⚡ Load forecast", "☀️ PV forecast", "🌦️ Weather", "📊 Forecast analytics"])
with tab_load:
    target_tab("load", "Demand (MW)")
with tab_pv:
    target_tab("pv", "Solar generation (MW)")

with tab_w:
    wz = zone if zone != TOTAL else MARKET.zones[0].code
    if zone == TOTAL:
        st.caption(f"Weather is per zone — showing {ZONES[wz]}. Pick a zone in the sidebar.")
    w = q("SELECT * FROM weather WHERE zone = ? AND ts >= ?", (wz, since))
    cols = st.multiselect("Variables", VARS, default=["temperature_2m", "apparent_temperature", "shortwave_radiation"],
                          format_func=WEATHER_LABELS.get)
    for v in cols:
        fig = px.line(w, x="ts", y=v, labels={"ts": "", v: WEATHER_LABELS[v]})
        fig.add_vline(x=run_ts, line=dict(color="#868e96", dash="dash"))
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    st.caption("Left of the dashed line: observed (ERA5 / Open-Meteo). Right: Open-Meteo forecast.")

with tab_an:
    src = st.radio("Source", ["Live forecasts", "Backtest (last training holdout)"], horizontal=True,
                   help="Live = forecasts actually issued by the daily pipeline, scored against what happened. "
                        "Backtest = the 28-day holdout from the last training, run on observed weather (a bit optimistic).")
    table = "forecasts" if src.startswith("Live") else "backtests"
    target = st.radio("Target", ["load", "pv"], horizontal=True, format_func={"load": "Load", "pv": "PV"}.get)
    ev = q(f"""SELECT f.ts, f.zone, f.model, f.value, a.{target} AS actual FROM {table} f
               JOIN (SELECT ts, zone, max(run_ts) r FROM {table} WHERE target = ? AND run_ts <= datetime(ts, '-12 hours')
                     GROUP BY ts, zone) m ON f.ts = m.ts AND f.zone = m.zone AND f.run_ts = m.r
               JOIN actuals a ON a.ts = f.ts AND a.zone = f.zone
               WHERE f.target = ? AND a.{target} IS NOT NULL""", (target, target))
    if ev.empty:
        st.info("Nothing to score yet: live forecasts are scored once the day has happened. "
                "Switch to *Backtest* to see the holdout performance.")
    else:
        ev = zone_filter(ev, zone, ["value", "actual"])
        ev["abs_err"] = (ev["value"] - ev["actual"]).abs()
        wape = lambda g: g["abs_err"].sum() / g["actual"].abs().sum() * 100  # noqa: E731
        by_model = ev.groupby("model").apply(wape).rename("WAPE %").reset_index().sort_values("WAPE %")
        mae = ev.groupby("model")["abs_err"].mean().rename("MAE (MW)").reset_index()
        summary = by_model.merge(mae, on="model")
        cols = st.columns(len(summary))
        for c, (_, r) in zip(cols, summary.iterrows()):
            c.metric(r["model"], f"{r['WAPE %']:.2f} %", f"MAE {r['MAE (MW)']:,.0f} MW", delta_color="off")
        ev["day"] = ev["ts"].dt.date
        daily = ev.groupby(["day", "model"]).apply(wape).rename("WAPE %").reset_index()
        fig = px.line(daily, x="day", y="WAPE %", color="model", color_discrete_map=COLORS, markers=True,
                      title="Daily WAPE (weighted absolute percentage error)")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        ev["hour"] = ev["ts"].dt.hour
        hourly = ev.groupby(["hour", "model"])["abs_err"].mean().rename("MAE (MW)").reset_index()
        fig = px.line(hourly, x="hour", y="MAE (MW)", color="model", color_discrete_map=COLORS, title="Error by hour of day")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        c1.plotly_chart(fig, width="stretch")
        if zone == TOTAL:
            allz = q(f"""SELECT f.zone, f.value, a.{target} AS actual FROM {table} f
                        JOIN actuals a ON a.ts = f.ts AND a.zone = f.zone
                        WHERE f.target = ? AND f.model = 'ensemble' AND a.{target} IS NOT NULL""", (target,))
            allz["abs_err"] = (allz["value"] - allz["actual"]).abs()
            bz = allz.groupby("zone").apply(wape).rename("WAPE %").reset_index()
            bz["zone"] = bz["zone"].map(ZONES)
            fig = px.bar(bz.sort_values("WAPE %"), x="WAPE %", y="zone", orientation="h", title="Ensemble WAPE by zone")
        else:
            e = ev[ev["model"] == "ensemble"]
            fig = px.scatter(e, x="actual", y="value", opacity=0.4, title="Ensemble: forecast vs actual",
                             labels={"value": "forecast (MW)", "actual": "actual (MW)"})
            lim = [0, float(max(e["actual"].max(), e["value"].max()))]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="#868e96"), showlegend=False))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        c2.plotly_chart(fig, width="stretch")
