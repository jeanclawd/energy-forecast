"""France: RTE eco2mix regional data via the ODRÉ open-data portal (no API key).

Two datasets are stitched together: the consolidated/definitive history
(`eco2mix-regional-cons-def`, half-hourly) and the real-time feed
(`eco2mix-regional-tr`, 15-minute) that covers the months not yet consolidated.
Both are averaged to hourly MW.
"""
import io
from datetime import datetime, timedelta

import pandas as pd
import requests

from .base import Market, Zone

API = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{ds}/exports/csv"
DATASETS = ("eco2mix-regional-cons-def", "eco2mix-regional-tr")


class France(Market):
    name = "france"
    timezone = "Europe/Paris"
    holidays_country = "FR"
    # One representative city per region for the weather.
    zones = [
        Zone("11", "Île-de-France", 48.86, 2.35),
        Zone("24", "Centre-Val de Loire", 47.90, 1.90),
        Zone("27", "Bourgogne-Franche-Comté", 47.32, 5.04),
        Zone("28", "Normandie", 49.44, 1.10),
        Zone("32", "Hauts-de-France", 50.63, 3.06),
        Zone("44", "Grand Est", 48.69, 6.18),
        Zone("52", "Pays de la Loire", 47.22, -1.55),
        Zone("53", "Bretagne", 48.11, -1.68),
        Zone("75", "Nouvelle-Aquitaine", 44.84, -0.58),
        Zone("76", "Occitanie", 43.60, 1.44),
        Zone("84", "Auvergne-Rhône-Alpes", 45.76, 4.84),
        Zone("93", "Provence-Alpes-Côte d'Azur", 43.30, 5.37),
    ]

    def _fetch(self, ds: str, start: datetime, end: datetime) -> pd.DataFrame:
        where = (f"date_heure >= '{start:%Y-%m-%dT%H:%M:%S}Z' and date_heure < '{end:%Y-%m-%dT%H:%M:%S}Z'"
                 " and consommation is not null")
        r = requests.get(API.format(ds=ds), params={
            "select": "code_insee_region,date_heure,consommation,solaire",
            "where": where, "delimiter": ";", "timezone": "UTC",
        }, timeout=300)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep=";", dtype={"code_insee_region": str})
        return df

    def fetch_actuals(self, start: datetime, end: datetime) -> pd.DataFrame:
        frames = []
        # Chunk by ~6 months to keep each export reasonable.
        cur = start
        while cur < end:
            nxt = min(cur + timedelta(days=183), end)
            for ds in DATASETS:
                frames.append(self._fetch(ds, cur, nxt))
            cur = nxt
        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            return pd.DataFrame(columns=["ts", "zone", "load", "pv"])
        df["ts"] = pd.to_datetime(df["date_heure"], utc=True).dt.tz_localize(None).dt.floor("h")
        df = df.rename(columns={"code_insee_region": "zone", "consommation": "load", "solaire": "pv"})
        df["pv"] = pd.to_numeric(df["pv"], errors="coerce").clip(lower=0)
        df["load"] = pd.to_numeric(df["load"], errors="coerce")
        out = df.groupby(["ts", "zone"], as_index=False)[["load", "pv"]].mean()
        # Hours with fewer sub-hourly points at the feed's edge are still averaged; fine for MW.
        return out[out["zone"].isin({z.code for z in self.zones})]
