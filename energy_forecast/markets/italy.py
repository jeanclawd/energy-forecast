"""Italy: ENTSO-E Transparency Platform, per bidding zone (needs a free token).

Set ENTSOE_TOKEN (register at transparency.entsoe.eu, then email
transparency@entsoe.eu for REST API access) and ENERGY_MARKET=italy.

Load: documentType A65 (system total load), processType A16 (realised).
PV:   documentType A75 (actual generation per type), psrType B16 (solar).

Written against the ENTSO-E API guide but not yet run against the live
API (no token on hand when it was written): run `python -m energy_forecast update`
once with a token and check the counts before trusting it.
"""
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import pandas as pd
import requests

from .base import Market, Zone

API = "https://web-api.tp.entsoe.eu/api"


class Italy(Market):
    name = "italy"
    timezone = "Europe/Rome"
    holidays_country = "IT"
    # code = bidding-zone EIC; lat/lon = a representative city for the weather
    zones = [
        Zone("10Y1001A1001A73I", "Nord", 45.46, 9.19),
        Zone("10Y1001A1001A70O", "Centro-Nord", 43.77, 11.25),
        Zone("10Y1001A1001A71M", "Centro-Sud", 41.90, 12.50),
        Zone("10Y1001A1001A788", "Sud", 41.12, 16.87),
        Zone("10Y1001C--00096J", "Calabria", 38.90, 16.60),
        Zone("10Y1001A1001A75E", "Sicilia", 37.50, 15.09),
        Zone("10Y1001A1001A74G", "Sardegna", 39.22, 9.12),
    ]

    def __init__(self):
        self.token = os.environ.get("ENTSOE_TOKEN", "")

    def _series(self, params: dict, start: datetime, end: datetime) -> pd.Series:
        if not self.token:
            raise RuntimeError("ENTSOE_TOKEN is not set — the Italian market needs an ENTSO-E API token")
        r = requests.get(API, params={"securityToken": self.token, "periodStart": f"{start:%Y%m%d%H%M}",
                                      "periodEnd": f"{end:%Y%m%d%H%M}", **params}, timeout=120)
        if r.status_code == 400 and "No matching data" in r.text:
            return pd.Series(dtype=float)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {"n": root.tag.split("}")[0].strip("{")}
        out = {}
        for period in root.iterfind(".//n:Period", ns):
            t0 = pd.Timestamp(period.find("n:timeInterval/n:start", ns).text).tz_convert(None)
            res = pd.Timedelta(period.find("n:resolution", ns).text.replace("PT", "").lower().replace("m", "min"))
            for pt in period.iterfind("n:Point", ns):
                ts = t0 + (int(pt.find("n:position", ns).text) - 1) * res
                out[ts] = float(pt.find("n:quantity", ns).text)
        s = pd.Series(out).sort_index()
        return s.resample("h").mean() if len(s) else s

    def fetch_actuals(self, start: datetime, end: datetime) -> pd.DataFrame:
        frames = []
        for z in self.zones:
            loads, pvs = [], []
            cur = start
            while cur < end:                      # the API serves at most one year per call
                nxt = min(cur + timedelta(days=360), end)
                loads.append(self._series({"documentType": "A65", "processType": "A16",
                                           "outBiddingZone_Domain": z.code}, cur, nxt))
                pvs.append(self._series({"documentType": "A75", "processType": "A16", "psrType": "B16",
                                         "in_Domain": z.code}, cur, nxt))
                cur = nxt
            df = pd.DataFrame({"load": pd.concat(loads), "pv": pd.concat(pvs)})
            df.index.name = "ts"
            df = df.reset_index()
            df["zone"] = z.code
            frames.append(df)
        return pd.concat(frames, ignore_index=True)[["ts", "zone", "load", "pv"]]
