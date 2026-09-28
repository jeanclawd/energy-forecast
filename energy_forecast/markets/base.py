from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class Zone:
    code: str
    name: str
    lat: float
    lon: float


class Market:
    """A market is a set of zones plus a way to fetch hourly actuals.

    `fetch_actuals` returns a DataFrame with columns
    ts (UTC, hourly, tz-naive), zone, load (MW), pv (MW).
    """

    name: str = ""
    timezone: str = "UTC"
    holidays_country: str = ""
    zones: list[Zone] = []

    def fetch_actuals(self, start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError

    def zone(self, code: str) -> Zone:
        return next(z for z in self.zones if z.code == code)
