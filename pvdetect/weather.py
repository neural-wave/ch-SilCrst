"""Weather coherence: does the midday withdrawal of a customer follow the regional weather?

PV output follows the clouds, so after the installation the midday withdrawal of a PV
customer drops on the same days as at the other PV customers of its region. Loads with a
fixed schedule (evening-only consumption, lunch breaks, seasonal activities) can also fall
below the night standby at midday, which the main detection rule takes as PV, but not in
step with the weather.

The daily weather of the regions is an aggregate index built from the unlabelled PV
customers (scripts/build_weather.py). The regions come from the consumption data alone:
customers whose sunny and cloudy days coincide form one region. No weather, irradiance or
location data is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri

from .config import DEFAULT_CONFIG, Config

MIN_DAYS = 120  # fewer usable days after the installation: no verdict


def _easter(year: int) -> np.datetime64:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    day = (h + m - 7 * n + 114) % 31 + 1
    return np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", "D")


@lru_cache(maxsize=None)
def _holidays(year: int) -> frozenset:
    """Day numbers of Swiss public holidays (national and widely observed cantonal ones) and of
    the Christmas break: business closures shared by many customers, unrelated to the weather."""
    e = _easter(year)
    moving = [e + np.timedelta64(k, "D") for k in (-2, 1, 39, 50, 60)]  # Good Friday, Easter Monday, Ascension, Whit Monday, Corpus Christi
    fixed = ["01-01", "01-02", "01-06", "03-19", "05-01", "06-29", "08-01", "08-15", "11-01", "12-08"]
    fixed += [f"12-{d}" for d in range(24, 32)]
    days = moving + [np.datetime64(f"{year:04d}-{md}", "D") for md in fixed]
    return frozenset(int(d.astype(int)) for d in days)


def holiday_mask(days: np.ndarray) -> np.ndarray:
    years = days.astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    hol = set().union(*(_holidays(int(y)) for y in np.unique(years))) if days.size else set()
    return np.isin(days, list(hol))


def _usable(valid: np.ndarray, days: np.ndarray, start: int) -> np.ndarray:
    return valid & (np.arange(valid.size) >= start) & ~holiday_mask(days)


def rank_signal(u: np.ndarray, ok: np.ndarray, days: np.ndarray) -> np.ndarray | None:
    """Daily values u on the days `ok` -> deseasonalised normal scores (NaN elsewhere); None if too few.

    The day-type level (weekday / Saturday / Sunday) is removed within each month, then the values
    are ranked within the month and mapped to normal scores, which removes the seasons and the
    scale of the customer."""
    if ok.sum() < MIN_DAYS:
        return None
    weekday = (days + 3) % 7  # 1970-01-01 (day 0) was a Thursday
    df = pd.DataFrame({
        "pos": np.flatnonzero(ok),
        "u": u[ok],
        "type": np.where(weekday[ok] < 5, 0, weekday[ok] - 4),
        "month": days[ok].astype("datetime64[D]").astype("datetime64[M]").astype(int),
    })
    df = df[np.isfinite(df["u"])]
    df["u"] -= df.groupby(["month", "type"])["u"].transform("median")
    size = df.groupby("month")["u"].transform("size")
    df = df[size >= 10]
    if len(df) < MIN_DAYS:
        return None
    rank = df.groupby("month")["u"].rank(method="average")
    size = df.groupby("month")["u"].transform("size")
    out = np.full(u.size, np.nan)
    out[df["pos"].to_numpy()] = ndtri(((rank - 0.5) / size).to_numpy())
    return out


def daily_signal(x: np.ndarray, valid: np.ndarray, days: np.ndarray, start: int,
                 cfg: Config = DEFAULT_CONFIG) -> np.ndarray | None:
    """(n_days,) midday sunniness signal from day index `start` on, NaN elsewhere; None if too short.

    u = -log((midday + eps) / (night + eps)): high when little is withdrawn at midday."""
    ok = _usable(valid, days, start)
    if ok.sum() < MIN_DAYS:
        return None
    with np.errstate(all="ignore"):
        mid = np.nanmean(x[:, list(cfg.midday_slots)], axis=1)
        dark = np.nanmean(x[:, list(cfg.dark_slots)], axis=1)
    level = np.nanmedian(dark[ok])
    eps = max(0.05 * level, 1e-4) if np.isfinite(level) else 1e-4
    return rank_signal(-np.log((mid + eps) / (dark + eps)), ok, days)


def width_signal(width_h: np.ndarray, valid: np.ndarray, days: np.ndarray, start: int) -> np.ndarray | None:
    """Daily signal of the hours below standby over the whole daylight span (features.day_stats).

    With PV the window where production covers the load widens on sunny days, also when a
    daytime business load hides the PV at midday (it then shows in the early morning, the
    evening and on idle days)."""
    return rank_signal(width_h.astype(float), _usable(valid, days, start), days)


def correlation(a: np.ndarray, b: np.ndarray, min_days: int = MIN_DAYS) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_days or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


@dataclass
class WeatherIndex:
    day0: int             # UTC day number of column 0
    index: np.ndarray     # (n_regions, n_days) mean normal score of the region's PV customers
    members: np.ndarray   # (n_regions,) customers behind each region

    def aligned(self, day0: int, n_days: int) -> np.ndarray:
        """(n_regions, n_days) index on the days day0 .. day0 + n_days - 1, NaN outside."""
        out = np.full((self.index.shape[0], n_days), np.nan)
        lo, hi = max(day0, self.day0), min(day0 + n_days, self.day0 + self.index.shape[1])
        if hi > lo:
            out[:, lo - day0:hi - day0] = self.index[:, lo - self.day0:hi - self.day0]
        return out

    def coherence(self, signal: np.ndarray, day0: int, min_days: int = MIN_DAYS) -> float:
        """Best correlation of a customer's daily signal with any region (NaN if too little overlap)."""
        r = [correlation(signal, row, min_days) for row in self.aligned(day0, signal.size)]
        r = [v for v in r if np.isfinite(v)]
        return max(r) if r else np.nan

    def save(self, path: str | Path) -> None:
        np.savez(path, day0=self.day0, index=self.index.astype(np.float32), members=self.members)

    @classmethod
    def load(cls, path: str | Path) -> "WeatherIndex":
        with np.load(path) as z:
            return cls(int(z["day0"]), z["index"].astype(float), z["members"])
