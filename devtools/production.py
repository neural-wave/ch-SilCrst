"""Data-derived PV production model, used only to inject synthetic PV (training data).

Everything comes from the consumption series themselves — no weather, irradiance or
location data:
- a regional daily sunniness index: how wide, relative to the customer's own best days of
  that month, the below-standby midday window of PV customers is on each date;
- the clear-day diurnal shape of production per month, from "uncensored" PV customers
  (PV smaller than the load, so the drop in withdrawal is the PV output itself);
- a daily clearness factor from the same uncensored customers.
The result is AIL-derived: it lives in cache/ and is never committed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pvdetect.config import DEFAULT_CONFIG, Config
from pvdetect.features import bell_curve, day_of_year, day_stats, day_type, month_of
from pvdetect.io import CustomerSeries


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*a, **k)


def counterfactual_load(s: CustomerSeries, tau: int, cfg: Config = DEFAULT_CONFIG,
                        pre_days: int = 730, post_days: int = 365) -> np.ndarray:
    """Expected load without PV for post-installation days: median profile of pre-installation
    days of the same weekday/weekend type within ±2 weeks of the year, scaled by the change in
    night load (unaffected by PV). NaN where it cannot be estimated."""
    x, n = s.x, s.n_days
    days = s.days()
    week = np.minimum((day_of_year(days) - 1) // 7, 51)
    kind = (day_type(days) > 0).astype(int)
    valid = day_stats(x, cfg).valid
    i = np.arange(n)
    pre = valid & (i < tau) & (i >= tau - pre_days)
    post = valid & (i >= tau) & (i < tau + post_days)
    dark_day = _quiet(np.nanmedian, x[:, list(cfg.dark_slots)], axis=1)

    ref = np.full((52, 2, 96), np.nan)
    ref_dark = np.full((52, 2), np.nan)
    for w in range(52):
        near = np.minimum(np.abs(week - w), 52 - np.abs(week - w)) <= 2
        for k in (0, 1):
            sel = pre & near & (kind == k)
            if sel.sum() >= 3:
                ref[w, k] = _quiet(np.nanmedian, x[sel], axis=0)
                ref_dark[w, k] = np.nanmedian(dark_day[sel])

    level = pd.Series(np.where(valid, dark_day, np.nan)).rolling(15, center=True, min_periods=5).median().to_numpy()
    out = np.full((n, 96), np.nan)
    p = np.flatnonzero(post)
    scale = np.clip(level[p] / ref_dark[week[p], kind[p]], 0.5, 2.0)
    out[p] = ref[week[p], kind[p]] * scale[:, None]
    return out


def fit_bell(y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit y(t) ~ b + A * max(0, cos(pi (t - t0) / D))^g over the 96 UTC slots (grid search).

    t0 (time of peak) and D (production window) are free, so the day length comes out of
    the data rather than from a latitude. Returns the bell without the offset b.
    """
    t = (np.arange(96) + 0.5) / 4
    ok = np.isfinite(y)
    best = (np.inf, None, None)
    for g in (1.0, 1.5, 2.0):
        for t0 in np.arange(10.0, 13.01, 0.125):
            for d in np.arange(5.0, 19.01, 0.25):
                bell = np.where(np.abs(t - t0) < d / 2, np.cos(np.pi * (t - t0) / d), 0.0) ** g
                X = np.column_stack([bell[ok], np.ones(ok.sum())])
                coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
                if coef[0] <= 0:
                    continue
                sse = float(((X @ coef - y[ok]) ** 2).sum())
                if sse < best[0]:
                    best = (sse, coef[0] * bell, {"t0": t0, "D": d, "g": g, "A": coef[0], "b": coef[1]})
    if best[1] is None:
        return np.zeros(96), {}
    return best[1], best[2]


def _season(doy) -> np.ndarray:
    return np.cos(2 * np.pi * (np.asarray(doy) - 172) / 365.25)


def _smooth_fit(doy: np.ndarray, y: np.ndarray, w: np.ndarray, max_resid: float) -> tuple[float, float]:
    """Weighted fit y = a + b * season(doy), refitted once without outliers."""
    keep = np.isfinite(y) & (w > 0)
    for _ in range(2):
        X = np.column_stack([np.ones(keep.sum()), _season(doy[keep])]) * np.sqrt(w[keep])[:, None]
        a, b = np.linalg.lstsq(X, y[keep] * np.sqrt(w[keep]), rcond=None)[0]
        keep &= np.abs(y - (a + b * _season(doy))) <= max_resid
    return float(a), float(b)


@dataclass
class ProductionProfile:
    day0: int                 # first UTC day number of the daily arrays
    sunniness: np.ndarray     # (n_days,) regional index, ~0 overcast .. ~1 as clear as the best days
    params: np.ndarray        # [t0, g, D_a, D_b, logA_a, logA_b]: smooth seasonal bell
    n_sun: np.ndarray         # customers behind each sunniness value
    fits: list                # raw per-month bell fits (for reporting)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, day0=self.day0, sunniness=self.sunniness, params=self.params, n_sun=self.n_sun)

    @classmethod
    def load(cls, path: str | Path) -> "ProductionProfile":
        with np.load(path) as z:
            return cls(int(z["day0"]), z["sunniness"], z["params"], z["n_sun"], [])

    def clear_day(self, doy: np.ndarray) -> np.ndarray:
        """(len(doy), 96) clear-day production shape; 1 = peak at the summer solstice."""
        return bell_curve(doy, self.params)

    @property
    def shape(self) -> np.ndarray:
        """(12, 96) clear-day shape at mid-month, for reporting."""
        return self.clear_day(np.array([15 + 30.5 * m for m in range(12)]))

    def daily_factor(self, days: np.ndarray) -> np.ndarray:
        """Share of clear-day production on the given UTC days (diffuse floor 0.1; 0.5 if unknown)."""
        i = np.clip(days - self.day0, 0, self.sunniness.size - 1)
        s = self.sunniness[i]
        return np.where(np.isfinite(s), 0.1 + 0.9 * s, 0.5)


def build_profile(pv: list[tuple[CustomerSeries, int]], uncensored: list[tuple[CustomerSeries, int]],
                  cfg: Config = DEFAULT_CONFIG, min_customers: int = 10) -> ProductionProfile:
    """pv / uncensored: (series, installation day index); 0 means 'already there at the start'."""
    day0 = min(s.day0 for s, _ in pv)
    n_days = max(s.day0 + s.n_days for s, _ in pv) - day0

    # 1. Sunniness: width of the below-standby window, relative to the customer's best days.
    rel = np.full((len(pv), n_days), np.nan, np.float32)
    for j, (s, tau) in enumerate(pv):
        st = day_stats(s.x, cfg)
        i = np.arange(s.n_days)
        post = st.valid & (i >= tau) & (i < tau + 730)
        w = np.where(post, st.width_h, np.nan)
        months = month_of(s.days())
        r = np.full(s.n_days, np.nan)
        for m in range(1, 13):
            sel = months == m
            q = _quiet(np.nanpercentile, w[sel], 95) if np.isfinite(w[sel]).any() else np.nan
            if np.isfinite(q) and q >= 1.0:
                r[sel] = np.clip(w[sel] / q, 0, 1)
        rel[j, s.day0 - day0 : s.day0 - day0 + s.n_days] = r
    n_sun = np.isfinite(rel).sum(axis=0)
    sunniness = np.where(n_sun >= min_customers, _quiet(np.nanmedian, rel, axis=0), np.nan)

    # 2. Clear-day production per month from uncensored PV customers (output = drop in withdrawal),
    #    each normalised by its own brightest month so that the seasonal amplitude is kept.
    all_days = np.arange(day0, day0 + n_days)
    month_all = month_of(all_days)
    sunny_cut = np.array([_quiet(np.nanpercentile, sunniness[month_all == m], 80) for m in range(1, 13)])
    profiles = []
    for s, tau in uncensored:
        red = counterfactual_load(s, tau, cfg) - s.x
        days = s.days()
        months = month_of(days)
        sun = sunniness[np.clip(days - day0, 0, n_days - 1)]
        prof = np.full((12, 96), np.nan)
        for m in range(1, 13):
            sel = (months == m) & np.isfinite(red).all(axis=1) & (sun >= sunny_cut[m - 1])
            if sel.sum() >= 3:
                prof[m - 1] = np.convolve(_quiet(np.nanmean, red[sel], axis=0), np.ones(3) / 3, mode="same")
        peak = _quiet(np.nanmax, prof)
        if np.isfinite(peak) and peak > 0:
            profiles.append(prof / peak)

    median = _quiet(np.nanmedian, np.stack(profiles), axis=0) if profiles else np.full((12, 96), np.nan)
    fits = []
    for m in range(12):
        _, fit = fit_bell(median[m]) if np.isfinite(median[m]).sum() > 20 else (None, {})
        fits.append(fit | {"n": int(sum(np.isfinite(p[m]).any() for p in profiles))})
    mid_doy = np.array([15 + 30.5 * m for m in range(12)])
    get = lambda k: np.array([f.get(k, np.nan) for f in fits], float)
    n = get("n")
    d_a, d_b = _smooth_fit(mid_doy, get("D"), n, max_resid=2.5)
    a_a, a_b = _smooth_fit(mid_doy, np.log(get("A")), n, max_resid=0.7)
    ok = np.isfinite(get("t0"))
    t0 = float(np.median(get("t0")[ok])) if ok.any() else 11.5
    g = float(np.median(get("g")[ok])) if ok.any() else 1.5
    params = np.array([t0, g, d_a, d_b, a_a, a_b])
    return ProductionProfile(day0, sunniness, params, n_sun, fits)


def load_bell(path: str | Path) -> list[float] | None:
    """Clear-day production parameters of a saved profile, or None if it does not exist."""
    return ProductionProfile.load(path).params.tolist() if Path(path).exists() else None
