"""Physics-informed PV features, computed per customer without any label.

Core idea: the grid withdrawal of a household without PV never drops below its own
night-time standby load. With PV, sunny middays fall well below it (often to ~0),
because self-consumed production covers the load. Days showing that pattern are
"PV-like"; the installation is the point after which they start to appear.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, Config
from .io import CustomerSeries
from .weather import WeatherIndex, daily_signal, holiday_mask, width_signal


@dataclass
class DayStats:
    valid: np.ndarray     # day has enough data in the dark and midday windows
    base: np.ndarray      # reference standby load (kWh / 15 min)
    z_mid: np.ndarray     # fraction of midday slots below standby
    z_dark: np.ndarray    # fraction of dark slots below standby
    width_h: np.ndarray   # hours of daylight below standby
    pv_like: np.ndarray   # 1.0 / 0.0, NaN for invalid days
    base_median: float


def _nan_quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kwargs)


def day_stats(x: np.ndarray, cfg: Config = DEFAULT_CONFIG) -> DayStats:
    dark = x[:, list(cfg.dark_slots)]
    mid = x[:, list(cfg.midday_slots)]
    daylight = x[:, list(cfg.daylight_slots)]
    n_dark = (~np.isnan(dark)).sum(axis=1)
    n_mid = (~np.isnan(mid)).sum(axis=1)
    valid = (n_dark >= cfg.min_dark_valid) & (n_mid >= cfg.min_midday_valid)

    base_day = _nan_quiet(np.nanpercentile, dark, cfg.base_percentile, axis=1)
    base_median = float(np.nanmedian(base_day[valid])) if valid.any() else np.nan
    base = np.fmax(base_day, cfg.base_floor_frac * base_median)
    thr = np.fmax(cfg.below_frac * base, cfg.abs_floor_kwh)[:, None]

    z_mid = (mid < thr).sum(axis=1) / np.maximum(n_mid, 1)
    z_dark = (dark < thr).sum(axis=1) / np.maximum(n_dark, 1)
    width_h = (daylight < thr).sum(axis=1) / 4.0
    pv_like = (z_mid >= cfg.pv_day_min_midday_frac) & (z_dark <= cfg.pv_day_max_dark_frac)
    return DayStats(valid, base, z_mid, z_dark, width_h, np.where(valid, pv_like.astype(float), np.nan), base_median)


def season_weight(doy: np.ndarray) -> np.ndarray:
    """Expected relative PV visibility over the year: 1 at the summer solstice, 0.3 in winter."""
    return 0.65 + 0.35 * np.cos(2 * np.pi * (doy - 172) / 365.25)


def day_of_year(days: np.ndarray) -> np.ndarray:
    d = days.astype("datetime64[D]")
    return (d - d.astype("datetime64[Y]")).astype(int) + 1


def month_of(days: np.ndarray) -> np.ndarray:
    d = days.astype("datetime64[D]")
    return (d.astype("datetime64[M]") - d.astype("datetime64[Y]")).astype(int) + 1


# Nominal clear-sky output at the June noon peak, kWh per 15 min per kWp (~0.75 kW/kWp),
# used to inject synthetic PV.
NOMINAL_PEAK_KWH_PER_KWP = 0.75 * 0.25


def bell_curve(doy: np.ndarray, bell) -> np.ndarray:
    """(len(doy), 96) clear-day production shape, 1 = peak at the summer solstice.

    bell = [t0, g, D_a, D_b, logA_a, logA_b] from devtools/production.py (derived from the
    consumption data: time of peak, exponent, seasonal window length and amplitude)."""
    t0, g, d_a, d_b, la, lb = bell
    season = np.cos(2 * np.pi * (np.asarray(doy) - 172) / 365.25)
    d = d_a + d_b * season
    amp = np.exp(lb * (season - 1))
    t = (np.arange(96) + 0.5) / 4
    arg = (t[None, :] - t0) / d[:, None]
    return amp[:, None] * np.where(np.abs(arg) < 0.5, np.cos(np.pi * arg), 0.0) ** g


def shape_check(s: CustomerSeries, valid: np.ndarray, tau: int, bell, cfg: Config = DEFAULT_CONFIG) -> tuple[float, float]:
    """Is the drop in withdrawal after `tau` shaped like PV output?

    Compares summer profiles within ~400 days before / after (night-load adjusted) separately
    for weekdays and weekends. PV: the drop follows the clear-day production curve and is the
    same on weekends. Business changes follow working hours and mostly hit weekdays.
    Returns (min correlation with the curve over the two groups, weekend / weekday amplitude)."""
    i = np.arange(s.n_days)
    days = s.days()
    doy, weekend = day_of_year(days), day_type(days) > 0
    curve = bell_curve(np.arange(cfg.summer_doy[0], cfg.summer_doy[1] + 1), bell).mean(axis=0)
    light = curve > 0.05 * curve.max()
    summer = (doy >= cfg.summer_doy[0]) & (doy <= cfg.summer_doy[1])
    pre = valid & summer & (i < tau) & (i >= tau - 400)
    post = valid & summer & (i >= tau) & (i < tau + 400)
    dark = list(cfg.dark_slots)
    # The night load must survive the step: a customer that stops consuming is not PV.
    night = _nan_quiet(np.nanmedian, s.x[:, dark], axis=1)
    before_all, after_all = valid & (i < tau) & (i >= tau - 400), valid & (i >= tau) & (i < tau + 400)
    if before_all.any() and after_all.any() and np.nanmedian(night[after_all]) < 0.5 * np.nanmedian(night[before_all]):
        return np.nan, np.nan
    res = []
    for group in (~weekend, weekend):
        a, b = pre & group, post & group
        if a.sum() < 6 or b.sum() < 6:
            return np.nan, np.nan
        scale = np.nanmedian(s.x[b][:, dark]) / max(np.nanmedian(s.x[a][:, dark]), 1e-6)
        drop = (_nan_quiet(np.nanmedian, s.x[a], axis=0) * scale - _nan_quiet(np.nanmedian, s.x[b], axis=0))[light]
        if not np.isfinite(drop).all() or drop.std() == 0:
            return np.nan, np.nan
        res.append((float(np.corrcoef(drop, curve[light])[0, 1]), float(drop @ curve[light] / (curve[light] @ curve[light]))))
    (c_wd, a_wd), (c_we, a_we) = res
    return min(c_wd, c_we), (a_we / a_wd if a_wd > 0 else np.nan)


def sun_sensitivity(x: np.ndarray, valid: np.ndarray, days: np.ndarray, use: np.ndarray, day0: int,
                    weather: WeatherIndex, bell, cfg: Config = DEFAULT_CONFIG) -> tuple[float, float]:
    """Shape and size of the sunny-day reduction of the withdrawal on the days `use`.

    Per hour, the slope of the withdrawal on the regional weather index (month and day-type
    means removed, so seasons and weekdays do not leak in). Returns, for the region where it is
    most PV-like: the correlation of the reduction with the clear-day PV curve, and its size on
    that curve in kWh per 15 min per unit of the index (a self-consumed PV: its output)."""
    ok = use & valid & ~holiday_mask(days)
    if ok.sum() < 120:
        return np.nan, np.nan
    month = days[ok].astype("datetime64[D]").astype("datetime64[M]").astype(int)
    key = month * 2 + ((days[ok] + 3) % 7 >= 5)
    X = x[ok].astype(float)
    Xc = X - pd.DataFrame(X).groupby(key).transform("mean").to_numpy()
    curve = bell_curve(day_of_year(days[ok]), bell).mean(axis=0).reshape(24, 4).mean(axis=1)
    light = curve > 0.05 * curve.max()
    best = (np.nan, np.nan)
    for w in weather.aligned(day0, x.shape[0]):
        sel = np.isfinite(w[ok])
        if sel.sum() < 120:
            continue
        wc = w[ok][sel] - pd.Series(w[ok][sel]).groupby(key[sel]).transform("mean").to_numpy()
        slope = _nan_quiet(np.nansum, Xc[sel] * wc[:, None], axis=0) / max(float(wc @ wc), 1e-9)
        drop = -slope.reshape(24, 4).mean(axis=1)[light]
        if drop.std() == 0:
            continue
        r = float(np.corrcoef(drop, curve[light])[0, 1])
        if not np.isfinite(best[0]) or r > best[0]:
            best = (r, float(drop @ curve[light] / (curve[light] @ curve[light])))
    return best


def _dst_sundays(years) -> list[tuple[int, int]]:
    """(day number, sign) of the last Sundays of March (+1 h, local events move 1 h earlier in UTC:
    sign -1) and October (sign +1), the CET/CEST changes."""
    out = []
    for y in years:
        for month, sign in ((3, -1), (10, +1)):
            last = int((np.datetime64(f"{y}-{month + 1:02d}-01", "D") - np.timedelta64(1, "D")).astype(int))
            out.append((last - (last + 3 - 6) % 7, sign))  # 1970-01-01 was a Thursday
    return out


def dip_timing(x: np.ndarray, st: DayStats, days: np.ndarray, post: np.ndarray) -> tuple[float, float]:
    """Where in the day the withdrawal falls below standby, on PV-like days of `post`.

    Returns (centroid of the below-standby daylight slots in UTC hours, median shift of that
    centroid across the DST changes in the local-time direction, hours). A PV dip sits around
    solar noon and does not move at the DST changes; a schedule follows local time and moves by
    one hour."""
    light = slice(16, 80)  # 04:00-20:00 UTC
    hours = (np.arange(16, 80) + 0.5) / 4
    thr = np.fmax(DEFAULT_CONFIG.below_frac * st.base, DEFAULT_CONFIG.abs_floor_kwh)[:, None]
    below = (x[:, light] < thr) & st.valid[:, None]
    use = post & ~holiday_mask(days)
    pv_days = use & (st.pv_like == 1)
    centroid = np.nan
    if pv_days.sum() >= 10:
        p = below[pv_days].mean(axis=0)
        centroid = float(p @ hours / p.sum())
    n = np.arange(x.shape[0])
    years = np.unique(days.astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970)
    shifts = []
    for day, sign in _dst_sundays(years):
        t = day - int(days[0])
        b, a = use & (n >= t - 21) & (n <= t - 3), use & (n >= t + 3) & (n <= t + 21)
        if b.sum() < 8 or a.sum() < 8:
            continue
        pb, pa = below[b].mean(axis=0), below[a].mean(axis=0)
        if pb.sum() >= 4 and pa.sum() >= 4:  # about an hour below standby on average
            shifts.append(sign * (pa @ hours / pa.sum() - pb @ hours / pb.sum()))
    return centroid, (float(np.median(shifts)) if shifts else np.nan)


def _rcumsum(a: np.ndarray) -> np.ndarray:
    return np.cumsum(a[::-1])[::-1]


def day_type(days: np.ndarray) -> np.ndarray:
    """0 = Monday-Friday, 1 = Saturday, 2 = Sunday (1970-01-01, day 0, was a Thursday)."""
    weekday = (days + 3) % 7
    return np.where(weekday < 5, 0, weekday - 4)


def coarse_changepoint(y: np.ndarray, valid: np.ndarray, g: np.ndarray, min_after: int):
    """Least-squares fit of y_d = mu + delta * g_d * [d >= tau] over all tau (O(n)).

    Returns (tau index, delta, explained fraction, t-statistic of delta) or None when
    no positive step exists. Residuals are autocorrelated, so t is a ranking score
    rather than a calibrated test.
    """
    idx = np.flatnonzero(valid)
    n = idx.size
    if n < 2 * min_after:
        return None
    yv, gv = y[idx], g[idx]
    sw, sww, swy = _rcumsum(gv), _rcumsum(gv * gv), _rcumsum(gv * yv)
    sy, syy = yv.sum(), (yv * yv).sum()
    det = n * sww - sw**2
    ok = (det > 1e-6 * n * n) & (np.arange(n) <= n - min_after)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = (sww * sy - sw * swy) / det
        delta = (n * swy - sw * sy) / det
        sse = syy - (mu * sy + delta * swy)
    ok &= delta > 0
    if not ok.any():
        return None
    k = int(np.argmin(np.where(ok, sse, np.inf)))
    sse_null = syy - sy * sy / n
    gain = float(1 - sse[k] / sse_null) if sse_null > 0 else 0.0
    se = np.sqrt(max(sse[k], 1e-12) / (n - 2) * n / det[k])
    return int(idx[k]), float(delta[k]), gain, float(delta[k] / se)


def refine_mean_shift(y: np.ndarray, valid: np.ndarray, tau: int, before: int, after: int) -> int:
    """Best single mean shift (upwards) of a continuous signal within [tau - before, tau + after]."""
    days = np.arange(y.size)
    w = np.flatnonzero(valid & (days >= tau - before) & (days <= tau + after))
    if w.size < 10:
        return tau
    yw = y[w]
    k = np.arange(1, w.size)
    c1, c2 = np.cumsum(yw)[:-1], np.cumsum(yw**2)[:-1]
    t1, t2 = yw.sum(), (yw**2).sum()
    sse = (c2 - c1**2 / k) + ((t2 - c2) - (t1 - c1) ** 2 / (w.size - k))
    up = (t1 - c1) / (w.size - k) > c1 / k
    if not up.any():
        return tau
    return int(w[k[np.argmin(np.where(up, sse, np.inf))]])


def midday_ratio_signal(x: np.ndarray, valid: np.ndarray, days: np.ndarray, cfg: Config = DEFAULT_CONFIG) -> np.ndarray:
    """-log(midday / night load) per day, centred per day type: rises when PV covers part of
    the midday load, even when the withdrawal never drops below standby (large flat loads)."""
    mid = _nan_quiet(np.nanmean, x[:, list(cfg.midday_slots)], axis=1)
    dark = _nan_quiet(np.nanmean, x[:, list(cfg.dark_slots)], axis=1)
    level = np.nanmedian(dark[valid]) if valid.any() else np.nan
    eps = max(0.05 * level, 1e-4) if np.isfinite(level) else 1e-4
    y = -np.log((mid + eps) / (dark + eps))
    types = day_type(days)
    for t in range(3):
        m = valid & (types == t)
        if m.any():
            y[m] -= np.median(y[m])
    if valid.any():
        lo, hi = np.percentile(y[valid], [1, 99])
        y = np.clip(y, lo, hi)
    return np.where(valid, y, 0.0)


def refine_changepoint(y: np.ndarray, valid: np.ndarray, tau: int, cfg: Config = DEFAULT_CONFIG) -> int:
    """Maximum-likelihood switch between two Bernoulli rates of PV-like days, near `tau`.

    A PV-like day before the switch is very unlikely (p0 small) while a non-PV-like day
    after it is common (cloudy weather), so the estimate lands on the first PV-like day
    of the sustained run, ignoring isolated outliers.
    """
    days = np.arange(y.size)
    before = valid & (days < tau - 30)
    after = valid & (days >= tau) & (days < tau + 90)
    p0 = float(np.clip(y[before].mean() if before.sum() >= 20 else 0.02, 0.002, 0.3))
    p1 = float(np.clip(y[after].mean() if after.sum() >= 10 else 0.5, 0.05, 0.98))
    if p1 <= 1.5 * p0:
        return tau
    w = np.flatnonzero(valid & (days >= tau - cfg.refine_before_days) & (days <= tau + cfg.refine_after_days))
    if w.size == 0:
        return tau
    yw = y[w]
    l0 = np.where(yw == 1, np.log(p0), np.log1p(-p0))
    l1 = np.where(yw == 1, np.log(p1), np.log1p(-p1))
    ll = np.concatenate([[0.0], np.cumsum(l0)[:-1]]) + _rcumsum(l1)
    return int(w[int(np.argmax(ll))])


def refine_weak_start(st: DayStats, tau: int, cfg: Config = DEFAULT_CONFIG) -> int:
    """Earliest start of a sustained run of weak PV days (>= 8% of midday below standby) before `tau`.

    Same Bernoulli switch as refine_changepoint, on the weak indicator. Can only move the
    date earlier, and only when weak days are rare long before and common after."""
    days = np.arange(st.z_mid.size)
    weak = np.where(st.valid, ((st.z_mid >= cfg.weak_min_midday_frac) & (st.z_dark <= cfg.pv_day_max_dark_frac)).astype(float), 0.0)
    back = cfg.weak_lookback_days
    far = st.valid & (days < tau - back) & (days >= tau - back - 365)
    post = st.valid & (days >= tau) & (days < tau + 60)
    if post.sum() < 10 or (far.sum() < 30 and not cfg.weak_short_history):
        return tau
    p0 = float(np.clip(weak[far].mean(), 0.002, 0.3)) if far.sum() >= 30 else 0.02
    p1 = float(np.clip(weak[post].mean(), 0.05, 0.98))
    if p1 <= 3 * p0:
        return tau
    w = np.flatnonzero(st.valid & (days >= tau - back) & (days <= tau + 5))
    if w.size == 0:
        return tau
    yw = weak[w]
    l0 = np.where(yw == 1, np.log(p0), np.log1p(-p0))
    l1 = np.where(yw == 1, np.log(p1), np.log1p(-p1))
    ll = np.concatenate([[0.0], np.cumsum(l0)[:-1]]) + _rcumsum(l1)
    return min(tau, int(w[int(np.argmax(ll))]))


def _date(day_number) -> str | None:
    return None if day_number is None else str(np.datetime64(int(day_number), "D"))


def customer_features(s: CustomerSeries, cfg: Config = DEFAULT_CONFIG, bell=None,
                      weather: WeatherIndex | None = None) -> dict:
    """bell: clear-day production parameters (model.bell); enables the PV-shape check.
    weather: regional weather index (model.weather); enables the weather-coherence check."""
    out = {"customer_id": s.customer_id, "n_days": s.n_days}
    if s.n_days == 0:
        return out | {"n_valid_days": 0}
    st = day_stats(s.x, cfg)
    days = s.days()
    doy = day_of_year(days)
    valid = st.valid
    idx = np.flatnonzero(valid)
    out |= {"n_valid_days": int(idx.size), "base_kwh": st.base_median,
            # Share of exactly-zero night intervals: a PV customer always has a standby load.
            "dark_zero_share": float(np.mean(s.x[valid][:, list(cfg.dark_slots)] == 0)) if valid.any() else np.nan}
    if idx.size < cfg.min_valid_days:
        return out
    first = int(idx[0])
    out |= {"first_date": _date(days[first]), "last_date": _date(days[idx[-1]])}
    y = st.pv_like
    y0 = np.nan_to_num(y)
    out["pv_day_rate"] = float(y[valid].mean())

    # Detection score: best rate of PV-like days over a rolling window.
    r = cfg.rolling_days
    cy = np.concatenate([[0.0], np.cumsum(y0)])
    cv = np.concatenate([[0.0], np.cumsum(valid)])
    if s.n_days >= r:
        sy, sv = cy[r:] - cy[:-r], cv[r:] - cv[:-r]
        rate = np.where(sv >= cfg.rolling_min_valid, sy / np.maximum(sv, 1), np.nan)
        out["rate60_max"] = float(np.nanmax(rate)) if np.isfinite(rate).any() else out["pv_day_rate"]
    else:
        out["rate60_max"] = out["pv_day_rate"]

    g = season_weight(doy)
    n = np.arange(s.n_days)
    summer = (doy >= cfg.summer_doy[0]) & (doy <= cfg.summer_doy[1])

    # Relative signal: step in the midday / night load ratio (catches PV smaller than the load).
    yr = midday_ratio_signal(s.x, valid, days, cfg)
    rcp = coarse_changepoint(yr, valid, g, cfg.min_days_after_change)
    rtau = None
    if rcp is not None:
        rtau_c, rdelta, _, rt = rcp
        r_pre = rtau_c - first < cfg.preexisting_days
        rtau = first if r_pre else refine_mean_shift(yr, valid, rtau_c, cfg.refine_before_days, cfg.refine_after_days)
        out |= {"ratio_delta": rdelta, "ratio_t": rt, "ratio_tau_date": _date(days[rtau]), "ratio_preexisting": bool(r_pre)}
        if bell is not None and not r_pre:
            out["shape_corr"], out["shape_amp_ratio"] = shape_check(s, valid, rtau, bell, cfg)

    # Installation date from the PV-like days.
    cp = coarse_changepoint(y0, valid, g, cfg.min_days_after_change)
    tau = None
    if cp is not None:
        tau_c, delta, gain, _ = cp
        preexisting = tau_c - first < cfg.preexisting_days
        tau = first if preexisting else refine_changepoint(y0, valid, tau_c, cfg)
        if cfg.weak_refine and not preexisting:
            tau = refine_weak_start(st, tau, cfg)
        before, after = valid & summer & (n < tau), valid & summer & (n >= tau)
        out |= {
            "step_delta": delta,
            "step_gain": gain,
            "tau_coarse_date": _date(days[tau_c]),
            "tau_date": _date(days[tau]),
            "preexisting": bool(preexisting),
            "rate_before_summer": float(y[before].mean()) if before.sum() >= 20 else np.nan,
            "rate_after_summer": float(y[after].mean()) if after.sum() >= 20 else np.nan,
        }
        winter_after = valid & ((doy >= 305) | (doy <= 59)) & (n >= tau)  # November-February
        out["rate_after_winter"] = float(y[winter_after].mean()) if winter_after.sum() >= 20 else np.nan
    if tau is None:
        tau = rtau
    if tau is None:
        return out

    # Does the midday withdrawal after the installation follow the regional weather? (pvdetect/weather.py)
    # Summer only too: PV follows the weather most in summer, heating loads only in winter.
    if weather is not None:
        start = tau + 30 if tau > first else first
        for name, z in (("weather_r", daily_signal(s.x, valid, days, start, cfg)),
                        ("weather_r_width", width_signal(st.width_h, valid, days, start))):
            if z is not None:
                out[name] = weather.coherence(z, s.day0)
                out[name + "_summer"] = weather.coherence(np.where(summer, z, np.nan), s.day0, min_days=60)
        # Weak PV before the onset (an extension mistaken for the installation):
        # does the midday withdrawal already follow the weather before the PV-like days start?
        if tau > first:
            z = daily_signal(s.x, valid, days, first, cfg)
            if z is not None:
                zb = np.where(n < tau - 30, z, np.nan)
                out["pre_weather_r"] = weather.coherence(zb, s.day0)
                out["pre_weather_r_summer"] = weather.coherence(np.where(summer, zb, np.nan), s.day0, min_days=60)
                out["pre_weather_r_first"] = weather.coherence(np.where(n < first + 365, zb, np.nan), s.day0, min_days=60)
            if bell is not None:
                out["pre_shape_r"], out["pre_sun_drop_kwh"] = sun_sensitivity(s.x, valid, days, n < tau - 30, s.day0, weather, bell, cfg)

    # Where the dip sits in the day and whether it follows local time (schedule) at the DST changes.
    out["dip_centroid_h"], out["dip_dst_shift_h"] = dip_timing(s.x, st, days, valid & (n >= tau))

    # Capacity features, first year after the installation.
    window = valid & (n >= tau) & (n < tau + cfg.capacity_window_days)
    pv_days = window & (y == 1)
    sel = pv_days & summer if (pv_days & summer).sum() >= 5 else pv_days
    if sel.sum() >= 5:
        out["w90"] = float(np.percentile(st.width_h[sel], 90))
    post_summer = window & summer
    out["post_days"], out["post_summer_days"] = int(window.sum()), int(post_summer.sum())
    # Sunny-day reduction of the withdrawal in the first year: the PV output where it is self-consumed.
    if weather is not None and bell is not None:
        out["post_shape_r"], out["post_sun_drop_kwh"] = sun_sensitivity(s.x, valid, days, window, s.day0, weather, bell, cfg)
    if post_summer.sum() >= 20:
        out["z_mid_post"] = float(np.median(st.z_mid[post_summer]))
    dark_load = _nan_quiet(np.nanmean, s.x[window][:, list(cfg.dark_slots)])
    if np.isfinite(dark_load) and dark_load > 0:
        out["log_load_dark"] = float(np.log(dark_load))
    prior = valid & (n >= tau - 365) & (n < tau)
    if prior.sum() >= 60:
        out["daily_kwh_before"] = float(_nan_quiet(np.nanmean, s.x[prior]) * 96)

    # Midday reduction vs the same season before, adjusted for the night-load trend (weekdays).
    weekday = day_type(days) == 0
    pre = valid & summer & weekday & (n < tau) & (n >= tau - 730)
    post = valid & summer & weekday & (n >= tau) & (n < tau + cfg.capacity_window_days)
    if pre.sum() >= 20 and post.sum() >= 20:
        mid, dark = list(cfg.midday_slots), list(cfg.dark_slots)
        dark_pre, dark_post = np.nanmedian(s.x[pre][:, dark]), np.nanmedian(s.x[post][:, dark])
        if dark_pre > 0 and dark_post > 0:
            expected = np.nanmedian(s.x[pre][:, mid]) * dark_post / dark_pre
            drop = expected - np.nanmedian(s.x[post][:, mid])
            if expected > 0:
                out["mid_drop_frac"] = float(drop / expected)
            if drop > 0:
                out["log_mid_drop"] = float(np.log(drop))
    return out


FEATURE_COLUMNS = [
    "customer_id", "n_days", "n_valid_days", "base_kwh", "dark_zero_share", "first_date", "last_date",
    "pv_day_rate", "rate60_max", "step_delta", "step_gain", "tau_coarse_date", "tau_date",
    "preexisting", "rate_before_summer", "rate_after_summer", "rate_after_winter",
    "ratio_delta", "ratio_t", "ratio_tau_date", "ratio_preexisting", "shape_corr", "shape_amp_ratio",
    "w90", "z_mid_post", "post_days", "post_summer_days", "log_load_dark", "daily_kwh_before", "mid_drop_frac", "log_mid_drop",
    "weather_r", "weather_r_width", "weather_r_summer", "weather_r_width_summer",
    "pre_weather_r", "pre_weather_r_summer", "pre_weather_r_first", "pre_shape_r", "pre_sun_drop_kwh",
    "dip_centroid_h", "dip_dst_shift_h", "post_shape_r", "post_sun_drop_kwh",
]


def extract_features(data: dict[str, CustomerSeries], cfg: Config = DEFAULT_CONFIG, bell=None,
                     weather: WeatherIndex | None = None) -> pd.DataFrame:
    rows = [customer_features(s, cfg, bell, weather) for s in data.values()]
    df = pd.DataFrame(rows).reindex(columns=FEATURE_COLUMNS)
    df["customer_id"] = df["customer_id"].astype(str)
    df["preexisting"] = df["preexisting"].eq(True)
    df["ratio_preexisting"] = df["ratio_preexisting"].eq(True)
    return df
