"""Synthetic data with the same schema/quirks as the AIL files — for testing code only.

Scores on these data say nothing about real performance. The PV generator uses a rough
solar model at a fixed latitude/longitude: fine for fake test data, never used by the
pipeline (pvdetect/ must not depend on location or irradiation).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Zurich"
LAT, LON = np.radians(46.5), 8.5
STEP = 300  # simulate at 5 min, aggregate to 15 min


def _sin_elevation(t: pd.DatetimeIndex) -> np.ndarray:
    doy = t.dayofyear.to_numpy()
    hours = (t.hour + t.minute / 60).to_numpy()
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    b = 2 * np.pi * (doy - 81) / 364
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    hour_angle = np.radians(15 * (hours + LON / 15 + eot / 60 - 12))
    return np.sin(LAT) * np.sin(decl) + np.cos(LAT) * np.cos(decl) * np.cos(hour_angle)


def _weather(rng: np.random.Generator, first_day: int, n_days: int) -> np.ndarray:
    """Daily clearness index shared by all customers (one region), cloudier in winter."""
    doy = ((np.arange(first_day, first_day + n_days).astype("datetime64[D]")
            - np.arange(first_day, first_day + n_days).astype("datetime64[D]").astype("datetime64[Y]")).astype(int))
    p_sunny = 0.42 + 0.15 * np.cos(2 * np.pi * (doy - 190) / 365)
    u = rng.random(n_days)
    return np.where(u < p_sunny, rng.uniform(0.85, 1.0, n_days),
                    np.where(u < p_sunny + 0.3, rng.uniform(0.4, 0.85, n_days), rng.uniform(0.05, 0.35, n_days)))


def _gauss(h, mu, sd):
    return np.exp(-0.5 * ((h - mu) / sd) ** 2)


def _customer(rng, start: pd.Timestamp, end: pd.Timestamp, clear: np.ndarray, clear_day0: int, pv: dict | None):
    t0 = int(start.tz_localize(TZ).timestamp())
    t1 = int((end + pd.Timedelta(days=1)).tz_localize(TZ).timestamp())
    epoch = np.arange(t0, t1, STEP, dtype=np.int64)
    utc = pd.to_datetime(epoch, unit="s", utc=True)
    local = utc.tz_convert(TZ)
    lh = (local.hour + local.minute / 60).to_numpy()
    weekday = local.dayofweek.to_numpy() < 5
    doy = local.dayofyear.to_numpy()
    n = epoch.size

    commercial = rng.random() < 0.15
    base = rng.lognormal(np.log(0.12 if not commercial else 0.8), 0.4)
    act = (rng.lognormal(np.log(0.4), 0.3) * _gauss(lh, np.where(weekday, 7.0, 8.5), 0.8)
           + rng.lognormal(np.log(0.3), 0.3) * _gauss(lh, 12.5, 1.2) * np.where(weekday, 1.0, 1.6)
           + rng.lognormal(np.log(0.8), 0.3) * _gauss(lh, 19.5, 1.6))
    if commercial:
        open_ = 1 / (1 + np.exp(-(lh - 7.5) * 3)) / (1 + np.exp(-(18 - lh) * 3))
        act = act + rng.uniform(2, 15) * open_ * weekday
    heat = 1 + rng.uniform(0, 0.8) * np.clip(np.cos(2 * np.pi * (doy - 15) / 365.25), 0, None)
    act = act * heat * rng.gamma(3, 1 / 3, n)
    for year in range(start.year, end.year + 1):  # holidays: standby only
        if rng.random() < 0.6:
            v0 = pd.Timestamp(year, int(rng.integers(1, 12)), 1, tz=TZ) + pd.Timedelta(days=int(rng.integers(0, 28)))
            off = (local >= v0) & (local < v0 + pd.Timedelta(days=int(rng.integers(7, 22))))
            act = np.where(off, 0.0, act)
    load = base * np.clip(1 + 0.15 * rng.standard_normal(n), 0.5, None) + act
    load += np.where(rng.random(n) < 0.015, rng.uniform(1, 3, n), 0.0)  # kettle, oven, ...
    if rng.random() < 0.3:  # night-tariff boiler
        load += 2.0 * ((lh >= 23) | (lh < 1))

    prod = np.zeros(n)
    if pv is not None:
        day = epoch // 86400 - clear_day0
        c = np.clip(clear[day] + rng.normal(0, 0.05, n), 0.02, 1.0)
        partly = (c > 0.35) & (c < 0.85)
        c = c * np.where(partly & (rng.random(n) < 0.4), rng.uniform(0.3, 1.0, n), 1.0)
        kwp = np.where(utc >= pv["date"].tz_localize(TZ), pv["kwp"], 0.0)
        for e_date, e_kwp in pv.get("extensions", []):
            kwp = np.where(utc >= e_date.tz_localize(TZ), e_kwp, kwp)
        prod = kwp * 0.8 * np.clip(_sin_elevation(utc), 0, None) ** 1.2 * c

    kwh = (np.maximum(load - prod, 0) * STEP / 3600).reshape(-1, 3).sum(axis=1)
    return epoch[::3], kwh


def generate(out_dir: str | Path, n_customers: int = 40, start: str = "2022-01-01", end: str = "2025-12-31",
             pv_share: float = 0.55, seed: int = 0) -> None:
    """Write energy_consumption_dev.csv, pv_labels_dev.csv, pv_extensions_labels_dev.csv to out_dir."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    clear_day0 = int(start_ts.value // 86_400_000_000_000) - 2
    clear = _weather(rng, clear_day0, (end_ts - start_ts).days + 5)

    frames, labels, extensions = [], [], []
    for i in range(n_customers):
        cid = f"{i + 1:05d}"  # leading zeros must survive
        c_start = start_ts if rng.random() < 0.7 else start_ts + pd.Timedelta(days=int(rng.integers(0, 540)))
        c_end = end_ts if rng.random() < 0.8 else max(c_start + pd.Timedelta(days=400), end_ts - pd.Timedelta(days=int(rng.integers(0, 300))))
        c_end = min(c_end, end_ts)
        pv = None
        if rng.random() < pv_share:
            span = (c_end - c_start).days
            if rng.random() < 0.2:  # installed before the series starts
                date = c_start - pd.Timedelta(days=int(rng.integers(30, 700)))
            else:
                date = c_start + pd.Timedelta(days=int(rng.integers(60, max(span - 90, 61))))
            pv = {"date": date.normalize(), "kwp": round(float(np.clip(rng.lognormal(np.log(9), 0.5), 2, 40)), 1)}
            if not extensions and date > c_start and (c_end - date).days > 420:
                e_date = (date + pd.Timedelta(days=int(rng.integers(300, (c_end - date).days - 60)))).normalize()
                pv["extensions"] = [(e_date, round(pv["kwp"] * float(rng.uniform(1.3, 1.8)), 1))]
                extensions.append((cid, e_date, pv["extensions"][0][1]))
        epoch, kwh = _customer(rng, c_start, c_end, clear, clear_day0, pv)
        keep = np.ones(epoch.size, bool)
        if rng.random() < 0.3:  # a data gap
            g0 = int(rng.integers(0, epoch.size - 3000))
            keep[g0 : g0 + int(rng.integers(3, 30)) * 96] = False
        frames.append(pd.DataFrame({"customer_id": cid, "energy_consumption_kwh": np.round(kwh[keep], 3),
                                    "timestamp_epoch": epoch[keep]}))
        labels.append((cid, pv is not None, pv["date"] if pv else None, pv["kwp"] if pv else None))

    df = pd.concat(frames, ignore_index=True)
    local = pd.to_datetime(df["timestamp_epoch"], unit="s", utc=True).dt.tz_convert(TZ)
    df.insert(2, "timestamp_local", local.dt.strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(out / "energy_consumption_dev.csv", sep=";", index=False)

    with open(out / "pv_labels_dev.csv", "w", encoding="utf-8") as f:
        f.write("customer_id;pv_installed;pv_installation_date;pv_capacity_kwp\n")
        for cid, has, date, kwp in labels:
            f.write(f"{cid};{'TRUE' if has else 'FALSE'};{date.strftime('%Y-%m-%d') if has else ''};{kwp if has else ''}\n")
    with open(out / "pv_extensions_labels_dev.csv", "w", encoding="utf-8") as f:
        f.write("customer_id,extension_date,pv_capacity_after_extension_kwp\n")  # ',' header, ';' rows (as in the real file)
        for cid, date, kwp in extensions:
            f.write(f"{cid};{date.strftime('%Y-%m-%d')};{kwp}\n")
