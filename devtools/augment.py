"""Synthetic labelled examples: inject PV into real series without PV.

withdrawal' = max(0, withdrawal - kWp * peak_kwh_per_kwp * clear_day(doy, slot) * daily_factor(day) * noise)

The production model comes from devtools.production (derived from the consumption data
only). Its absolute scale is nominal: a single log-offset is calibrated on the real labels.
"""
from __future__ import annotations

import numpy as np

from pvdetect.config import DEFAULT_CONFIG, Config
from pvdetect.features import NOMINAL_PEAK_KWH_PER_KWP as PEAK_KWH_PER_KWP
from pvdetect.features import bell_curve, day_of_year, day_stats, day_type
from pvdetect.io import CustomerSeries

from .production import ProductionProfile



def midday_load_kw(s: CustomerSeries, cfg: Config = DEFAULT_CONFIG) -> float:
    """Median weekday midday withdrawal in kW (the reference for PV size relative to the load)."""
    valid = day_stats(s.x, cfg).valid & (day_type(s.days()) == 0)
    if not valid.any():
        return np.nan
    return float(np.nanmedian(s.x[valid][:, list(cfg.midday_slots)]) * 4)


def size_prior(dev: dict, labels) -> tuple[float, float]:
    """Mean / sd of log(kWp / midday load kW) over labelled PV customers, load taken before installation."""
    r = []
    for _, row in labels[labels["pv_installed"]].iterrows():
        s = dev[row["customer_id"]]
        tau = int(np.datetime64(row["pv_installation_date"].date(), "D").astype(int)) - s.day0
        if tau >= 180:
            load = midday_load_kw(CustomerSeries(s.customer_id, s.day0, s.x[:tau]))
            if np.isfinite(load) and load > 0:
                r.append(np.log(row["pv_capacity_kwp"] / load))
    return float(np.mean(r)), float(max(np.std(r), 0.5))


def random_plant(rng: np.random.Generator) -> dict:
    """Plant diversity: orientation (peak earlier for east, later for west), broader or narrower
    daily curve, and inverter clipping (AC limit below the DC peak)."""
    u = rng.random()
    shift = 0.0 if u < 0.6 else (-1 if u < 0.8 else 1) * rng.uniform(0.5, 1.5)
    return {"shift_h": shift, "width": rng.uniform(0.9, 1.2), "clip": rng.uniform(0.65, 1.0) if rng.random() < 0.5 else None}


def inject(s: CustomerSeries, tau: int, kwp: float, prof: ProductionProfile, rng: np.random.Generator,
           new_id: str, plant: dict | None = None) -> CustomerSeries:
    days = s.days()
    p = np.flatnonzero(np.arange(s.n_days) >= tau)
    if plant is None:
        clear = prof.clear_day(day_of_year(days[p]))
    else:
        t0, g, d_a, d_b, la, lb = prof.params
        params = [t0 + plant["shift_h"], g, d_a * plant["width"], d_b * plant["width"], la, lb]
        clear = bell_curve(day_of_year(days[p]), params)
    f = prof.daily_factor(days[p])
    partly = ((f > 0.3) & (f < 0.85))[:, None]
    noise = np.where(partly, rng.uniform(0.4, 1.3, clear.shape), rng.normal(1.0, 0.05, clear.shape))
    prod = kwp * PEAK_KWH_PER_KWP * clear * f[:, None] * np.clip(noise, 0, None)
    if plant is not None and plant["clip"] is not None:
        prod = np.minimum(prod, kwp * PEAK_KWH_PER_KWP * plant["clip"])
    x = s.x.copy()
    x[p] = np.maximum(x[p] - prod, 0.0).astype(np.float32)
    return CustomerSeries(new_id, s.day0, x)


def make_examples(hosts: list[CustomerSeries], prof: ProductionProfile, ratio_mu: float, ratio_sd: float,
                  draws: int = 4, preexisting_share: float = 0.15, seed: int = 0, vary: bool = False):
    """Yield (series, label dict). PV size: log(kWp / midday load kW) ~ N(ratio_mu, ratio_sd)."""
    rng = np.random.default_rng(seed)
    for h in hosts:
        load = midday_load_kw(h)
        if not np.isfinite(load) or load <= 0 or h.n_days < 400:
            continue
        for k in range(draws):
            kwp = float(np.exp(np.log(load) + rng.normal(ratio_mu, ratio_sd)))
            if rng.random() < preexisting_share:
                tau, date_day = 0, h.day0 - int(rng.integers(30, 700))
            else:
                tau = int(rng.integers(60, h.n_days - 120))
                date_day = h.day0 + tau
            s = inject(h, tau, kwp, prof, rng, f"{h.customer_id}~{k}", random_plant(rng) if vary else None)
            yield s, {"customer_id": s.customer_id, "host_id": h.customer_id, "pv_installed": True,
                      "pv_installation_date": np.datetime64(date_day, "D"), "pv_capacity_kwp": kwp}


def make_late_examples(hosts: list[CustomerSeries], prof: ProductionProfile, ratio_mu: float, ratio_sd: float,
                       seed: int = 707):
    """Yield (series, label dict): PV installed 45-200 days before the end of the series, the case
    where few days are left to check the detection."""
    rng = np.random.default_rng(seed)
    for h in hosts:
        load = midday_load_kw(h)
        if not np.isfinite(load) or load <= 0 or h.n_days < 400:
            continue
        kwp = float(np.exp(np.log(load) + rng.normal(ratio_mu, ratio_sd)))
        tau = int(h.n_days - rng.integers(45, 200))
        s = inject(h, tau, kwp, prof, rng, f"{h.customer_id}~late")
        yield s, {"customer_id": s.customer_id, "host_id": h.customer_id, "pv_installed": True,
                  "pv_installation_date": np.datetime64(h.day0 + tau, "D"), "pv_capacity_kwp": kwp}


def make_weak_strong_examples(hosts: list[CustomerSeries], prof: ProductionProfile, draws: int = 2,
                              pre_share: float = 0.6, seed: int = 21):
    """Yield (series, initial label dict, extension events): a small initial PV (peak 10-60% of the
    midday load, so the midday withdrawal stays above standby) later extended to a large one (peak
    1.5-6x the load). The case of the weak-onset rule,
    rare in make_extension_examples. A `pre_share` of the initial systems predates the series."""
    rng = np.random.default_rng(seed)
    kw_per_kwp = PEAK_KWH_PER_KWP * 4
    for h in hosts:
        load = midday_load_kw(h)
        if not np.isfinite(load) or load <= 0 or h.n_days < 1000:
            continue
        for k in range(draws):
            cid = f"{h.customer_id}~w{k}"
            kwp0 = float(np.exp(rng.uniform(np.log(0.1), np.log(0.6)))) * load / kw_per_kwp
            kwp1 = float(np.exp(rng.uniform(np.log(1.5), np.log(6.0)))) * load / kw_per_kwp
            pre = rng.random() < pre_share
            tau0 = 0 if pre else int(rng.integers(60, 300))
            t1 = int(rng.integers(tau0 + 365, h.n_days - 240))
            s = inject(h, tau0, kwp0, prof, rng, cid)
            s = inject(s, t1, kwp1 - kwp0, prof, rng, cid)
            date0 = h.day0 - int(rng.integers(30, 700)) if pre else h.day0 + tau0
            yield s, {"customer_id": cid, "host_id": h.customer_id, "pv_installed": True,
                      "pv_installation_date": np.datetime64(date0, "D"), "pv_capacity_kwp": kwp0}, \
                [{"customer_id": cid, "extension_date": np.datetime64(h.day0 + t1, "D"), "pv_capacity_after_extension_kwp": kwp1}]


def make_extension_examples(hosts: list[CustomerSeries], prof: ProductionProfile, ratio_mu: float, ratio_sd: float,
                            draws: int = 2, share: float = 0.25, seed: int = 1):
    """Yield (series, initial label dict, extension events). A `share` of the examples gets one
    extension (x1.2-2.5 capacity, 150-900 days later), a quarter of those a second one."""
    rng = np.random.default_rng(seed)
    for h in hosts:
        load = midday_load_kw(h)
        if not np.isfinite(load) or load <= 0 or h.n_days < 700:
            continue
        for k in range(draws):
            cid = f"{h.customer_id}~e{k}"
            kwp = float(np.exp(np.log(load) + rng.normal(ratio_mu, ratio_sd)))
            tau = 0 if rng.random() < 0.15 else int(rng.integers(60, h.n_days - 400))
            s = inject(h, tau, kwp, prof, rng, cid)
            events, total, t = [], kwp, tau
            n_ext = (1 + (rng.random() < 0.25)) if rng.random() < share else 0
            for _ in range(n_ext):
                t = t + int(rng.integers(150, 900))
                if t > h.n_days - 90:
                    break
                new_total = total * float(rng.uniform(1.2, 2.5))
                s = inject(s, t, new_total - total, prof, rng, cid)
                total = new_total
                events.append({"customer_id": cid, "extension_date": np.datetime64(h.day0 + t, "D"),
                               "pv_capacity_after_extension_kwp": total})
            date_day = h.day0 + tau if tau > 0 else h.day0 - int(rng.integers(30, 700))
            yield s, {"customer_id": cid, "pv_installed": True, "pv_installation_date": np.datetime64(date_day, "D"),
                      "pv_capacity_kwp": kwp}, events
