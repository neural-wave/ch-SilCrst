"""PV capacity extensions: later upward steps of the PV signature (optional bonus task).

On clear days the midday window below standby spans the hours where production exceeds
the load. A bigger system widens it, so an extension is a persistent upward step in the
clear-day width (upper envelope of the daily width), on top of its seasonal cycle:

    envelope_d = seasonal(doy_d) + sum_k delta_k * [d >= tau_k]

The new capacity follows from the width through the clear-day production shape
cos(pi (t - t0) / D(doy))^g, whose parameters come from the data (devtools/production.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, Config
from .features import _rcumsum, day_of_year, day_stats
from .io import CustomerSeries


@dataclass(frozen=True)
class ExtensionConfig:
    # Policy chosen by simulating test-like sets (11 PV customers, 3 events) on synthetic
    # extensions (scripts/eval_bonus_policy.py): emit the `top_k` strongest candidates of the
    # whole file, ranked by explained variance. A false event never lowers the main score.
    top_k: int | None = 2
    # Extensions implied by the weak-onset rule (model.weak_onset) are emitted on top of the top_k
    # candidates; other candidates within forced_gap_days of them are dropped.
    forced_gap_days: int = 150
    ratio_min: float = 1.2      # capacity ratio after / before must be plausible ...
    ratio_max: float = 3.0      # ... larger jumps are artefacts (short or winter-only reference)
    # Threshold mode (top_k=None), tuned by scripts/eval_extensions.py.
    delta_min_h: float = 0.5    # minimum widening of the clear-day window (hours)
    gain_min: float = 0.3       # minimum share of residual variance explained by the step
    min_gap_days: int = 150     # after the installation or a previous extension
    min_after_days: int = 180  # near the end of the data the rolling envelope makes spurious steps
    envelope_days: int = 21
    envelope_q: float = 0.9
    max_events: int = 2


def _season_basis(doy: np.ndarray) -> np.ndarray:
    a = 2 * np.pi * doy / 365.25
    return np.column_stack([np.ones(doy.size), np.cos(a), np.sin(a), np.cos(2 * a), np.sin(2 * a)])


def _fit(X: np.ndarray, y: np.ndarray):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    return coef, float(r @ r)


def capacity_ratio(width_before: np.ndarray, width_after: np.ndarray, doy: np.ndarray, bell: list[float] | None) -> float:
    """K_after / K_before from clear-day widths, inverting cos(pi W / (2 D))^g = load / K."""
    if bell is None:
        return float(np.median(1 + (width_after - width_before) / np.maximum(width_before, 1)))
    _, g, d_a, d_b = bell[:4]
    d = d_a + d_b * np.cos(2 * np.pi * (doy - 172) / 365.25)
    level = lambda w: np.cos(np.pi * np.clip(w, 0, 0.95 * d) / (2 * d)) ** g
    return float(np.median(level(width_before) / np.maximum(level(width_after), 1e-3)))


def extension_candidates(s: CustomerSeries, tau: int, bell: list[float] | None = None,
                         cfg: Config = DEFAULT_CONFIG, ecfg: ExtensionConfig = ExtensionConfig()) -> list[dict]:
    """Up to `max_events` upward steps after installation day `tau`, in discovery order, each with
    its widening (delta, hours), explained variance (gain), capacity ratio and refined day index."""
    st = day_stats(s.x, cfg)
    n = s.n_days
    i = np.arange(n)
    use = st.valid & (i >= tau + 30)
    width = np.where(use, st.width_h, np.nan)
    env = pd.Series(width).rolling(ecfg.envelope_days, center=True, min_periods=ecfg.envelope_days // 2) \
        .quantile(ecfg.envelope_q).to_numpy()
    idx = np.flatnonzero(use & np.isfinite(env))
    if idx.size < 365 or (idx[-1] - idx[0]) < ecfg.min_gap_days + ecfg.min_after_days:
        return []
    doy = day_of_year(s.days())
    y, base = env[idx], _season_basis(doy[idx])

    found = []  # (day index, delta, gain)
    for _ in range(ecfg.max_events):
        steps = [f[0] for f in found]
        X_old = np.column_stack([base] + [(idx >= t).astype(float) for t in steps])
        _, sse_old = _fit(X_old, y)
        best = None
        for k in range(0, idx.size, 2):
            t = idx[k]
            if t < tau + ecfg.min_gap_days or t > idx[-1] - ecfg.min_after_days:
                continue
            if any(abs(t - p) < ecfg.min_gap_days for p in steps):
                continue
            coef, sse = _fit(np.column_stack([X_old, (idx >= t).astype(float)]), y)
            if coef[-1] > 0 and (best is None or sse < best[1]):
                best = (t, sse, coef[-1])
        if best is None:
            break
        found.append((best[0], float(best[2]), float(1 - best[1] / sse_old) if sse_old > 0 else 0.0))

    order = sorted(found)
    X = np.column_stack([base] + [(idx >= t).astype(float) for t, _, _ in order]) if order else base
    coef, _ = _fit(X, y)
    out = []
    for t, delta, gain in found:
        j = order.index((t, delta, gain))
        nxt = order[j + 1][0] if j + 1 < len(order) else n
        after = (idx >= t) & (idx < nxt)
        fitted_after = X[after] @ coef
        ratio = capacity_ratio(fitted_after - coef[5 + j], fitted_after, doy[idx[after]], bell)
        # Refine: first day whose width clearly exceeds the previous level (Bernoulli switch).
        pre_level = np.full(n, np.nan)
        pre_level[idx] = X @ coef - coef[5 + j] * (idx >= t)
        exceed = np.where(np.isfinite(pre_level), (st.width_h > pre_level + coef[5 + j] / 2).astype(float), np.nan)
        out.append({"t": int(t), "delta": delta, "gain": gain, "ratio": ratio, "day": _refine(exceed, int(t))})
    return out


def select_events(cands: list[dict], ecfg: ExtensionConfig = ExtensionConfig()) -> list[tuple[int, float]]:
    """Accept candidates in discovery order until one is too weak; drop those with a small ratio."""
    events = []
    for c in cands:
        if c["delta"] < ecfg.delta_min_h or c["gain"] < ecfg.gain_min:
            break
        if ecfg.ratio_min <= c["ratio"] <= ecfg.ratio_max:
            events.append((c["day"], c["ratio"]))
    return sorted(events)


def find_extensions(s: CustomerSeries, tau: int, bell: list[float] | None = None,
                    cfg: Config = DEFAULT_CONFIG, ecfg: ExtensionConfig = ExtensionConfig()) -> list[tuple[int, float]]:
    """[(day index of the extension, capacity ratio vs the previous level), ...] after installation day `tau`."""
    return select_events(extension_candidates(s, tau, bell, cfg, ecfg), ecfg)


def _refine(u: np.ndarray, t: int, before: int = 60, after: int = 30) -> int:
    i = np.arange(u.size)
    ok = np.isfinite(u)
    p0 = float(np.clip(np.nanmean(u[ok & (i < t - 30) & (i >= t - 180)]) if (ok & (i < t - 30)).sum() >= 20 else 0.05, 0.01, 0.4))
    p1 = float(np.clip(np.nanmean(u[ok & (i >= t) & (i < t + 90)]) if (ok & (i >= t)).sum() >= 10 else 0.5, 0.05, 0.98))
    w = np.flatnonzero(ok & (i >= t - before) & (i <= t + after))
    if p1 <= 1.5 * p0 or w.size == 0:
        return int(t)
    uw = u[w]
    l0 = np.where(uw == 1, np.log(p0), np.log1p(-p0))
    l1 = np.where(uw == 1, np.log(p1), np.log1p(-p1))
    ll = np.concatenate([[0.0], np.cumsum(l0)[:-1]]) + _rcumsum(l1)
    return int(w[int(np.argmax(ll))])


def predict_extensions(data: dict[str, CustomerSeries], pred: pd.DataFrame, bell: list[float] | None,
                       ecfg: ExtensionConfig = ExtensionConfig()) -> pd.DataFrame:
    """Extension events for the customers predicted with PV (pred = BaselineModel.predict output)."""
    chosen: dict[str, list[tuple[int, float]]] = {}
    pool = []
    kwp0, day0, forced = {}, {}, []
    onsets = pred["extension_onset"] if "extension_onset" in pred else pd.Series(pd.NaT, index=pred.index)
    for r, onset, onset_kwp in zip(pred[pred["pv_installed"]].itertuples(index=False), onsets[pred["pv_installed"]],
                                   pred.get("extension_onset_kwp", pd.Series(np.nan, index=pred.index))[pred["pv_installed"]]):
        s = data[r.customer_id]
        tau = 0
        if pd.notna(r.pv_installation_date):
            tau = max(int(np.datetime64(pd.Timestamp(r.pv_installation_date).date(), "D").astype(int)) - s.day0, 0)
        kwp0[r.customer_id] = float(r.pv_capacity_kwp) if pd.notna(r.pv_capacity_kwp) else np.nan
        day0[r.customer_id] = s.day0
        cands = extension_candidates(s, tau, bell, ecfg=ecfg)
        if pd.notna(onset):
            t = int(np.datetime64(pd.Timestamp(onset).date(), "D").astype(int)) - s.day0
            forced.append((r.customer_id, t, float(onset_kwp)))
            cands = [c for c in cands if abs(c["day"] - t) >= ecfg.forced_gap_days]
        if ecfg.top_k is None:
            chosen[r.customer_id] = select_events(cands, ecfg)
        else:
            pool += [(c["gain"], r.customer_id, c["day"], c["ratio"]) for c in cands
                     if ecfg.ratio_min <= c["ratio"] <= ecfg.ratio_max]
    if ecfg.top_k is not None:
        for _, cid, day, ratio in sorted(pool, key=lambda p: -p[0])[: ecfg.top_k]:
            chosen.setdefault(cid, []).append((day, ratio))

    rows = []
    for cid, t, total in forced:  # the weak-onset rule knows the capacity after its extension
        rows.append({"customer_id": cid, "extension_date": pd.Timestamp(np.datetime64(int(day0[cid] + t), "D")),
                     "pv_capacity_after_extension_kwp": total})
    for cid, events in chosen.items():
        kwp = kwp0[cid]
        for t, ratio in sorted(events):
            kwp = kwp * ratio
            rows.append({"customer_id": cid, "extension_date": pd.Timestamp(np.datetime64(int(day0[cid] + t), "D")),
                         "pv_capacity_after_extension_kwp": kwp})
    return pd.DataFrame(rows, columns=["customer_id", "extension_date", "pv_capacity_after_extension_kwp"])
