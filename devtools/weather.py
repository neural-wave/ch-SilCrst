"""Weather regions from the consumption data alone (training only, see pvdetect/weather.py).

The daily sunniness signals of PV customers share common factors: one per weather region.
A two-factor model separates the regions (direction of each customer's loadings), then the
customers that follow neither region (fixed-schedule loads mistaken for PV) are dropped and
each region's index is the mean signal of its remaining members.
"""
from __future__ import annotations

import numpy as np

from pvdetect.weather import WeatherIndex, correlation


def factor_loadings(X: np.ndarray, k: int = 2, iters: int = 60, lam: float = 1e-3, seed: int = 0) -> np.ndarray:
    """(n_customers, k) loadings of X ~ L F^T fitted by alternating least squares on observed entries."""
    obs = np.isfinite(X)
    Z = np.nan_to_num(X)
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(X.shape[0], k)) * 0.1
    L[:, 0] = 1.0
    F = np.zeros((X.shape[1], k))
    eye = lam * np.eye(k)
    for _ in range(iters):
        for d in range(X.shape[1]):
            m = obs[:, d]
            if m.sum() > k:
                F[d] = np.linalg.solve(L[m].T @ L[m] + eye, L[m].T @ Z[m, d])
        for i in range(X.shape[0]):
            m = obs[i]
            if m.sum() > k:
                L[i] = np.linalg.solve(F[m].T @ F[m] + eye, F[m].T @ Z[i, m])
    return L


def split_regions(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Two groups from the angle of the 2-factor loadings (1-D 2-means); 0 = the larger group."""
    L = factor_loadings(X, 2, seed=seed)
    ang = np.arctan2(L[:, 1], L[:, 0])
    c = np.quantile(ang, [0.25, 0.75])
    lab = np.zeros(len(ang), int)
    for _ in range(100):
        lab = np.abs(ang[:, None] - c[None]).argmin(axis=1)
        c_new = np.array([np.median(ang[lab == g]) if (lab == g).any() else c[g] for g in range(2)])
        if np.allclose(c_new, c):
            break
        c = c_new
    return lab if (lab == 0).sum() >= (lab == 1).sum() else 1 - lab


def mean_index(X: np.ndarray, members: np.ndarray, min_n: int = 5) -> np.ndarray:
    s, n = np.nan_to_num(X[members]).sum(axis=0), np.isfinite(X[members]).sum(axis=0)
    return np.where(n >= min_n, s / np.maximum(n, 1), np.nan)


def build_index(X: np.ndarray, day0: int, min_r: float = 0.2, min_members: int = 20) -> tuple[WeatherIndex, dict]:
    """Regional weather index from the signals X (customers x days, NaN = unused)."""
    lab = split_regions(X)
    r_own = np.full(len(X), np.nan)
    for i in range(len(X)):
        others = (lab == lab[i]) & (np.arange(len(X)) != i)
        r_own[i] = correlation(X[i], mean_index(X, others))
    keep = np.nan_to_num(r_own, nan=-1) >= min_r
    regions = [g for g in (0, 1) if (keep & (lab == g)).sum() >= min_members]
    index = np.stack([mean_index(X, keep & (lab == g)) for g in regions])
    members = np.array([int((keep & (lab == g)).sum()) for g in regions])
    info = {"customers": len(X), "group_sizes": [int((lab == g).sum()) for g in (0, 1)],
            "dropped_incoherent": int((~keep).sum()), "members": members.tolist(),
            "between_regions_r": correlation(index[0], index[1]) if len(regions) == 2 else np.nan}
    return WeatherIndex(day0, index, members), info
