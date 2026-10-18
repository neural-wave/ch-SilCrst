"""Official challenge scoring (PDF §4.2-4.3)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def score_main(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Main score = F1 * (70 + 15 D + 15 P) over the customers in `truth`.

    Both frames: customer_id, pv_installed (bool), pv_installation_date (datetime), pv_capacity_kwp (float).
    A customer missing from `pred` counts as predicted FALSE.
    """
    m = truth.merge(pred, on="customer_id", how="left", suffixes=("_true", "_pred"))
    t = m["pv_installed_true"].astype(bool).to_numpy()
    p = m["pv_installed_pred"].eq(True).to_numpy()
    tp, fp, fn, tn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum()), int((~t & ~p).sum())
    f1 = _f1(tp, fp, fn)

    hit = m[t & p]
    err_days = (pd.to_datetime(hit["pv_installation_date_pred"]) - pd.to_datetime(hit["pv_installation_date_true"])).dt.days.abs()
    d_i = np.clip(1 - err_days.to_numpy(float) / 30, 0, None)
    d_i = np.where(np.isnan(d_i), 0.0, d_i)  # missing / invalid date scores 0
    true_kwp = hit["pv_capacity_kwp_true"].to_numpy(float)
    pred_kwp = pd.to_numeric(hit["pv_capacity_kwp_pred"], errors="coerce").to_numpy(float)
    ape = np.abs(pred_kwp - true_kwp) / true_kwp
    p_i = np.clip(1 - ape / 0.5, 0, None)
    p_i = np.where(np.isnan(p_i) | (pred_kwp <= 0), 0.0, p_i)

    d = float(d_i.mean()) if tp else 0.0
    p_score = float(p_i.mean()) if tp else 0.0
    return {
        "main": f1 * (70 + 15 * d + 15 * p_score) if tp else 0.0,
        "f1": f1, "D": d, "P": p_score,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "date_abs_err_median": float(np.nanmedian(err_days)) if err_days.notna().any() else np.nan,
        "date_within_7": int((err_days <= 7).sum()),
        "date_within_30": int((err_days < 30).sum()),
        "ape_median": float(np.nanmedian(ape)) if np.isfinite(ape).any() else np.nan,
        "ape_within_25": int((ape <= 0.25).sum()),
        "ape_within_50": int((ape < 0.5).sum()),
    }


def _best_matching(true_days: tuple, pred_days: tuple) -> tuple[int, list[tuple[int, int]]]:
    """Maximise the number of ±30-day one-to-one matches, then minimise total date error."""

    @lru_cache(maxsize=None)
    def best(i: int, used: int) -> tuple[int, int, tuple]:
        if i == len(true_days):
            return 0, 0, ()
        result = best(i + 1, used)  # leave true event i unmatched
        for j, pd_ in enumerate(pred_days):
            err = abs(pd_ - true_days[i])
            if not used >> j & 1 and err <= 30:
                n, e, pairs = best(i + 1, used | 1 << j)
                if (n + 1, -(e + err)) > (result[0], -result[1]):
                    result = (n + 1, e + err, ((i, j),) + pairs)
        return result

    n, _, pairs = best(0, 0)
    return n, list(pairs)


def score_extensions(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Bonus = F1_ext * (7 + 1.5 D_ext + 1.5 P_ext), events matched per customer within ±30 days."""
    tp = fp = fn = 0
    d_scores, p_scores = [], []
    pred = pred.dropna(subset=["extension_date"])  # a missing date prevents matching
    for cid in set(truth["customer_id"]) | set(pred["customer_id"]):
        t = truth[truth["customer_id"] == cid].reset_index(drop=True)
        p = pred[pred["customer_id"] == cid].reset_index(drop=True)
        t_days = tuple(t["extension_date"].values.astype("datetime64[D]").astype(int))
        p_days = tuple(p["extension_date"].values.astype("datetime64[D]").astype(int))
        n, pairs = _best_matching(t_days, p_days)
        tp, fp, fn = tp + n, fp + len(p_days) - n, fn + len(t_days) - n
        for i, j in pairs:
            d_scores.append(max(0.0, 1 - abs(p_days[j] - t_days[i]) / 30))
            true_kwp = float(t.loc[i, "pv_capacity_after_extension_kwp"])
            pred_kwp = pd.to_numeric(p.loc[j, "pv_capacity_after_extension_kwp"], errors="coerce")
            ok = pd.notna(pred_kwp) and pred_kwp > 0
            p_scores.append(max(0.0, 1 - abs(pred_kwp - true_kwp) / true_kwp / 0.5) if ok else 0.0)
    f1 = _f1(tp, fp, fn)
    d = float(np.mean(d_scores)) if tp else 0.0
    p_score = float(np.mean(p_scores)) if tp else 0.0
    return {"bonus": f1 * (7 + 1.5 * d + 1.5 * p_score) if tp else 0.0,
            "f1_ext": f1, "D_ext": d, "P_ext": p_score, "tp": tp, "fp": fp, "fn": fn}
