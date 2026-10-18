"""Statistical differences of the model's features.

Usage: python scripts/feature_stats.py DATA_DIR [--augmented cache/augmented_w0.csv ...] [--cache cache]

A. Capacity model (the trained model): are the synthetic training examples distributed like real PV?
   Synthetic: the augmented sets. Real: the 22 labelled PV of the dev set. Per feature of the
   model's input matrix: share of missing values, Kolmogorov-Smirnov and Mann-Whitney tests (Holm
   correction over the features), standardised mean difference. Overall: a classifier trained to tell synthetic from real
   (adversarial validation, cross-validated ROC AUC; 0.5 = indistinguishable).
B. Detection features: do they separate PV from non-PV? Development labels (22 PV, 18 non-PV):
   ROC AUC per feature (0.5 = no separation, 1 or 0 = perfect) and Mann-Whitney p-value.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pvdetect import ENERGY_DEV, LABELS_DEV  # noqa: E402
from pvdetect.capacity import CAPACITY_FEATURES, capacity_matrix  # noqa: E402
from pvdetect.features import extract_features  # noqa: E402
from pvdetect.io import load_consumption, read_labels, read_table  # noqa: E402
from pvdetect.model import BaselineModel  # noqa: E402

DETECTION_FEATURES = ("rate60_max", "pv_day_rate", "step_delta", "rate_after_summer", "ratio_t", "shape_corr",
                      "weather_r", "weather_r_width", "weather_r_summer", "dip_centroid_h", "dip_dst_shift_h", "dark_zero_share")


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--augmented", nargs="+", default=[f"cache/augmented_w{k}.csv" for k in range(5)])
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    d = Path(a.data_dir)
    model = BaselineModel.load("models/baseline.json")
    dev, _ = load_consumption(d / ENERGY_DEV, a.cache)
    labels = read_labels(d / LABELS_DEV)
    dev_f = extract_features(dev, bell=model.bell, weather=model.weather).merge(labels, on="customer_id")
    real = dev_f[dev_f["pv_installed"]].reset_index(drop=True)
    syn = pd.concat([read_table(p) for p in a.augmented], ignore_index=True)
    syn["preexisting"] = syn["preexisting"].eq(True)
    Xs, Xr = capacity_matrix(syn), capacity_matrix(real)

    print(f"== A. Capacity model features: synthetic ({len(syn)} examples, {len(a.augmented)} sets) "
          f"vs real PV ({len(real)} labelled development customers)")
    rows = []
    for j, name in enumerate(CAPACITY_FEATURES):
        s, r = Xs[:, j], Xr[:, j]
        s_ok, r_ok = s[np.isfinite(s)], r[np.isfinite(r)]
        row = {"feature": name, "missing syn": np.mean(~np.isfinite(s)), "missing real": np.mean(~np.isfinite(r)),
               "median syn": np.median(s_ok) if s_ok.size else np.nan, "median real": np.median(r_ok) if r_ok.size else np.nan}
        if s_ok.size > 5 and r_ok.size > 5 and np.ptp(np.concatenate([s_ok, r_ok])) > 0:
            pooled = np.sqrt((s_ok.var(ddof=1) + r_ok.var(ddof=1)) / 2)
            row |= {"std. mean diff": (r_ok.mean() - s_ok.mean()) / pooled if pooled > 0 else np.nan,
                    "KS": ks_2samp(s_ok, r_ok).statistic, "p KS": ks_2samp(s_ok, r_ok).pvalue,
                    "p MW": mannwhitneyu(s_ok, r_ok).pvalue}
        rows.append(row)
    t = pd.DataFrame(rows)
    for c in ("p KS", "p MW"):
        ok = t[c].notna()
        t.loc[ok, c + " Holm"] = holm(t.loc[ok, c].to_numpy())
    pd.set_option("display.width", 200)
    print(t.drop(columns=["p KS", "p MW"]).round(3).to_string(index=False))
    print("   (std. mean diff: real minus synthetic in pooled standard deviations; |d| > 0.5 is a medium difference)")

    X = np.vstack([Xs, Xr])
    y = np.r_[np.zeros(len(Xs)), np.ones(len(Xr))]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=0)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    rng = np.random.default_rng(0)
    null = [roc_auc_score(rng.permutation(y), proba) for _ in range(500)]
    print(f"adversarial validation (classifier synthetic vs real, 5-fold): ROC AUC {auc:.3f} "
          f"(0.5 = indistinguishable; permutation 99th percentile {np.percentile(null, 99):.3f})")
    single = []
    for j, name in enumerate(CAPACITY_FEATURES):
        p_j = cross_val_predict(clf, X[:, [j]], y, cv=cv, method="predict_proba")[:, 1]
        single.append((roc_auc_score(y, p_j), name))
    print("   features that alone tell synthetic from real best: " + ", ".join(f"{n} {v:.2f}" for v, n in sorted(single, reverse=True)[:4]))

    print(f"\n== B. Detection features, development labels ({int(dev_f['pv_installed'].sum())} PV vs {int((~dev_f['pv_installed']).sum())} non-PV)")
    rows = []
    yb = dev_f["pv_installed"].to_numpy(bool)
    for name in DETECTION_FEATURES:
        v = pd.to_numeric(dev_f[name], errors="coerce").to_numpy(float)
        ok = np.isfinite(v)
        if ok.sum() < 10 or yb[ok].all() or not yb[ok].any():
            rows.append({"feature": name, "available PV / non-PV": f"{int((ok & yb).sum())} / {int((ok & ~yb).sum())}"})
            continue
        rows.append({"feature": name, "available PV / non-PV": f"{int((ok & yb).sum())} / {int((ok & ~yb).sum())}",
                     "median PV": np.median(v[ok & yb]), "median non-PV": np.median(v[ok & ~yb]),
                     "ROC AUC": roc_auc_score(yb[ok], v[ok]), "p MW": mannwhitneyu(v[ok & yb], v[ok & ~yb]).pvalue})
    t = pd.DataFrame(rows)
    ok = t["p MW"].notna()
    t.loc[ok, "p MW Holm"] = holm(t.loc[ok, "p MW"].to_numpy())
    print(t.drop(columns=["p MW"]).round(4).to_string(index=False))
    print("   (several weather and timing features exist only for customers with a detected onset, i.e. mostly PV)")


if __name__ == "__main__":
    main()
