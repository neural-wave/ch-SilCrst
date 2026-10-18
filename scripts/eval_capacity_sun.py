"""Does the sunny-day reduction help the capacity model? Paired comparison.

Usage: python scripts/eval_capacity_sun.py DATA_DIR [--seeds 0 1 2 3 4] [--cache cache]

For each seed, a synthetic set built like scripts/augment.py (4 draws per confident non-PV host) but
with the weather index, so that post_sun_drop_kwh exists. Two capacity models (CatBoost, as shipped)
differ only by the feature log_sun_drop:
- leave-one-out on the 22 labelled PV (synthetic + the 21 others, log offset on the 21 others);
- synthetic held-out: trained on the examples of half of the hosts, scored on the other half.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from devtools.augment import make_examples, size_prior  # noqa: E402
from devtools.production import ProductionProfile  # noqa: E402
from pvdetect import ENERGY_DEV, ENERGY_UNLABELLED, LABELS_DEV  # noqa: E402
from pvdetect.capacity import CAPACITY_FEATURES, CapacityModel  # noqa: E402
from pvdetect.features import customer_features, extract_features  # noqa: E402
from pvdetect.io import customers_only, load_consumption, read_labels, read_table  # noqa: E402
from pvdetect.model import BaselineModel  # noqa: E402

VARIANTS = {"shipped features": CAPACITY_FEATURES, "+ log_sun_drop": CAPACITY_FEATURES + ("log_sun_drop",)}


def p_score(pred, true) -> np.ndarray:
    return np.clip(1 - np.abs(pred - true) / true / 0.5, 0, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--pseudo", default="outputs/pseudo_labels.csv")
    a = ap.parse_args()
    d = Path(a.data_dir)
    model = BaselineModel.load("models/baseline.json")
    dev, _ = load_consumption(d / ENERGY_DEV, a.cache)
    labels = read_labels(d / LABELS_DEV)
    real = extract_features(dev, bell=model.bell, weather=model.weather).merge(labels, on="customer_id")
    real = real[real["pv_installed"] & (real["pv_capacity_kwp"] > 0)].reset_index(drop=True)
    prof = ProductionProfile.load(Path(a.cache) / "production_profile.npz")
    mu, sd = size_prior(dev, labels)
    cols = extract_features({}).columns.tolist()
    print(f"real PV: {len(real)} | sunny-day reduction available for {int(real['post_sun_drop_kwh'].notna().sum())}")

    loo = {k: [] for k in VARIANTS}
    loo_pred = {k: [] for k in VARIANTS}
    held = {k: [] for k in VARIANTS}
    for seed in a.seeds:
        path = Path(a.cache) / f"aug_sun_s{seed}.csv"  # AIL-derived, git-ignored
        if path.exists():
            aug = read_table(path)
        else:
            unl, _ = load_consumption(d / ENERGY_UNLABELLED, a.cache)
            unl, _ = customers_only(unl)
            pl = read_table(a.pseudo)
            hosts = [unl[c] for c in pl.loc[pl["group"] == "non_pv", "customer_id"] if c in unl]
            rows = [customer_features(s, bell=model.bell, weather=model.weather) | lab
                    for s, lab in make_examples(hosts, prof, mu, sd, draws=4, seed=seed)]
            aug = pd.DataFrame(rows)
            aug.to_csv(path, index=False)
        aug = aug.reindex(columns=list(dict.fromkeys(cols + list(aug.columns))))
        aug["preexisting"] = aug["preexisting"].eq(True)
        host = aug["host_id"].astype(str)
        even = host.isin(sorted(host.unique())[::2]).to_numpy()
        for name, feats in VARIANTS.items():
            pred = np.zeros(len(real))
            for i in range(len(real)):
                rest = real.drop(index=i)
                cap = CapacityModel(seed=seed, features=feats).fit(pd.concat([aug, rest], ignore_index=True), calibrate_on=rest)
                pred[i] = cap.predict(real.iloc[[i]])[0]
            loo[name].append(p_score(pred, real["pv_capacity_kwp"].to_numpy(float)))
            loo_pred[name].append(np.log(pred))
            cap = CapacityModel(seed=seed, features=feats).fit(aug[even])
            true = pd.to_numeric(aug.loc[~even, "pv_capacity_kwp"]).to_numpy(float)
            held[name].append(float(p_score(cap.predict(aug[~even]), true).mean()))
        print(f"seed {seed}: " + " | ".join(f"{k}: LOO P {loo[k][-1].mean():.3f}, synthetic held-out P {held[k][-1]:.3f}" for k in VARIANTS))

    names = list(VARIANTS)
    per_customer = {k: np.mean(loo[k], axis=0) for k in names}
    diff = per_customer[names[1]] - per_customer[names[0]]
    rng = np.random.default_rng(0)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
    print(f"\nmean over seeds: " + " | ".join(f"{k}: LOO P {per_customer[k].mean():.3f}, held-out P {np.mean(held[k]):.3f}" for k in names))
    true = real["pv_capacity_kwp"].to_numpy(float)
    ens = {k: p_score(np.exp(np.mean(loo_pred[k], axis=0)), true) for k in names}
    ediff = ens[names[1]] - ens[names[0]]
    eboot = [ediff[rng.integers(0, len(ediff), len(ediff))].mean() for _ in range(5000)]
    print("mean of the models of all seeds (geometric mean of the predictions): " + " | ".join(f"{k}: LOO P {ens[k].mean():.3f}" for k in names)
          + f" | difference {ediff.mean():+.3f} [95% {np.percentile(eboot, 2.5):+.3f}, {np.percentile(eboot, 97.5):+.3f}], in favour {np.mean(np.array(eboot) > 0):.0%}")
    print(f"paired LOO difference (+ log_sun_drop - shipped): {diff.mean():+.3f} [95% {np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]"
          f", resamples in favour {np.mean(np.array(boot) > 0):.0%} | customers better / worse: {(diff > 0.02).sum()} / {(diff < -0.02).sum()}")


if __name__ == "__main__":
    main()
