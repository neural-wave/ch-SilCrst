"""Build the synthetic labelled set: PV injected into confident non-PV unlabelled series.

Usage: python scripts/augment.py DATA_DIR [--draws 4] [--cache cache] [--pseudo outputs/pseudo_labels.csv]

Needs pseudo_label.py and build_production.py first. Writes features + labels of every
synthetic example to cache/augmented.csv (AIL-derived, git-ignored). Prints summary checks.
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
from pvdetect.features import customer_features, extract_features  # noqa: E402
from pvdetect.io import load_consumption, read_labels, read_table  # noqa: E402
from pvdetect.model import BaselineModel  # noqa: E402

AUGMENTED = "augmented.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--pseudo", default="outputs/pseudo_labels.csv")
    ap.add_argument("--vary", action="store_true", help="random orientation, curve width and inverter clipping")
    ap.add_argument("--out", default=AUGMENTED)
    ap.add_argument("--seed", type=int, default=0, help="random injections (different seeds = different synthetic sets)")
    a = ap.parse_args()
    d = Path(a.data_dir)

    dev, _ = load_consumption(d / ENERGY_DEV, a.cache)
    unl, _ = load_consumption(d / ENERGY_UNLABELLED, a.cache)
    labels = read_labels(d / LABELS_DEV)
    pl = read_table(a.pseudo)
    prof = ProductionProfile.load(Path(a.cache) / "production_profile.npz")
    mu, sd = size_prior(dev, labels)
    hosts = [unl[c] for c in pl.loc[pl["group"] == "non_pv", "customer_id"]]

    weather = BaselineModel.load("models/baseline.json").weather  # weather features, if built
    rows = []
    for s, lab in make_examples(hosts, prof, mu, sd, draws=a.draws, seed=a.seed, vary=a.vary):
        rows.append(customer_features(s, bell=prof.params.tolist(), weather=weather) | lab)
    aug = pd.DataFrame(rows)
    aug.to_csv(Path(a.cache) / a.out, index=False)

    model = BaselineModel().fit(extract_features(dev), labels)
    det = model.predict(aug.drop(columns=["pv_installed", "pv_installation_date", "pv_capacity_kwp"]).assign(
        preexisting=aug["preexisting"].eq(True), ratio_preexisting=aug["ratio_preexisting"].eq(True)))
    pre = aug["pv_installation_date"] < pd.to_datetime(aug["first_date"])
    err = (pd.to_datetime(det["pv_installation_date"]) - pd.to_datetime(aug["pv_installation_date"])).dt.days.abs()
    print(f"synthetic examples: {len(aug)} from {len(hosts)} hosts")
    print(f"  detected by the dev-fitted model: {det['pv_installed'].mean():.0%}")
    print(f"  date error (detected, installed inside the series): median {err[det['pv_installed'] & ~pre].median():.0f} d, "
          f"<=7 d: {(err[det['pv_installed'] & ~pre] <= 7).mean():.0%}")
    inside = det["pv_installed"] & ~pre
    print(f"  date score D on synthetic (detected, inside the series): {np.clip(1 - err[inside] / 30, 0, None).mean():.3f} "
          f"| errors >= 30 d: {(err[inside] >= 30).mean():.0%}")
    print(f"  censored share (z_mid_post >= 0.5): {(aug['z_mid_post'] >= 0.5).mean():.0%}")
    print(f"-> {Path(a.cache) / a.out}  (AIL-derived, git-ignored)")


if __name__ == "__main__":
    main()
