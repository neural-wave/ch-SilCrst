"""Calibrate the schedule guard (dip that follows local time or sits outside PV hours).

Usage: python scripts/calibrate_schedule.py DATA_DIR [--cache cache] [--pseudo outputs/pseudo_labels.csv]

Cost, real PV removed: synthetic PV (plain, varied orientation and clipping, installed in the last
45-200 days of the series) and the dev PV.
Benefit: the unlabelled detections removed. Thresholds are set on the cost side.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from devtools.augment import make_examples, make_late_examples, size_prior  # noqa: E402
from devtools.production import ProductionProfile  # noqa: E402
from pvdetect import ENERGY_DEV, ENERGY_UNLABELLED, LABELS_DEV  # noqa: E402
from pvdetect.features import customer_features, extract_features  # noqa: E402
from pvdetect.io import customers_only, load_consumption, read_labels, read_table  # noqa: E402
from pvdetect.model import BaselineModel  # noqa: E402

GRID = [(1.0, (8.5, 15.0), 0.4), (0.8, (8.5, 15.0), 0.4), (1.0, (9.0, 14.5), 0.4), (1.0, (8.5, 15.0), None), (0.8, (9.0, 14.5), None)]


def variant(model: BaselineModel, **params) -> BaselineModel:
    m = replace(model, **params)
    m._cap, m._weather = model._cap, model._weather
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--pseudo", default="outputs/pseudo_labels.csv")
    a = ap.parse_args()
    d = Path(a.data_dir)
    model = BaselineModel.load("models/baseline.json")
    dev, _ = load_consumption(d / ENERGY_DEV, a.cache)
    labels = read_labels(d / LABELS_DEV)
    unl, _ = load_consumption(d / ENERGY_UNLABELLED, a.cache)
    unl, _ = customers_only(unl)
    pl = read_table(a.pseudo)
    prof = ProductionProfile.load(Path(a.cache) / "production_profile.npz")
    mu, sd = size_prior(dev, labels)
    hosts = [unl[c] for c in pl.loc[pl["group"] == "non_pv", "customer_id"] if c in unl]
    cols = extract_features({}).columns.tolist()
    tables = {}
    for name, gen in (("synthetic plain", make_examples(hosts, prof, mu, sd, draws=1, seed=505)),
                      ("synthetic varied", make_examples(hosts, prof, mu, sd, draws=1, seed=606, vary=True)),
                      ("synthetic late", make_late_examples(hosts, prof, mu, sd, seed=707))):
        f = pd.DataFrame([customer_features(s, bell=model.bell, weather=model.weather) for s, _ in gen]).reindex(columns=cols)
        f["preexisting"], f["ratio_preexisting"] = f["preexisting"].eq(True), f["ratio_preexisting"].eq(True)
        tables[name] = f
    tables["dev PV"] = extract_features(dev, bell=model.bell, weather=model.weather).merge(labels, on="customer_id").query("pv_installed")
    unl_f = extract_features(unl, bell=model.bell, weather=model.weather)
    tables["unlabelled"] = unl_f

    off = variant(model, schedule_dst_h=None)
    base = {n: off.predict(t)["pv_installed"].to_numpy() for n, t in tables.items()}
    print("detections without the guard: " + " | ".join(f"{n} {b.sum()}" for n, b in base.items()))
    print("\nremoved by the guard (dst shift >= h | centroid outside range, unless weather r >= keep):")
    for dst, rng, keep in GRID:
        m = variant(model, schedule_dst_h=dst, schedule_centroid_h=rng, schedule_keep_weather_r=keep if keep is not None else np.inf)
        cells = [f"{n} {int((base[n] & ~m.predict(t)['pv_installed'].to_numpy()).sum())}" for n, t in tables.items()]
        print(f"   dst {dst} h, centroid {rng[0]}-{rng[1]} UTC, keep r {keep}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
