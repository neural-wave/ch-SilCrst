"""Compare capacity estimators on the real labelled PV customers.

Usage: python scripts/eval_capacity.py DATA_DIR [--cache cache]

A: current linear model fitted on the real labels (leave-one-out)
B: gradient boosting trained on the synthetic set only, plus a log-offset calibrated on the
   real labels (leave-one-out)
C: gradient boosting trained on synthetic + real (leave-one-out)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pvdetect import ENERGY_DEV, LABELS_DEV  # noqa: E402
from pvdetect.capacity import CAPACITY_FEATURES, CapacityModel  # noqa: E402
from pvdetect.features import extract_features  # noqa: E402
from pvdetect.io import load_consumption, read_labels, read_table  # noqa: E402
from pvdetect.model import BaselineModel  # noqa: E402


def p_score(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    ape = np.abs(pred - true) / true
    return float(np.mean(np.clip(1 - ape / 0.5, 0, None))), float(np.median(ape))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    d = Path(a.data_dir)

    dev, _ = load_consumption(d / ENERGY_DEV, a.cache)
    labels = read_labels(d / LABELS_DEV)
    feats = extract_features(dev)
    real = feats.merge(labels, on="customer_id")
    pv = real[real["pv_installed"]].reset_index(drop=True)
    true = pv["pv_capacity_kwp"].to_numpy(float)
    aug = read_table(Path(a.cache) / "augmented.csv")

    preds = {k: np.zeros(len(pv)) for k in "ABC"}
    for i in range(len(pv)):
        train_lab = labels[labels["customer_id"] != pv.loc[i, "customer_id"]]
        held = pv.iloc[[i]]
        preds["A"][i] = BaselineModel().fit(feats, train_lab).predict_capacity(held)[0]
        rest = pv.drop(index=i)
        preds["B"][i] = CapacityModel().fit(aug, calibrate_on=rest).predict(held)[0]
        preds["C"][i] = CapacityModel().fit(pd.concat([aug, rest]), calibrate_on=rest).predict(held)[0]

    print(f"real PV customers: {len(pv)} | synthetic examples: {len(aug)} | features: {', '.join(CAPACITY_FEATURES)}")
    for k, name in [("A", "linear, real labels only"), ("B", "boosting, synthetic + calibration"),
                    ("C", "boosting, synthetic + real")]:
        p, ape = p_score(preds[k], true)
        print(f"  {k}: {name:<36} P {p:.3f} | median APE {ape:.0%} | within 25%: {(np.abs(preds[k] - true) / true <= .25).sum()}/{len(pv)}")
    p, ape = p_score(CapacityModel().fit(aug).predict(pv), true)
    print(f"  synthetic only, no calibration:            P {p:.3f} | median APE {ape:.0%}")


if __name__ == "__main__":
    main()
