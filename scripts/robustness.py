"""How does the final model hold up on shorter or gappier series?

Usage: python scripts/robustness.py DATA_DIR [--cache cache]

The development series are degraded (later start, earlier end, short random windows, random
gaps) and scored with the official metric. Labels follow the window: a PV installed after the
last day counts as no PV. The model saw these customers, so absolute scores are optimistic;
the drop from the full-data score is what matters.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pvdetect import ENERGY_DEV, LABELS_DEV  # noqa: E402
from pvdetect.features import extract_features  # noqa: E402
from pvdetect.io import CustomerSeries, load_consumption, read_labels  # noqa: E402
from pvdetect.metrics import score_main  # noqa: E402
from pvdetect.model import BaselineModel  # noqa: E402


def day(date: str) -> int:
    return int(np.datetime64(date, "D").astype(int))


def window(s: CustomerSeries, start: int | None = None, end: int | None = None) -> CustomerSeries:
    a = 0 if start is None else max(start - s.day0, 0)
    b = s.n_days if end is None else min(end - s.day0 + 1, s.n_days)
    return CustomerSeries(s.customer_id, s.day0 + a, s.x[a:b].copy()) if b > a else CustomerSeries(s.customer_id, 0, s.x[:0])


def with_gaps(s: CustomerSeries, rng: np.random.Generator, day_share: float = 0.15, slot_share: float = 0.02) -> CustomerSeries:
    x = s.x.copy()
    target = int(day_share * s.n_days)
    while target > 0 and s.n_days > 30:
        length = int(min(rng.integers(3, 31), target))
        a = int(rng.integers(0, s.n_days - length))
        x[a : a + length] = np.nan
        target -= length
    x[rng.random(x.shape) < slot_share] = np.nan
    return CustomerSeries(s.customer_id, s.day0, x)


def scenarios(data: dict, rng: np.random.Generator) -> dict:
    out = {
        "full data": data,
        "start 2023-01-01": {c: window(s, start=day("2023-01-01")) for c, s in data.items()},
        "start 2024-01-01": {c: window(s, start=day("2024-01-01")) for c, s in data.items()},
        "end 2024-12-31": {c: window(s, end=day("2024-12-31")) for c, s in data.items()},
    }
    short = {}
    for c, s in data.items():
        if s.n_days > 548:
            a = s.day0 + int(rng.integers(0, s.n_days - 548))
            short[c] = window(s, start=a, end=a + 547)
        else:
            short[c] = s
    out["random 18-month window"] = short
    out["random gaps (15% days, 2% slots)"] = {c: with_gaps(s, rng) for c, s in data.items()}
    out["start 2024 + gaps"] = {c: with_gaps(window(s, start=day("2024-01-01")), rng) for c, s in data.items()}
    return out


def labels_for(labels: pd.DataFrame, data: dict) -> pd.DataFrame:
    """PV counts only if installed before the last observed day."""
    last = {c: np.datetime64(s.day0 + s.n_days - 1, "D") for c, s in data.items() if s.n_days}
    lab = labels.copy()
    end = pd.to_datetime(lab["customer_id"].map(lambda c: last.get(c)))
    after = lab["pv_installed"] & (lab["pv_installation_date"] > end)
    lab.loc[after, "pv_installed"] = False
    lab.loc[after, ["pv_installation_date", "pv_capacity_kwp"]] = [pd.NaT, np.nan]
    return lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    data, _ = load_consumption(Path(a.data_dir) / ENERGY_DEV, a.cache)
    labels = read_labels(Path(a.data_dir) / LABELS_DEV)
    model = BaselineModel.load("models/baseline.json")
    print("scenario                          | PV | MAIN   | F1    (FP FN) | D     | D inside window | P")
    for name, d in scenarios(data, np.random.default_rng(0)).items():
        d = {c: s for c, s in d.items() if s.n_days > 0}  # a customer with no data left is not in this test
        lab = labels_for(labels[labels["customer_id"].isin(d)], d)
        pred = model.predict(extract_features(d, bell=model.bell, weather=model.weather))
        s = score_main(pred, lab)
        m = lab.merge(pred, on="customer_id", suffixes=("_t", "_p"))
        first = pd.to_datetime(m["customer_id"].map(lambda c: np.datetime64(d[c].day0, "D") if d[c].n_days else np.datetime64("NaT")))
        inside = m["pv_installed_t"] & m["pv_installed_p"] & (m["pv_installation_date_t"] >= first)
        err = (pd.to_datetime(m.loc[inside, "pv_installation_date_p"]) - m.loc[inside, "pv_installation_date_t"]).dt.days.abs()
        d_in = np.clip(1 - err / 30, 0, None).mean() if inside.any() else np.nan
        print(f"{name:<33} | {int(lab['pv_installed'].sum()):2d} | {s['main']:6.2f} | {s['f1']:.3f} ({s['fp']}  {s['fn']}) | "
              f"{s['D']:.3f} | {d_in:.3f} (n={int(inside.sum())})      | {s['P']:.3f}")


if __name__ == "__main__":
    main()
