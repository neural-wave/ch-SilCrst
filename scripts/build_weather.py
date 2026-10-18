"""Build the regional weather index used by the weather-coherence check.

Usage: python scripts/build_weather.py DATA_DIR [--cache cache] [--pseudo outputs/pseudo_labels.csv]
       [--out models/weather_index.npz]

Signals of the confident PV customers of the unlabelled set (after their installation) ->
two weather regions -> mean daily signal per region, without the customers that follow
neither. The output is an aggregate over 60-140 customers per day; it is shipped with the
model because inference needs it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from devtools.weather import build_index  # noqa: E402
from pvdetect import ENERGY_UNLABELLED  # noqa: E402
from pvdetect.features import day_stats  # noqa: E402
from pvdetect.io import customers_only, load_consumption, read_table  # noqa: E402
from pvdetect.weather import daily_signal  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--pseudo", default="outputs/pseudo_labels.csv")
    ap.add_argument("--out", default="models/weather_index.npz")
    a = ap.parse_args()
    unl, _ = load_consumption(Path(a.data_dir) / ENERGY_UNLABELLED, a.cache)
    unl, _ = customers_only(unl)
    pl = read_table(a.pseudo)
    pv = pl[(pl["group"] == "pv") & pl["customer_id"].isin(set(unl))]

    series = [unl[c] for c in pv["customer_id"]]
    day0 = min(s.day0 for s in series)
    n_days = max(s.day0 + s.n_days for s in series) - day0
    rows = []
    for s, r in zip(series, pv.itertuples()):
        tau = 0 if r.preexisting or pd.isna(r.tau_date) else int(np.datetime64(r.tau_date, "D").astype(int)) - s.day0
        z = daily_signal(s.x, day_stats(s.x).valid, s.days(), tau + 30 if tau > 0 else 0)
        if z is not None:
            row = np.full(n_days, np.nan)
            row[s.day0 - day0:s.day0 - day0 + s.n_days] = z
            rows.append(row)
    index, info = build_index(np.stack(rows), day0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    index.save(a.out)
    covered = np.isfinite(index.index).all(axis=0)
    print(f"confident PV customers with a usable signal: {info['customers']} | two-factor groups: {info['group_sizes']}")
    print(f"dropped (follow neither region): {info['dropped_incoherent']} | region members: {info['members']} "
          f"| correlation between the regions' weather: {info['between_regions_r']:.2f}")
    print(f"days covered by every region: {covered.sum()} "
          f"({np.datetime64(day0 + int(np.flatnonzero(covered)[0]), 'D')} .. {np.datetime64(day0 + int(np.flatnonzero(covered)[-1]), 'D')})")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
