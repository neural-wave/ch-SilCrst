"""Build the data-derived PV production profile used for synthetic injection.

Usage: python scripts/build_production.py DATA_DIR [--cache cache] [--pseudo outputs/pseudo_labels.csv]

Needs scripts/pseudo_label.py to have run. Prints physical sanity checks only
(time of peak, production window, monthly sunniness), no customer data.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from devtools.production import build_profile, month_of  # noqa: E402
from pvdetect import ENERGY_DEV, ENERGY_UNLABELLED, LABELS_DEV  # noqa: E402
from pvdetect.features import extract_features  # noqa: E402
from pvdetect.io import load_consumption, read_labels, read_table  # noqa: E402

PROFILE = "production_profile.npz"


def day_index(s, date) -> int:
    return int(np.datetime64(pd.Timestamp(date).date(), "D").astype(int)) - s.day0


def is_uncensored(f: pd.DataFrame) -> pd.Series:
    return (f["z_mid_post"] < 0.2) & f["mid_drop_frac"].between(0.05, 0.8)


def hhmm(slot: int) -> str:
    return f"{slot // 4:02d}:{slot % 4 * 15:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--pseudo", default="outputs/pseudo_labels.csv")
    a = ap.parse_args()
    d = Path(a.data_dir)

    dev, _ = load_consumption(d / ENERGY_DEV, a.cache)
    unl, _ = load_consumption(d / ENERGY_UNLABELLED, a.cache)
    labels = read_labels(d / LABELS_DEV).set_index("customer_id")
    dev_feats = extract_features(dev).set_index("customer_id")
    pl = read_table(a.pseudo).set_index("customer_id")

    pv, unc = [], []
    for cid, row in labels[labels["pv_installed"]].iterrows():
        s = dev[cid]
        tau = max(day_index(s, row["pv_installation_date"]), 0)
        pv.append((s, tau))
        if is_uncensored(dev_feats.loc[[cid]]).iloc[0] and tau > 0:
            unc.append((s, tau))
    for cid, row in pl[pl["group"] == "pv"].iterrows():
        s = unl[cid]
        date = row["tau_date"] if not row["preexisting"] and pd.notna(row["tau_date"]) else row["first_date"]
        if pd.isna(date):
            continue
        tau = max(day_index(s, date), 0)
        pv.append((s, tau))
        if row["uncensored"] and tau > 0:
            unc.append((s, tau))

    prof = build_profile(pv, unc)
    prof.save(Path(a.cache) / PROFILE)

    days = np.arange(prof.day0, prof.day0 + prof.sunniness.size)
    months = month_of(days)
    s = pd.Series(prof.sunniness)
    print(f"PV customers: {len(pv)} | uncensored PV customers: {len(unc)}")
    print(f"sunniness: {np.isfinite(prof.sunniness).mean():.0%} of days covered (median {int(np.median(prof.n_sun))} customers/day) "
          f"| lag-1 autocorr {s.autocorr(1):.2f}")
    t0, g = prof.params[:2]
    print(f"smooth bell: peak at {int(t0):02d}:{int(round(t0 % 1 * 60)):02d} UTC, exponent {g:.1f}")
    print("month  sunniness  customers  raw window(h)  smooth window(h)  window(UTC)   clear-day energy (rel.)")
    shape = prof.shape
    energy = shape.sum(axis=1)
    for m in range(1, 13):
        f = prof.fits[m - 1]
        on = np.flatnonzero(shape[m - 1] > 0)
        win = f"{hhmm(on[0])}-{hhmm(on[-1] + 1)}" if on.size else "-"
        print(f"{m:5d}  {np.nanmean(prof.sunniness[months == m]):9.2f}  {f['n']:9d}  {f.get('D', np.nan):13.2f}  "
              f"{on.size / 4:16.2f}  {win:>12}  {energy[m - 1] / max(energy.max(), 1e-9):10.2f}")
    print(f"profile -> {Path(a.cache) / PROFILE}  (AIL-derived, git-ignored)")


if __name__ == "__main__":
    main()
