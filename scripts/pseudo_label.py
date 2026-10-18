"""Pseudo-label the unlabelled series with the baseline fitted on the development labels.

Usage: python scripts/pseudo_label.py DATA_DIR [--cache cache] [--out outputs/pseudo_labels.csv]

Groups every unlabelled customer as confident PV, confident non-PV or uncertain, and
flags PV customers whose midday withdrawal stays above standby ("uncensored": the PV
output is fully self-consumed, so the drop in withdrawal measures it directly).
Prints the size of each group; the per-customer file holds AIL data (git-ignored).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pvdetect import ENERGY_DEV, ENERGY_UNLABELLED, LABELS_DEV  # noqa: E402
from pvdetect.features import extract_features  # noqa: E402
from pvdetect.io import customers_only, load_consumption, read_labels  # noqa: E402
from pvdetect.model import BaselineModel  # noqa: E402


def pseudo_labels(feats: pd.DataFrame, model: BaselineModel, dev_neg_ratio_median: float) -> pd.DataFrame:
    pred = model.predict(feats)
    score = feats["rate60_max"].fillna(-1)
    ratio_t = feats["ratio_t"].fillna(-np.inf)
    group = np.select(
        [score >= 3 * model.threshold,
         (score <= 0.02) & (ratio_t <= dev_neg_ratio_median) & (feats["n_valid_days"] >= 365)],
        ["pv", "non_pv"], "uncertain",
    )
    uncensored = (group == "pv") & (feats["z_mid_post"] < 0.2) & feats["mid_drop_frac"].between(0.05, 0.8)
    return feats.assign(
        group=group,
        uncensored=uncensored,
        pred_pv=pred["pv_installed"].to_numpy(),
        pred_date=pred["pv_installation_date"].to_numpy(),
        pred_kwp=pred["pv_capacity_kwp"].to_numpy(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="outputs/pseudo_labels.csv")
    a = ap.parse_args()
    d = Path(a.data_dir)

    dev, _ = load_consumption(d / ENERGY_DEV, a.cache)
    labels = read_labels(d / LABELS_DEV)
    dev_feats = extract_features(dev)
    model = BaselineModel().fit(dev_feats, labels)
    neg = dev_feats.merge(labels, on="customer_id").query("~pv_installed")
    ratio_med = float(neg["ratio_t"].median())

    unl, _ = load_consumption(d / ENERGY_UNLABELLED, a.cache)
    unl, not_customers = customers_only(unl)  # organizers: keep only ids starting with "POD"
    overlap = len(set(unl) & set(labels["customer_id"]))
    unl = {k: v for k, v in unl.items() if k not in set(labels["customer_id"])}
    pl = pseudo_labels(extract_features(unl), model, ratio_med)

    counts = pl["group"].value_counts()
    pv = pl[pl["group"] == "pv"]
    years = pd.to_datetime(pv.loc[~pv["preexisting"], "tau_date"]).dt.year.value_counts().sort_index()
    print(f"unlabelled customers: {len(pl)} (non-customer series dropped: {not_customers}, "
          f"overlap with dev labels removed: {overlap})")
    print(f"  confident PV: {counts.get('pv', 0)} | confident non-PV: {counts.get('non_pv', 0)} | "
          f"uncertain: {counts.get('uncertain', 0)} | model says PV overall: {int(pl['pred_pv'].sum())}")
    print(f"  confident PV already there at series start: {int(pv['preexisting'].sum())}; "
          f"installed per year: {years.to_dict()}")
    print(f"  confident PV with pre-installation summer data (mid_drop known): {int(pv['mid_drop_frac'].notna().sum())}")
    print(f"  uncensored PV (midday stays above standby): {int(pl['uncensored'].sum())}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pl.to_csv(a.out, index=False)
    print(f"per-customer pseudo-labels -> {a.out}  (AIL data, git-ignored)")


if __name__ == "__main__":
    main()
