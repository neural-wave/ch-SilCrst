"""Tune and evaluate the extension detector (bonus task).

Usage: python scripts/eval_extensions.py DATA_DIR [--cache cache] [--pseudo outputs/pseudo_labels.csv]

1. Synthetic: PV injected into confident non-PV series, a quarter with 1-2 later extensions.
   Thresholds are tuned on half of the hosts and reported on the other half.
2. Real: the development set (one labelled extension) and the share of unlabelled PV
   customers that would get an extension.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from devtools.augment import make_extension_examples, size_prior  # noqa: E402
from devtools.production import ProductionProfile  # noqa: E402
from pvdetect import ENERGY_DEV, ENERGY_UNLABELLED, EXTENSIONS_DEV, LABELS_DEV  # noqa: E402
from pvdetect.extensions import ExtensionConfig, extension_candidates, select_events  # noqa: E402
from pvdetect.features import customer_features, extract_features  # noqa: E402
from pvdetect.io import load_consumption, read_extension_labels, read_labels, read_table  # noqa: E402
from pvdetect.metrics import score_extensions  # noqa: E402
from pvdetect.model import BaselineModel  # noqa: E402

GRID = {"delta_min_h": [0.25, 0.5, 0.75, 1.0, 1.5], "gain_min": [0.05, 0.1, 0.2, 0.3, 0.4], "ratio_min": [1.1, 1.2, 1.3]}


def candidates_for(series: dict, pred: pd.DataFrame, bell) -> dict:
    """customer_id -> (day0, initial kWp, candidates) for customers predicted with PV."""
    out = {}
    for r in pred[pred["pv_installed"]].itertuples(index=False):
        s = series[r.customer_id]
        tau = 0
        if pd.notna(r.pv_installation_date):
            tau = max(int(np.datetime64(pd.Timestamp(r.pv_installation_date).date(), "D").astype(int)) - s.day0, 0)
        out[r.customer_id] = (s.day0, r.pv_capacity_kwp, extension_candidates(s, tau, bell))
    return out


def events_from(cands: dict, ecfg: ExtensionConfig, ids=None) -> pd.DataFrame:
    rows = []
    for cid, (day0, kwp, c) in cands.items():
        if ids is not None and cid not in ids:
            continue
        for t, ratio in select_events(c, ecfg):
            kwp = kwp * ratio
            rows.append({"customer_id": cid, "extension_date": pd.Timestamp(np.datetime64(int(day0 + t), "D")),
                         "pv_capacity_after_extension_kwp": kwp})
    return pd.DataFrame(rows, columns=["customer_id", "extension_date", "pv_capacity_after_extension_kwp"])


def fmt(s: dict) -> str:
    return (f"bonus {s['bonus']:.2f}/10 | F1_ext {s['f1_ext']:.2f} (TP {s['tp']} FP {s['fp']} FN {s['fn']}) "
            f"| D_ext {s['D_ext']:.2f} | P_ext {s['P_ext']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--pseudo", default="outputs/pseudo_labels.csv")
    a = ap.parse_args()
    d = Path(a.data_dir)

    dev, _ = load_consumption(d / ENERGY_DEV, a.cache)
    unl, _ = load_consumption(d / ENERGY_UNLABELLED, a.cache)
    labels = read_labels(d / LABELS_DEV)
    ext_labels = read_extension_labels(d / EXTENSIONS_DEV)
    pl = read_table(a.pseudo)
    prof = ProductionProfile.load(Path(a.cache) / "production_profile.npz")
    bell = prof.params[:4].tolist()
    model = BaselineModel.load("models/baseline.json")

    # 1. Synthetic set.
    mu, sd = size_prior(dev, labels)
    hosts = [unl[c] for c in pl.loc[pl["group"] == "non_pv", "customer_id"]]
    series, rows, true_events = {}, [], []
    for s, lab, ev in make_extension_examples(hosts, prof, mu, sd):
        series[s.customer_id] = s
        rows.append(customer_features(s))
        true_events += ev
    feats = pd.DataFrame(rows).reindex(columns=extract_features({}).columns)
    feats["preexisting"] = feats["preexisting"].eq(True)
    feats["ratio_preexisting"] = feats["ratio_preexisting"].eq(True)
    true_ev = pd.DataFrame(true_events, columns=["customer_id", "extension_date", "pv_capacity_after_extension_kwp"])
    true_ev["extension_date"] = pd.to_datetime(true_ev["extension_date"])
    cands = candidates_for(series, model.predict(feats), bell)

    host_of = {cid: cid.split("~")[0] for cid in series}
    hosts_sorted = sorted(set(host_of.values()))
    half_a = {cid for cid, h in host_of.items() if hosts_sorted.index(h) % 2 == 0}
    half_b = set(series) - half_a
    sub = lambda ev, ids: ev[ev["customer_id"].isin(ids)]
    n_ext_customers = true_ev["customer_id"].nunique()
    print(f"synthetic: {len(series)} PV examples, {n_ext_customers} with extensions, {len(true_ev)} extension events")

    best = None
    for vals in itertools.product(*GRID.values()):
        ecfg = ExtensionConfig(**dict(zip(GRID, vals)))
        s = score_extensions(events_from(cands, ecfg, half_a), sub(true_ev, half_a))
        if best is None or s["bonus"] > best[1]["bonus"]:
            best = (ecfg, s)
    ecfg = best[0]
    print(f"  tuned on half A: delta_min {ecfg.delta_min_h} h, gain_min {ecfg.gain_min}, ratio_min {ecfg.ratio_min}")
    print(f"  half A (tuning):  {fmt(best[1])}")
    pred_b = events_from(cands, ecfg, half_b)
    print(f"  half B (held out): {fmt(score_extensions(pred_b, sub(true_ev, half_b)))}")
    no_ext_b = half_b - set(true_ev["customer_id"])
    print(f"  half B customers without extension that get one: {pred_b['customer_id'].isin(no_ext_b).sum()} events "
          f"on {pred_b.loc[pred_b['customer_id'].isin(no_ext_b), 'customer_id'].nunique()}/{len(no_ext_b)} customers")
    default = ExtensionConfig()
    print(f"  half B with the shipped defaults: {fmt(score_extensions(events_from(cands, default, half_b), sub(true_ev, half_b)))}")

    # 2. Real data.
    dev_feats = extract_features(dev)
    dev_cands = candidates_for(dev, model.predict(dev_feats), bell)
    dev_pred = events_from(dev_cands, ecfg)
    s = score_extensions(dev_pred, ext_labels)
    matched = dev_pred.merge(ext_labels, on="customer_id", suffixes=("_p", "_t"))
    err = (pd.to_datetime(matched["extension_date_p"]) - pd.to_datetime(matched["extension_date_t"])).dt.days.abs()
    print(f"\ndev (real, {len(ext_labels)} labelled extension): {fmt(s)}")
    print(f"  predicted events: {len(dev_pred)} on {dev_pred['customer_id'].nunique()} customers | "
          f"closest date error on the labelled customer: {f'{err.min()} d' if len(err) else 'no event predicted'}")
    lab_cid = set(ext_labels["customer_id"])
    for name, cfg_ in (("default", ExtensionConfig()), ("permissive", ExtensionConfig(delta_min_h=0, gain_min=0, ratio_min=1.0))):
        p_ = events_from({c: v for c, v in dev_cands.items() if c in lab_cid}, cfg_)
        e_ = (pd.to_datetime(p_["extension_date"]).values.astype("datetime64[D]").astype(int)[:, None]
              - pd.to_datetime(ext_labels["extension_date"]).values.astype("datetime64[D]").astype(int)[None, :])
        print(f"  labelled customer, {name} thresholds: {len(p_)} events, closest |error| "
              f"{np.abs(e_).min() if e_.size else 'n/a'} d")
    pv_ids = set(pl.loc[pl["group"] == "pv", "customer_id"])
    unl_feats = extract_features({c: unl[c] for c in pv_ids})
    unl_pred = events_from(candidates_for(unl, model.predict(unl_feats), bell), ecfg)
    print(f"unlabelled confident PV: {unl_pred['customer_id'].nunique()}/{len(pv_ids)} customers get an extension "
          f"({len(unl_pred)} events)")
    print(f"\nchosen thresholds: {ecfg}")


if __name__ == "__main__":
    main()
