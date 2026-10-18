"""Calibrate the weak-onset rule (an extension mistaken for the installation).

Usage: python scripts/calibrate_weak_onset.py DATA_DIR [--cache cache] [--pseudo outputs/pseudo_labels.csv]

Synthetic customers, PV injected into confident non-PV series:
- weak -> strong (devtools.augment.make_weak_strong_examples): a small PV, later extended to a
  large one;
- standard extensions (make_extension_examples) and plain PV (make_examples).
1. Cost: synthetic PV with an inside onset whose date the rule moves; how often it fires on the
   dev PV and on the unlabelled PV.
2. The kWp of the weak PV: weak_kwp_per_drop fitted on half of the weak -> strong hosts, checked
   on the other half.
3. Bonus and main score on test-like sets (11 PV, one customer with one extension and one with
   two), with the extension customers drawn from the standard or the weak -> strong kind.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from devtools.augment import make_examples, make_extension_examples, make_weak_strong_examples, size_prior  # noqa: E402
from devtools.production import ProductionProfile  # noqa: E402
from pvdetect import ENERGY_DEV, ENERGY_UNLABELLED, LABELS_DEV  # noqa: E402
from pvdetect.extensions import ExtensionConfig, extension_candidates  # noqa: E402
from pvdetect.features import customer_features, extract_features  # noqa: E402
from pvdetect.io import customers_only, load_consumption, read_labels, read_table  # noqa: E402
from pvdetect.metrics import score_extensions  # noqa: E402
from pvdetect.model import BaselineModel  # noqa: E402

RULE = dict(weak_min_r=0.30, weak_min_shape=0.8)


def variant(model: BaselineModel, **params) -> BaselineModel:
    m = replace(model, **params)
    m._cap, m._weather = model._cap, model._weather
    return m


def tidy(rows: list[dict]) -> pd.DataFrame:
    cols = extract_features({}).columns.tolist()
    df = pd.DataFrame(rows).reindex(columns=list(dict.fromkeys(cols + list(rows[0]))))
    df["preexisting"] = df["preexisting"].eq(True)
    df["ratio_preexisting"] = df["ratio_preexisting"].eq(True)
    return df


def p_score(pred, true) -> np.ndarray:
    return np.clip(1 - np.abs(pred - true) / true / 0.5, 0, None)


def d_score(pred: pd.Series, true: pd.Series) -> np.ndarray:
    return np.clip(1 - (pd.to_datetime(pred) - pd.to_datetime(true)).dt.days.abs() / 30, 0, None).fillna(0).to_numpy()


def synthetic(a, model: BaselineModel) -> tuple[pd.DataFrame, dict]:
    """Features, truth and extension candidates of every synthetic customer."""
    d = Path(a.data_dir)
    dev, _ = load_consumption(d / ENERGY_DEV, a.cache)
    labels = read_labels(d / LABELS_DEV)
    unl, _ = load_consumption(d / ENERGY_UNLABELLED, a.cache)
    unl, _ = customers_only(unl)
    pl = read_table(a.pseudo)
    prof = ProductionProfile.load(Path(a.cache) / "production_profile.npz")
    mu, sd = size_prior(dev, labels)
    hosts = [unl[c] for c in pl.loc[pl["group"] == "non_pv", "customer_id"] if c in unl]
    series, rows, events = {}, [], []
    gens = [("weak", make_weak_strong_examples(hosts, prof, draws=2, seed=21)),
            ("standard", make_extension_examples(hosts, prof, mu, sd, draws=2, share=0.5, seed=404)),
            ("plain", ((s, lab, []) for s, lab in make_examples(hosts, prof, mu, sd, draws=1, seed=303)))]
    for kind, gen in gens:
        for s, lab, ev in gen:
            series[s.customer_id] = s
            rows.append(customer_features(s, bell=model.bell, weather=model.weather)
                        | {"kind": kind, "n_ext": len(ev), "true_date": lab["pv_installation_date"],
                           "true_kwp": lab["pv_capacity_kwp"], "host": s.customer_id.split("~")[0]})
            events += ev
    feats = tidy(rows)
    true_ev = pd.DataFrame(events, columns=["customer_id", "extension_date", "pv_capacity_after_extension_kwp"])
    true_ev["extension_date"] = pd.to_datetime(true_ev["extension_date"])
    return feats, {"series": series, "events": true_ev}


def candidates(feats: pd.DataFrame, pred: pd.DataFrame, series: dict, bell, ecfg: ExtensionConfig) -> dict:
    """customer_id -> (forced event or None, regular candidates as (gain, day, ratio, kwp_after))."""
    out = {}
    for f, r in zip(feats.itertuples(index=False), pred.itertuples(index=False)):
        if not r.pv_installed:
            continue
        s = series[r.customer_id]
        tau = max(int(np.datetime64(pd.Timestamp(r.pv_installation_date).date(), "D").astype(int)) - s.day0, 0) \
            if pd.notna(r.pv_installation_date) else 0
        cands = extension_candidates(s, tau, bell, ecfg=ecfg)
        forced = None
        if pd.notna(r.extension_onset):
            t = int(np.datetime64(pd.Timestamp(r.extension_onset).date(), "D").astype(int)) - s.day0
            forced = (pd.Timestamp(r.extension_onset), float(r.extension_onset_kwp))
            cands = [c for c in cands if abs(c["day"] - t) >= ecfg.forced_gap_days]
        regular = [(c["gain"], pd.Timestamp(np.datetime64(int(s.day0 + c["day"]), "D")), c["ratio"], r.pv_capacity_kwp * c["ratio"])
                   for c in cands if ecfg.ratio_min <= c["ratio"] <= ecfg.ratio_max]
        out[r.customer_id] = (forced, regular)
    return out


def emit(test: list[str], cands: dict, policy: str, k: int) -> pd.DataFrame:
    rows, pool = [], []
    for cid in test:
        forced, regular = cands.get(cid, (None, []))
        if forced is not None and policy != "ignore forced":
            rows.append((np.inf, cid, forced[0], forced[1]))
        pool += [(g, cid, day, kwp) for g, day, _, kwp in regular]
    pool.sort(key=lambda p: -p[0])
    if policy == "forced within k":
        pool = pool[: max(k - len(rows), 0)]
    else:
        pool = pool[:k]
    return pd.DataFrame([{"customer_id": c, "extension_date": d, "pv_capacity_after_extension_kwp": kw} for _, c, d, kw in rows + pool],
                        columns=["customer_id", "extension_date", "pv_capacity_after_extension_kwp"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--pseudo", default="outputs/pseudo_labels.csv")
    a = ap.parse_args()
    model = BaselineModel.load("models/baseline.json")
    assert model.weather is not None and model.bell is not None
    feats, extra = synthetic(a, model)
    off, on = variant(model, weak_min_r=None), variant(model, **RULE)
    p_off, p_on = off.predict(feats), on.predict(feats)
    kind = feats["kind"].to_numpy()
    first = pd.to_datetime(feats["first_date"])
    true_date = pd.to_datetime(feats["true_date"])
    inside = (true_date >= first).to_numpy()
    fired = (p_on["extension_onset"].notna() & p_on["pv_installed"]).to_numpy()

    # 1. Cost: dates the rule moves where the PV really started inside the series.
    print("== 1. Where the rule fires")
    for kname in ("weak", "standard", "plain"):
        sel = (kind == kname) & p_off["pv_installed"].to_numpy()
        print(f"   synthetic {kname:<8}: fires on {fired[sel].sum()}/{sel.sum()} detected"
              f" | of which the PV started inside the series: {(fired & inside)[sel].sum()}")
    d = Path(a.data_dir)
    dev, _ = load_consumption(d / ENERGY_DEV, a.cache)
    labels = read_labels(d / LABELS_DEV)
    dev_f = extract_features(dev, bell=model.bell, weather=model.weather).merge(labels, on="customer_id")
    dev_on = on.predict(dev_f)
    print(f"   dev: fires on {int(dev_on['extension_onset'].notna().sum())} of {int(dev_on['pv_installed'].sum())} PV predictions")
    unl, _ = load_consumption(d / ENERGY_UNLABELLED, a.cache)
    unl, _ = customers_only(unl)
    unl_f = extract_features(unl, bell=model.bell, weather=model.weather)
    unl_f.to_csv(Path(a.cache) / "weak_calib_unlabelled.csv", index=False)  # AIL-derived, git-ignored
    unl_on = on.predict(unl_f)
    print(f"   unlabelled: fires on {int(unl_on['extension_onset'].notna().sum())} of {int(unl_on['pv_installed'].sum())} PV predictions")

    # 2. kWp of the weak PV.
    hosts = sorted(set(feats["host"]))
    half = feats["host"].map({h: i % 2 for i, h in enumerate(hosts)}).to_numpy()
    drop = pd.to_numeric(feats["pre_sun_drop_kwh"], errors="coerce").to_numpy()
    k0 = feats["true_kwp"].to_numpy(float)
    ok = fired & (kind == "weak") & np.isfinite(drop) & (drop > 0)
    k_fit = float(np.exp(np.median(np.log(k0[ok & (half == 0)] / drop[ok & (half == 0)]))))
    held = ok & (half == 1)
    print(f"\n== 2. kWp of the weak PV: weak_kwp_per_drop = {k_fit:.1f} (fitted on {int((ok & (half == 0)).sum())}) | held out ({held.sum()}): "
          f"P {p_score(k_fit * drop[held], k0[held]).mean():.3f} against {p_score(p_off['pv_capacity_kwp'].to_numpy(float)[held], k0[held]).mean():.3f} "
          f"when the total capacity is given")
    on = variant(model, **RULE, weak_kwp_per_drop=k_fit)
    p_on = on.predict(feats)

    # 3. Test-like sets.
    ecfg = ExtensionConfig(**model.extension)
    c_off = candidates(feats, p_off, extra["series"], model.bell[:4], ecfg)
    c_on = candidates(feats, p_on, extra["series"], model.bell[:4], ecfg)
    ids = feats["customer_id"].to_numpy()
    true_ev = extra["events"]
    n_ext = feats["n_ext"].to_numpy()
    pool_plain = ids[(kind == "plain") & (half == 1)]
    one = {"standard": ids[(kind == "standard") & (n_ext == 1) & (half == 1)], "weak": ids[(kind == "weak") & (half == 1)]}
    two = ids[(kind == "standard") & (n_ext == 2) & (half == 1)]
    idx = {c: i for i, c in enumerate(ids)}
    D_off, D_on = d_score(p_off["pv_installation_date"], true_date), d_score(p_on["pv_installation_date"], true_date)
    P_off = np.nan_to_num(p_score(p_off["pv_capacity_kwp"].to_numpy(float), k0))
    P_on = np.nan_to_num(p_score(p_on["pv_capacity_kwp"].to_numpy(float), k0))
    det_off, det_on = p_off["pv_installed"].to_numpy(), p_on["pv_installed"].to_numpy()
    print("\n== 3. Test-like sets (held-out hosts): 9 plain PV + one customer with 1 extension + one with 2 (standard)")
    rng = np.random.default_rng(0)
    for mix in ("standard", "weak"):
        res = {}
        for _ in range(1500):
            test = list(rng.choice(pool_plain, 9, replace=False)) + [rng.choice(one[mix]), rng.choice(two)]
            truth = true_ev[true_ev["customer_id"].isin(test)]
            ii = [idx[c] for c in test]
            dmain = 15 / 11 * (np.sum((D_on[ii] * det_on[ii]) - (D_off[ii] * det_off[ii])) + np.sum((P_on[ii] * det_on[ii]) - (P_off[ii] * det_off[ii])))
            for name, cands, policy, k in (("rule off, top-2              ", c_off, "ignore forced", 2),
                                           ("rule on, forced only", c_on, "forced + k", 0),
                                           ("rule on, forced + top-2", c_on, "forced + k", 2),
                                           ("rule on, forced + top-1", c_on, "forced + k", 1),
                                           ("rule on, forced within 2", c_on, "forced within k", 2)):
                b = score_extensions(emit(test, cands, policy, k), truth)["bonus"]
                res.setdefault(name, []).append((b, dmain if "on" in name.split(",")[0] else 0.0))
        print(f"   one-extension customer of the {mix} kind:")
        for name, v in res.items():
            v = np.array(v)
            print(f"      {name:<28} bonus {v[:, 0].mean():.2f} (P>0 {np.mean(v[:, 0] > 0):.0%}) | main score change {v[:, 1].mean():+.2f}"
                  f" (5th pct {np.percentile(v[:, 1], 5):+.2f})")


if __name__ == "__main__":
    main()
