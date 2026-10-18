"""Calibrate the weather-coherence check.

Usage: python scripts/calibrate_weather.py DATA_DIR [--cache cache] [--pseudo outputs/pseudo_labels.csv] [--reuse]

Every rule variant in VARIANTS is applied on the same feature tables:
- cost, real PV wrongly removed: synthetic PV injected into confident non-PV series (fresh
  seeds, plain and varied plants, two weather sequences: the production model's sunniness,
  a median over both regions, pessimistic; and the main region's day-to-day weather), the
  22 labelled PV of the dev set;
- benefit, false detections removed: the number of unlabelled detections removed.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from devtools.augment import make_examples, midday_load_kw, size_prior  # noqa: E402
from devtools.production import ProductionProfile  # noqa: E402
from pvdetect import ENERGY_DEV, ENERGY_UNLABELLED, LABELS_DEV  # noqa: E402
from pvdetect.features import customer_features, extract_features  # noqa: E402
from pvdetect.io import customers_only, load_consumption, read_labels, read_table  # noqa: E402
from pvdetect.model import BaselineModel  # noqa: E402

OFF = dict(weather_min_r=None)
VARIANTS = {
    "no check": OFF,
    "v1: midday < 0.10, summer exemption": dict(weather_rescue_r=None, weather_exempt_season_gap=None, weather_summer_min_r=None),
    "v2: v1 + width rescue 0.15 (shipped)": dict(weather_exempt_season_gap=None, weather_summer_min_r=None),
    "v2 without summer exemption": dict(weather_summer_exempt=np.inf, weather_exempt_season_gap=None, weather_summer_min_r=None),
    "v2, exemption needs summer - winter >= 0.1": dict(weather_exempt_season_gap=0.1, weather_summer_min_r=None),
    "v2, exemption needs summer - winter >= 0.2": dict(weather_exempt_season_gap=0.2, weather_summer_min_r=None),
    "v2, exemption needs summer - winter >= 0.3": dict(weather_exempt_season_gap=0.3, weather_summer_min_r=None),
    "v2 + summer veto 0.05": dict(weather_exempt_season_gap=None, weather_summer_min_r=0.05),
    "v2 + summer veto 0.10": dict(weather_exempt_season_gap=None, weather_summer_min_r=0.10),
    "v2 + summer veto 0.15": dict(weather_exempt_season_gap=None, weather_summer_min_r=0.15),
    "v2, gap 0.2 + summer veto 0.10": dict(weather_exempt_season_gap=0.2, weather_summer_min_r=0.10),
    "v2, gap 0.2 + summer veto 0.15": dict(weather_exempt_season_gap=0.2, weather_summer_min_r=0.15),
    "v2 no exemption + summer veto 0.03": dict(weather_summer_exempt=np.inf, weather_summer_min_r=0.03),
    "v2 no exemption + summer veto 0.05": dict(weather_summer_exempt=np.inf, weather_summer_min_r=0.05),
    "v2 no exemption + summer veto 0.10": dict(weather_summer_exempt=np.inf, weather_summer_min_r=0.10),
    "v2, gap 0.5": dict(weather_exempt_season_gap=0.5, weather_summer_min_r=None),
    "v3: v2, gap 0.5 + summer veto 0.05": dict(weather_exempt_season_gap=0.5, weather_summer_min_r=0.05),
}


def variant(model: BaselineModel, params: dict) -> BaselineModel:
    m = replace(model, **params)
    m._cap, m._weather = model._cap, model._weather
    return m


def regional_profile(prof: ProductionProfile, model: BaselineModel) -> ProductionProfile:
    """Same monthly distribution of sunniness, but in the day order of the main region's weather."""
    days = np.arange(prof.day0, prof.day0 + prof.sunniness.size)
    w = model.weather.aligned(prof.day0, days.size)[0]
    sun = prof.sunniness.copy()
    month = days.astype("datetime64[D]").astype("datetime64[M]").astype(int)
    for m in np.unique(month):
        sel = np.flatnonzero((month == m) & np.isfinite(w) & np.isfinite(sun))
        if sel.size >= 5:
            sun[sel[np.argsort(w[sel])]] = np.sort(sun[sel])
    return replace(prof, sunniness=sun)


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=list(dict.fromkeys(extract_features({}).columns.tolist() + df.columns.tolist())))
    df["preexisting"] = df["preexisting"].eq(True)
    df["ratio_preexisting"] = df["ratio_preexisting"].eq(True)
    return df


def build_tables(a, model: BaselineModel) -> dict:
    d = Path(a.data_dir)
    dev, _ = load_consumption(d / ENERGY_DEV, a.cache)
    labels = read_labels(d / LABELS_DEV)
    unl, _ = load_consumption(d / ENERGY_UNLABELLED, a.cache)
    unl, _ = customers_only(unl)
    pl = read_table(a.pseudo)
    prof = ProductionProfile.load(Path(a.cache) / "production_profile.npz")
    mu, sd = size_prior(dev, labels)
    hosts = [unl[c] for c in pl.loc[pl["group"] == "non_pv", "customer_id"] if c in unl]
    load = {h.customer_id: midday_load_kw(h) for h in hosts}
    rows = []
    for wname, p in (("mixed", prof), ("regional", regional_profile(prof, model))):
        for vary in (False, True):
            for s, lab in make_examples(hosts, p, mu, sd, draws=2, seed=101 + vary, vary=vary):
                f = customer_features(s, bell=model.bell, weather=model.weather)
                rows.append(f | {"variant": f"{wname}|{'varied' if vary else 'plain'}",
                                 "size_ratio": lab["pv_capacity_kwp"] / load[lab["host_id"]]})
    tables = {
        "synthetic": tidy(pd.DataFrame(rows)),
        "dev": extract_features(dev, bell=model.bell, weather=model.weather).merge(labels, on="customer_id"),
        "unlabelled": extract_features(unl, bell=model.bell, weather=model.weather),
    }
    for name, t in tables.items():
        t.to_csv(Path(a.cache) / f"weather_calib_{name}.csv", index=False)  # AIL-derived, git-ignored
    return tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--pseudo", default="outputs/pseudo_labels.csv")
    ap.add_argument("--reuse", action="store_true", help="reuse the feature tables of the previous run")
    a = ap.parse_args()
    model = BaselineModel.load("models/baseline.json")
    assert model.weather is not None, "run scripts/build_weather.py first"
    if a.reuse:
        tables = {n: tidy(read_table(Path(a.cache) / f"weather_calib_{n}.csv")) for n in ("synthetic", "dev", "unlabelled")}
    else:
        tables = build_tables(a, model)
    syn, dev, unl = tables["synthetic"], tables["dev"], tables["unlabelled"]

    base = {n: variant(model, OFF).predict(t)["pv_installed"].to_numpy() for n, t in tables.items()}
    giants = syn["size_ratio"].to_numpy(float) >= 10
    print(f"synthetic PV detected without the check: {base['synthetic'].sum()} (giants >= 10x the load: "
          f"{(base['synthetic'] & giants).sum()}) | dev PV: {base['dev'].sum()} | unlabelled detections: {base['unlabelled'].sum()}")
    print("\n                                          | cost: real PV removed        | benefit")
    print("variant                                   | synthetic (giants)  dev      | unlabelled")
    for name, params in VARIANTS.items():
        m = variant(model, params)
        rm = {n: base[n] & ~m.predict(t)["pv_installed"].to_numpy() for n, t in tables.items()}
        line = (f"{name:<42}| {rm['synthetic'].sum():3d}/{base['synthetic'].sum()} ({(rm['synthetic'] & giants).sum():2d})"
                f"      {rm['dev'].sum():2d}/22")
        print(line + f"    | {rm['unlabelled'].sum():3d}")

    q = lambda v: " ".join(f"{p}%:{np.nanpercentile(v, p):+.2f}" for p in (1, 5, 50))
    det = base["synthetic"]
    print(f"\nsummer coherence, synthetic PV detected: midday {q(syn.loc[det, 'weather_r_summer'])} | "
          f"width {q(syn.loc[det, 'weather_r_width_summer'])}")
    print(f"summer coherence, dev PV: midday {q(dev.loc[dev['pv_installed'], 'weather_r_summer'])} | "
          f"width {q(dev.loc[dev['pv_installed'], 'weather_r_width_summer'])}")
    print(f"summer - winter PV-like rate: synthetic PV {q(syn.loc[det, 'rate_after_summer'] - syn.loc[det, 'rate_after_winter'])} | "
          f"dev PV {q(dev.loc[dev['pv_installed'], 'rate_after_summer'] - dev.loc[dev['pv_installed'], 'rate_after_winter'])}")


if __name__ == "__main__":
    main()
