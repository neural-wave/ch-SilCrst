"""The numpy ViT capacity port on synthetic series (never on AIL data)."""
from pathlib import Path

import numpy as np
import pandas as pd

from devtools.synthetic import generate
from pvdetect import ENERGY_DEV, LABELS_DEV
from pvdetect.io import read_consumption, read_labels
from pvdetect.vit_capacity import VitCapacity, summer_time, with_vit_capacity

ROOT = Path(__file__).resolve().parent.parent
VIT = VitCapacity.load(ROOT / "models" / "vit_capacity.npz")


def test_summer_time_switches_on_the_last_sundays():
    days = np.array(["2024-03-30", "2024-03-31", "2024-10-26", "2024-10-27"], dtype="datetime64[D]")
    assert summer_time(days).tolist() == [0, 1, 1, 0]


def test_kwp_is_finite_and_scales_with_the_customer(tmp_path):
    generate(tmp_path, n_customers=6, start="2023-01-01", end="2024-12-31", seed=2)
    data, _ = read_consumption(tmp_path / ENERGY_DEV)
    s = next(iter(data.values()))
    kwp = VIT.kwp(s)
    assert np.isfinite(kwp) and kwp > 0
    bigger = VIT.kwp(type(s)(s.customer_id, s.day0, s.x * 2))
    assert 1.5 < bigger / kwp < 2.7  # the size comes from the scale token: twice the load, about twice the kWp


def test_only_the_pv_customers_outside_the_weak_onset_rule_change(tmp_path):
    generate(tmp_path, n_customers=4, start="2023-01-01", end="2024-12-31", seed=3)
    data, _ = read_consumption(tmp_path / ENERGY_DEV)
    ids = sorted(read_labels(tmp_path / LABELS_DEV)["customer_id"])
    pred = pd.DataFrame({
        "customer_id": ids, "pv_installed": [True, True, False, True],
        "pv_installation_date": pd.to_datetime(["2024-01-01"] * 4), "pv_capacity_kwp": [10.0, 20.0, np.nan, 30.0],
        "extension_onset_kwp": [np.nan, np.nan, np.nan, 45.0],  # the last one: weak-onset rule
    })
    out = with_vit_capacity(pred, data, VIT, (1.0, 2000.0))
    assert out["pv_capacity_kwp"][0] == VIT.kwp(data[ids[0]]) and out["pv_capacity_kwp"][1] == VIT.kwp(data[ids[1]])
    assert np.isnan(out["pv_capacity_kwp"][2]) and out["pv_capacity_kwp"][3] == 30.0
    assert out.drop(columns="pv_capacity_kwp").equals(pred.drop(columns="pv_capacity_kwp"))
