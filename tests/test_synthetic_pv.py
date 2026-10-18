"""Synthetic PV components (production profile, injection) on synthetic inputs, never on AIL data."""
import numpy as np
import pandas as pd

from devtools.production import ProductionProfile, fit_bell
from devtools.augment import inject
from pvdetect.capacity import CapacityModel
from pvdetect.io import CustomerSeries
from pvdetect.model import BaselineModel


def test_fit_bell_recovers_peak_time_and_window():
    t = (np.arange(96) + 0.5) / 4
    true = np.where(np.abs(t - 11.5) < 7, np.cos(np.pi * (t - 11.5) / 14), 0) ** 1.5
    y = 0.05 + 0.8 * true + np.random.default_rng(0).normal(0, 0.02, 96)
    bell, fit = fit_bell(y)
    assert abs(fit["t0"] - 11.5) <= 0.25
    assert np.abs(bell - 0.8 * true).max() < 0.06  # (window, exponent) trade off, the curve must match


def test_injection_only_lowers_withdrawal_after_installation():
    prof = ProductionProfile(19000, np.full(800, 0.8), np.array([11.5, 1.2, 12.5, 2.5, 0.0, 0.5]), np.full(800, 50), [])
    x = np.full((400, 96), 0.5, np.float32)
    s = inject(CustomerSeries("h", 19100, x), tau=200, kwp=10.0, prof=prof, rng=np.random.default_rng(0), new_id="h~0")
    assert np.array_equal(s.x[:200], x[:200])
    assert (s.x[200:] <= x[200:]).all() and (s.x[200:] >= 0).all()
    assert s.x[200:, 44:48].mean() < 0.25  # strong midday reduction
    assert np.allclose(s.x[200:, :8], 0.5)  # nothing at night


def test_capacity_model_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    load = rng.normal(0, 1, n)
    kwp = np.exp(3 + 0.8 * load + rng.normal(0, 0.1, n))
    df = pd.DataFrame({"log_load_dark": load, "w90": rng.uniform(5, 14, n), "z_mid_post": rng.uniform(0, 1, n),
                       "mid_drop_frac": np.nan, "log_mid_drop": np.nan, "daily_kwh_before": np.nan,
                       "rate_after_summer": np.nan, "ratio_delta": np.nan, "preexisting": False, "pv_capacity_kwp": kwp})
    m = CapacityModel().fit(df.iloc[:250])
    ape = np.abs(m.predict(df.iloc[250:]) - kwp[250:]) / kwp[250:]
    assert np.median(ape) < 0.15
    m.save(tmp_path / "cap.pkl")
    assert np.allclose(CapacityModel.load(tmp_path / "cap.pkl").predict(df.iloc[250:]), m.predict(df.iloc[250:]))


def test_baseline_model_saves_and_loads_capacity_model(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"log_load_dark": rng.normal(0, 1, 100), "w90": rng.uniform(5, 14, 100), "z_mid_post": 0.5,
                       "mid_drop_frac": np.nan, "log_mid_drop": np.nan, "daily_kwh_before": 100.0,
                       "rate_after_summer": 0.5, "ratio_delta": 0.1, "preexisting": False,
                       "pv_capacity_kwp": rng.uniform(20, 500, 100)})
    model = BaselineModel(cap_clip_kwp=(1.0, 1e4))
    model.save(tmp_path / "plain.json")  # no capacity model -> no pickle
    assert not list(tmp_path.glob("*.pkl")) and BaselineModel.load(tmp_path / "plain.json")._cap is None
    model._cap = CapacityModel().fit(df)
    model.save(tmp_path / "m.json")
    loaded = BaselineModel.load(tmp_path / "m.json")
    assert loaded.capacity_file == "m_capacity.pkl"
    assert np.allclose(loaded.predict_capacity(df), model.predict_capacity(df))


def test_extension_detector_finds_a_later_step():
    from pvdetect.extensions import ExtensionConfig, find_extensions

    rng = np.random.default_rng(3)
    n = 1100
    # Flat 4 kW load with mild noise; clear-ish weather every day.
    x = (1.0 + rng.normal(0, 0.03, (n, 96))).astype(np.float32)
    prof = ProductionProfile(19000, np.full(n + 10, 0.9), np.array([11.5, 1.2, 12.5, 2.5, 0.0, 0.5]), np.full(n + 10, 50), [])
    s = inject(CustomerSeries("h", 19000, x), tau=100, kwp=40.0, prof=prof, rng=rng, new_id="h")
    s = inject(s, tau=600, kwp=40.0, prof=prof, rng=rng, new_id="h")  # doubles the capacity
    events = find_extensions(s, 100, bell=prof.params[:4].tolist(), ecfg=ExtensionConfig())
    assert len(events) == 1
    day, ratio = events[0]
    assert abs(day - 600) <= 30 and 1.4 <= ratio <= 3.0
    assert find_extensions(inject(CustomerSeries("h", 19000, x), 100, 40.0, prof, rng, "h"), 100,
                           bell=prof.params[:4].tolist()) == []


def test_top_k_policy_emits_the_strongest_extensions():
    from pvdetect.extensions import ExtensionConfig, predict_extensions

    rng = np.random.default_rng(5)
    n = 1100
    prof = ProductionProfile(19000, np.full(n + 10, 0.9), np.array([11.5, 1.2, 12.5, 2.5, 0.0, 0.5]), np.full(n + 10, 50), [])
    data, rows = {}, []
    for cid, factor in (("weak", 1.25), ("strong", 2.0), ("none", None)):
        x = (1.0 + rng.normal(0, 0.03, (n, 96))).astype(np.float32)
        s = inject(CustomerSeries(cid, 19000, x), 100, 40.0, prof, rng, cid)
        if factor:
            s = inject(s, 600, 40.0 * (factor - 1), prof, rng, cid)
        data[cid] = s
        rows.append({"customer_id": cid, "pv_installed": True, "pv_installation_date": pd.Timestamp(np.datetime64(19100, "D")),
                     "pv_capacity_kwp": 40.0})
    pred = pd.DataFrame(rows)
    ev = predict_extensions(data, pred, prof.params[:4].tolist(), ExtensionConfig(top_k=1))
    assert ev["customer_id"].tolist() == ["strong"]
    assert abs((ev["extension_date"].iloc[0] - pd.Timestamp(np.datetime64(19600, "D"))).days) <= 30
    assert 1.2 * 40 <= ev["pv_capacity_after_extension_kwp"].iloc[0] <= 3.0 * 40
    assert len(predict_extensions(data, pred, prof.params[:4].tolist(), ExtensionConfig(top_k=5))) <= 5


def test_model_load_ignores_unknown_parameters(tmp_path):
    import json
    BaselineModel().save(tmp_path / "m.json")
    params = json.loads((tmp_path / "m.json").read_text())
    params["parameter_from_another_version"] = 1
    (tmp_path / "m.json").write_text(json.dumps(params))
    assert BaselineModel.load(tmp_path / "m.json").threshold == BaselineModel().threshold
