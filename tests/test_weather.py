"""Weather-coherence check, on made-up series (never AIL data)."""
import numpy as np
import pandas as pd

from pvdetect.features import day_stats, extract_features
from pvdetect.model import BaselineModel
from pvdetect.weather import WeatherIndex, _easter, daily_signal, holiday_mask

DAY0 = int(np.datetime64("2023-01-01", "D").astype(int))
N = 730


def weather(seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros(N)
    for d in range(1, N):
        w[d] = 0.5 * w[d - 1] + rng.normal()
    return w


def series(midday_drop, seed):
    rng = np.random.default_rng(seed)
    x = 1.0 + 0.05 * rng.normal(size=(N, 96))
    x[:, 36:56] = np.clip(x[:, 36:56] - midday_drop[:, None], 0, None)
    return x.astype(np.float32)


def test_easter_and_holidays():
    assert str(_easter(2024)) == "2024-03-31" and str(_easter(2025)) == "2025-04-20"
    days = np.array([np.datetime64(d, "D").astype(int) for d in ("2025-08-01", "2025-04-21", "2025-08-04")])
    assert holiday_mask(days).tolist() == [True, True, False]


def test_pv_follows_the_weather_and_a_schedule_does_not():
    w = weather()
    days = np.arange(DAY0, DAY0 + N)
    index = WeatherIndex(DAY0, w[None, :], np.array([50]))
    sun = 1 / (1 + np.exp(-w))  # sunnier days -> bigger midday drop
    pv = series(0.9 * sun, seed=1)
    weekday = (days + 3) % 7 < 5
    rng = np.random.default_rng(2)
    schedule = series(np.where(weekday & (rng.random(N) < 0.7), 0.9, 0.0), seed=3)
    r = [index.coherence(daily_signal(x, day_stats(x).valid, days, 0), DAY0) for x in (pv, schedule)]
    assert r[0] > 0.5 and abs(r[1]) < 0.1
    assert daily_signal(pv, day_stats(pv).valid, days, N - 50) is None  # too few days: no verdict


def test_veto_only_main_rule_detections_without_summer_evidence():
    f = extract_features({}).reindex(range(6))
    f["customer_id"] = [f"c{i}" for i in range(6)]
    f["rate60_max"] = 0.5
    f["dark_zero_share"] = 0.0
    f["preexisting"] = f["ratio_preexisting"] = False
    f["first_date"] = "2023-01-01"
    f["tau_date"] = "2023-06-01"
    f["weather_r"] = [0.02, 0.02, np.nan, 0.5, 0.02, 0.02]
    f["rate_after_summer"] = [0.1, 0.8, 0.1, 0.1, np.nan, 0.1]
    f["weather_r_width"] = [0.0, 0.0, 0.0, 0.0, np.nan, 0.4]  # last: PV hidden at midday by a business load
    pv = BaselineModel(threshold=0.125).predict(f)["pv_installed"].tolist()
    assert pv == [False, True, True, True, False, True]
    assert BaselineModel(threshold=0.125, weather_min_r=None).predict(f)["pv_installed"].all()


def test_seasonal_exemption_and_summer_veto():
    f = extract_features({}).reindex(range(4))
    f["customer_id"] = [f"c{i}" for i in range(4)]
    f["rate60_max"], f["dark_zero_share"] = 0.9, 0.0
    f["preexisting"] = f["ratio_preexisting"] = False
    f["first_date"], f["tau_date"] = "2023-01-01", "2023-06-01"
    # timer load all year | giant PV on a winter heating load | heating load | ordinary PV
    f["weather_r"] = [0.02, 0.02, 0.6, 0.6]
    f["weather_r_width"] = 0.0
    f["rate_after_summer"] = [0.9, 0.9, 0.1, 0.9]
    f["rate_after_winter"] = [0.9, 0.2, 0.2, 0.2]
    f["weather_r_summer"] = [0.0, 0.0, 0.0, 0.5]  # the heating load follows the weather only in winter
    f["weather_r_width_summer"] = 0.0
    pv = BaselineModel(threshold=0.125).predict(f)["pv_installed"].tolist()
    assert pv == [False, True, False, True]
    f["rate_after_summer"] = [0.9, 0.1, 0.1, 0.9]  # without summer PV-like days the giant is not exempt
    assert BaselineModel(threshold=0.125).predict(f)["pv_installed"].tolist() == [False, False, False, True]


def test_weak_onset_moves_the_date_and_implies_an_extension():
    from pvdetect.extensions import ExtensionConfig, predict_extensions
    from pvdetect.io import CustomerSeries

    f = extract_features({}).reindex(range(3))
    f["customer_id"] = ["weak", "plain", "no_shape"]
    f["rate60_max"], f["dark_zero_share"] = 0.9, 0.0
    f["preexisting"] = f["ratio_preexisting"] = False
    f["first_date"], f["tau_date"] = "2022-01-01", "2024-03-20"
    f["weather_r"], f["weather_r_width"], f["weather_r_summer"] = 0.7, 0.7, 0.7
    f["pre_weather_r"] = f["pre_weather_r_summer"] = f["pre_weather_r_first"] = [0.5, 0.05, 0.5]
    f["pre_shape_r"] = [0.95, 0.95, 0.2]
    f["pre_sun_drop_kwh"] = 2.0
    model = BaselineModel(threshold=0.125, weak_min_r=0.3, weak_min_shape=0.8, weak_kwp_per_drop=10.0,
                          cap_default_kwp=150.0, cap_clip_kwp=(5.0, 1000.0))
    pred = model.predict(f)
    assert pred["pv_installation_date"].dt.strftime("%Y-%m-%d").tolist() == ["2022-01-01", "2024-03-20", "2024-03-20"]
    assert pred["pv_capacity_kwp"].tolist() == [20.0, 150.0, 150.0]  # 10 kWp per kWh of sunny-day drop
    assert pred["extension_onset"].notna().tolist() == [True, False, False]
    assert pred["extension_onset_kwp"].iloc[0] == 150.0  # the capacity seen after the onset

    day0 = int(np.datetime64("2022-01-01", "D").astype(int))
    data = {c: CustomerSeries(c, day0, np.ones((900, 96), np.float32)) for c in f["customer_id"]}
    ext = predict_extensions(data, pred, None, ExtensionConfig())
    assert ext[["customer_id", "pv_capacity_after_extension_kwp"]].values.tolist() == [["weak", 150.0]]
    assert ext["extension_date"].iloc[0] == pd.Timestamp("2024-03-20")
    assert not BaselineModel(threshold=0.125, weak_min_r=None).predict(f)["extension_onset"].notna().any()


def test_dip_timing_tells_a_local_time_schedule_from_pv():
    from pvdetect.features import _dst_sundays, dip_timing

    day0 = int(np.datetime64("2023-01-01", "D").astype(int))
    days = np.arange(day0, day0 + 730)
    summer_time = np.zeros(days.size, bool)
    for y in (2023, 2024):
        (start, _), (end, _) = _dst_sundays([y])
        summer_time |= (days >= start) & (days < end)
    t = (np.arange(96) + 0.5) / 4
    rng = np.random.default_rng(0)
    office = 1.0 + 0.05 * rng.random((days.size, 96))
    for i, cest in enumerate(summer_time):  # closed 07:00-16:00 local time
        utc0 = 7 - (2 if cest else 1)
        office[i, (t >= utc0) & (t < utc0 + 9)] = 0.2
    bell = np.clip(np.cos(np.pi * (t - 11.7) / 10), 0, None) * (np.abs(t - 11.7) < 5)
    pv = 1.0 + 0.05 * rng.random((days.size, 96)) - 0.9 * bell[None, :]
    out = {}
    for name, x in (("office", office), ("pv", pv)):
        st = day_stats(x.astype(np.float32))
        out[name] = dip_timing(x, st, days, st.valid)
    assert out["office"][1] > 0.8 and abs(out["pv"][1]) < 0.2
    assert abs(out["pv"][0] - 11.7) < 0.3


def test_schedule_veto_needs_two_signs():
    f = extract_features({}).reindex(range(4))
    f["customer_id"] = [f"c{i}" for i in range(4)]
    f["rate60_max"], f["dark_zero_share"] = 0.9, 0.0
    f["preexisting"] = f["ratio_preexisting"] = False
    f["first_date"], f["tau_date"] = "2022-01-01", "2023-06-01"
    f["weather_r"] = [0.2, 0.7, 0.2, np.nan]  # weather evidence keeps the second one
    f["weather_r_width"] = 0.5
    f["dip_dst_shift_h"] = [1.2, 1.2, 0.1, np.nan]
    f["dip_centroid_h"] = [11.0, 11.0, 11.8, 16.0]
    pv = BaselineModel(threshold=0.125).predict(f)["pv_installed"].tolist()
    assert pv == [False, True, True, False]
    assert BaselineModel(threshold=0.125, schedule_dst_h=None).predict(f)["pv_installed"].all()
