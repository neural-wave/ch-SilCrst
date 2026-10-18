"""Capacity ensemble on made-up features (never AIL data)."""
import numpy as np
import pandas as pd

from pvdetect.capacity import CAPACITY_FEATURES, CapacityEnsemble, capacity_matrix


def frame(n, seed):
    rng = np.random.default_rng(seed)
    load = rng.uniform(0, 3, n)
    df = pd.DataFrame({c: rng.normal(size=n) for c in CAPACITY_FEATURES if c not in ("log_daily_before", "preexisting", "log_sun_drop")})
    df["log_load_dark"] = load
    df["daily_kwh_before"] = np.exp(load) * 96
    df["preexisting"] = False
    df["post_sun_drop_kwh"] = np.exp(load) * 0.05
    df["pv_capacity_kwp"] = np.exp(load + 2 + 0.1 * rng.normal(size=n))
    return df


def test_ensemble_is_the_geometric_mean_and_uses_the_sun_drop():
    sets, real = [frame(200, 1), frame(200, 2)], frame(20, 3)
    ens = CapacityEnsemble.fit(sets, real)
    x = frame(5, 4)
    each = np.array([m.predict(x) for m in ens.models])
    assert np.allclose(ens.predict(x), np.exp(np.log(each).mean(axis=0)))
    assert "log_sun_drop" in ens.models[0].features
    assert np.isfinite(capacity_matrix(x)[:, list(CAPACITY_FEATURES).index("log_sun_drop")]).all()
