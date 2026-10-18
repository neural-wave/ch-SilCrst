"""Capacity model: gradient boosting on label-free features, trained on log(kWp).

Trained on the synthetic set (PV injected into real non-PV series) plus, optionally, the
real labels; a single log-offset is then calibrated on the real labelled PV customers,
because the synthetic production scale is nominal.

CatBoost with its default settings replaced scikit-learn's HistGradientBoostingRegressor:
leave-one-out P 0.458 vs 0.418, mean of 5 synthetic sets with a leak-free size prior.
"hgb" keeps the old regressor for comparisons.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

CAPACITY_FEATURES = (
    "log_load_dark", "w90", "z_mid_post", "mid_drop_frac", "log_mid_drop",
    "log_daily_before", "rate_after_summer", "ratio_delta", "preexisting",
    "post_days", "post_summer_days",  # how much data the other features rest on
    "log_sun_drop",  # sunny-day reduction in the first year: the self-consumed PV output
)


def capacity_matrix(df: pd.DataFrame, features=CAPACITY_FEATURES) -> np.ndarray:
    daily = pd.to_numeric(df["daily_kwh_before"], errors="coerce")
    cols = {c: pd.to_numeric(df[c], errors="coerce") for c in CAPACITY_FEATURES if c in df}
    cols["log_daily_before"] = np.log(daily.where(daily > 0))
    cols["preexisting"] = df["preexisting"].eq(True).astype(float)
    if "post_sun_drop_kwh" in df:  # optional feature
        drop = pd.to_numeric(df["post_sun_drop_kwh"], errors="coerce")
        cols["log_sun_drop"] = np.log(drop.where(drop > 0))
    return pd.DataFrame(cols).reindex(columns=list(features)).to_numpy(float)  # missing -> NaN


class CapacityModel:
    def __init__(self, seed: int = 0, features: tuple = CAPACITY_FEATURES, regressor: str = "catboost"):
        self.seed = seed
        self.features = tuple(features)
        self.regressor = regressor
        self.gbm: CatBoostRegressor | HistGradientBoostingRegressor | None = None
        self.log_offset = 0.0

    def _new_regressor(self):
        if self.regressor == "hgb":
            return HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                l2_regularization=1.0, random_state=self.seed,
            )
        # Default settings (1000 trees, depth 6, learning rate chosen by CatBoost); no files written.
        return CatBoostRegressor(random_seed=self.seed, verbose=0, allow_writing_files=False)

    def fit(self, train: pd.DataFrame, calibrate_on: pd.DataFrame | None = None) -> "CapacityModel":
        y = np.log(pd.to_numeric(train["pv_capacity_kwp"], errors="coerce").to_numpy(float))
        X = capacity_matrix(train, self.features)
        ok = np.isfinite(y)
        self.gbm = self._new_regressor().fit(X[ok], y[ok])
        self.log_offset = 0.0
        if calibrate_on is not None and len(calibrate_on):
            true = np.log(calibrate_on["pv_capacity_kwp"].to_numpy(float))
            self.log_offset = float(np.nanmedian(true - self.gbm.predict(capacity_matrix(calibrate_on, self.features))))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.gbm.predict(capacity_matrix(df, self.features)) + self.log_offset)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> "CapacityModel | CapacityEnsemble":
        return pickle.loads(Path(path).read_bytes())


class CapacityEnsemble:
    """Geometric mean of capacity models trained on different synthetic sets.

    The synthetic draw alone moves the leave-one-out P of one model by up to 0.1 on the 22 labelled
    PV; the mean of the models of 5 draws is steadier (and, with log_sun_drop, better)."""

    def __init__(self, models: list[CapacityModel]):
        self.models = list(models)

    @classmethod
    def fit(cls, synthetic_sets: list[pd.DataFrame], real: pd.DataFrame, features: tuple = CAPACITY_FEATURES) -> "CapacityEnsemble":
        """One model per synthetic set (seed = its position), each on that set plus the real PV,
        each with its own log offset calibrated on the real PV."""
        return cls([CapacityModel(seed=k, features=features).fit(pd.concat([aug, real], ignore_index=True), calibrate_on=real)
                    for k, aug in enumerate(synthetic_sets)])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(np.mean([np.log(m.predict(df)) for m in self.models], axis=0))

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))
