"""Baseline model: a few parameters learned on top of the label-free features."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd

from .capacity import CapacityEnsemble, CapacityModel
from .extensions import ExtensionConfig, predict_extensions
from .weather import WeatherIndex

SCORE_FEATURES = ("rate60_max", "step_delta", "rate_after_summer")
CAPACITY_FEATURES = ("log_load_dark", "w90")


def best_threshold(score: np.ndarray, truth: np.ndarray) -> float:
    """Threshold maximising F1 for `score >= thr`; ties go to the widest gap between classes."""
    ok = np.isfinite(score)
    s, t = score[ok], truth[ok].astype(bool)
    u = np.unique(s)
    if u.size < 2 or not t.any():
        return float(u[0]) if u.size else 0.5
    cands = (u[:-1] + u[1:]) / 2
    gaps = np.diff(u)
    best, best_key = cands[0], (-1.0, -1.0)
    for c, gap in zip(cands, gaps):
        p = s >= c
        tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if (f1, gap) > best_key:
            best, best_key = c, (f1, gap)
    return float(best)


@dataclass
class BaselineModel:
    score_feature: str = "rate60_max"
    threshold: float = 0.2
    # Second rule, for PV smaller than the load (withdrawal never below standby): a step in the
    # midday/night ratio whose shape matches PV output on weekdays and weekends. Thresholds are
    # calibrated on ~850 unlabelled customers + synthetic small PV (scripts/calibrate_ratio.py),
    # not fitted on the 40 labels: false alarm rate ~1%.
    use_ratio_rule: bool = True
    ratio_threshold: float = 10.0
    shape_min_corr: float = 0.8
    shape_amp_range: tuple[float, float] = (0.5, 2.0)
    # Date reconciliation: when the ratio step (passing the PV-shape check) comes more than this
    # many days before the first PV-like days, the PV was there earlier but its output stayed
    # below the load (validated on synthetic dates and the dev set).
    reconcile_gap_days: int = 30
    # Dead-meter guard: a meter that is mostly zero at night is not a PV customer (PV customers always have a
    # standby load); such meters produced absurd detections (up to 1672 kWp).
    max_dark_zero_share: float = 0.4
    # Weather coherence: a main-rule detection whose midday withdrawal
    # does not follow the regional weather (correlation < weather_min_r) is a fixed-schedule load, not
    # PV, unless the width of its below-standby window follows the weather (>= weather_rescue_r: PV
    # hidden at midday by a daytime business load) or it is clearly seasonal (PV-like on most summer
    # days and far more often in summer than in winter: giant PV on a winter heating load). Calibrated
    # on synthetic PV and the dev set (cost) and on the unlabelled detections it removes (benefit). The index lives in `weather_file` next to the JSON.
    weather_file: str | None = None
    weather_min_r: float | None = 0.10
    weather_summer_exempt: float = 0.5
    weather_rescue_r: float | None = 0.15
    # Summer exemption only with summer - winter PV-like rate >= this (timer and night-only loads are
    # below standby all year); and a second veto for detections that follow the weather in no summer
    # signal (correlation < weather_summer_min_r: heating loads follow it only in winter). None = off.
    weather_exempt_season_gap: float | None = 0.5
    weather_summer_min_r: float | None = 0.05
    # Schedule guard: a main-rule detection whose below-standby dip moves with
    # local time at the DST changes (>= schedule_dst_h) or sits where no PV orientation puts it (centroid
    # outside schedule_centroid_h, UTC) is a schedule, unless the midday withdrawal clearly follows the
    # weather (>= schedule_keep_weather_r): two independent signs. None = off.
    schedule_dst_h: float | None = 1.0
    schedule_centroid_h: tuple[float, float] = (8.5, 15.0)
    schedule_keep_weather_r: float = 0.4
    # Weak PV before the onset: when the midday withdrawal already followed
    # the weather before the PV-like days (all-year, summer and first-year coherence >= weak_min_r)
    # with a PV-shaped sunny-day reduction (>= weak_min_shape), the PV predates the series and the
    # onset is an extension. The initial kWp is weak_kwp_per_drop x the sunny-day reduction (a
    # self-consumed PV); the extension keeps the capacity estimated after the onset. Thresholds set
    # on synthetic weak -> strong extensions. None = off.
    weak_min_r: float | None = 0.30
    weak_min_shape: float = 0.8
    weak_kwp_per_drop: float | None = 27.0
    date_offset_days: int = 0
    cap_coef: list[float] | None = None  # [intercept, *CAPACITY_FEATURES] for log(kWp)
    cap_default_kwp: float = 10.0
    cap_clip_kwp: tuple[float, float] = (1.0, 200.0)
    capacity_file: str | None = None  # boosting capacity model saved next to the JSON
    # Clear-day production parameters [t0, g, D_a, D_b, logA_a, logA_b], derived from the
    # consumption data (devtools/production.py); used by the crossing and extension features.
    bell: list[float] | None = None
    # Extension detector thresholds {"delta_min_h": .., "gain_min": .., "ratio_min": ..}; None = off.
    extension: dict | None = None
    info: dict = field(default_factory=dict)
    _cap: CapacityModel | CapacityEnsemble | None = field(default=None, init=False, repr=False)
    _weather: WeatherIndex | None = field(default=None, init=False, repr=False)

    @property
    def weather(self) -> WeatherIndex | None:
        """Regional weather index for extract_features (None: the coherence check is off)."""
        return self._weather

    # ------------------------------------------------------------ fit
    def fit(self, feats: pd.DataFrame, labels: pd.DataFrame, augmented: pd.DataFrame | list[pd.DataFrame] | None = None) -> "BaselineModel":
        """`augmented`: synthetic labelled examples (scripts/augment.py) for the capacity model; a list of
        synthetic sets gives a CapacityEnsemble (one model per set, geometric mean)."""
        m = feats.merge(labels, on="customer_id", how="inner")
        truth = m["pv_installed"].to_numpy(bool)
        score = m[self.score_feature].to_numpy(float)
        self.threshold = best_threshold(score, truth)

        # Systematic lag between the first PV-like day and the labelled commissioning date.
        pos = m[truth & ~m["preexisting"] & m["tau_date"].notna() & m["pv_installation_date"].notna()]
        lag = (pos["pv_installation_date"] - pd.to_datetime(pos["tau_date"])).dt.days
        lag = lag[lag.abs() <= 120]  # ignore gross change-point failures
        self.date_offset_days = int(np.clip(round(lag.median()), -30, 30)) if len(lag) >= 3 else 0

        # log(kWp) ~ intercept + CAPACITY_FEATURES, small ridge for stability.
        kwp = m["pv_capacity_kwp"].to_numpy(float)
        has = truth & np.isfinite(kwp) & (kwp > 0)
        if has.any():
            self.cap_default_kwp = float(np.median(kwp[has]))
            self.cap_clip_kwp = (float(kwp[has].min() / 2), float(kwp[has].max() * 2))
        X = m[list(CAPACITY_FEATURES)].to_numpy(float)
        rows = has & np.isfinite(X).all(axis=1)
        self.cap_coef = None
        if rows.sum() >= max(8, 3 * (X.shape[1] + 1)):
            A = np.column_stack([np.ones(rows.sum()), X[rows]])
            reg = 1e-2 * rows.sum() * np.diag([0.0] + [1.0] * X.shape[1])
            self.cap_coef = np.linalg.solve(A.T @ A + reg, A.T @ np.log(kwp[rows])).tolist()
        self._cap = None
        if augmented is not None:
            real = m[has]
            if isinstance(augmented, (list, tuple)):
                self._cap = CapacityEnsemble.fit(list(augmented), real)
            else:
                self._cap = CapacityModel().fit(pd.concat([augmented, real], ignore_index=True), calibrate_on=real)
        self.info = {"n_train": int(len(m)), "n_pv": int(truth.sum()), "n_capacity_fit": int(rows.sum()),
                     "n_synthetic": 0 if augmented is None else int(sum(len(a) for a in augmented) if isinstance(augmented, (list, tuple)) else len(augmented)),
                     "n_synthetic_sets": 0 if augmented is None else (len(augmented) if isinstance(augmented, (list, tuple)) else 1)}
        return self

    # ------------------------------------------------------------ predict
    def weather_veto(self, feats: pd.DataFrame) -> np.ndarray:
        """Customers whose midday withdrawal does not follow the weather, that are not PV-like on most
        summer days and whose below-standby window does not follow it either. No verdict (False)
        when the midday coherence is unknown."""
        if self.weather_min_r is None:
            return np.zeros(len(feats), bool)
        r, summer = _num(feats, "weather_r"), _num(feats, "rate_after_summer")
        exempt = summer >= self.weather_summer_exempt
        if self.weather_exempt_season_gap is not None:  # unknown winter rate: summer evidence suffices
            exempt &= ~(summer - _num(feats, "rate_after_winter") < self.weather_exempt_season_gap)
        veto = r < self.weather_min_r
        if self.weather_rescue_r is not None:
            veto &= ~(_num(feats, "weather_r_width") >= self.weather_rescue_r)
        if self.weather_summer_min_r is not None:
            q = self.weather_summer_min_r
            rs, ws = _num(feats, "weather_r_summer"), _num(feats, "weather_r_width_summer")
            veto |= np.isfinite(rs) & ~(rs >= q) & ~(ws >= q)
        return veto & ~exempt

    def schedule_veto(self, feats: pd.DataFrame) -> np.ndarray:
        """Detections whose dip follows local time or sits outside any PV hours, without weather evidence."""
        if self.schedule_dst_h is None:
            return np.zeros(len(feats), bool)
        dst, centroid = _num(feats, "dip_dst_shift_h"), _num(feats, "dip_centroid_h")
        lo, hi = self.schedule_centroid_h
        schedule = (dst >= self.schedule_dst_h) | (centroid < lo) | (centroid > hi)
        return schedule & ~(_num(feats, "weather_r") >= self.schedule_keep_weather_r)

    def weak_onset(self, feats: pd.DataFrame) -> np.ndarray:
        """Detections whose onset follows a weak PV already visible in the weather (see weak_min_r)."""
        if self.weak_min_r is None:
            return np.zeros(len(feats), bool)
        r = np.column_stack([_num(feats, c) for c in ("pre_weather_r", "pre_weather_r_summer", "pre_weather_r_first")])
        return (np.nan_to_num(r, nan=-1) >= self.weak_min_r).all(axis=1) & (_num(feats, "pre_shape_r") >= self.weak_min_shape)

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        score = pd.to_numeric(feats[self.score_feature], errors="coerce").to_numpy(float)
        dead = pd.to_numeric(feats["dark_zero_share"], errors="coerce").to_numpy(float) >= self.max_dark_zero_share
        main = (np.nan_to_num(score, nan=-np.inf) >= self.threshold) & ~dead & ~self.weather_veto(feats) & ~self.schedule_veto(feats)
        by_ratio = np.zeros(len(feats), bool)
        if self.use_ratio_rule:
            num = lambda c: _num(feats, c)
            amp = num("shape_amp_ratio")
            by_ratio = (~main & ~dead & (np.nan_to_num(num("ratio_t"), nan=-np.inf) >= self.ratio_threshold)
                        & (np.nan_to_num(num("shape_corr"), nan=-np.inf) >= self.shape_min_corr)
                        & (amp >= self.shape_amp_range[0]) & (amp <= self.shape_amp_range[1]))
        pv = main | by_ratio

        first = pd.to_datetime(feats["first_date"])
        offset = np.timedelta64(int(self.date_offset_days), "D")  # pd.Timedelta + Series warns under numpy >= 2.5
        tau = (pd.to_datetime(feats["tau_date"]) + offset).where(~feats["preexisting"] & feats["tau_date"].notna(), first)
        rtau = (pd.to_datetime(feats["ratio_tau_date"]) + offset).where(~feats["ratio_preexisting"] & feats["ratio_tau_date"].notna(), first)
        pv_shaped = ((pd.to_numeric(feats["ratio_t"], errors="coerce") >= self.ratio_threshold)
                     & (pd.to_numeric(feats["shape_corr"], errors="coerce") >= self.shape_min_corr)).to_numpy()
        earlier = ((tau - rtau).dt.days > self.reconcile_gap_days).to_numpy() & ~feats["preexisting"].to_numpy()
        use_ratio_date = by_ratio | (main & pv_shaped & earlier)
        date = tau.where(~use_ratio_date, rtau)

        kwp = self.predict_capacity(feats)

        # Weak PV before the onset: the series starts with PV, the onset becomes an extension.
        weak = self.weak_onset(feats) & main & ~use_ratio_date & ~feats["preexisting"].to_numpy(bool)
        onset = date.where(weak)
        date = date.where(~weak, first)
        total = np.where(weak, kwp, np.nan)
        if self.weak_kwp_per_drop is not None:
            initial = np.clip(self.weak_kwp_per_drop * _num(feats, "pre_sun_drop_kwh"), *self.cap_clip_kwp)
            kwp = np.where(weak & np.isfinite(initial), initial, kwp)

        return pd.DataFrame({
            "customer_id": feats["customer_id"].astype(str).to_numpy(),
            "pv_installed": pv,
            "pv_installation_date": date.where(pv),
            "pv_capacity_kwp": np.where(pv, kwp, np.nan),
            "extension_onset": onset.where(pv),        # weak-onset rule: date and total kWp of the
            "extension_onset_kwp": np.where(pv, total, np.nan),  # extension it implies (else NaN)
        })

    def predict_capacity(self, feats: pd.DataFrame) -> np.ndarray:
        """kWp for every row, whether or not PV is detected."""
        kwp = np.full(len(feats), self.cap_default_kwp)
        if self._cap is not None:
            kwp = self._cap.predict(feats)
        elif self.cap_coef is not None:
            X = feats[list(CAPACITY_FEATURES)].to_numpy(float)
            ok = np.isfinite(X).all(axis=1)
            kwp[ok] = np.exp(self.cap_coef[0] + X[ok] @ np.asarray(self.cap_coef[1:]))
        return np.clip(kwp, *self.cap_clip_kwp)

    def predict_extensions(self, data: dict, pred: pd.DataFrame) -> pd.DataFrame:
        """Extension events for the customers predicted with PV (empty if the detector is off)."""
        if not self.extension or not self.bell:
            return pd.DataFrame(columns=["customer_id", "extension_date", "pv_capacity_after_extension_kwp"])
        return predict_extensions(data, pred, self.bell[:4] if self.bell else None, ExtensionConfig(**self.extension))

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.capacity_file = None
        if self._cap is not None:
            self.capacity_file = f"{path.stem}_capacity.pkl"
            self._cap.save(path.parent / self.capacity_file)
        params = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BaselineModel":
        path = Path(path)
        params = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls) if f.init}
        params = {k: v for k, v in params.items() if k in known}  # tolerate models saved by other versions
        params["cap_clip_kwp"] = tuple(params.get("cap_clip_kwp", (1.0, 200.0)))
        params["shape_amp_range"] = tuple(params.get("shape_amp_range", (0.5, 2.0)))
        params["schedule_centroid_h"] = tuple(params.get("schedule_centroid_h", (8.5, 15.0)))
        model = cls(**params)
        if model.capacity_file:
            model._cap = CapacityModel.load(path.parent / model.capacity_file)
        if model.weather_file and (path.parent / model.weather_file).exists():
            model._weather = WeatherIndex.load(path.parent / model.weather_file)
        return model


def _num(feats: pd.DataFrame, col: str) -> np.ndarray:
    """Numeric column, all NaN when absent (features computed without the optional inputs)."""
    if col not in feats:
        return np.full(len(feats), np.nan)
    return pd.to_numeric(feats[col], errors="coerce").to_numpy(float)
