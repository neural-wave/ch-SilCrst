"""Initial PV capacity from a vision transformer over the whole history, in numpy.

The network is the team's multi-task ViT (synthetic-supervision stage, seed 0): one token per
UTC day from a conv stem over its 96 slots, a scale token with the absolute levels, 4 pre-norm
transformer blocks. Only the parts the capacity head needs run here: the date posterior, the
contrast pooling "after minus before" the installation it implies, and a Laplace distribution
on log kWp. The submitted kWp maximises the expected P under that distribution.

The weights never saw a development label or series: masked pretraining on the unlabelled
series, then supervision on synthetic PV injected into PV-free unlabelled customers
(scripts/export_vit_capacity.py converts them). On the 1,187 dev and unlabelled customers this
port gives the torch network's kWp (median relative difference 5e-7), so inference needs no
torch. The only differences come from the reader: negative readings are missing here (io.py),
measured values in the team's converter (3 unlabelled customers).

The PV verdict, the date and the extensions stay with the standby model; only the initial kWp
of its PV customers comes from here (model.py keeps its own for the weak-onset rule).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf

from .io import CustomerSeries

NIGHT_SLOTS = np.r_[80:96, 0:14]  # 20:00-03:30 UTC
MIDDAY_SLOTS = np.arange(36, 56)  # 09:00-14:00 UTC
LEVEL_FLOOR_KW = 0.01
NEGATIVE = -1e4
QUANTILES = (np.arange(400) + 0.5) / 400
CAPACITY_GRID = 600
CAPACITY_TOLERANCE = 0.5  # P is zero past a 50% error (spec §4.2)


def summer_time(days: np.ndarray) -> np.ndarray:
    """1 where Swiss clocks are on CEST at 12:00 UTC: last Sunday of March to the day before the
    last Sunday of October."""
    days = days.astype("datetime64[D]")
    years = days.astype("datetime64[Y]")

    def last_sunday(month_offset: int) -> np.ndarray:
        last_day = (years + np.timedelta64(month_offset, "M")).astype("datetime64[D]") + np.timedelta64(30, "D")
        weekday = (last_day.astype(np.int64) + 3) % 7
        return last_day - ((weekday - 6) % 7).astype("timedelta64[D]")

    return ((days >= last_sunday(2)) & (days < last_sunday(9))).astype(np.int64)


def _log_level(kwh: np.ndarray) -> float:
    kw = 4 * float(np.median(kwh)) if len(kwh) else 0.0
    return float(np.log(max(kw, 0.0) + LEVEL_FLOOR_KW))


def _gelu(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1 + erf(x / math.sqrt(2)))).astype(np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Conv1d with kernel 5, stride 2, padding 2: (n, c_in, l) -> (n, c_out, l / 2)."""
    n, _, length = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (2, 2)))
    starts = 2 * np.arange(length // 2)
    windows = padded[:, :, starts[:, None] + np.arange(5)]  # (n, c_in, l/2, 5)
    return np.einsum("nclk,ock->nol", windows, w, optimize=True) + b[None, :, None]


@dataclass
class VitCapacity:
    w: dict[str, np.ndarray]
    n_heads: int = 4

    @classmethod
    def load(cls, path: str | Path) -> VitCapacity:
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def _linear(self, x: np.ndarray, name: str) -> np.ndarray:
        return x @ self.w[f"{name}.weight"].T + self.w[f"{name}.bias"]

    def _mlp(self, x: np.ndarray, name: str) -> np.ndarray:
        return self._linear(_gelu(self._linear(x, f"{name}.0")), f"{name}.3")

    def _tokens(self, s: CustomerSeries) -> dict[str, np.ndarray]:
        mask = np.isfinite(s.x)
        values = np.where(mask, s.x, 0.0).astype(np.float32)
        measured = values[mask]
        mean = max(float(measured.mean()), 1e-3) if len(measured) else 1.0
        relative = values / mean
        tok = np.where(mask, np.sign(relative) * np.log1p(np.abs(relative)), 0.0).astype(np.float32)
        night = values[:, NIGHT_SLOTS][mask[:, NIGHT_SLOTS]]
        midday = values[:, MIDDAY_SLOTS][mask[:, MIDDAY_SLOTS]]
        scale = np.array([np.log(4 * mean + LEVEL_FLOOR_KW), _log_level(night), _log_level(midday)], np.float32)
        days = s.days().astype("datetime64[D]")
        return {"values": tok, "mask": mask, "day_of_year": (days - days.astype("datetime64[Y]")).astype(np.int64),
                "weekday": (days.astype(np.int64) + 3) % 7, "summer_time": summer_time(days), "scale": scale}

    def _positions(self, n: int, d: int) -> np.ndarray:
        position = np.arange(n, dtype=np.float32)[:, None]
        rate = np.exp(np.arange(0, d, 2, dtype=np.float32) * np.float32(-math.log(10000.0) / d))
        table = np.zeros((n, d), np.float32)
        table[:, 0::2] = np.sin(position * rate)
        table[:, 1::2] = np.cos(position * rate)
        return table

    def _block(self, x: np.ndarray, i: int) -> np.ndarray:
        p = f"blocks.{i}"
        t, d = x.shape
        h = self.n_heads
        qkv = self._linear(_layer_norm(x, self.w[f"{p}.norm1.weight"], self.w[f"{p}.norm1.bias"]), f"{p}.qkv")
        q, k, v = qkv.reshape(t, 3, h, d // h).transpose(1, 2, 0, 3)  # each (h, t, d/h)
        att = _softmax(q @ k.transpose(0, 2, 1) / np.float32(math.sqrt(d // h))) @ v
        x = x + self._linear(att.transpose(1, 0, 2).reshape(t, d), f"{p}.proj")
        y = _layer_norm(x, self.w[f"{p}.norm2.weight"], self.w[f"{p}.norm2.bias"])
        return x + self._linear(_gelu(self._linear(y, f"{p}.mlp.0")), f"{p}.mlp.3")

    def log_kwp(self, s: CustomerSeries) -> tuple[float, float]:
        """Laplace (mu, b) on log kWp for one customer's whole history."""
        tok = self._tokens(s)
        values, mask = tok["values"], tok["mask"].astype(np.float32)
        t = len(values)
        day_in = np.stack([values * mask, mask], axis=1)  # (t, 2, 96)
        stem = _gelu(_conv1d(day_in, self.w["stem.0.weight"], self.w["stem.0.bias"]))
        stem = _gelu(_conv1d(stem, self.w["stem.2.weight"], self.w["stem.2.bias"]))
        days = self._linear(stem.reshape(t, -1), "stem_out")
        angle = (2 * math.pi * tok["day_of_year"].astype(np.float32) / 365.25).astype(np.float32)
        days = (days + self._linear(np.stack([np.sin(angle), np.cos(angle)], axis=-1), "calendar")
                + self.w["weekday.weight"][tok["weekday"]] + self.w["summer.weight"][tok["summer_time"]]
                + self._positions(t, days.shape[1]))
        x = np.concatenate([self._mlp(tok["scale"][None], "scale_in"), days]).astype(np.float32)
        n_blocks = 1 + max(int(k.split(".")[1]) for k in self.w if k.startswith("blocks."))
        for i in range(n_blocks):
            x = self._block(x, i)
        x = _layer_norm(x, self.w["norm.weight"], self.w["norm.bias"])
        scale_state, states = x[0], x[1:]

        has_data = tok["mask"].any(axis=1)
        date = np.concatenate([self._linear(scale_state, "date_before"), self._linear(states, "date_day")[:, 0]])
        p = _softmax(date)
        installed = np.clip(p[0] + np.cumsum(p[1:]), 0, 1)
        weight = has_data.astype(np.float32)

        def weighted_mean(w: np.ndarray) -> np.ndarray:
            return (w[:, None] * states).sum(0) / (w.sum() + 1e-3)

        after, before = weighted_mean(installed * weight), weighted_mean((1 - installed) * weight)
        score = np.where(has_data, self._linear(states, "pool_score")[:, 0], NEGATIVE)
        pooled = _softmax(score) @ states
        cap = self._mlp(np.concatenate([after - before, after, scale_state, pooled, tok["scale"]]), "capacity")
        return float(cap[0] + self.w["real_offset"]), float(np.exp(np.clip(cap[1], -4, 2)))

    def kwp(self, s: CustomerSeries) -> float:
        """argmax_c E[max(0, 1 - |c - y| / (0.5 y))] for y = exp(Laplace(mu, b))."""
        mu, b = self.log_kwp(s)
        q = QUANTILES
        y = np.exp(mu - b * np.sign(q - 0.5) * np.log(1 - 2 * np.abs(q - 0.5)))
        grid = np.exp(np.linspace(np.log(y.min()), np.log(y.max()), CAPACITY_GRID))
        expected = np.clip(1 - np.abs(grid[:, None] - y[None, :]) / (CAPACITY_TOLERANCE * y[None, :]), 0, None).mean(1)
        return float(grid[np.argmax(expected)])


def with_vit_capacity(pred: pd.DataFrame, data: dict[str, CustomerSeries], vit: VitCapacity,
                      clip_kwp: tuple[float, float]) -> pd.DataFrame:
    """`pred` (BaselineModel.predict) with the ViT's initial kWp for its PV customers. The
    weak-onset rule's customers keep theirs: the rule sets the initial kWp and the extension."""
    out = pred.copy()
    weak = out["extension_onset_kwp"].notna() if "extension_onset_kwp" in out else pd.Series(False, index=out.index)
    rows = out["pv_installed"].astype(bool) & ~weak & out["customer_id"].map(lambda c: c in data and data[c].n_days > 0)
    out.loc[rows, "pv_capacity_kwp"] = [float(np.clip(vit.kwp(data[c]), *clip_kwp)) for c in out.loc[rows, "customer_id"]]
    return out
