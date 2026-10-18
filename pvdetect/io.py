"""Readers for the challenge CSV files and a local cache of the parsed series.

The files are not fully consistent (pv_extensions_labels_dev.csv has a ',' header
with ';' rows), so separator and decimal mark are detected per file.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DAY_SECONDS, SLOT_SECONDS, SLOTS_PER_DAY

CONSUMPTION_COLUMNS = ("customer_id", "energy_consumption_kwh", "timestamp_epoch")
# Missing-value tokens for numeric columns only: ids such as "NA" or "null" must stay strings.
NA_TOKENS = ["", "#N/A", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null", "<NA>", "-nan", "-NaN"]
CACHE_VERSION = 2
# Organizers' note: the unlabelled file also holds a few non-customer curves and records with a
# null customer_id; only ids starting with "POD" are real customers.
CUSTOMER_ID_PREFIX = "POD"


def customers_only(data: dict, prefix: str = CUSTOMER_ID_PREFIX) -> tuple[dict, int]:
    """Keep the series whose id starts with `prefix`; returns (kept, number dropped).
    Used on training data only: inference keeps every id of the test file."""
    kept = {k: v for k, v in data.items() if str(k).startswith(prefix)}
    return kept, len(data) - len(kept)
DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d", "%d/%m/%Y")


@dataclass
class CustomerSeries:
    """x[d, s] = kWh measured in UTC slot s (15 min) of UTC day `day0 + d`; NaN = missing."""

    customer_id: str
    day0: int
    x: np.ndarray

    @property
    def n_days(self) -> int:
        return self.x.shape[0]

    def days(self) -> np.ndarray:
        return np.arange(self.day0, self.day0 + self.n_days)


@dataclass
class ReadReport:
    rows: int = 0
    bad_timestamps: int = 0
    missing_values: int = 0
    negative_values: int = 0
    off_grid: int = 0
    duplicates: int = 0
    epoch_unit: str = "s"


# ---------------------------------------------------------------- format sniffing

def _split_header(line: str) -> list[str]:
    return [c.strip().strip('"').lower() for c in re.split(r"[;,]", line.lstrip("﻿").strip())]


def _sniff(path: Path, value_col: str | None = None) -> tuple[list[str], str, str]:
    """Return (lower-cased header columns, row separator, decimal mark)."""
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline()
        rows = [line.strip() for line in (f.readline() for _ in range(20)) if line.strip()]
    cols = _split_header(header)
    sep = ";" if any(";" in r for r in rows) or (not rows and ";" in header) else ","
    decimal = "."
    if sep == ";" and value_col in cols:
        i = cols.index(value_col)
        fields = [r.split(";") for r in rows]
        if any(len(f) > i and "," in f[i] for f in fields):
            decimal = ","
    return cols, sep, decimal


# ---------------------------------------------------------------- consumption series

def read_consumption(path: str | Path, chunksize: int = 2_000_000) -> tuple[dict[str, CustomerSeries], ReadReport]:
    """Parse a consumption CSV (any size) into one UTC day x slot matrix per customer."""
    path = Path(path)
    cols, sep, decimal = _sniff(path, "energy_consumption_kwh")
    missing = [c for c in CONSUMPTION_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing} (found {cols})")
    pos = [cols.index(c) for c in CONSUMPTION_COLUMNS]

    report = ReadReport()
    epochs: dict[str, list[np.ndarray]] = defaultdict(list)
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    epoch_scale = None
    reader = pd.read_csv(
        path, sep=sep, decimal=decimal, header=None, skiprows=1, usecols=pos,
        dtype={pos[0]: str}, chunksize=chunksize, encoding="utf-8-sig",
        keep_default_na=False, na_values={pos[1]: NA_TOKENS, pos[2]: NA_TOKENS},
    )
    for chunk in reader:
        ids = chunk[pos[0]].astype(str).str.strip().to_numpy()
        val = pd.to_numeric(chunk[pos[1]], errors="coerce").to_numpy(float)
        ep = pd.to_numeric(chunk[pos[2]], errors="coerce").to_numpy(float)
        if epoch_scale is None and np.isfinite(ep).any():
            # Seconds are ~1.7e9 today; milliseconds ~1.7e12.
            epoch_scale = 1000.0 if np.nanmax(ep) > 1e11 else 1.0
            report.epoch_unit = "ms" if epoch_scale > 1 else "s"
        ep = ep / (epoch_scale or 1.0)
        report.rows += len(chunk)

        codes, uniques = pd.factorize(ids)
        order = np.argsort(codes, kind="stable")
        for grp in np.split(order, np.flatnonzero(np.diff(codes[order])) + 1):
            if grp.size:
                cid = uniques[codes[grp[0]]]
                epochs[cid].append(ep[grp])
                values[cid].append(val[grp])

    data = {cid: _to_grid(cid, np.concatenate(epochs[cid]), np.concatenate(values[cid]), report) for cid in epochs}
    return data, report


def _to_grid(cid: str, ep: np.ndarray, val: np.ndarray, report: ReadReport) -> CustomerSeries:
    ok = np.isfinite(ep)
    report.bad_timestamps += int((~ok).sum())
    ep, val = np.floor(ep[ok]).astype(np.int64), val[ok]
    report.missing_values += int(np.isnan(val).sum())
    negative = val < 0
    report.negative_values += int(negative.sum())
    val = np.where(negative, np.nan, val)
    if ep.size == 0:
        return CustomerSeries(cid, 0, np.full((0, SLOTS_PER_DAY), np.nan, np.float32))

    rem = ep % SLOT_SECONDS
    report.off_grid += int((rem != 0).sum())
    ep = ep - rem
    day, slot = ep // DAY_SECONDS, (ep % DAY_SECONDS) // SLOT_SECONDS
    day0 = int(day.min())
    n_days = int(day.max()) - day0 + 1
    flat = (day - day0) * SLOTS_PER_DAY + slot
    report.duplicates += int(flat.size - np.unique(flat).size)

    good = ~np.isnan(val)
    total = np.bincount(flat[good], weights=val[good], minlength=n_days * SLOTS_PER_DAY)
    count = np.bincount(flat[good], minlength=n_days * SLOTS_PER_DAY)
    x = np.full(n_days * SLOTS_PER_DAY, np.nan)
    np.divide(total, count, out=x, where=count > 0)  # duplicates are averaged
    return CustomerSeries(cid, day0, x.reshape(n_days, SLOTS_PER_DAY).astype(np.float32))


def load_consumption(path: str | Path, cache_dir: str | Path | None = None) -> tuple[dict[str, CustomerSeries], ReadReport]:
    """read_consumption with an optional .npz cache, invalidated when the CSV changes.

    The cache holds AIL data: keep cache_dir git-ignored.
    """
    path = Path(path)
    if cache_dir is None:
        return read_consumption(path)
    stat = path.stat()
    signature = f"v{CACHE_VERSION}-{stat.st_size}-{int(stat.st_mtime)}"
    tag = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
    cache = Path(cache_dir) / f"{path.stem}-{tag}.npz"
    if cache.exists():
        with np.load(cache) as z:
            if str(z["signature"]) == signature:
                ids, day0s = z["ids"], z["day0s"]
                data = {str(c): CustomerSeries(str(c), int(d), z[f"x{i}"]) for i, (c, d) in enumerate(zip(ids, day0s))}
                return data, ReadReport(**json.loads(str(z["report"])))

    data, report = read_consumption(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    series = list(data.values())
    np.savez(
        cache,
        signature=np.array(signature),
        report=np.array(json.dumps(asdict(report))),
        ids=np.array([s.customer_id for s in series], dtype=str),
        day0s=np.array([s.day0 for s in series], dtype=np.int64),
        **{f"x{i}": s.x for i, s in enumerate(series)},
    )
    return data, report


# ---------------------------------------------------------------- labels / predictions

def read_table(path: str | Path) -> pd.DataFrame:
    """Read one of our own CSV outputs, keeping customer_id exactly as written ("NA" stays "NA")."""
    df = pd.read_csv(path, dtype={"customer_id": str})
    df["customer_id"] = pd.read_csv(path, usecols=["customer_id"], dtype=str, keep_default_na=False)["customer_id"]
    return df


def read_small_table(path: str | Path) -> pd.DataFrame:
    """Read a small CSV as strings, tolerating a header separator that differs from the rows'."""
    lines = [line for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame()
    cols = _split_header(lines[0])
    rows = lines[1:]
    sep = ";" if any(";" in r for r in rows) else ","
    data = [([v.strip().strip('"') for v in r.split(sep)] + [""] * len(cols))[: len(cols)] for r in rows]
    return pd.DataFrame(data, columns=cols, dtype="string")


def parse_bool(s: pd.Series) -> pd.Series:
    return s.fillna("").str.strip().str.upper().isin(["TRUE", "1", "YES", "Y", "T"])


def parse_date(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        todo = out.isna() & (s != "")
        if not todo.any():
            break
        out[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    return out


def parse_float(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s.fillna("").str.strip().str.replace(",", ".", regex=False), errors="coerce")
    return out.astype(float)  # plain NaN instead of pd.NA


def read_labels(path: str | Path) -> pd.DataFrame:
    """Initial-installation labels (or predictions.csv, same format)."""
    t = read_small_table(path)
    if t.empty:
        return pd.DataFrame(columns=["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"])
    return pd.DataFrame({
        "customer_id": t["customer_id"].str.strip().astype(str),
        "pv_installed": parse_bool(t["pv_installed"]),
        "pv_installation_date": parse_date(t["pv_installation_date"]),
        "pv_capacity_kwp": parse_float(t["pv_capacity_kwp"]),
    })


def read_extension_labels(path: str | Path) -> pd.DataFrame:
    """Extension labels (or pv_extensions_predictions.csv, same format)."""
    t = read_small_table(path)
    if t.empty:
        return pd.DataFrame({
            "customer_id": pd.Series(dtype=str),
            "extension_date": pd.Series(dtype="datetime64[ns]"),
            "pv_capacity_after_extension_kwp": pd.Series(dtype=float),
        })
    return pd.DataFrame({
        "customer_id": t["customer_id"].str.strip().astype(str),
        "extension_date": parse_date(t["extension_date"]),
        "pv_capacity_after_extension_kwp": parse_float(t["pv_capacity_after_extension_kwp"]),
    })
