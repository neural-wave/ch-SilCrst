"""Data-quality checks of the parsed series."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SLOT_SECONDS, SLOTS_PER_DAY
from .io import CustomerSeries, ReadReport

TZ = "Europe/Zurich"


def validation_report(data: dict[str, CustomerSeries], report: ReadReport) -> list[str]:
    """Check parsing and the DST-aware completeness of local days (92 / 96 / 100 intervals)."""
    lines = [
        f"customers: {len(data)} | rows: {report.rows:,} | epoch unit: {report.epoch_unit}",
        f"bad timestamps: {report.bad_timestamps:,} | missing kWh: {report.missing_values:,} | "
        f"negative kWh: {report.negative_values:,} | off-grid timestamps: {report.off_grid:,} | "
        f"duplicate slots: {report.duplicates:,}",
    ]
    series = [s for s in data.values() if s.n_days]
    if not series:
        return lines + ["no usable series"]

    lo = min(s.day0 for s in series)
    hi = max(s.day0 + s.n_days for s in series)
    grid = np.arange(lo * SLOTS_PER_DAY, hi * SLOTS_PER_DAY, dtype=np.int64) * SLOT_SECONDS
    local = pd.to_datetime(grid, unit="s", utc=True).tz_convert(TZ).tz_localize(None)
    local_day = local.to_numpy().astype("datetime64[D]").astype(np.int64)
    code = local_day - local_day.min()
    expected = np.bincount(code)  # 92 / 96 / 100 grid slots per local day

    in_range = complete = empty = short_ok = long_ok = 0
    days_per_customer = []
    for s in series:
        start = (s.day0 - lo) * SLOTS_PER_DAY
        c = code[start : start + s.n_days * SLOTS_PER_DAY]
        observed = np.bincount(c[~np.isnan(s.x.ravel())], minlength=expected.size)
        days = np.arange(c.min() + 1, c.max())  # first/last local day are only partly on the grid
        exp, obs = expected[days], observed[days]
        in_range += days.size
        complete += int((obs == exp).sum())
        empty += int((obs == 0).sum())
        short_ok += int(((exp == 92) & (obs == exp)).sum())
        long_ok += int(((exp == 100) & (obs == exp)).sum())
        days_per_customer.append(int((obs > 0).sum()))

    first = np.datetime64(int(local_day.min()), "D")
    last = np.datetime64(int(local_day.max()), "D")
    lines += [
        f"period: {first} .. {last} | days with data per customer: "
        f"min {min(days_per_customer)}, median {int(np.median(days_per_customer))}, max {max(days_per_customer)}",
        f"local days in range: {in_range:,} | complete: {complete / max(in_range, 1):.1%} | "
        f"partial: {in_range - complete - empty:,} | without data: {empty:,}",
        f"complete DST days: {short_ok} with 92 intervals, {long_ok} with 100 intervals",
    ]
    return lines
