"""Submission files (PDF §4.1): UTF-8, decimal point, exact column names and order."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# §4.1 text asks for semicolons (its examples use commas; the text wins).
OUTPUT_SEP = ";"

MAIN_COLUMNS = ("customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp")
EXTENSION_COLUMNS = ("customer_id", "extension_date", "pv_capacity_after_extension_kwp")


def _date(v) -> str:
    return "" if pd.isna(v) else pd.Timestamp(v).strftime("%Y-%m-%d")


def _kwp(v) -> str:
    return "" if pd.isna(v) else f"{float(v):.2f}"


def write_predictions(pred: pd.DataFrame, path: str | Path, sep: str = OUTPUT_SEP) -> None:
    lines = [sep.join(MAIN_COLUMNS)]
    for r in pred.itertuples(index=False):
        pv = bool(r.pv_installed)
        lines.append(sep.join([
            str(r.customer_id),
            "TRUE" if pv else "FALSE",
            _date(r.pv_installation_date) if pv else "",
            _kwp(r.pv_capacity_kwp) if pv else "",
        ]))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_extensions(ext: pd.DataFrame | None, path: str | Path, sep: str = OUTPUT_SEP) -> None:
    lines = [sep.join(EXTENSION_COLUMNS)]
    for r in ([] if ext is None else ext.itertuples(index=False)):
        lines.append(sep.join([str(r.customer_id), _date(r.extension_date), _kwp(r.pv_capacity_after_extension_kwp)]))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
