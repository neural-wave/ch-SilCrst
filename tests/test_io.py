import numpy as np
import pandas as pd

from pvdetect.io import load_consumption, read_consumption, read_extension_labels, read_labels
from pvdetect.validate import validation_report


def write_series(path, start_local, days, sep=";", decimal=".", epoch_ms=False, customer="007"):
    start = pd.Timestamp(start_local, tz="Europe/Zurich")
    end = start + pd.Timedelta(days=days)
    epoch = np.arange(int(start.timestamp()), int(end.timestamp()), 900)
    local = pd.to_datetime(epoch, unit="s", utc=True).tz_convert("Europe/Zurich").strftime("%Y-%m-%d %H:%M:%S")
    lines = [sep.join(["customer_id", "energy_consumption_kwh", "timestamp_local", "timestamp_epoch"])]
    for i, (e, t) in enumerate(zip(epoch, local)):
        v = f"{0.1 + (i % 4) / 100:.3f}".replace(".", decimal)
        lines.append(sep.join([customer, v, t, str(e * 1000 if epoch_ms else e)]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return epoch


def test_semicolon_decimal_comma_ms_epoch_and_leading_zeros(tmp_path):
    p = tmp_path / "e.csv"
    epoch = write_series(p, "2023-01-01", 3, decimal=",", epoch_ms=True)
    data, report = read_consumption(p)
    assert list(data) == ["007"]
    s = data["007"]
    assert report.epoch_unit == "ms" and report.rows == epoch.size
    assert np.nansum(s.x) > 0 and np.isfinite(s.x).sum() == epoch.size
    assert np.isclose(np.nanmin(s.x), 0.1)


def test_comma_separated_file(tmp_path):
    p = tmp_path / "e.csv"
    write_series(p, "2023-01-01", 2, sep=",")
    data, _ = read_consumption(p)
    assert np.isfinite(data["007"].x).sum() == 2 * 96


def test_dst_days_are_complete(tmp_path):
    p = tmp_path / "e.csv"
    write_series(p, "2023-03-20", 230)  # covers both DST transitions of 2023
    data, report = read_consumption(p)
    text = "\n".join(validation_report(data, report))
    assert "complete: 100.0%" in text
    assert "1 with 92 intervals, 1 with 100 intervals" in text


def test_cache_roundtrip(tmp_path):
    p = tmp_path / "e.csv"
    write_series(p, "2023-01-01", 3)
    a, _ = load_consumption(p, tmp_path / "cache")
    b, report = load_consumption(p, tmp_path / "cache")  # served from cache
    assert list(b) == list(a) and report.rows == 3 * 96
    assert np.array_equal(a["007"].x, b["007"].x, equal_nan=True) and a["007"].day0 == b["007"].day0


def test_labels_with_mixed_separators(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("customer_id,pv_installed,pv_installation_date,pv_capacity_kwp\n"
                 "0012;TRUE;2023-04-05;9,5\n0013;FALSE;;\n", encoding="utf-8")
    t = read_labels(p)
    assert t["customer_id"].tolist() == ["0012", "0013"]
    assert t["pv_installed"].tolist() == [True, False]
    assert t.loc[0, "pv_installation_date"] == pd.Timestamp("2023-04-05")
    assert t.loc[0, "pv_capacity_kwp"] == 9.5 and np.isnan(t.loc[1, "pv_capacity_kwp"])


def test_header_only_extension_file(tmp_path):
    p = tmp_path / "ext.csv"
    p.write_text("customer_id;extension_date;pv_capacity_after_extension_kwp\n", encoding="utf-8")
    assert read_extension_labels(p).empty


def test_na_like_customer_ids_are_kept(tmp_path):
    p = tmp_path / "e.csv"
    write_series(p, "2023-01-01", 2, customer="NA")
    with open(p, "a", encoding="utf-8") as f:
        f.write("null;;2023-01-03 00:00:00;1672700400\n")  # missing value, id "null"
    data, report = read_consumption(p)
    assert set(data) == {"NA", "null"}
    assert np.isfinite(data["NA"].x).sum() == 2 * 96 and report.missing_values == 1
