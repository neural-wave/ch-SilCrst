"""End-to-end checks on a small synthetic set (never on AIL data)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from devtools.synthetic import generate
from pvdetect import ENERGY_DEV, LABELS_DEV
from pvdetect.features import extract_features
from pvdetect.io import read_consumption, read_labels
from pvdetect.metrics import score_main
from pvdetect.model import BaselineModel

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    generate(out, n_customers=14, start="2023-01-01", end="2024-12-31", seed=1)
    return out


def test_features_model_and_score(synthetic):
    data, _ = read_consumption(synthetic / ENERGY_DEV)
    labels = read_labels(synthetic / LABELS_DEV)
    feats = extract_features(data)
    assert len(feats) == len(labels) == 14
    s = score_main(BaselineModel().fit(feats, labels).predict(feats), labels)
    assert s["f1"] >= 0.9
    assert s["date_within_7"] >= 0.6 * s["tp"]


def test_inference_cli_writes_submission_files(synthetic, tmp_path):
    subprocess.run([sys.executable, str(ROOT / "inference.py"), str(synthetic / ENERGY_DEV)], cwd=tmp_path, check=True)
    lines = (tmp_path / "predictions.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "customer_id;pv_installed;pv_installation_date;pv_capacity_kwp"
    rows = [line.split(";") for line in lines[1:]]
    assert len(rows) == 14 and all(len(r) == 4 for r in rows)
    assert rows[0][0] == "00001"  # leading zeros preserved
    for cid, pv, date, kwp in rows:
        assert pv in ("TRUE", "FALSE")
        if pv == "TRUE":
            pd.Timestamp(date) and float(kwp)
        else:
            assert date == "" and kwp == ""
    ext = (tmp_path / "pv_extensions_predictions.csv").read_text(encoding="utf-8").splitlines()
    assert ext[0] == "customer_id;extension_date;pv_capacity_after_extension_kwp"
    for row in ext[1:]:
        cid, date, kwp = row.split(";")
        assert cid in {r[0] for r in rows} and pd.Timestamp(date) and float(kwp) > 0


def test_inference_rejects_wrong_arguments(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "inference.py")], cwd=tmp_path, capture_output=True)
    assert r.returncode == 2


def test_degenerate_series_do_not_crash():
    from pvdetect.io import CustomerSeries

    rng = np.random.default_rng(0)
    normal = rng.gamma(2, 0.05, (400, 96)).astype(np.float32)
    gappy = normal.copy()
    gappy[50:300] = np.nan
    cases = {
        "empty": CustomerSeries("empty", 0, np.full((0, 96), np.nan, np.float32)),
        "short": CustomerSeries("short", 19000, normal[:10]),
        "all_nan": CustomerSeries("all_nan", 19000, np.full((100, 96), np.nan, np.float32)),
        "all_zero": CustomerSeries("all_zero", 19000, np.zeros((400, 96), np.float32)),
        "gappy": CustomerSeries("gappy", 19000, gappy),
        "normal": CustomerSeries("normal", 19000, normal),
    }
    feats = extract_features(cases)
    assert feats["customer_id"].tolist() == list(cases)
    pred = BaselineModel(threshold=-1.0).predict(feats)  # force PV everywhere it has a score
    assert len(pred) == len(cases)
    dead = feats["dark_zero_share"].to_numpy(float) >= 0.4  # all-zero meter: never PV
    has_score = feats["rate60_max"].notna().to_numpy() & ~dead
    assert dead[list(cases).index("all_zero")]
    assert (pred["pv_installed"].to_numpy() == has_score).all()
    assert pred.loc[has_score, "pv_installation_date"].notna().all()
    assert (pred.loc[has_score, "pv_capacity_kwp"] > 0).all()
