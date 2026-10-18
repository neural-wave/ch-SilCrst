import numpy as np
import pandas as pd
import pytest

from pvdetect.metrics import score_extensions, score_main


def frame(rows):
    return pd.DataFrame({
        "customer_id": [r[0] for r in rows],
        "pv_installed": [r[1] for r in rows],
        "pv_installation_date": pd.to_datetime([r[2] for r in rows]),
        "pv_capacity_kwp": [r[3] for r in rows],
    })


def test_pdf_example():
    # 9 TP with 6-day date error (D_i = 0.8) and 15% capacity error (P_i = 0.7), 1 FP, 1 FN -> F1 = 0.9.
    truth, pred = [], []
    for i in range(9):
        truth.append((f"c{i}", True, "2023-05-01", 10.0))
        pred.append((f"c{i}", True, "2023-05-07", 11.5))
    truth += [("fn", True, "2023-01-01", 5.0), ("fp", False, None, np.nan)]
    pred += [("fn", False, None, np.nan), ("fp", True, "2023-01-01", 5.0)]
    s = score_main(frame(pred), frame(truth))
    assert s["f1"] == pytest.approx(0.9)
    assert s["D"] == pytest.approx(0.8)
    assert s["P"] == pytest.approx(0.7)
    assert s["main"] == pytest.approx(83.25)


def test_invalid_values_score_zero_and_missing_customer_is_false():
    truth = frame([("a", True, "2023-05-01", 10.0), ("b", True, "2023-05-01", 10.0)])
    pred = frame([("a", True, None, np.nan)])
    s = score_main(pred, truth)
    assert (s["tp"], s["fn"]) == (1, 1)
    assert s["D"] == 0 and s["P"] == 0
    assert s["main"] == pytest.approx(2 / 3 * 70)


def test_no_true_positive_scores_zero():
    truth = frame([("a", True, "2023-05-01", 10.0)])
    pred = frame([("a", False, None, np.nan)])
    assert score_main(pred, truth)["main"] == 0


def ext(rows):
    return pd.DataFrame({
        "customer_id": [r[0] for r in rows],
        "extension_date": pd.to_datetime([r[1] for r in rows]),
        "pv_capacity_after_extension_kwp": [r[2] for r in rows],
    })


def test_extension_matching_is_one_to_one_within_30_days():
    truth = ext([("a", "2023-06-01", 15.0), ("a", "2024-07-01", 18.0)])
    pred = ext([("a", "2023-06-06", 15.0), ("a", "2023-06-11", 15.0), ("a", "2024-08-15", 18.0)])
    s = score_extensions(pred, truth)
    # Only the first true event can be matched (once); the 2024 prediction is 45 days off.
    assert (s["tp"], s["fp"], s["fn"]) == (1, 2, 1)
    assert s["f1_ext"] == pytest.approx(2 / 5)
    assert s["D_ext"] == pytest.approx(1 - 5 / 30)
    assert s["bonus"] == pytest.approx(0.4 * (7 + 1.5 * (1 - 5 / 30) + 1.5))


def test_extension_matching_prefers_more_matches_over_smaller_error():
    truth = ext([("a", "2023-06-01", 10.0), ("a", "2023-07-10", 12.0)])
    # Greedy would pair the first prediction with the closer second event and lose a match.
    pred = ext([("a", "2023-06-25", 10.0), ("a", "2023-08-05", 12.0)])
    assert score_extensions(pred, truth)["tp"] == 2


def test_empty_extensions_give_zero_bonus():
    assert score_extensions(ext([]), ext([("a", "2023-06-01", 15.0)]))["bonus"] == 0
