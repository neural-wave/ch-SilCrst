"""Objective performance report from leave-one-out predictions.

Usage: python scripts/report.py [outputs/loo_details.csv] [--draws 2000]

Run scripts/evaluate.py first (with --augmented for the final model). Reports the official
metric with its components, trivial baselines for reference, a bootstrap confidence interval
and the score distribution of simulated 20-customer test sets.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pvdetect.io import read_table  # noqa: E402
from pvdetect.metrics import score_main  # noqa: E402

COLS = ["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"]


def frames(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = d[["customer_id"] + [f"{c}_true" for c in COLS[1:]]].set_axis(COLS, axis=1).copy()
    pred = d[["customer_id"] + [f"{c}_pred" for c in COLS[1:]]].set_axis(COLS, axis=1).copy()
    for f in (truth, pred):
        f["pv_installed"] = f["pv_installed"].astype(bool)
        f["pv_installation_date"] = pd.to_datetime(f["pv_installation_date"])
    return truth, pred


def loo_median_kwp(truth: pd.DataFrame) -> np.ndarray:
    """Median kWp of the labelled PV customers, excluding each customer in turn."""
    k = truth["pv_capacity_kwp"].where(truth["pv_installed"])
    return np.array([k.drop(index=i).median() for i in truth.index])


def line(name: str, s: dict) -> str:
    return f"  {name:<44} MAIN {s['main']:6.2f} | F1 {s['f1']:.3f} | D {s['D']:.3f} | P {s['P']:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("details", nargs="?", default="outputs/loo_details.csv")
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    d = read_table(a.details)
    truth, pred = frames(d)
    s = score_main(pred, truth)
    t, p = truth["pv_installed"].to_numpy(), pred["pv_installed"].to_numpy()
    n_pv = int(t.sum())

    print(f"== Final model, leave-one-out on {len(truth)} customers ({n_pv} with PV)")
    print(line("official score", s))
    print(f"  max possible with this F1: {s['f1'] * 100:.1f} | points lost: detection {100 - s['f1'] * 100:.1f}, "
          f"date {s['f1'] * 15 * (1 - s['D']):.1f}, capacity {s['f1'] * 15 * (1 - s['P']):.1f}")

    print("\n== Detection")
    print(f"  confusion: TP {s['tp']}  FN {s['fn']}  FP {s['fp']}  TN {s['tn']}")
    print(f"  precision {s['precision']:.2f} | recall {s['recall']:.2f} | specificity {s['tn'] / max(s['tn'] + s['fp'], 1):.2f} "
          f"| accuracy {(s['tp'] + s['tn']) / len(truth):.2f}")

    hit = t & p
    err = (pred["pv_installation_date"] - truth["pv_installation_date"]).dt.days[hit]
    before = (truth["pv_installation_date"] < pd.to_datetime(d["first_date"]))[hit]
    print("\n== Commissioning date (correctly detected PV)")
    print(f"  |error| days: median {err.abs().median():.0f} | mean {err.abs().mean():.0f} | 90th pct {err.abs().quantile(.9):.0f}")
    print(f"  within 3 d: {(err.abs() <= 3).sum()}/{hit.sum()} | 7 d: {(err.abs() <= 7).sum()} | 15 d: {(err.abs() <= 15).sum()} "
          f"| <30 d (scores > 0): {(err.abs() < 30).sum()}")
    print(f"  installed before the series starts (date not observable): {int(before.sum())}; "
          f"errors >= 30 d among the others: {int(((err.abs() >= 30) & ~before).sum())}")
    print(f"  bias: predicted later than true in {(err > 0).sum()}, earlier in {(err < 0).sum()}, exact in {(err == 0).sum()}")

    kt, kp = truth["pv_capacity_kwp"][hit].to_numpy(), pred["pv_capacity_kwp"][hit].to_numpy()
    ape = np.abs(kp - kt) / kt
    log_ratio = np.log(kp / kt)
    print("\n== Capacity (correctly detected PV)")
    print(f"  APE: median {np.median(ape):.0%} | mean {ape.mean():.0%} | within 10%: {(ape <= .1).sum()}/{hit.sum()} "
          f"| 25%: {(ape <= .25).sum()} | <50% (scores > 0): {(ape < .5).sum()}")
    print(f"  bias: median predicted/true {np.exp(np.median(log_ratio)):.2f} | over-estimates {(log_ratio > 0).sum()}, "
          f"under-estimates {(log_ratio < 0).sum()}")
    rank = pd.Series(kp).rank().corr(pd.Series(kt).rank())
    print(f"  rank correlation predicted vs true (Spearman): {rank:.2f}")

    print("\n== Reference baselines (same customers, leave-one-out)")
    none = pred.assign(pv_installed=False)
    print(line("everything FALSE", score_main(none, truth)))
    mid = pd.to_datetime(d["first_date"]) + (pd.to_datetime(d["last_date"]) - pd.to_datetime(d["first_date"])) / 2
    allpv = pred.assign(pv_installed=True, pv_installation_date=mid, pv_capacity_kwp=loo_median_kwp(truth))
    print(line("everything TRUE, mid-series date, median kWp", score_main(allpv, truth)))
    const_kwp = pred.assign(pv_capacity_kwp=loo_median_kwp(truth))
    print(line("our detection + dates, median kWp", score_main(const_kwp, truth)))

    rng = np.random.default_rng(0)
    boot = [score_main(pred.loc[i], truth.loc[i])["main"]
            for i in (rng.choice(truth.index, len(truth), replace=True) for _ in range(a.draws))]
    sims = [score_main(pred.loc[i], truth.loc[i])["main"]
            for i in (rng.choice(truth.index, 20, replace=False) for _ in range(a.draws))]
    print("\n== Uncertainty")
    print(f"  bootstrap 95% interval of the 40-customer score: {np.percentile(boot, 2.5):.1f} - {np.percentile(boot, 97.5):.1f}")
    print(f"  simulated 20-customer test sets: median {np.median(sims):.1f} | 5th-95th pct "
          f"{np.percentile(sims, 5):.1f} - {np.percentile(sims, 95):.1f} | P(score >= 80) {np.mean(np.array(sims) >= 80):.0%}")


if __name__ == "__main__":
    main()
