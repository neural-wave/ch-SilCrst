"""Leave-one-out evaluation of the baseline on the development set, with the official score.

Usage: python scripts/evaluate.py DATA_DIR [--cache cache] [--details outputs/loo_details.csv]

The per-customer details file holds AIL data: it stays in outputs/ (git-ignored).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from devtools.production import load_bell  # noqa: E402
from pvdetect import ENERGY_DEV, EXTENSIONS_DEV, LABELS_DEV  # noqa: E402
from pvdetect.features import extract_features  # noqa: E402
from pvdetect.io import load_consumption, read_extension_labels, read_labels, read_table  # noqa: E402
from pvdetect.metrics import score_extensions, score_main  # noqa: E402
from pvdetect.model import SCORE_FEATURES, BaselineModel  # noqa: E402
from pvdetect.validate import validation_report  # noqa: E402


def leave_one_out(feats: pd.DataFrame, labels: pd.DataFrame, augmented: pd.DataFrame | None = None,
                  **model_args) -> pd.DataFrame:
    preds = []
    for cid in labels["customer_id"]:
        train = labels[labels["customer_id"] != cid]
        model = BaselineModel(**model_args).fit(feats, train, augmented)
        preds.append(model.predict(feats[feats["customer_id"] == cid]))
    return pd.concat(preds, ignore_index=True)


def fmt(s: dict) -> str:
    return (f"MAIN {s['main']:6.2f} | F1 {s['f1']:.3f} (P {s['precision']:.2f} R {s['recall']:.2f}) "
            f"| D {s['D']:.3f} | P {s['P']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--details", default="outputs/loo_details.csv")
    ap.add_argument("--augmented", nargs="+", help="synthetic set(s) from scripts/augment.py; several sets give a capacity ensemble")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)

    data, report = load_consumption(data_dir / ENERGY_DEV, a.cache)
    labels = read_labels(data_dir / LABELS_DEV)
    ext_labels = read_extension_labels(data_dir / EXTENSIONS_DEV)
    print("== Data validation")
    for line in validation_report(data, report):
        print("  " + line)
    missing = sorted(set(labels["customer_id"]) - set(data))
    print(f"  labels: {len(labels)} customers, {int(labels['pv_installed'].sum())} with PV, "
          f"{len(missing)} without series | extension events: {len(ext_labels)}")

    weather = BaselineModel.load("models/baseline.json").weather  # None if no index: check off
    feats = extract_features(data, bell=load_bell(Path(a.cache) / "production_profile.npz"), weather=weather)
    labelled = feats[feats["customer_id"].isin(labels["customer_id"])]

    print(f"\n== Leave-one-out over {len(labels)} customers")
    results = {}
    for sf in SCORE_FEATURES:
        for ratio in (False, True):
            args = {"score_feature": sf, "use_ratio_rule": ratio}
            results[(sf, ratio)] = score_main(leave_one_out(labelled, labels, **args), labels)
            print(f"  score={sf:<18} ratio rule={'on ' if ratio else 'off'}  {fmt(results[(sf, ratio)])}")
    best_sf, best_ratio = max(results, key=lambda k: results[k]["main"])
    best_args = {"score_feature": best_sf, "use_ratio_rule": best_ratio}
    aug = None
    if a.augmented:
        sets = [read_table(p) for p in a.augmented]
        aug = sets[0] if len(sets) == 1 else sets  # several sets: capacity ensemble
    preds = leave_one_out(labelled, labels, aug, **best_args)
    s = score_main(preds, labels)

    cap = "boosting on synthetic + real" if aug is not None else "linear on real"
    print(f"\n== Best: score={best_sf}, ratio rule={'on' if best_ratio else 'off'}, capacity: {cap}")
    print(f"  {fmt(s)}")
    print(f"  detection: TP {s['tp']}  FP {s['fp']}  FN {s['fn']}  TN {s['tn']}")
    print(f"  date (TP): median |error| {s['date_abs_err_median']:.0f} d | <=7 d: {s['date_within_7']}/{s['tp']} "
          f"| <30 d: {s['date_within_30']}/{s['tp']}")
    print(f"  capacity (TP): median APE {s['ape_median']:.0%} | <=25%: {s['ape_within_25']}/{s['tp']} "
          f"| <50%: {s['ape_within_50']}/{s['tp']}")
    saved = BaselineModel.load("models/baseline.json")
    ext_pred = saved.predict_extensions(data, preds)
    ext = score_extensions(ext_pred, ext_labels)
    print(f"  extension bonus: {ext['bonus']:.2f}/10 (TP {ext['tp']} FP {ext['fp']} FN {ext['fn']}; "
          f"detector {'on' if saved.extension else 'off'} in models/baseline.json)")
    in_sample = score_main(BaselineModel(**best_args).fit(labelled, labels, aug).predict(labelled), labels)
    print(f"  in-sample (fit on all 40, optimistic): {fmt(in_sample)}")

    details = labels.merge(preds, on="customer_id", suffixes=("_true", "_pred")).merge(labelled, on="customer_id")
    Path(a.details).parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(a.details, index=False)
    print(f"\nper-customer details -> {a.details}  (AIL data, git-ignored)")


if __name__ == "__main__":
    main()
