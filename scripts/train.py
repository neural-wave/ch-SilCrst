"""Fit the baseline on the whole development set and save it for inference.py.

Usage: python scripts/train.py DATA_DIR [--augmented cache/augmented.csv] [--out models/baseline.json] [--cache cache]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from devtools.production import load_bell  # noqa: E402
from pvdetect import ENERGY_DEV, LABELS_DEV  # noqa: E402
from pvdetect.extensions import ExtensionConfig  # noqa: E402
from pvdetect.features import extract_features  # noqa: E402
from pvdetect.io import load_consumption, read_labels, read_table  # noqa: E402
from pvdetect.metrics import score_main  # noqa: E402
from pvdetect.model import SCORE_FEATURES, BaselineModel  # noqa: E402
from pvdetect.weather import WeatherIndex  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--score-feature", default="rate60_max", choices=SCORE_FEATURES)
    ap.add_argument("--no-ratio-rule", action="store_true")
    ap.add_argument("--augmented", nargs="+", help="synthetic set(s) from scripts/augment.py; several sets give a capacity ensemble")
    ap.add_argument("--production", default="cache/production_profile.npz",
                    help="production profile from build_production.py (enables the extension detector)")
    ap.add_argument("--ext-top-k", type=int, default=ExtensionConfig.top_k,
                    help="extension events emitted per file (scripts/eval_bonus_policy.py); 0 = threshold mode")
    ap.add_argument("--out", default="models/baseline.json")
    ap.add_argument("--weather", default="weather_index.npz",
                    help="regional weather index from build_weather.py, in the folder of --out (coherence check)")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    data, _ = load_consumption(Path(a.data_dir) / ENERGY_DEV, a.cache)
    labels = read_labels(Path(a.data_dir) / LABELS_DEV)
    bell = load_bell(a.production)
    aug = None
    if a.augmented:
        sets = [read_table(p) for p in a.augmented]
        aug = sets[0] if len(sets) == 1 else sets  # several sets: capacity ensemble
    # Same features as at inference (weather ones included), or the capacity model is trained on
    # columns that inference fills differently.
    weather_path = Path(a.out).parent / a.weather
    weather = WeatherIndex.load(weather_path) if weather_path.exists() else None
    feats = extract_features(data, bell=bell, weather=weather)
    model = BaselineModel(score_feature=a.score_feature, use_ratio_rule=not a.no_ratio_rule).fit(feats, labels, aug)
    model.bell = bell
    if bell is not None:
        e = ExtensionConfig()
        model.extension = {"top_k": a.ext_top_k or None, "ratio_min": e.ratio_min, "ratio_max": e.ratio_max,
                           "delta_min_h": e.delta_min_h, "gain_min": e.gain_min}
    if (Path(a.out).parent / a.weather).exists():
        model.weather_file = a.weather
    else:
        print(f"no {a.weather} next to {a.out}: weather-coherence check off (run scripts/build_weather.py)")
    model.save(a.out)
    s = score_main(BaselineModel.load(a.out).predict(feats), labels)  # reloaded: exactly what inference.py uses
    print(f"in-sample check (optimistic, on the 40 development customers): MAIN {s['main']:.2f} | F1 {s['f1']:.3f} | D {s['D']:.3f} | P {s['P']:.3f}")
    print(f"model fitted on {model.info['n_train']} customers -> {a.out}")
