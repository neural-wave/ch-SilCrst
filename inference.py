"""Submission entry point.

    python inference.py path/to/test_set.csv

Writes predictions.csv and pv_extensions_predictions.csv to the current working directory.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pvdetect.features import extract_features  # noqa: E402
from pvdetect.io import read_consumption  # noqa: E402
from pvdetect.model import BaselineModel  # noqa: E402
from pvdetect.output import write_extensions, write_predictions  # noqa: E402
from pvdetect.vit_capacity import VitCapacity, with_vit_capacity  # noqa: E402

MODEL_PATH = HERE / "models" / "baseline.json"
VIT_PATH = HERE / "models" / "vit_capacity.npz"  # initial kWp of the PV customers


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python inference.py path/to/test_set.csv", file=sys.stderr)
        return 2
    data, _ = read_consumption(argv[1])
    model = BaselineModel.load(MODEL_PATH)
    if model.weather_file and model.weather is None:
        print(f"warning: models/{model.weather_file} not found: weather-coherence check off", file=sys.stderr)
    feats = extract_features(data, bell=model.bell, weather=model.weather)
    pred = model.predict(feats)
    if VIT_PATH.exists():  # before the extensions: their kWp is the initial one times a ratio
        pred = with_vit_capacity(pred, data, VitCapacity.load(VIT_PATH), model.cap_clip_kwp)
    else:
        print(f"warning: models/{VIT_PATH.name} not found: initial kWp from the boosting model", file=sys.stderr)
    ext = model.predict_extensions(data, pred)

    write_predictions(pred, Path.cwd() / "predictions.csv")
    write_extensions(ext, Path.cwd() / "pv_extensions_predictions.csv")
    print(f"{len(pred)} customers, {int(pred['pv_installed'].sum())} predicted with PV, "
          f"{len(ext)} extension events -> predictions.csv, pv_extensions_predictions.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
