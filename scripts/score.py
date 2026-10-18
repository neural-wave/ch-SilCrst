"""Score submission files against labels with the official formula.

Usage: python scripts/score.py predictions.csv LABELS.csv [pv_extensions_predictions.csv EXTENSION_LABELS.csv]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pvdetect.io import read_extension_labels, read_labels  # noqa: E402
from pvdetect.metrics import score_extensions, score_main  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) not in (3, 5):
        sys.exit(__doc__)
    s = score_main(read_labels(sys.argv[1]), read_labels(sys.argv[2]))
    print(f"MAIN {s['main']:.2f} | F1 {s['f1']:.3f} | D {s['D']:.3f} | P {s['P']:.3f} "
          f"| TP {s['tp']} FP {s['fp']} FN {s['fn']} TN {s['tn']}")
    if len(sys.argv) == 5:
        e = score_extensions(read_extension_labels(sys.argv[3]), read_extension_labels(sys.argv[4]))
        print(f"BONUS {e['bonus']:.2f} | F1_ext {e['f1_ext']:.3f} | D_ext {e['D_ext']:.3f} | P_ext {e['P_ext']:.3f}")
