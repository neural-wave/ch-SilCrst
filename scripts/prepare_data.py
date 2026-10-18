"""Parse every consumption CSV of a data folder into the local cache and validate it.

Usage: python scripts/prepare_data.py DATA_DIR [--cache cache]

Prints data-quality checks. The cache holds AIL data: it lives in cache/ (git-ignored).
The unlabelled file (~8 GB) takes a few minutes and a few GB of RAM.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pvdetect.io import CUSTOMER_ID_PREFIX, customers_only, load_consumption  # noqa: E402
from pvdetect.validate import validation_report  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    for csv in sorted(Path(a.data_dir).glob("energy_consumption*.csv")):
        t = time.time()
        data, report = load_consumption(csv, a.cache)
        print(f"== {csv.name} ({time.time() - t:.0f} s)")
        for line in validation_report(data, report):
            print("  " + line)
        print(f"  series whose id does not start with {CUSTOMER_ID_PREFIX!r} (not customers, excluded from training): "
              f"{customers_only(data)[1]}")
