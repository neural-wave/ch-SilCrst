"""Generate a fake development set with the AIL schema (for testing code without AIL data).

Usage: python scripts/make_synthetic.py [out_dir] [--customers 40] [--start 2022-01-01] [--end 2025-12-31] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from devtools.synthetic import generate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="synthetic_data")
    ap.add_argument("--customers", type=int, default=40)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(a.out_dir, a.customers, a.start, a.end, seed=a.seed)
    print(f"synthetic development set written to {a.out_dir}/")
