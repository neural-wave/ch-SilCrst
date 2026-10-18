"""Export the ViT weights the capacity port needs (pvdetect/vit_capacity.py) from a torch bundle.

Usage: python scripts/export_vit_capacity.py path/to/base_0.pt [--out models/vit_capacity.npz]

Needs torch (development only; inference reads the .npz with numpy). The bundle is the team's
`models/vit/base_0.pt` (synthetic stage of the multi-task ViT, seed 0, clean pretraining).
"""
import argparse

import numpy as np
import torch

# Heads the capacity port does not run: reconstruction, presence, verdict, extensions.
UNUSED = ("recon.", "presence.", "detect.", "ext_day.", "ext_none.", "ext_capacity.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--out", default="models/vit_capacity.npz")
    a = ap.parse_args()
    bundle = torch.load(a.bundle, map_location="cpu", weights_only=True)
    cfg = bundle["config"]
    assert (cfg["d_model"], cfg["n_heads"]) == (128, 4), f"the port assumes width 128 and 4 heads, got {cfg}"
    weights = {k: v.float().numpy() for k, v in bundle["state_dict"].items() if not k.startswith(UNUSED)}
    np.savez_compressed(a.out, **weights)
    print(f"{len(weights)} arrays, {sum(v.size for v in weights.values()):,} parameters -> {a.out} ({bundle['meta']})")


if __name__ == "__main__":
    main()
