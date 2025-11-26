"""Lightweight smoke tests for classifier_NN components.

Run with:
    python -m classifier_NN.smoke_test
"""

import sys

import torch

from .config import get_default_cfg, CFG
from .datasets import create_dataloaders
from .models import build_model


def run_smoke_tests():
    # Build a small cfg for quick CPU smoke test
    cfg = get_default_cfg()
    cfg["image_size"] = 64
    cfg["batch_size"] = 2
    cfg["num_workers"] = 0
    cfg["use_flow"] = False
    cfg["use_seq"] = False
    cfg["use_diff_attention"] = False

    # Update global CFG so datasets/model see it
    CFG.update(cfg)

    # Dataloader smoke
    try:
        dls = create_dataloaders()
        batch = next(iter(dls["Train"]))
        x, y, meta = batch
        print("[SMOKE] Dataloader ok - batch shapes:",
              (x[0].shape, x[1].shape) if isinstance(x, (list, tuple)) else x.shape,
              "labels shape:", y.shape)
    except Exception as e:
        print("[SMOKE] Dataloader failed:", e)
        return 1

    # Model forward smoke
    try:
        model = build_model(num_classes=2)
        model.eval()
        with torch.no_grad():
            if isinstance(x, (list, tuple)):
                out = model(tuple(xi for xi in x))
            else:
                out = model(x)
        print("[SMOKE] Model forward ok - output shape:", out.shape)
    except Exception as e:
        print("[SMOKE] Model forward failed:", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_tests())
