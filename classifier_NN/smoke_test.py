"""Lightweight smoke tests for classifier_NN components.

This module intentionally supports two levels of smoke tests:

- Default (data-free): validates imports, model construction, and a forward pass
  with synthetic input. This should succeed on fresh environments even when the
  dataset is not present yet.
- Optional (data-backed): validates that the WebDataset shards are reachable and
  that a real dataloader batch can be produced.

Run with:
    python -m classifier_NN.smoke_test
"""

import sys
import os

import torch

from .config import get_default_cfg, CFG, SPLIT_DIRS
from .models import build_model


def _have_train_shards() -> bool:
    """Return True if Train split directory exists and has at least one .tar shard."""
    train_dir = SPLIT_DIRS.get("Train")
    if not train_dir or not os.path.isdir(train_dir):
        return False
    try:
        return any(name.endswith(".tar") for name in os.listdir(train_dir))
    except OSError:
        return False


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

    # Model forward smoke (data-free, always runs)
    try:
        model = build_model(cfg=cfg, num_classes=2)
        model.eval()
        with torch.no_grad():
            backbone = str(cfg.get("backbone", "")).lower()

            # Default config uses a 3D backbone; generate the correct synthetic
            # input shape based on backbone family.
            video_backbones = {
                "simple3dcnn",
                "3d_cnn",
                "resnet3d_simple",
                "r3d_18",
                "r3d18",
                "slowfast",
                "slowfast_r50",
                "video_transformer",
                "timeformer",
                "video_vit",
            }

            if backbone in video_backbones:
                t = int(cfg.get("seq_T", 3))
                x = torch.randn(cfg["batch_size"], t, 1, cfg["image_size"], cfg["image_size"])
            else:
                x = torch.randn(cfg["batch_size"], 1, cfg["image_size"], cfg["image_size"])

            out = model(x)
        print("[SMOKE] Model forward ok - output shape:", out.shape)
    except Exception as e:
        print("[SMOKE] Model forward failed:", e)
        return 1

    # Optional dataloader smoke (only if shards exist)
    if _have_train_shards():
        try:
            from .datasets import create_dataloaders
            dls = create_dataloaders()
            batch = next(iter(dls["Train"]))
            x, y, meta = batch
            print(
                "[SMOKE] Dataloader ok - batch shapes:",
                (x[0].shape, x[1].shape) if isinstance(x, (list, tuple)) else x.shape,
                "labels shape:", y.shape,
            )
        except Exception as e:
            print("[SMOKE] Dataloader failed:", e)
            return 1
    else:
        print(f"[SMOKE] Dataloader skipped (no shards found under: {SPLIT_DIRS.get('Train')})")

    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_tests())
