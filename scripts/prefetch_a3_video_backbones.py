"""Prefetch weights for A3 video backbones into TORCH_HOME.

Run this on a login node with internet, with caches pointed at scratch:

  export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
  python scripts/prefetch_a3_video_backbones.py --r3d18 --slowfast

Notes:
- r3d_18 weights come from torchvision.
- SlowFast weights come from torch.hub (facebookresearch/pytorchvideo).
"""

from __future__ import annotations

import argparse


def _prefetch_r3d18() -> None:
    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights
        _ = r3d_18(weights=R3D_18_Weights.DEFAULT)
        print("✅ Prefetched torchvision r3d_18 weights")
    except Exception as e:
        print(f"❌ Failed to prefetch r3d_18: {e}")


def _prefetch_slowfast() -> None:
    try:
        import torch

        _ = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "slowfast_r50",
            pretrained=True,
        )
        print("✅ Prefetched torch.hub slowfast_r50 weights")
    except Exception as e:
        print(f"❌ Failed to prefetch SlowFast: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r3d18", action="store_true")
    ap.add_argument("--slowfast", action="store_true")
    args = ap.parse_args()

    if not (args.r3d18 or args.slowfast):
        raise SystemExit("Pick at least one: --r3d18, --slowfast")

    if args.r3d18:
        _prefetch_r3d18()
    if args.slowfast:
        _prefetch_slowfast()


if __name__ == "__main__":
    main()
