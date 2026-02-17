"""Prefetch timm backbone weights into the local cache.

Run this on a machine *with outbound internet access* (often a login node).
It will download weights for the backbones used in A1 so compute nodes can run
with OFFLINE=1 and PRETRAINED=1.

Usage:
  source /home/dgarmaev/envs/ar-flares/bin/activate
  export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
  export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
  python scripts/prefetch_timm_backbones.py
"""

from __future__ import annotations

import os

import argparse

import timm


def _prefetch(name: str):
    print(f"Prefetching {name}...")
    m = timm.create_model(name, pretrained=True, num_classes=0)
    # Touch model to ensure weights are loaded
    _ = getattr(m, "num_features", None)
    print(f"OK: {name}")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backbone",
        action="append",
        default=None,
        help="timm backbone name to prefetch; repeat for multiple",
    )
    args = ap.parse_args(argv)

    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("TORCH_HOME:", os.environ.get("TORCH_HOME"))

    backbones = args.backbone or [
        "resnet50",
        "efficientnet_b0",
        "convnext_tiny",
        "swin_tiny_patch4_window7_224",
    ]

    for bb in backbones:
        _prefetch(bb)


if __name__ == "__main__":
    main()
