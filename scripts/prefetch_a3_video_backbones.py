"""Prefetch weights for A3 video backbones into TORCH_HOME.

Run this on a login node with internet, with caches pointed at scratch:

  export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
        python scripts/prefetch_a3_video_backbones.py --r3d18 --r2plus1d18 --videoswin

Notes:
- r3d_18 weights come from torchvision.
- r2plus1d_18 weights come from torchvision.
- MViT weights come from torchvision.
- SlowFast weights come from torch.hub (facebookresearch/pytorchvideo).
- Video Swin (swin3d_t) weights come from torchvision.
- TimeSformer weights come from HuggingFace transformers cache.
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


def _prefetch_r2plus1d18() -> None:
    try:
        from torchvision.models.video import r2plus1d_18

        try:
            # Newer torchvision
            from torchvision.models.video import R2Plus1D_18_Weights

            _ = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        except Exception:
            # Older torchvision fallback
            _ = r2plus1d_18(pretrained=True)
        print("✅ Prefetched torchvision r2plus1d_18 weights")
    except Exception as e:
        print(f"❌ Failed to prefetch r2plus1d_18: {e}")


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


def _prefetch_videoswin() -> None:
    try:
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights

        _ = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)
        print("✅ Prefetched torchvision swin3d_t weights")
    except Exception as e:
        print(f"❌ Failed to prefetch Video Swin (swin3d_t): {e}")


def _prefetch_mvit(variant: str) -> None:
    tried = []
    try:
        from torchvision.models import video as tv_video
    except Exception as e:
        print(f"❌ Failed to import torchvision video models for MViT prefetch: {e}")
        return

    candidates = []
    requested = str(variant).lower().strip()
    if requested in {"mvit", "mvit_v2_s", "mvitv2_s"}:
        candidates.append(("mvit_v2_s", "MViT_V2_S_Weights"))
    elif requested in {"mvit_v1_b", "mvitv1_b"}:
        candidates.append(("mvit_v1_b", "MViT_V1_B_Weights"))
    else:
        candidates.append((requested, None))
    for candidate in [("mvit_v2_s", "MViT_V2_S_Weights"), ("mvit_v1_b", "MViT_V1_B_Weights")]:
        if candidate not in candidates:
            candidates.append(candidate)

    for fn_name, weight_name in candidates:
        try:
            builder = getattr(tv_video, fn_name)
            if weight_name is not None and hasattr(tv_video, weight_name):
                weights_enum = getattr(tv_video, weight_name)
                _ = builder(weights=weights_enum.DEFAULT)
            else:
                _ = builder(pretrained=True)
            print(f"✅ Prefetched torchvision {fn_name} weights")
            return
        except Exception as e:
            tried.append(f"{fn_name}: {e}")

    print(f"❌ Failed to prefetch MViT ({variant}). Tried: {'; '.join(tried)}")


def _prefetch_timesformer(model_id: str) -> None:
    try:
        from transformers import TimesformerForVideoClassification

        _ = TimesformerForVideoClassification.from_pretrained(model_id)
        print(f"✅ Prefetched HF TimeSformer weights: {model_id}")
    except Exception as e:
        print(f"❌ Failed to prefetch TimeSformer ({model_id}): {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r3d18", action="store_true")
    ap.add_argument("--r2plus1d18", action="store_true")
    ap.add_argument("--slowfast", action="store_true")
    ap.add_argument("--videoswin", action="store_true")
    ap.add_argument("--mvit", action="store_true")
    ap.add_argument("--mvit-model-id", type=str, default="mvit_v2_s")
    ap.add_argument("--timesformer", action="store_true")
    ap.add_argument("--timesformer-model-id", type=str, default="facebook/timesformer-base-finetuned-k400")
    args = ap.parse_args()

    if not (args.r3d18 or args.r2plus1d18 or args.slowfast or args.videoswin or args.mvit or args.timesformer):
        raise SystemExit("Pick at least one: --r3d18, --r2plus1d18, --slowfast, --videoswin, --mvit, --timesformer")

    if args.r3d18:
        _prefetch_r3d18()
    if args.r2plus1d18:
        _prefetch_r2plus1d18()
    if args.slowfast:
        _prefetch_slowfast()
    if args.videoswin:
        _prefetch_videoswin()
    if args.mvit:
        _prefetch_mvit(str(args.mvit_model_id))
    if args.timesformer:
        _prefetch_timesformer(str(args.timesformer_model_id))


if __name__ == "__main__":
    main()
