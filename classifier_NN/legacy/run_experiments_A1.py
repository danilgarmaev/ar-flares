"""Experiment A1: Spatial resolution sweep for single-frame models.

Backbones: ResNet-50, EfficientNet-B0
Resolutions: 224, 112, 56, 28 (downsample->upsample into 224x224)

Runs each configuration for N seeds and reports mean±std TSS.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from statistics import mean, stdev

from ..config import get_default_cfg
from ..train import main


RESOLUTIONS = [224, 112, 56, 28]
BACKBONES = ["resnet50", "efficientnet_b0"]
N_RUNS_DEFAULT = 5


def _resolution_factor(resolution: int) -> int:
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Unsupported resolution: {resolution}")
    return 224 // resolution


def _build_cfg(name: str, overrides: dict, common_overrides: dict | None = None) -> dict:
    cfg = get_default_cfg()
    if common_overrides:
        cfg.update(common_overrides)
    cfg.update(overrides)
    cfg["model_name"] = name
    cfg["run_id"] = name
    return cfg


COMMON_A1_OVERRIDES = {
    # single-frame only
    "use_seq": False,
    "use_flow": False,
    "two_stream": False,
    "use_diff": False,
    "use_diff_attention": False,

    # input resolution (always feed 224x224 into model)
    "image_size": 224,

    # training
    "batch_size": 64,
    "optimizer": "adam_paper",
    "lr": 1e-3,
    "epochs": 50,

    # model
    # Default to pretrained transfer learning; override with --no-pretrained or --smoke.
    "pretrained": True,
    "freeze_backbone": True,

    # imbalance handling
    "balance_classes": False,
    "loss_type": "ce_weighted",

    # no augmentation
    "use_aug": False,

    # model selection / logging
    "model_selection": "tss",
    "redirect_log": True,

    # early stopping
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.0,
}


def run_single(*, backbone: str, resolution: int, seed: int, epochs: int | None = None):
    factor = _resolution_factor(resolution)
    name = f"A1-{resolution}_{backbone}_seed{seed}"
    overrides = {
        "backbone": backbone,
        "spatial_downsample_factor": factor,
        "seed": seed,
        "notes": (
            "A1 resolution sweep (single-frame): downsample->upsample before backbone. "
            f"Backbone={backbone}, resolution={resolution}, seed={seed}."
        ),
    }
    if epochs is not None:
        overrides["epochs"] = int(epochs)

    print("\n" + "=" * 80)
    print(f"Launching: {name} | spatial_downsample_factor={factor}")
    print("=" * 80 + "\n")

    cfg = _build_cfg(name, overrides, COMMON_A1_OVERRIDES)
    # If we're not using pretrained weights, freezing the backbone would make
    # training ineffective (random frozen features). Flip to full finetune.
    if not bool(cfg.get("pretrained", True)):
        cfg["freeze_backbone"] = False
    exp_dir, results = main(cfg)
    return exp_dir, results


def run_sweep(
    *,
    backbones: list[str] | None = None,
    resolutions: list[int] | None = None,
    seeds: list[int] | None = None,
    epochs: int | None = None,
    common_overrides: dict | None = None,
    write_summary: bool = True,
):
    backbones = backbones or list(BACKBONES)
    resolutions = resolutions or list(RESOLUTIONS)
    seeds = seeds or list(range(N_RUNS_DEFAULT))

    summary = {
        "metric": "TSS",
        "resolutions": resolutions,
        "backbones": {},
        "n_runs": len(seeds),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    for backbone in backbones:
        summary["backbones"][backbone] = {}
        for resolution in resolutions:
            tss_values = []
            for seed in seeds:
                try:
                    if common_overrides:
                        old_common = dict(COMMON_A1_OVERRIDES)
                        try:
                            COMMON_A1_OVERRIDES.update(common_overrides)
                            _, results = run_single(
                                backbone=backbone,
                                resolution=resolution,
                                seed=seed,
                                epochs=epochs,
                            )
                        finally:
                            COMMON_A1_OVERRIDES.clear()
                            COMMON_A1_OVERRIDES.update(old_common)
                    else:
                        _, results = run_single(
                            backbone=backbone,
                            resolution=resolution,
                            seed=seed,
                            epochs=epochs,
                        )
                    tss_values.append(float(results["TSS"]))
                except Exception as e:
                    print(f"❌ Run failed: backbone={backbone}, res={resolution}, seed={seed} -> {e}")
            if not tss_values:
                continue
            tss_mean = mean(tss_values)
            tss_std = stdev(tss_values) if len(tss_values) > 1 else 0.0
            summary["backbones"][backbone][str(resolution)] = {
                "tss_values": tss_values,
                "mean_tss": tss_mean,
                "std_tss": tss_std,
                "n": len(tss_values),
            }
            print(
                f"\n[{backbone} | {resolution}] TSS = {tss_mean:.4f} ± {tss_std:.4f} (n={len(tss_values)})\n"
            )

    if write_summary:
        # Save summary to results directory
        cfg = get_default_cfg()
        out_dir = cfg.get("results_base", ".")
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, "summary_a1_resolution_sweep.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {summary_path}")
        return summary_path
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backbone",
        action="append",
        choices=BACKBONES,
        default=None,
        help="Backbone(s) to run. Repeat for multiple backbones.",
    )
    ap.add_argument(
        "--resolution",
        action="append",
        type=int,
        default=None,
        help="Resolution(s) to run (224, 112, 56, 28). Repeat for multiple.",
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=N_RUNS_DEFAULT,
        help="Number of runs per config (used to generate seeds 0..N-1).",
    )
    ap.add_argument(
        "--seed",
        action="append",
        type=int,
        default=None,
        help="Explicit seed(s) to use. Overrides --runs if provided.",
    )
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs for all runs")
    ap.add_argument("--smoke", action="store_true", help="Tiny smoke run (1 epoch, shard/batch limits, no pretrained)")
    ap.add_argument(
        "--single",
        action="store_true",
        help="Run exactly one (backbone,resolution,seed) config and do not write the sweep summary JSON",
    )
    ap.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=None,
        help="Use pretrained weights (requires cached weights or network access)",
    )
    ap.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        default=None,
        help="Do not use pretrained weights (safe default on compute nodes)",
    )
    args = ap.parse_args()

    resolutions = args.resolution if args.resolution else None
    backbones = args.backbone if args.backbone else None
    seeds = args.seed if args.seed is not None else list(range(int(args.runs)))

    common_overrides = None
    if args.smoke:
        common_overrides = {
            "epochs": 1,
            "batch_size": 16,
            "num_workers": 0,
            "persistent_workers": False,
            "redirect_log": False,
            "val_max_batches": 5,
            "max_train_shards": 2,
            "max_val_shards": 1,
            "max_test_shards": 1,
            "early_stopping_patience": 2,
            "pretrained": False,
        }
    if args.pretrained is not None:
        if common_overrides is None:
            common_overrides = {}
        common_overrides["pretrained"] = bool(args.pretrained)

    if args.single:
        if not backbones or len(backbones) != 1:
            raise SystemExit("--single requires exactly one --backbone")
        if not resolutions or len(resolutions) != 1:
            raise SystemExit("--single requires exactly one --resolution")
        if args.seed is None or len(args.seed) != 1:
            raise SystemExit("--single requires exactly one --seed")
        run_single(
            backbone=backbones[0],
            resolution=int(resolutions[0]),
            seed=int(args.seed[0]),
            epochs=args.epochs,
        )
    else:
        run_sweep(
            backbones=backbones,
            resolutions=resolutions,
            seeds=seeds,
            epochs=args.epochs,
            common_overrides=common_overrides,
            write_summary=True,
        )
