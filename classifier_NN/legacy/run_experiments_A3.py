"""Experiment A3: Temporal sampling interval study for spatio-temporal models.

This script runs controlled sweeps for sequence models with T=16 frames.

Two studies:
- cadence: fixed window (6h), vary sampling interval by snapping sampled times
  to multiples of the desired cadence.
- context: fixed number of frames (T=16), vary cadence -> total temporal
  coverage increases (3h/6h/12h/24h for 12/24/48/96 min).

Models: use existing sequence-capable backbones in `classifier_NN.models`:
- r3d_18 (torchvision r3d_18 wrapper)
- slowfast (torch.hub facebookresearch/pytorchvideo slowfast_r50)
- video_transformer (2D backbone + temporal transformer)

Notes
-----
- This repo's shard ordering is assumed to be native 12-minute cadence.
- We rely on dataset-side padding (repeat earliest AR frame) for short histories.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from statistics import mean, stdev

import numpy as np

from ..config import get_default_cfg
from ..train import main as train_main


INTERVALS_MIN = [12, 24, 48, 96]
T_DEFAULT = 16
IMG_SIZE_DEFAULT = 112
FIXED_WINDOW_MIN = 6 * 60


def _sanitize(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def _interval_to_stride(interval_min: int) -> int:
    if interval_min % 12 != 0:
        raise ValueError(f"interval_min must be a multiple of 12, got {interval_min}")
    return int(interval_min // 12)


def _offsets_context(*, interval_min: int, T: int) -> list[int]:
    """Fixed T, stride determines coverage: offsets = [-15*s, ..., 0]."""
    s = _interval_to_stride(interval_min)
    return list(range(-(T - 1) * s, 1, s))


def _offsets_cadence_fixed_window(*, interval_min: int, window_min: int, T: int) -> list[int]:
    """Fixed window (e.g. 6h), uniformly sample T timestamps in [-window,0].

    Then snap each sampled time to the nearest multiple of the desired cadence.
    Offsets are in units of 12-minute steps.

    This can create duplicates for coarse cadences; duplicates naturally repeat
    the same frame (effective temporal padding within the fixed window).
    """
    window_steps = int(round(window_min / 12))
    stride = _interval_to_stride(interval_min)

    # Uniform positions in base 12-min steps
    xs = np.linspace(-window_steps, 0, T)
    # Snap to nearest cadence multiple
    snapped = [int(round(x / stride) * stride) for x in xs]
    # Ensure last is exactly 0
    snapped[-1] = 0
    # Clamp to [-window_steps, 0]
    snapped = [max(-window_steps, min(0, int(v))) for v in snapped]
    # Make non-decreasing (avoid slight rounding inversions)
    for i in range(1, len(snapped)):
        if snapped[i] < snapped[i - 1]:
            snapped[i] = snapped[i - 1]
    return snapped


def _build_cfg(*, name: str, common: dict, overrides: dict) -> dict:
    cfg = get_default_cfg()
    cfg.update(common)
    cfg.update(overrides)
    cfg["model_name"] = name
    cfg["run_id"] = name
    return cfg


COMMON_A3 = {
    # sequence mode
    "use_seq": True,
    "use_flow": False,
    "two_stream": False,
    "use_diff": False,
    "use_diff_attention": False,

    # data / model
    "image_size": IMG_SIZE_DEFAULT,
    "seq_T": T_DEFAULT,

    # training defaults (override as needed)
    "batch_size": 16,
    "num_workers": 2,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "lr": 1e-4,
    "epochs": 50,
    "scheduler": "cosine",

    # imbalance handling
    "balance_classes": False,
    "loss_type": "ce_weighted",

    # augmentation (keep simple for sequences initially)
    "use_aug": False,

    # model selection / early stopping
    "model_selection": "tss",
    "redirect_log": True,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.0,
}


def run_one(
    *,
    backbone: str,
    study: str,
    interval_min: int,
    seed: int,
    name_suffix: str | None,
    fixed_window_min: int,
    T: int,
    img_size: int,
    common_overrides: dict | None,
):
    study = str(study).lower()
    if study not in {"context", "cadence"}:
        raise ValueError("study must be 'context' or 'cadence'")

    if study == "context":
        offsets = _offsets_context(interval_min=interval_min, T=T)
        window_desc = f"context={(T - 1) * interval_min}min"
    else:
        offsets = _offsets_cadence_fixed_window(interval_min=interval_min, window_min=fixed_window_min, T=T)
        window_desc = f"window={fixed_window_min}min"

    stride_steps = _interval_to_stride(interval_min)

    name = f"A3-{study}-{interval_min}min_{backbone}_seed{seed}"
    if name_suffix:
        name = f"{name}_{_sanitize(name_suffix)}"

    overrides = {
        "backbone": backbone,
        "seed": seed,
        "image_size": int(img_size),
        "seq_T": int(T),
        "seq_offsets": list(map(int, offsets)),
        # Reduce overlap / sample explosion: advance start indices at cadence stride.
        "seq_stride": int(stride_steps),
        "seq_stride_steps": int(stride_steps),
        "notes": (
            f"A3 temporal sampling study ({study}): interval={interval_min}min, {window_desc}, "
            f"T={T}, img={img_size}, seed={seed}."
        ),
    }

    common = dict(COMMON_A3)
    if common_overrides:
        common.update(common_overrides)

    cfg = _build_cfg(name=name, common=common, overrides=overrides)
    exp_dir, results = train_main(cfg)
    return exp_dir, results


def run_sweep(
    *,
    backbones: list[str],
    study: str,
    intervals_min: list[int],
    seeds: list[int],
    name_suffix: str | None,
    fixed_window_min: int,
    T: int,
    img_size: int,
    common_overrides: dict | None,
):
    summary = {
        "study": study,
        "intervals_min": intervals_min,
        "T": T,
        "image_size": img_size,
        "metric": "TSS",
        "backbones": {},
        "n_runs": len(seeds),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    for bb in backbones:
        summary["backbones"][bb] = {}
        for interval in intervals_min:
            values = []
            exp_dirs = []
            for seed in seeds:
                try:
                    exp_dir, res = run_one(
                        backbone=bb,
                        study=study,
                        interval_min=int(interval),
                        seed=int(seed),
                        name_suffix=name_suffix,
                        fixed_window_min=fixed_window_min,
                        T=T,
                        img_size=img_size,
                        common_overrides=common_overrides,
                    )
                    exp_dirs.append(exp_dir)
                    values.append(float(res.get("test_tss_at_val_threshold", res.get("TSS"))))
                except Exception as e:
                    print(f"❌ failed: bb={bb} interval={interval} seed={seed}: {e}")
            if not values:
                continue
            m = mean(values)
            s = stdev(values) if len(values) > 1 else 0.0
            summary["backbones"][bb][str(interval)] = {
                "tss_values": values,
                "mean_tss": m,
                "std_tss": s,
                "n": len(values),
                "exp_dirs": exp_dirs,
            }
            print(f"[{bb} | {interval}min] TSS={m:.4f} ± {s:.4f} (n={len(values)})")

    cfg0 = get_default_cfg()
    out_dir = cfg0.get("results_base", ".")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"summary_a3_{study}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written: {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment A3: temporal sampling interval study")
    ap.add_argument("--study", choices=["context", "cadence"], default="context")
    ap.add_argument("--backbone", action="append", default=None, help="Backbone(s): r3d_18, slowfast, video_transformer")
    ap.add_argument("--interval-min", action="append", type=int, default=None, help="Intervals (min): 12/24/48/96")
    ap.add_argument("--runs", type=int, default=3, help="Number of seeds (0..runs-1)")
    ap.add_argument("--seed", action="append", type=int, default=None, help="Explicit seed(s); overrides --runs")
    ap.add_argument("--T", type=int, default=T_DEFAULT)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE_DEFAULT)
    ap.add_argument("--fixed-window-min", type=int, default=FIXED_WINDOW_MIN)
    ap.add_argument("--single", action="store_true", help="Run one config (requires exactly one backbone/interval/seed)")
    ap.add_argument("--name-suffix", type=str, default=None)

    # Training knobs (optional)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--optimizer", type=str, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--scheduler", type=str, default=None)
    ap.add_argument("--pretrained-3d", action="store_true", default=None, help="Use pretrained weights for 3D backbones")
    ap.add_argument("--no-pretrained-3d", dest="pretrained_3d", action="store_false", default=None)

    args = ap.parse_args()

    backbones = args.backbone or ["r3d_18", "slowfast"]
    intervals = args.interval_min or list(INTERVALS_MIN)
    seeds = args.seed if args.seed is not None else list(range(int(args.runs)))

    common_overrides: dict = {}
    for k, v in {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
    }.items():
        if v is not None:
            common_overrides[k] = v
    if args.pretrained_3d is not None:
        common_overrides["pretrained_3d"] = bool(args.pretrained_3d)

    if not common_overrides:
        common_overrides = None

    if args.single:
        if len(backbones) != 1 or len(intervals) != 1 or len(seeds) != 1:
            raise SystemExit("--single requires exactly one backbone, one interval, one seed")
        run_one(
            backbone=backbones[0],
            study=args.study,
            interval_min=int(intervals[0]),
            seed=int(seeds[0]),
            name_suffix=args.name_suffix,
            fixed_window_min=int(args.fixed_window_min),
            T=int(args.T),
            img_size=int(args.img_size),
            common_overrides=common_overrides,
        )
    else:
        run_sweep(
            backbones=backbones,
            study=args.study,
            intervals_min=[int(x) for x in intervals],
            seeds=[int(x) for x in seeds],
            name_suffix=args.name_suffix,
            fixed_window_min=int(args.fixed_window_min),
            T=int(args.T),
            img_size=int(args.img_size),
            common_overrides=common_overrides,
        )


if __name__ == "__main__":
    main()
