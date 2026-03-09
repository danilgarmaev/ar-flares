"""Experiment: 16-frame sequences with varying cadences (36, 72, 108 minutes).

This script tests video models with 16 frames at different temporal sampling rates:
- 36 min cadence: 9 hours total temporal context
- 72 min cadence: 18 hours total temporal context (matches macros project)
- 108 min cadence: 27 hours total temporal context

Models to test:
- r2plus1d_18 (R(2+1)D from torchvision)
- VideoMAE (when available, needs to be implemented)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from statistics import mean, stdev

from ..config import get_default_cfg
from ..train import main as train_main


def _interval_to_stride(interval_min: int) -> int:
    """Convert cadence interval in minutes to stride in 12-minute steps."""
    if interval_min % 12 != 0:
        raise ValueError(f"interval_min must be a multiple of 12, got {interval_min}")
    return int(interval_min // 12)


def _generate_offsets(interval_min: int, T: int = 16) -> list[int]:
    """Generate offsets for T frames with given cadence interval.
    
    Args:
        interval_min: Cadence in minutes (e.g. 36, 72, 108)
        T: Number of frames (default 16)
    
    Returns:
        List of offsets in 12-minute steps, e.g. [-90, -84, ..., -6, 0]
    
    Examples:
        36 min (stride=3): [-45, -42, -39, ..., -3, 0] = 9 hours
        72 min (stride=6): [-90, -84, -78, ..., -6, 0] = 18 hours  
        108 min (stride=9): [-135, -126, -117, ..., -9, 0] = 27 hours
    """
    stride = _interval_to_stride(interval_min)
    offsets = list(range(-(T - 1) * stride, 1, stride))
    return offsets


COMMON_CONFIG = {
    # Sequence mode settings
    "use_seq": True,
    "use_flow": False,
    "two_stream": False,
    "use_diff": False,
    "use_diff_attention": False,

    # Fixed for this experiment
    "seq_T": 16,
    "image_size": 112,  # 112x112 works well for video models

    # Training defaults
    "batch_size": 8,  # Reduce for memory-intensive video models
    "num_workers": 2,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "lr": 1e-4,
    "epochs": 50,
    "scheduler": "cosine",

    # Imbalance handling
    "balance_classes": False,
    "loss_type": "ce_weighted",

    # Augmentation (configurable via --use-aug / --no-aug)
    "use_aug": False,  # Default: no augmentation for video sequences

    # Model selection
    "model_selection": "tss",
    "redirect_log": True,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.0,

    # Checkpointing
    "save_last_checkpoint": True,
    "save_last_full_checkpoint": True,
    "save_best_f1": False,
}


def run_experiment(
    backbone: str,
    interval_min: int,
    seed: int = 0,
    common_overrides: dict | None = None,
) -> tuple[str, dict]:
    """Run a single experiment with specified backbone, cadence, and seed.
    
    Args:
        backbone: Model name (e.g., "r2plus1d_18", "videomae")
        interval_min: Cadence in minutes (36, 72, or 108)
        seed: Random seed
        common_overrides: Optional dict to override common config values
    
    Returns:
        Tuple of (experiment_directory, results_dict)
    """
    offsets = _generate_offsets(interval_min, T=16)
    stride = _interval_to_stride(interval_min)
    total_hours = (len(offsets) - 1) * interval_min / 60
    
    # Build experiment name
    exp_name = f"16frames_cadence{interval_min}min_{backbone}_seed{seed}"
    
    # Build config
    cfg = get_default_cfg()
    cfg.update(COMMON_CONFIG)
    if common_overrides:
        cfg.update(common_overrides)
    
    # Set experiment-specific parameters
    cfg.update({
        "backbone": backbone,
        "seed": seed,
        "seq_offsets": offsets,
        "seq_stride": stride,
        "seq_stride_steps": stride,
        "model_name": exp_name,
        "run_id": exp_name,
        "notes": (
            f"16-frame sequence with {interval_min}min cadence "
            f"({total_hours:.1f}h total context). "
            f"Offsets: {offsets[:3]}...{offsets[-3:]} "
            f"Seed: {seed}"
        ),
    })
    
    print(f"\n{'='*80}")
    print(f"Running: {exp_name}")
    print(f"  Cadence: {interval_min} min (stride={stride})")
    print(f"  Temporal context: {total_hours:.1f} hours")
    print(f"  Offsets: {offsets}")
    print(f"{'='*80}\n")
    
    exp_dir, results = train_main(cfg)
    return exp_dir, results


def run_sweep(
    backbones: list[str],
    intervals_min: list[int],
    seeds: list[int],
    common_overrides: dict | None = None,
) -> str:
    """Run experiments for all combinations of backbones, intervals, and seeds.
    
    Args:
        backbones: List of model names (e.g., ["r2plus1d_18"])
        intervals_min: List of cadences in minutes (e.g., [36, 72, 108])
        seeds: List of random seeds (e.g., [0, 1, 2])
        common_overrides: Optional dict to override common config values
    
    Returns:
        Path to summary JSON file
    """
    summary = {
        "experiment": "16-frame cadence sweep",
        "T": 16,
        "image_size": 112,
        "intervals_min": intervals_min,
        "metric": "TSS",
        "backbones": {},
        "n_seeds": len(seeds),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    for bb in backbones:
        summary["backbones"][bb] = {}
        
        for interval in intervals_min:
            tss_values = []
            exp_dirs = []
            
            for seed in seeds:
                try:
                    exp_dir, results = run_experiment(
                        backbone=bb,
                        interval_min=interval,
                        seed=seed,
                        common_overrides=common_overrides,
                    )
                    exp_dirs.append(exp_dir)
                    
                    # Extract TSS from results
                    tss = float(results.get("test_tss_at_val_threshold", results.get("TSS", 0.0)))
                    tss_values.append(tss)
                    
                    print(f"✅ {bb} | {interval}min | seed{seed} | TSS={tss:.4f}")
                    
                except Exception as e:
                    print(f"❌ Failed: {bb} | {interval}min | seed{seed}")
                    print(f"   Error: {e}")
                    continue
            
            if not tss_values:
                print(f"⚠️ No successful runs for {bb} | {interval}min")
                continue
            
            # Calculate statistics
            mean_tss = mean(tss_values)
            std_tss = stdev(tss_values) if len(tss_values) > 1 else 0.0
            
            summary["backbones"][bb][str(interval)] = {
                "tss_values": tss_values,
                "mean_tss": mean_tss,
                "std_tss": std_tss,
                "n": len(tss_values),
                "exp_dirs": exp_dirs,
            }
            
            print(f"\n📊 Summary [{bb} | {interval}min]: TSS = {mean_tss:.4f} ± {std_tss:.4f} (n={len(tss_values)})\n")
    
    # Save summary
    cfg0 = get_default_cfg()
    out_dir = cfg0.get("results_base", ".")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary_16frames_cadence_sweep.json")
    
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Sweep complete! Summary saved to:")
    print(f"   {out_path}")
    print(f"{'='*80}\n")
    
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test 16-frame sequences with different temporal cadences"
    )
    
    # Model selection
    ap.add_argument(
        "--backbone",
        action="append",
        default=None,
        help="Backbone(s) to test: r2plus1d_18, r3d_18, timesformer, slowfast, etc. Can be specified multiple times.",
    )
    
    # Cadence selection
    ap.add_argument(
        "--interval-min",
        action="append",
        type=int,
        default=None,
        help="Cadence interval(s) in minutes: 36, 72, 108. Can be specified multiple times.",
    )
    
    # Seeds
    ap.add_argument(
        "--seed",
        action="append",
        type=int,
        default=None,
        help="Random seed(s). Can be specified multiple times. If not provided, uses [0].",
    )
    ap.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of seeds to run (0 to n-seeds-1). Ignored if --seed is provided.",
    )
    
    # Single run mode
    ap.add_argument(
        "--single",
        action="store_true",
        help="Run a single experiment (requires exactly one backbone, interval, and seed).",
    )
    
    # Training parameters
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None, help="Limit steps per epoch for quick testing")
    ap.add_argument("--val-max-batches", type=int, default=None, help="Limit validation batches")
    ap.add_argument("--pretrained-3d", action="store_true", default=False, help="Use pretrained 3D weights")
    
    # Augmentation
    ap.add_argument("--use-aug", action="store_true", default=None, help="Enable augmentation (±30° rotation, flips)")
    ap.add_argument("--no-aug", dest="use_aug", action="store_false", default=None, help="Disable augmentation (default)")
    
    args = ap.parse_args()
    
    # Set defaults
    backbones = args.backbone or ["r2plus1d_18"]
    intervals = args.interval_min or [36, 72, 108]
    seeds = args.seed if args.seed else list(range(args.n_seeds))
    
    # Validate single mode
    if args.single:
        if len(backbones) != 1 or len(intervals) != 1 or len(seeds) != 1:
            raise ValueError(
                "--single requires exactly one backbone, interval, and seed. "
                f"Got {len(backbones)} backbone(s), {len(intervals)} interval(s), {len(seeds)} seed(s)."
            )
    
    # Build overrides from command-line args
    overrides = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.steps_per_epoch is not None:
        overrides["steps_per_epoch"] = args.steps_per_epoch
    if args.use_aug is not None:
        overrides["use_aug"] = args.use_aug
    if args.val_max_batches is not None:
        overrides["val_max_batches"] = args.val_max_batches
    if args.pretrained_3d:
        overrides["pretrained_3d"] = True
    
    # Print configuration
    print("\n" + "="*80)
    print("16-Frame Cadence Sweep Configuration")
    print("="*80)
    print(f"Backbones: {backbones}")
    print(f"Cadences: {intervals} minutes")
    print(f"Seeds: {seeds}")
    print(f"Image size: 112x112")
    print(f"Frames per sequence: 16")
    if overrides:
        print(f"Overrides: {overrides}")
    print("="*80 + "\n")
    
    # Run experiments
    if args.single:
        exp_dir, results = run_experiment(
            backbone=backbones[0],
            interval_min=intervals[0],
            seed=seeds[0],
            common_overrides=overrides if overrides else None,
        )
        print(f"\n✅ Single experiment complete!")
        print(f"   Directory: {exp_dir}")
        print(f"   TSS: {results.get('test_tss_at_val_threshold', results.get('TSS', 'N/A'))}")
    else:
        summary_path = run_sweep(
            backbones=backbones,
            intervals_min=intervals,
            seeds=seeds,
            common_overrides=overrides if overrides else None,
        )
        print(f"Summary available at: {summary_path}")


if __name__ == "__main__":
    main()
