"""Resume Experiment A1 runs to 20 total epochs (10 extra), selecting by best Val TSS.

This is a thin wrapper around `classifier_NN.resume_experiment`.
It targets the three A1 experiment directories and writes new artifacts
with suffix `best_tss_20_epochs` (no overwrites).

Usage:
python -m classifier_NN.run_resume_A1_20_epochs

Optionally force CPU:
CUDA_VISIBLE_DEVICES="" python -m classifier_NN.run_resume_A1_20_epochs --device cpu
"""

from __future__ import annotations

import argparse

from ..resume_experiment import resume_experiment


A1_DIRS = [
    "/teamspace/studios/this_studio/AR-flares/results/2025-12-18 14:33:09_A1-224_vgg16",
    "/teamspace/studios/this_studio/AR-flares/results/2025-12-18 17:14:03_A1-112_vgg16",
    "/teamspace/studios/this_studio/AR-flares/results/2025-12-18 20:10:00_A1-56_vgg16",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:0 (optional)")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--suffix", default="best_tss_20_epochs")
    parser.add_argument("--no-redirect-log", action="store_true")
    parser.add_argument("--append-epoch-metrics", action="store_true")
    args = parser.parse_args()

    for exp_dir in A1_DIRS:
        out = resume_experiment(
            exp_dir=exp_dir,
            warm_start_ckpt="best_val_loss.pt",
            baseline_best_tss_ckpt="best_tss.pt",
            max_epochs=args.max_epochs,
            suffix=args.suffix,
            device_str=args.device,
            redirect_log=not args.no_redirect_log,
            append_epoch_metrics=args.append_epoch_metrics,
        )
        print(f"\nDone: {exp_dir}")
        print(f"  ckpt:    {out['checkpoint']}")
        print(f"  metrics: {out['metrics']}")
        print(f"  plots:   {out['plots']}")
        print(f"  TSS: {out['TSS']:.4f} | AUC: {out['AUC']:.4f} | PR-AUC: {out['PR_AUC']:.4f}")


if __name__ == "__main__":
    main()
