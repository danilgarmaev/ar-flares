"""Plot mean±std TSS vs resolution for each backbone.

Usage:
  python scripts/plot_a1_resolution_sweep.py --summary results/summary_a1_resolution_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary",
        default="results/summary_a1_resolution_sweep.json",
        help="Path to summary JSON produced by run_experiments_A1.py",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output image path (default: alongside summary JSON)",
    )
    args = ap.parse_args()

    with open(args.summary, "r") as f:
        summary = json.load(f)

    resolutions = [int(r) for r in summary.get("resolutions", [])]
    backbones = summary.get("backbones", {})

    if not resolutions or not backbones:
        raise ValueError("Summary JSON missing resolutions/backbones data")

    plt.figure(figsize=(7, 5))
    for backbone, data in backbones.items():
        means = []
        stds = []
        for res in resolutions:
            entry = data.get(str(res), {})
            means.append(entry.get("mean_tss", 0.0))
            stds.append(entry.get("std_tss", 0.0))
        plt.errorbar(
            resolutions,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=backbone,
        )

    plt.gca().invert_xaxis()  # higher resolution on the left
    plt.xlabel("Resolution (input downsample size)")
    plt.ylabel("TSS (mean ± std)")
    plt.title("A1 Resolution Sweep: TSS vs Resolution")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    if args.out is None:
        base = os.path.splitext(args.summary)[0]
        out_path = f"{base}_plot.png"
    else:
        out_path = args.out

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
