#!/usr/bin/env python3
"""Regenerate Figure 2 from the June 10-seed A2 longitude metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_OUT = ROOT / "figures" / "a2_longitude_tss_std.png"
POINTS_OUT = ROOT / "figures" / "a2_longitude_tss_std_points.png"


def parse_task_seed(path: Path, metrics: dict) -> tuple[str, int]:
    name = str(metrics.get("exp_dir") or path.parent.name).lower()
    task = ">=C" if "cplus" in name else ">=M"
    match = re.search(r"mvit_seed(\d+)_", name)
    if not match:
        match = re.search(r"_seed(\d+)(?:_|$)", name)
    if not match:
        raise ValueError(f"Could not parse seed from {path}")
    return task, int(match.group(1))


def load_records() -> dict[tuple[str, int], dict[str, float]]:
    paths = sorted(RESULTS.glob("2026-06-11_*_A2_longitude_eval_*/a2_metrics.json"))
    records: dict[tuple[str, int], dict[str, float]] = {}
    for path in paths:
        if "A2-lowLon30" in path.parent.name:
            continue
        metrics = json.loads(path.read_text())
        exp_dir = str(metrics.get("exp_dir", ""))
        if "mvitv1_notemporal" not in exp_dir:
            continue
        if "low_lon_training" not in metrics:
            continue
        task, seed = parse_task_seed(path, metrics)
        key = (task, seed)
        if key in records:
            raise ValueError(f"Duplicate A2 baseline metrics for {key}: {path}")
        records[key] = {
            "baseline_inner": float(metrics["buckets"]["lon_le_30"]["TSS"]),
            "baseline_outer": float(metrics["buckets"]["lon_30_60"]["TSS"]),
            "central_inner": float(metrics["low_lon_training"]["buckets"]["lon_le_30"]["TSS"]),
            "central_outer": float(metrics["low_lon_training"]["buckets"]["lon_30_60"]["TSS"]),
        }
    expected = {(task, seed) for task in (">=C", ">=M") for seed in range(10)}
    missing = sorted(expected - set(records))
    extra = sorted(set(records) - expected)
    if missing or extra:
        raise SystemExit(f"Unexpected A2 records. missing={missing} extra={extra}")
    return records


def mean_std(records: dict[tuple[str, int], dict[str, float]], task: str, field: str) -> tuple[float, float, np.ndarray]:
    values = np.array([records[(task, seed)][field] for seed in range(10)], dtype=float)
    return float(values.mean()), float(values.std(ddof=1)), values


def plot_panel(ax, records, fields, labels, title, show_points: bool = False):
    tasks = [">=C", ">=M"]
    x = np.arange(len(tasks), dtype=float)
    width = 0.32
    colors = ["#327897", "#C97860"]
    offsets = [-width / 2, width / 2]

    for field, label, color, offset in zip(fields, labels, colors, offsets):
        means = []
        stds = []
        values_by_task = []
        for task in tasks:
            mean, std, values = mean_std(records, task, field)
            means.append(mean)
            stds.append(std)
            values_by_task.append(values)
        ax.bar(
            x + offset,
            means,
            width,
            label=label,
            color=color,
            alpha=0.88,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=stds,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
        )
        if show_points:
            for i, values in enumerate(values_by_task):
                ax.scatter(
                    np.full_like(values, x[i] + offset),
                    values,
                    s=18,
                    color="white",
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )

    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0.42, 0.82)
    ax.set_ylabel("TSS", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-points", action="store_true", help="Overlay individual seed values on each bar.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    args = parser.parse_args()

    records = load_records()
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.2), sharey=True, constrained_layout=True)
    plot_panel(
        axes[0],
        records,
        fields=["baseline_inner", "baseline_outer"],
        labels=[r"$|lon| \leq 30^\circ$", r"$30^\circ < |lon| \leq 60^\circ$"],
        title="A. Full-Training Model",
        show_points=args.with_points,
    )
    plot_panel(
        axes[1],
        records,
        fields=["central_inner", "central_outer"],
        labels=[r"$|lon| \leq 30^\circ$", r"$30^\circ < |lon| \leq 60^\circ$"],
        title="B. Central-Only Model",
        show_points=args.with_points,
    )
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    out = args.out or (POINTS_OUT if args.with_points else DEFAULT_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
