#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cnn_features"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
STEP_MINUTES = 12
STEP_SECONDS = STEP_MINUTES * 60

SPLIT_FILES = {
    "Train": DATA / "Train_Data_by_AR_png_224.csv",
    "Validation": DATA / "Validation_Data_by_AR_png_224.csv",
    "Test": DATA / "Test_Data_by_AR_png_224.csv",
}

FILENAME_RE = re.compile(r"(?P<ar>\d+)_hmi\.M_720s\.(?P<date>\d{8})_(?P<time>\d{6})_TAI")


def parse_entry(path: str) -> tuple[int, datetime] | None:
    name = os.path.basename(path)
    match = FILENAME_RE.search(name)
    if not match:
        return None
    ar = int(match.group("ar"))
    timestamp = datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S")
    return ar, timestamp


def longest_contiguous_span_hours(times: list[datetime]) -> tuple[int, float]:
    if not times:
        return 0, 0.0
    times = sorted(set(times))
    best = current = 1
    for prev, cur in zip(times, times[1:]):
        if (cur - prev).total_seconds() == STEP_SECONDS:
            current += 1
        else:
            best = max(best, current)
            current = 1
    best = max(best, current)
    span_hours = (best - 1) * STEP_MINUTES / 60.0
    return best, span_hours


def load_split(split: str, path: Path) -> list[dict[str, int | float | str]]:
    by_ar: dict[int, list[datetime]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed = parse_entry(row["filename"])
            if parsed is None:
                continue
            ar, timestamp = parsed
            by_ar[ar].append(timestamp)

    rows: list[dict[str, int | float | str]] = []
    for ar, times in sorted(by_ar.items()):
        best_frames, max_span_hours = longest_contiguous_span_hours(times)
        rows.append(
            {
                "split": split,
                "ar": ar,
                "n_observations": len(set(times)),
                "longest_contiguous_frames": best_frames,
                "max_contiguous_span_hours": max_span_hours,
            }
        )
    return rows


def summarize(rows: list[dict[str, int | float | str]]) -> list[dict[str, int | float | str]]:
    out = []
    for split in ["Train", "Validation", "Test", "All"]:
        if split == "All":
            vals = np.array([float(row["max_contiguous_span_hours"]) for row in rows])
        else:
            vals = np.array([float(row["max_contiguous_span_hours"]) for row in rows if row["split"] == split])
        if len(vals) == 0:
            continue
        out.append(
            {
                "split": split,
                "n_ar": len(vals),
                "median_hours": float(np.median(vals)),
                "q25_hours": float(np.percentile(vals, 25)),
                "q75_hours": float(np.percentile(vals, 75)),
                "max_hours": float(np.max(vals)),
                "pct_ge_15h": float(np.mean(vals >= 15) * 100),
                "pct_ge_30h": float(np.mean(vals >= 30) * 100),
                "pct_ge_45h": float(np.mean(vals >= 45) * 100),
                "pct_ge_60h": float(np.mean(vals >= 60) * 100),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_survival(rows: list[dict[str, int | float | str]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    colors = {"Train": "#26786C", "Validation": "#4D6FA9", "Test": "#C64F4F", "All": "#333333"}
    thresholds = np.arange(0, 121, 3)
    for split in ["Train", "Validation", "Test", "All"]:
        if split == "All":
            vals = np.array([float(row["max_contiguous_span_hours"]) for row in rows])
            lw = 2.8
        else:
            vals = np.array([float(row["max_contiguous_span_hours"]) for row in rows if row["split"] == split])
            lw = 2.0
        survival = [np.mean(vals >= threshold) * 100 for threshold in thresholds]
        ax.plot(thresholds, survival, label=split, color=colors[split], lw=lw)

    for span, label in [(15, "16 x 60 min"), (30, "16 x 120 min"), (45, "16 x 180 min")]:
        ax.axvline(span, color="#888888", lw=1.0, ls="--")
        ax.text(span + 1.0, 6, label, rotation=90, va="bottom", ha="left", fontsize=9, color="#555555")

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Required contiguous history (hours)")
    ax.set_ylabel("Active regions with at least this span (%)")
    ax.set_title("Available Contiguous Temporal Coverage per Active Region")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "intro_ar_temporal_coverage.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / "intro_ar_temporal_coverage.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, int | float | str]] = []
    for split, path in SPLIT_FILES.items():
        all_rows.extend(load_split(split, path))
    summary_rows = summarize(all_rows)
    write_csv(RESULTS / "intro_ar_temporal_coverage_by_ar.csv", all_rows)
    write_csv(RESULTS / "intro_ar_temporal_coverage_summary.csv", summary_rows)
    plot_survival(all_rows)

    for row in summary_rows:
        print(
            "{split}: n_ar={n_ar}, median={median_hours:.1f}h, q25={q25_hours:.1f}h, "
            "q75={q75_hours:.1f}h, max={max_hours:.1f}h, >=45h={pct_ge_45h:.1f}%".format(**row)
        )
    print(f"Saved {FIGURES / 'intro_ar_temporal_coverage.png'}")
    print(f"Saved {FIGURES / 'intro_ar_temporal_coverage.pdf'}")
    print(f"Saved {RESULTS / 'intro_ar_temporal_coverage_summary.csv'}")


if __name__ == "__main__":
    main()
