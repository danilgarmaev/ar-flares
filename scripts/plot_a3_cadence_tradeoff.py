#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from count_a3_sequence_sizes import summarize_counts
except ModuleNotFoundError:
    from scripts.count_a3_sequence_sizes import summarize_counts

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CADENCES = [60, 120, 180, 240, 300]
LABELS = ["C", "M"]
RUN_RE = re.compile(r"A3-16-cadence-(\d+)min_.*_seed(\d+)_", re.IGNORECASE)


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_dt(run_dir: Path, metrics: dict) -> datetime:
    prefix = run_dir.name.split("_", 1)[0]
    try:
        return datetime.strptime(prefix, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
    date_str = metrics.get("date")
    if isinstance(date_str, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                pass
    return datetime.min


def _collect_latest_fixed_tss() -> dict[tuple[int, str, int], tuple[datetime, float]]:
    latest: dict[tuple[int, str, int], tuple[datetime, float]] = {}
    for run_dir in RESULTS_DIR.iterdir():
        if not run_dir.is_dir() or "A3-16-cadence-" not in run_dir.name:
            continue
        cfg = _load_json(run_dir / "config.json")
        metrics = _load_json(run_dir / "metrics.json")
        if not cfg or not metrics:
            continue
        if str(cfg.get("backbone", "")).lower() != "mvit":
            continue
        if not bool(cfg.get("use_aug", False)):
            continue
        if int(cfg.get("seq_T", 0) or 0) != 16:
            continue
        if int(cfg.get("image_size", 0) or 0) != 224:
            continue
        m = RUN_RE.search(run_dir.name)
        if not m:
            continue
        cadence_min = int(m.group(1))
        seed = int(m.group(2))
        if cadence_min not in CADENCES:
            continue
        label = str(cfg.get("min_flare_class", "C") or "C").upper()
        if label not in LABELS:
            continue
        fixed_tss = metrics.get("fixed_TSS", metrics.get("test_tss_at_val_threshold"))
        if not isinstance(fixed_tss, (int, float)):
            continue
        key = (cadence_min, label, seed)
        dt = _parse_dt(run_dir, metrics)
        prev = latest.get(key)
        if prev is None or dt >= prev[0]:
            latest[key] = (dt, float(fixed_tss))
    return latest


def _aggregate_tss() -> dict[str, dict[int, tuple[float, float, int]]]:
    latest = _collect_latest_fixed_tss()
    out: dict[str, dict[int, tuple[float, float, int]]] = {label: {} for label in LABELS}
    for label in LABELS:
        for cadence in CADENCES:
            vals = [tss for (cad, lab, _seed), (_dt, tss) in latest.items() if cad == cadence and lab == label]
            if not vals:
                raise RuntimeError(f"Missing augmented A3 T=16 results for {label}+ at {cadence} min")
            out[label][cadence] = (mean(vals), stdev(vals) if len(vals) > 1 else 0.0, len(vals))
    return out


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tss_summary = _aggregate_tss()
    print("Aggregated fixed_TSS from A3 augmented runs", flush=True)
    count_rows = summarize_counts(show_progress=False)
    print("Loaded exact A3 cadence sample counts", flush=True)
    count_map = {(int(row["cadence_min"]), str(row["label"])): row for row in count_rows}

    c_means = [tss_summary["C"][c][0] for c in CADENCES]
    c_stds = [tss_summary["C"][c][1] for c in CADENCES]
    m_means = [tss_summary["M"][c][0] for c in CADENCES]
    m_stds = [tss_summary["M"][c][1] for c in CADENCES]

    train_effective = [int(count_map[(c, "C")]["train_effective_post_subsample"]) for c in CADENCES]
    c_pos = [int(count_map[(c, "C")]["train_pos"]) for c in CADENCES]
    m_pos = [int(count_map[(c, "M")]["train_pos"]) for c in CADENCES]

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]})

    ax1.errorbar(CADENCES, c_means, yerr=c_stds, marker="o", lw=2.2, capsize=4, color="#0B6E4F", label="C+ fixed TSS")
    ax1.errorbar(CADENCES, m_means, yerr=m_stds, marker="s", lw=2.2, capsize=4, color="#C84C09", label="M+ fixed TSS")
    ax1.set_ylabel("Fixed TSS")
    ax1.set_title("A3 Cadence Trade-off: Performance vs Usable Data")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best")

    ax2.plot(CADENCES, train_effective, marker="o", lw=2.2, color="#1F4E79", label="Train windows / epoch")
    ax2.plot(CADENCES, c_pos, marker="^", lw=2.0, color="#2A9D8F", label="C+ train positives")
    ax2.plot(CADENCES, m_pos, marker="D", lw=2.0, color="#E76F51", label="M+ train positives")
    ax2.set_xlabel("Cadence (minutes)")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="best")

    out_png = FIGURES_DIR / "a3_cadence_tradeoff.png"
    out_pdf = FIGURES_DIR / "a3_cadence_tradeoff.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
