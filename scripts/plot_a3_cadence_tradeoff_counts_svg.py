#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CACHE_COUNTS_PATH = RESULTS_DIR / "a3_sequence_size_counts.tsv"
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


def _load_cached_counts(path: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            parsed: dict[str, int | str] = {}
            for key, value in row.items():
                if key == "label":
                    parsed[key] = value or ""
                else:
                    parsed[key] = int(value or 0)
            rows.append(parsed)
    return rows


def _line_points(xs: list[float], ys: list[float]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def _svg_text(x: float, y: float, text: str, *, size: int = 14, anchor: str = "middle", weight: str = "normal", rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-weight="{weight}"{transform}>{text}</text>'
    )


def main() -> None:
    if not CACHE_COUNTS_PATH.exists():
        raise FileNotFoundError(f"Missing cached counts file: {CACHE_COUNTS_PATH}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tss_summary = _aggregate_tss()
    count_rows = _load_cached_counts(CACHE_COUNTS_PATH)
    count_map = {(int(row["cadence_min"]), str(row["label"])): row for row in count_rows}

    c_means = [tss_summary["C"][c][0] for c in CADENCES]
    c_stds = [tss_summary["C"][c][1] for c in CADENCES]
    m_means = [tss_summary["M"][c][0] for c in CADENCES]
    m_stds = [tss_summary["M"][c][1] for c in CADENCES]

    train_effective = [int(count_map[(c, "C")]["train_effective_post_subsample"]) for c in CADENCES]
    c_pos = [int(count_map[(c, "C")]["train_pos"]) for c in CADENCES]
    m_pos = [int(count_map[(c, "M")]["train_pos"]) for c in CADENCES]

    width = 900
    height = 760
    left = 90
    right = 40
    panel_w = width - left - right
    top_y0, top_y1 = 90, 320
    bot_y0, bot_y1 = 430, 680

    def x_map(cad: int) -> float:
        c0, c1 = CADENCES[0], CADENCES[-1]
        return left + (cad - c0) / (c1 - c0) * panel_w

    tss_min = min(min(c - s for c, s in zip(c_means, c_stds)), min(m - s for m, s in zip(m_means, m_stds))) - 0.02
    tss_max = max(max(c + s for c, s in zip(c_means, c_stds)), max(m + s for m, s in zip(m_means, m_stds))) + 0.02
    tss_min = max(0.55, tss_min)
    tss_max = min(0.86, tss_max)

    def y_map_top(val: float) -> float:
        return top_y1 - (val - tss_min) / (tss_max - tss_min) * (top_y1 - top_y0)

    count_max = max(train_effective + c_pos + m_pos)
    count_min = 0

    def y_map_bot(val: float) -> float:
        return bot_y1 - (val - count_min) / (count_max - count_min) * (bot_y1 - bot_y0)

    c_color = "#0B6E4F"
    m_color = "#C84C09"
    train_color = "#1F4E79"
    cpos_color = "#2A9D8F"
    mpos_color = "#E76F51"

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 36, "A3 Cadence Trade-off: Performance vs Exact Counts", size=22, weight="bold"),
    ]

    for y in [0.60, 0.65, 0.70, 0.75, 0.80]:
        yy = y_map_top(y)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#dddddd" stroke-dasharray="4 4"/>')
        parts.append(_svg_text(left - 12, yy + 5, f"{y:.2f}", size=12, anchor="end"))

    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        val = int(round(count_max * frac))
        yy = y_map_bot(val)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#dddddd" stroke-dasharray="4 4"/>')
        parts.append(_svg_text(left - 12, yy + 5, str(val), size=12, anchor="end"))

    parts.extend(
        [
            f'<rect x="{left}" y="{top_y0}" width="{panel_w}" height="{top_y1-top_y0}" fill="none" stroke="black"/>',
            f'<rect x="{left}" y="{bot_y0}" width="{panel_w}" height="{bot_y1-bot_y0}" fill="none" stroke="black"/>',
            _svg_text(28, (top_y0 + top_y1) / 2, "Fixed TSS", size=15, rotate=-90),
            _svg_text(28, (bot_y0 + bot_y1) / 2, "Count", size=15, rotate=-90),
            _svg_text(width / 2, height - 24, "Cadence (minutes)", size=16),
        ]
    )

    top_xs = [x_map(c) for c in CADENCES]
    c_ys = [y_map_top(v) for v in c_means]
    m_ys = [y_map_top(v) for v in m_means]
    parts.append(f'<polyline fill="none" stroke="{c_color}" stroke-width="3" points="{_line_points(top_xs, c_ys)}"/>')
    parts.append(f'<polyline fill="none" stroke="{m_color}" stroke-width="3" points="{_line_points(top_xs, m_ys)}"/>')

    for x, mean_v, std_v, color, marker in zip(top_xs, c_means, c_stds, [c_color] * len(top_xs), ["circle"] * len(top_xs)):
        y = y_map_top(mean_v)
        y0 = y_map_top(mean_v - std_v)
        y1 = y_map_top(mean_v + std_v)
        parts.append(f'<line x1="{x:.2f}" y1="{y0:.2f}" x2="{x:.2f}" y2="{y1:.2f}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<line x1="{x-5:.2f}" y1="{y0:.2f}" x2="{x+5:.2f}" y2="{y0:.2f}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<line x1="{x-5:.2f}" y1="{y1:.2f}" x2="{x+5:.2f}" y2="{y1:.2f}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}"/>')

    for x, mean_v, std_v in zip(top_xs, m_means, m_stds):
        y = y_map_top(mean_v)
        y0 = y_map_top(mean_v - std_v)
        y1 = y_map_top(mean_v + std_v)
        parts.append(f'<line x1="{x:.2f}" y1="{y0:.2f}" x2="{x:.2f}" y2="{y1:.2f}" stroke="{m_color}" stroke-width="2"/>')
        parts.append(f'<line x1="{x-5:.2f}" y1="{y0:.2f}" x2="{x+5:.2f}" y2="{y0:.2f}" stroke="{m_color}" stroke-width="2"/>')
        parts.append(f'<line x1="{x-5:.2f}" y1="{y1:.2f}" x2="{x+5:.2f}" y2="{y1:.2f}" stroke="{m_color}" stroke-width="2"/>')
        parts.append(f'<rect x="{x-4.5:.2f}" y="{y-4.5:.2f}" width="9" height="9" fill="{m_color}"/>')

    bot_train_ys = [y_map_bot(v) for v in train_effective]
    bot_cpos_ys = [y_map_bot(v) for v in c_pos]
    bot_mpos_ys = [y_map_bot(v) for v in m_pos]
    parts.append(f'<polyline fill="none" stroke="{train_color}" stroke-width="3" points="{_line_points(top_xs, bot_train_ys)}"/>')
    parts.append(f'<polyline fill="none" stroke="{cpos_color}" stroke-width="3" points="{_line_points(top_xs, bot_cpos_ys)}"/>')
    parts.append(f'<polyline fill="none" stroke="{mpos_color}" stroke-width="3" points="{_line_points(top_xs, bot_mpos_ys)}"/>')

    for x, y in zip(top_xs, bot_train_ys):
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{train_color}"/>')
    for x, y in zip(top_xs, bot_cpos_ys):
        parts.append(
            f'<polygon points="{x:.2f},{y-5:.2f} {x-5:.2f},{y+4:.2f} {x+5:.2f},{y+4:.2f}" fill="{cpos_color}"/>'
        )
    for x, y in zip(top_xs, bot_mpos_ys):
        parts.append(
            f'<polygon points="{x:.2f},{y-5:.2f} {x-5:.2f},{y:.2f} {x:.2f},{y+5:.2f} {x+5:.2f},{y:.2f}" fill="{mpos_color}"/>'
        )

    for cadence in CADENCES:
        x = x_map(cadence)
        parts.append(f'<line x1="{x:.2f}" y1="{top_y1}" x2="{x:.2f}" y2="{top_y1+6}" stroke="black"/>')
        parts.append(f'<line x1="{x:.2f}" y1="{bot_y1}" x2="{x:.2f}" y2="{bot_y1+6}" stroke="black"/>')
        parts.append(_svg_text(x, bot_y1 + 24, str(cadence), size=13))

    legend_x = width - 250
    legend_y = 58
    legend_items = [
        (c_color, "line-circle", "C+ fixed TSS"),
        (m_color, "line-square", "M+ fixed TSS"),
        (train_color, "line-circle", "Train windows / epoch"),
        (cpos_color, "line-triangle", "C+ train positives"),
        (mpos_color, "line-diamond", "M+ train positives"),
    ]
    for idx, (color, kind, label) in enumerate(legend_items):
        y = legend_y + idx * 20
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+18}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        if kind == "line-circle":
            parts.append(f'<circle cx="{legend_x+9}" cy="{y}" r="4" fill="{color}"/>')
        elif kind == "line-square":
            parts.append(f'<rect x="{legend_x+5}" y="{y-4}" width="8" height="8" fill="{color}"/>')
        elif kind == "line-triangle":
            parts.append(f'<polygon points="{legend_x+9},{y-5} {legend_x+4},{y+4} {legend_x+14},{y+4}" fill="{color}"/>')
        elif kind == "line-diamond":
            parts.append(f'<polygon points="{legend_x+9},{y-5} {legend_x+4},{y} {legend_x+9},{y+5} {legend_x+14},{y}" fill="{color}"/>')
        parts.append(_svg_text(legend_x + 26, y + 4, label, size=12, anchor="start"))

    parts.append("</svg>")
    svg = "\n".join(parts)

    out_svg = FIGURES_DIR / "a3_cadence_tradeoff_counts.svg"
    out_pdf = FIGURES_DIR / "a3_cadence_tradeoff_counts.pdf"
    out_png = FIGURES_DIR / "a3_cadence_tradeoff_counts.png"
    out_svg.write_text(svg, encoding="utf-8")

    subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(out_pdf), str(out_svg)], check=True)
    subprocess.run(["convert", str(out_svg), str(out_png)], check=True)

    print(f"Saved {out_svg}")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
