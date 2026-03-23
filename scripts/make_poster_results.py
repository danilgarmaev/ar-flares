from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "poster_results_20260321"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip()
    if text in {"", "NA", "None"}:
        return None
    return float(text.replace(",", "."))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [row for row in reader]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def render_table_png(
    rows: list[dict[str, Any]],
    columns: list[str],
    title: str,
    out_path: Path,
    fontsize: int = 10,
) -> None:
    cell_text: list[list[str]] = []
    for row in rows:
        vals = []
        for c in columns:
            val = row[c]
            vals.append(_fmt(val) if isinstance(val, float) else str(val))
        cell_text.append(vals)

    fig_height = max(2.2, 0.45 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=220)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    table = ax.table(cellText=cell_text, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.35)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#404040")
        if r == 0:
            cell.set_facecolor("#d9ead3")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f7f7f7")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_roc_points(
    rows: list[dict[str, Any]],
    title: str,
    out_path: Path,
    group_key: str | None = None,
    point_label: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=220)

    ax.plot([0, 1], [0, 1], "--", color="#8a8a8a", linewidth=1.2, label="Random")

    if group_key is None:
        for row in rows:
            fpr = row["FPR"]
            tpr = row["TPR"]
            label = row[point_label] if point_label else row["Model"]
            ax.scatter(fpr, tpr, s=45)
            ax.text(fpr + 0.01, tpr + 0.01, label, fontsize=8)
    else:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[group_key])].append(row)

        palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
        for idx, (name, g_rows) in enumerate(sorted(grouped.items())):
            g_rows = sorted(g_rows, key=lambda x: x.get("Cadence_min", 0))
            x = [r["FPR"] for r in g_rows]
            y = [r["TPR"] for r in g_rows]
            color = palette[idx % len(palette)]
            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4.5, color=color, label=name)
            for r in g_rows:
                ax.text(r["FPR"] + 0.008, r["TPR"] + 0.008, f"{int(r['Cadence_min'])}m", fontsize=7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_exp1_rows() -> list[dict[str, Any]]:
    metrics_paths = [
        (
            "ResNet50",
            224,
            RESULTS_DIR / "2026-02-16 19:12:43_A1-224_resnet50_seed0" / "metrics.json",
        ),
        (
            "Swin-Tiny",
            56,
            RESULTS_DIR / "2026-02-20 23:12:08_A1-56_swin_tiny_patch4_window7_224_seed0" / "metrics.json",
        ),
        (
            "Swin-Tiny",
            128,
            RESULTS_DIR / "2026-02-20 23:10:03_A1-128_swin_tiny_patch4_window7_224_seed0" / "metrics.json",
        ),
        (
            "Swin-Tiny",
            224,
            RESULTS_DIR / "2026-02-16 22:30:42_A1-224_swin_tiny_patch4_window7_224_seed0" / "metrics.json",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for model, res, path in metrics_paths:
        m = _load_json(path)
        tpr = float(m["TPR"])
        tnr = float(m["TNR"])
        rows.append(
            {
                "Model": model,
                "Resolution": res,
                "AUC": float(m["AUC"]),
                "TSS": float(m["TSS"]),
                "TPR": tpr,
                "FPR": max(0.0, min(1.0, 1.0 - tnr)),
                "Best_threshold": float(m["Best_threshold"]),
            }
        )
    return sorted(rows, key=lambda x: (x["Model"], x["Resolution"]))


def build_exp2_rows() -> list[dict[str, Any]]:
    full_metrics = _load_json(
        RESULTS_DIR / "2026-02-16 22:30:42_A1-224_swin_tiny_patch4_window7_224_seed0" / "metrics.json"
    )
    lon30_metrics = _load_json(
        RESULTS_DIR
        / "2026-02-21 16:20:09_A2-lowLon30_2026-02-16 22:30:42_A1-224_swin_tiny_patch4_window7_224_seed0"
        / "metrics.json"
    )

    rows = []
    for tag, m in [("All longitudes", full_metrics), ("|lon| <= 30 deg", lon30_metrics)]:
        tpr = float(m["TPR"])
        tnr = float(m["TNR"])
        rows.append(
            {
                "Subset": tag,
                "Model": "Swin-Tiny 224",
                "AUC": float(m["AUC"]),
                "TSS": float(m["TSS"]),
                "TPR": tpr,
                "FPR": max(0.0, min(1.0, 1.0 - tnr)),
                "Best_threshold": float(m["Best_threshold"]),
            }
        )
    return rows


def build_exp3_rows() -> list[dict[str, Any]]:
    rows = _read_tsv(RESULTS_DIR / "summary_video_excel.tsv")
    filtered = []
    for r in rows:
        if r.get("min_flare_class") != "C":
            continue
        if r.get("use_aug") != "True":
            continue
        if r.get("backbone") not in {"r2plus1d_18", "videomae", "videoswin"}:
            continue
        cadence = _to_float(r.get("interval_min"))
        if cadence not in {36.0, 72.0, 108.0}:
            continue
        tss = _to_float(r.get("TSS"))
        auc = _to_float(r.get("AUC"))
        tpr = _to_float(r.get("Recall"))
        if tss is None or auc is None or tpr is None:
            continue
        filtered.append(
            {
                "Model": r["backbone"],
                "Cadence_min": int(cadence),
                "AUC": float(auc),
                "TSS": float(tss),
                "TPR": float(tpr),
                "FPR": max(0.0, min(1.0, float(tpr) - float(tss))),
                "Run": r["run_id"],
            }
        )

    best_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for r in filtered:
        key = (r["Model"], r["Cadence_min"])
        prev = best_by_key.get(key)
        if prev is None or r["TSS"] > prev["TSS"]:
            best_by_key[key] = r

    return sorted(best_by_key.values(), key=lambda x: (x["Model"], x["Cadence_min"]))


def write_poster_text(exp1: list[dict[str, Any]], exp2: list[dict[str, Any]], exp3: list[dict[str, Any]]) -> None:
    best_exp1 = max(exp1, key=lambda x: x["TSS"])
    exp2_delta = exp2[1]["TSS"] - exp2[0]["TSS"]

    best_exp3_by_model: dict[str, dict[str, Any]] = {}
    for r in exp3:
        model = r["Model"]
        cur = best_exp3_by_model.get(model)
        if cur is None or r["TSS"] > cur["TSS"]:
            best_exp3_by_model[model] = r

    lines = [
        "# Poster Results (Compact)",
        "",
        "## Key points",
        f"- Resolution/backbone sweep: best TSS is {best_exp1['TSS']:.3f} with {best_exp1['Model']} at {best_exp1['Resolution']} px.",
        f"- Within-30-deg subset: TSS change vs all longitudes is {exp2_delta:+.3f}.",
        "- 16-frame 3D cadence sweep (C-class, augmented): best cadence by model:",
    ]
    for model in sorted(best_exp3_by_model):
        row = best_exp3_by_model[model]
        lines.append(
            f"  - {model}: cadence {row['Cadence_min']} min, TSS={row['TSS']:.3f}, AUC={row['AUC']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Figure notes",
            "- ROC panels are shown in ROC space using best-threshold operating points (FPR, TPR) from saved metrics.",
            "- This avoids over-wide figures while preserving model ranking for poster layout constraints.",
        ]
    )

    (OUT_DIR / "poster_results_text.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exp1 = build_exp1_rows()
    exp2 = build_exp2_rows()
    exp3 = build_exp3_rows()

    _write_csv(OUT_DIR / "exp1_resolution_table.csv", exp1, ["Model", "Resolution", "AUC", "TSS", "TPR", "FPR", "Best_threshold"])
    _write_csv(OUT_DIR / "exp2_lon30_table.csv", exp2, ["Subset", "Model", "AUC", "TSS", "TPR", "FPR", "Best_threshold"])
    _write_csv(OUT_DIR / "exp3_16frames_table.csv", exp3, ["Model", "Cadence_min", "AUC", "TSS", "TPR", "FPR", "Run"])

    render_table_png(
        exp1,
        ["Model", "Resolution", "AUC", "TSS", "TPR", "FPR"],
        "Experiment 1: Resolution and Backbone (A1)",
        OUT_DIR / "exp1_resolution_table.png",
    )
    render_table_png(
        exp2,
        ["Subset", "Model", "AUC", "TSS", "TPR", "FPR"],
        "Experiment 2: |lon| <= 30 deg vs Full Set (A2)",
        OUT_DIR / "exp2_lon30_table.png",
    )
    render_table_png(
        exp3,
        ["Model", "Cadence_min", "AUC", "TSS", "TPR", "FPR"],
        "Experiment 3: 16-Frame 3D Models Across Cadence (A3, C-class)",
        OUT_DIR / "exp3_16frames_table.png",
        fontsize=9,
    )

    render_roc_points(
        exp1,
        "ROC Space: Experiment 1 Operating Points",
        OUT_DIR / "exp1_roc_space.png",
        point_label="Resolution",
    )
    render_roc_points(
        exp2,
        "ROC Space: Experiment 2 Operating Points",
        OUT_DIR / "exp2_roc_space.png",
        point_label="Subset",
    )
    render_roc_points(
        exp3,
        "ROC Space: Experiment 3 Operating Points by Cadence",
        OUT_DIR / "exp3_roc_space.png",
        group_key="Model",
    )

    write_poster_text(exp1, exp2, exp3)

    manifest = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(OUT_DIR),
        "files": sorted([p.name for p in OUT_DIR.iterdir() if p.is_file()]),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()