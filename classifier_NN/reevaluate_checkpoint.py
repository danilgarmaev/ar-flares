"""Re-evaluate a saved checkpoint on the Test split.

This is intended for post-hoc evaluation without retraining.
It loads `config.json` from an experiment directory, rebuilds the model,
loads a specified checkpoint, and runs `evaluate_model` on the Test set.

Outputs are written with a suffix so existing artifacts are not overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch

from .config import CFG
from .datasets import create_dataloaders
from .metrics import evaluate_model
from .models import build_model


def _write_summary_md(path: str, exp_dir: str, cfg: dict, checkpoint_path: str, results: dict) -> None:
    lines = []
    lines.append("# AR-Flares Test Re-evaluation\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Experiment dir:** `{exp_dir}`\n")
    lines.append(f"**Checkpoint:** `{checkpoint_path}`\n")
    lines.append(f"**Backbone:** `{cfg.get('backbone')}`\n")
    lines.append(f"**Use Sequences:** `{cfg.get('use_seq', False)}`\n")
    lines.append(f"**Image size:** `{cfg.get('image_size', None)}`\n")
    if cfg.get("spatial_downsample_factor", 1) not in (None, 1):
        lines.append(f"**Spatial downsample factor:** `{cfg.get('spatial_downsample_factor')}`\n")
    lines.append("\n## Results (Test)\n")
    lines.append(f"- ROC AUC: {results['AUC']:.4f}\n")
    lines.append(f"- PR-AUC: {results['PR_AUC']:.4f}\n")
    lines.append(f"- Precision/Recall/F1: {results['Precision']:.4f} / {results['Recall']:.4f} / {results['F1']:.4f}\n")
    lines.append(f"- TSS/HSS: {results['TSS']:.4f} / {results['HSS']:.4f}\n")
    lines.append(f"- Best TSS: {results['Best_TSS']:.4f} @ threshold={results['Best_threshold']:.3f}\n")
    lines.append("\n## Confusion Matrix (best-TSS threshold)\n")
    lines.append(f"- TP/TN/FP/FN: {results['TP']} / {results['TN']} / {results['FP']} / {results['FN']}\n")

    with open(path, "w") as f:
        f.writelines(lines)


def reevaluate(exp_dir: str, checkpoint_name: str, suffix: str) -> dict:
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in {exp_dir}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    checkpoint_path = os.path.join(exp_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    # Ensure dataloaders/models see the run's cfg (helpers read global CFG)
    CFG.update(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg=cfg, num_classes=2).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    dls = create_dataloaders()

    plots_dir = os.path.join(exp_dir, f"plots_{suffix}")
    os.makedirs(plots_dir, exist_ok=True)

    model_name = f"{cfg.get('model_name', os.path.basename(exp_dir))}_{suffix}"
    results = evaluate_model(
        model,
        dls["Test"],
        device,
        plots_dir,
        model_name,
        save_pr_curve=bool(cfg.get("save_pr_curve", True)),
    )

    # Add metadata
    results.update(
        {
            "reeval": True,
            "checkpoint": checkpoint_name,
            "suffix": suffix,
            "exp_dir": exp_dir,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    metrics_path = os.path.join(exp_dir, f"metrics_{suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    summary_path = os.path.join(exp_dir, f"{cfg.get('model_name', 'experiment')}_{suffix}_summary.md")
    _write_summary_md(summary_path, exp_dir, cfg, checkpoint_path, results)

    return {
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "plots_dir": plots_dir,
        **results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dirs", nargs="+", help="One or more experiment directories")
    parser.add_argument("--checkpoint", default="best_tss.pt", help="Checkpoint filename inside exp_dir")
    parser.add_argument(
        "--suffix",
        default="best_tss",
        help="Suffix used for output files/folders (e.g. best_tss -> metrics_best_tss.json)",
    )
    args = parser.parse_args()

    for exp_dir in args.exp_dirs:
        out = reevaluate(exp_dir, args.checkpoint, args.suffix)
        print(f"Re-evaluated: {exp_dir}")
        print(f"  metrics: {out['metrics_path']}")
        print(f"  summary: {out['summary_path']}")
        print(f"  plots:   {out['plots_dir']}")
        print(f"  TSS: {out['TSS']:.4f} | AUC: {out['AUC']:.4f} | PR-AUC: {out['PR_AUC']:.4f}")


if __name__ == "__main__":
    main()
