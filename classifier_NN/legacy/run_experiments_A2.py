"""Experiment 4.2 (A2) – High Longitude Impact (evaluation only).

Goal
----
Measure whether a trained model's TSS degrades for high-longitude Active Regions,
without any projection correction.

Constraints
-----------
- Evaluation-only: do NOT retrain.
- Use existing Test dataloader that yields (x, label, meta).
- Bucket by SRS Location longitude using a fixed decision threshold.
    By default we use the model's Best_TSS threshold from the experiment outputs.

Usage
-----
python -m classifier_NN.legacy.run_experiments_A2 \
  --exp-dir "/path/to/experiment_dir" \
  --checkpoint best_tss.pt

If you want to avoid depending on exp-dir/config.json, you can pass --cfg-json.

Outputs
-------
Prints a concise report for:
- |lon| <= 30 degrees
- 30 < |lon| <= 60 degrees
and counts skipped samples missing SRS.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
except Exception:  # pragma: no cover
    roc_auc_score = None
    average_precision_score = None
    roc_curve = None
    precision_recall_curve = None

from ..config import CFG
from ..datasets import create_dataloaders
from ..models import build_model


LocationKey = Tuple[int, str]  # (ar_number, YYYYMMDD)


_LOCATION_RE = re.compile(r"\b[NS](\d{1,2})([EW])(\d{1,2})\b")


def _normalize_yyyymmdd(value: Any) -> Optional[str]:
    """Return a YYYYMMDD string extracted from value, or None if unavailable."""
    if value is None:
        return None
    s = str(value)
    m = re.search(r"(\d{8})", s)
    return m.group(1) if m else None


def _parse_longitude_from_location_token(token: str) -> Optional[int]:
    """Parse longitude from SRS Location token.

    Examples:
      N15E42 -> +42
      S08W12 -> -12
    """
    m = _LOCATION_RE.search(token.strip())
    if not m:
        return None
    hemisphere = m.group(2)
    deg = int(m.group(3))
    return deg if hemisphere == "E" else -deg


def _extract_date_from_srs_filename(filename: str) -> Optional[str]:
    """Extract YYYYMMDD from an SRS filename like YYYYMMDD SRS / YYYYMMDD...SRS.txt."""
    base = os.path.basename(filename)
    m = re.search(r"(\d{8})", base)
    return m.group(1) if m else None


def build_srs_longitude_lookup(srs_root: str) -> Dict[LocationKey, int]:
    """Parse SRS text files and build (ar_number, YYYYMMDD) -> longitude_deg lookup."""
    lookup: Dict[LocationKey, int] = {}

    if not os.path.isdir(srs_root):
        raise FileNotFoundError(f"SRS root not found: {srs_root}")

    for year_dir in sorted(os.listdir(srs_root)):
        if not year_dir.endswith("_SRS"):
            continue
        full_year_dir = os.path.join(srs_root, year_dir)
        if not os.path.isdir(full_year_dir):
            continue

        for fname in sorted(os.listdir(full_year_dir)):
            if not fname.lower().endswith(".txt"):
                continue
            path = os.path.join(full_year_dir, fname)
            date = _extract_date_from_srs_filename(fname)
            if not date:
                continue

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except OSError:
                continue

            for line in lines:
                # Location tokens look like N15E42 / S08W12
                loc_match = _LOCATION_RE.search(line)
                if not loc_match:
                    continue
                lon = _parse_longitude_from_location_token(loc_match.group(0))
                if lon is None:
                    continue

                # Try to find NOAA AR number on the same line.
                # Many SRS formats place the region number as the first integer token.
                tokens = line.strip().split()
                ar_num: Optional[int] = None
                for tok in tokens:
                    if tok.isdigit():
                        ar_num = int(tok)
                        break
                if ar_num is None:
                    # As a fallback, grab the first 4-6 digit number anywhere in the line.
                    m = re.search(r"\b(\d{4,6})\b", line)
                    if m:
                        ar_num = int(m.group(1))

                if ar_num is None:
                    continue

                key = (ar_num, date)
                # If there are duplicates, keep the first encountered.
                if key not in lookup:
                    lookup[key] = lon

    return lookup


def _to_prob_positive(outputs: torch.Tensor) -> torch.Tensor:
    """Convert model outputs to P(y=1).

    Handles common cases:
    - logits shape [B, 2] -> softmax -> [:, 1]
    - logits shape [B] or [B, 1] -> sigmoid
    - probabilities already in [0, 1]
    """
    if outputs.ndim == 2 and outputs.shape[1] == 2:
        # In this codebase we train with CrossEntropy-style 2-class logits.
        # Always apply softmax to get P(y=1).
        return torch.softmax(outputs, dim=1)[:, 1]

    if outputs.ndim == 2 and outputs.shape[1] == 1:
        outputs = outputs[:, 0]

    if outputs.ndim != 1:
        raise ValueError(f"Unsupported output shape: {tuple(outputs.shape)}")

    if outputs.min().item() >= 0.0 and outputs.max().item() <= 1.0:
        return outputs

    return torch.sigmoid(outputs)


@dataclass
class BucketStats:
    name: str
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def samples(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        preds_i = preds.to(torch.int64)
        labels_i = labels.to(torch.int64)
        self.tp += int(((labels_i == 1) & (preds_i == 1)).sum().item())
        self.tn += int(((labels_i == 0) & (preds_i == 0)).sum().item())
        self.fp += int(((labels_i == 0) & (preds_i == 1)).sum().item())
        self.fn += int(((labels_i == 1) & (preds_i == 0)).sum().item())

    def tpr(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def tnr(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0

    def tss(self) -> float:
        return self.tpr() + self.tnr() - 1.0

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    def recall(self) -> float:
        return self.tpr()

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def accuracy(self) -> float:
        total = self.samples
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "samples": int(self.samples),
            "TP": int(self.tp),
            "TN": int(self.tn),
            "FP": int(self.fp),
            "FN": int(self.fn),
            "TPR": float(self.tpr()),
            "TNR": float(self.tnr()),
            "TSS": float(self.tss()),
            "Precision": float(self.precision()),
            "Recall": float(self.recall()),
            "F1": float(self.f1()),
            "Accuracy": float(self.accuracy()),
        }

    def report(self) -> str:
        return (
            f"[{self.name}]\n"
            f"samples={self.samples}\n"
            f"TP={self.tp} TN={self.tn} FP={self.fp} FN={self.fn}\n"
            f"TPR={self.tpr():.4f} TNR={self.tnr():.4f} TSS={self.tss():.4f}\n"
            f"Precision={self.precision():.4f} Recall={self.recall():.4f} F1={self.f1():.4f} Acc={self.accuracy():.4f}\n"
        )


def _safe_total(dataloader) -> Optional[int]:
    try:
        return len(dataloader)
    except TypeError:
        return None


def _compute_auc_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {"ROC_AUC": float("nan"), "PR_AUC": float("nan")}
    if labels.size == 0:
        return out
    if roc_auc_score is None or average_precision_score is None:
        return out
    try:
        out["ROC_AUC"] = float(roc_auc_score(labels, probs))
    except Exception:
        pass
    try:
        out["PR_AUC"] = float(average_precision_score(labels, probs))
    except Exception:
        pass
    return out


def _plot_curves(out_dir: str, curves: dict[str, tuple[np.ndarray, np.ndarray]], *, title_suffix: str = "") -> None:
    """Plot ROC and PR curves for multiple series.

    curves: name -> (labels, probs)
    """
    if not curves:
        return

    if roc_curve is None or precision_recall_curve is None or roc_auc_score is None or average_precision_score is None:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    for name, (y, p) in curves.items():
        if y.size == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(y, p)
            auc_v = roc_auc_score(y, p)
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc_v:.3f})")
        except Exception:
            continue
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC{title_suffix}")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle="--", alpha=0.3)

    # PR
    for name, (y, p) in curves.items():
        if y.size == 0:
            continue
        try:
            prec, rec, _ = precision_recall_curve(y, p)
            ap = average_precision_score(y, p)
            ax_pr.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        except Exception:
            continue
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision–Recall{title_suffix}")
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=200)
    plt.close(fig)


def _plot_bucket_tss(out_dir: str, buckets: dict[str, BucketStats]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = list(buckets.keys())
    tss_vals = [buckets[n].tss() for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, tss_vals)
    ax.set_ylabel("TSS")
    ax.set_title("A2 Longitude Buckets")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bucket_tss.png"), dpi=200)
    plt.close(fig)


def _try_read_best_threshold_from_exp_dir(exp_dir: str) -> Optional[float]:
    """Try to infer best decision threshold from prior experiment artifacts.

    Expected sources:
    - metrics.json with key Best_threshold
    - log.txt with line like: Best TSS: ... @ threshold=0.350
    """
    # 1) metrics.json
    metrics_path = os.path.join(exp_dir, "metrics.json")
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            for k in ("Best_threshold", "best_threshold", "best_tss_threshold"):
                if k in metrics:
                    val = float(metrics[k])
                    if 0.0 <= val <= 1.0:
                        return val
    except Exception:
        pass

    # 2) log.txt
    log_path = os.path.join(exp_dir, "log.txt")
    try:
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            m = re.search(r"Best TSS:.*?threshold=([0-9]*\.?[0-9]+)", text)
            if m:
                val = float(m.group(1))
                if 0.0 <= val <= 1.0:
                    return val
    except Exception:
        pass

    return None


def evaluate_longitude_buckets(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    srs_lookup: Dict[LocationKey, int],
    threshold: float = 0.5,
) -> Tuple[BucketStats, BucketStats, int, dict[str, Any]]:
    lon_in_30 = BucketStats(name="Longitude ≤ 30°")
    lon_out_30 = BucketStats(name="30° < Longitude ≤ 60°")
    skipped_missing_srs = 0

    # Collect labels/probs for curves and AUC metrics
    all_labels: list[int] = []
    all_probs: list[float] = []
    in30_labels: list[int] = []
    in30_probs: list[float] = []
    out30_labels: list[int] = []
    out30_probs: list[float] = []

    model.eval()
    with torch.no_grad():
        is_tty = hasattr(getattr(__import__("sys"), "stdout"), "isatty") and __import__("sys").stdout.isatty()
        pbar = tqdm(total=_safe_total(test_loader), desc="A2 Test", leave=True, disable=not is_tty, dynamic_ncols=True)
        for inputs, labels, meta in test_loader:
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(device, non_blocking=True) for x in inputs)
            else:
                inputs = inputs.to(device, non_blocking=True)

            labels_t = labels
            if torch.is_tensor(labels_t):
                labels_t = labels_t.to(device, non_blocking=True)
            else:
                labels_t = torch.as_tensor(labels_t, device=device)

            outputs = model(inputs)
            probs = _to_prob_positive(outputs).detach()
            preds = (probs >= float(threshold)).to(torch.int64)

            probs_cpu = probs.detach().float().cpu().numpy()
            labels_cpu = labels_t.detach().to(torch.int64).cpu().numpy()

            # meta is expected to be a dict-like batch structure.
            # meta is usually a dict of batched fields. Some pipelines do not
            # include a separate 'date' field; in that case we try to extract
            # the date from the sample filename supplied in 'basename'.
            ars = meta.get("ar")
            dates = meta.get("date")
            basenames = meta.get("basename")

            if ars is None:
                raise KeyError("meta must contain key 'ar'")

            # Normalize/derive YYYYMMDD date strings.
            if dates is None:
                if basenames is None:
                    raise KeyError("meta must contain keys 'date' or 'basename' to extract date")
                if torch.is_tensor(basenames):
                    basenames_list = [b.decode() if isinstance(b, bytes) else str(b) for b in basenames.detach().cpu().tolist()]
                else:
                    basenames_list = list(basenames)
                dates = [_normalize_yyyymmdd(b) for b in basenames_list]
            else:
                if torch.is_tensor(dates):
                    dates_list = dates.detach().cpu().tolist()
                else:
                    dates_list = list(dates)
                dates = [_normalize_yyyymmdd(d) for d in dates_list]

            # Normalize to python lists for per-sample lookup
            if torch.is_tensor(ars):
                ars_list: Iterable = ars.detach().cpu().tolist()
            else:
                ars_list = list(ars)

            dates_list = list(dates)

            for i, (ar, date) in enumerate(zip(ars_list, dates_list)):
                try:
                    ar_i = int(ar)
                except Exception:
                    skipped_missing_srs += 1
                    continue
                date_s = str(date) if date is not None else ""
                if not date_s:
                    skipped_missing_srs += 1
                    continue
                key = (ar_i, date_s)
                lon = srs_lookup.get(key)
                if lon is None:
                    skipped_missing_srs += 1
                    continue

                abs_lon = abs(int(lon))
                if abs_lon <= 30:
                    lon_in_30.update(preds[i : i + 1], labels_t[i : i + 1])
                    in30_labels.append(int(labels_cpu[i]))
                    in30_probs.append(float(probs_cpu[i]))
                elif 30 < abs_lon <= 60:
                    lon_out_30.update(preds[i : i + 1], labels_t[i : i + 1])
                    out30_labels.append(int(labels_cpu[i]))
                    out30_probs.append(float(probs_cpu[i]))
                else:
                    # Should not exist per spec; skip if it does.
                    continue

                all_labels.append(int(labels_cpu[i]))
                all_probs.append(float(probs_cpu[i]))

            pbar.update(1)
        pbar.close()

    arrays = {
        "all": (np.asarray(all_labels, dtype=np.int64), np.asarray(all_probs, dtype=np.float32)),
        "in_30": (np.asarray(in30_labels, dtype=np.int64), np.asarray(in30_probs, dtype=np.float32)),
        "out_30_60": (np.asarray(out30_labels, dtype=np.int64), np.asarray(out30_probs, dtype=np.float32)),
    }
    return lon_in_30, lon_out_30, skipped_missing_srs, arrays


def _load_cfg_from_exp_dir(exp_dir: str) -> dict:
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in exp_dir: {exp_dir}")
    with open(config_path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4.2 (A2) – High Longitude Impact")
    parser.add_argument("--exp-dir", default=None, help="Experiment directory containing config.json and checkpoint")
    parser.add_argument("--cfg-json", default=None, help="Path to a cfg JSON (alternative to --exp-dir/config.json)")
    parser.add_argument("--checkpoint", default="best_tss.pt", help="Checkpoint filename inside exp_dir")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Full checkpoint path (overrides --exp-dir/--checkpoint)",
    )
    parser.add_argument(
        "--srs-root",
        default=str(Path(__file__).resolve().parents[2] / "data" / "SRS"),
        help="Root directory containing YYYY_SRS folders",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for evaluation batch size (useful for CPU-only runs)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override for DataLoader num_workers (useful for clusters / CPUs)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Decision threshold for positive class. "
            "If omitted, will try to use Best_threshold from exp_dir/metrics.json (or exp_dir/log.txt); "
            "falls back to 0.5."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write A2 outputs (defaults to a new timestamped folder under results_base)",
    )
    args = parser.parse_args()

    if args.exp_dir is None and args.cfg_json is None:
        raise ValueError("Provide --exp-dir or --cfg-json")

    if args.cfg_json is not None:
        with open(args.cfg_json, "r") as f:
            cfg = json.load(f)
    else:
        cfg = _load_cfg_from_exp_dir(args.exp_dir)

    # Threshold: default to the experiment's best-TSS threshold when available.
    if args.threshold is None:
        inferred = None
        if args.exp_dir is not None:
            inferred = _try_read_best_threshold_from_exp_dir(args.exp_dir)
        threshold = float(inferred) if inferred is not None else 0.5
    else:
        threshold = float(args.threshold)

    # Optional overrides for evaluation convenience
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["num_workers"] = int(args.num_workers)

    CFG.update(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg=cfg, num_classes=2).to(device)

    if args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    else:
        if args.exp_dir is None:
            raise ValueError("--exp-dir is required unless --checkpoint-path is provided")
        ckpt_path = os.path.join(args.exp_dir, args.checkpoint)

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and ("model_state_dict" in state or "state_dict" in state):
        model.load_state_dict(state.get("model_state_dict") or state.get("state_dict"))
    else:
        model.load_state_dict(state)

    dls = create_dataloaders()
    test_loader = dls["Test"]

    srs_lookup = build_srs_longitude_lookup(args.srs_root)

    lon_in_30, lon_out_30, skipped, arrays = evaluate_longitude_buckets(
        model=model,
        test_loader=test_loader,
        device=device,
        srs_lookup=srs_lookup,
        threshold=threshold,
    )

    print(lon_in_30.report())
    print(lon_out_30.report())
    print(f"skipped_missing_srs={skipped}")

    # Save outputs under results/
    if args.out_dir is None:
        base = CFG.get("results_base", str(Path(__file__).resolve().parents[2] / "results"))
        tag = "A2_longitude_eval"
        if args.exp_dir:
            tag += "_" + os.path.basename(args.exp_dir)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = os.path.join(base, f"{ts}_{tag}")
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Metrics dict
    buckets = {
        "lon_le_30": lon_in_30,
        "lon_30_60": lon_out_30,
    }
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "skipped_missing_srs": int(skipped),
        "checkpoint_path": ckpt_path,
        "exp_dir": args.exp_dir,
        "buckets": {k: v.as_dict() for k, v in buckets.items()},
    }

    # Add AUC metrics (overall + per bucket)
    curves = {}
    for name, (y, p) in arrays.items():
        metrics[name] = _compute_auc_metrics(y, p)
        curves_name = {
            "all": "All",
            "in_30": "|lon|≤30°",
            "out_30_60": "30°<|lon|≤60°",
        }.get(name, name)
        curves[curves_name] = (y, p)

    with open(os.path.join(out_dir, "a2_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    _plot_curves(out_dir, curves)
    _plot_bucket_tss(out_dir, buckets)

    # Human-readable report
    with open(os.path.join(out_dir, "a2_report.txt"), "w") as f:
        f.write(lon_in_30.report() + "\n")
        f.write(lon_out_30.report() + "\n")
        f.write(f"skipped_missing_srs={skipped}\n")
        f.write("\n")
        f.write(f"threshold={threshold:.6f}\n")
        f.write(f"ROC_AUC(all)={metrics['all']['ROC_AUC']:.4f} PR_AUC(all)={metrics['all']['PR_AUC']:.4f}\n")
    print(f"Saved A2 outputs to: {out_dir}")


if __name__ == "__main__":
    main()
