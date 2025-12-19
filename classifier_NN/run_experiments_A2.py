"""Experiment 4.2 (A2) – High Longitude Impact (evaluation only).

Goal
----
Measure whether a trained model's TSS degrades for high-longitude Active Regions,
without any projection correction.

Constraints
-----------
- Evaluation-only: do NOT retrain.
- Use existing Test dataloader that yields (x, label, meta).
- Bucket by SRS Location longitude using a fixed threshold=0.5.

Usage
-----
python -m classifier_NN.run_experiments_A2 \
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
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch

from .config import CFG
from .datasets import create_dataloaders
from .models import build_model


LocationKey = Tuple[int, str]  # (ar_number, YYYYMMDD)


_LOCATION_RE = re.compile(r"\b[NS](\d{1,2})([EW])(\d{1,2})\b")


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
        # Heuristic: if values aren't in [0,1] or rows don't sum ~1, treat as logits.
        if outputs.min().item() < 0.0 or outputs.max().item() > 1.0:
            return torch.softmax(outputs, dim=1)[:, 1]
        row_sums = outputs.sum(dim=1)
        if torch.any(torch.abs(row_sums - 1.0) > 1e-2):
            return torch.softmax(outputs, dim=1)[:, 1]
        return outputs[:, 1]

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

    def report(self) -> str:
        return (
            f"[{self.name}]\n"
            f"samples={self.samples}\n"
            f"TP={self.tp} TN={self.tn} FP={self.fp} FN={self.fn}\n"
            f"TPR={self.tpr():.4f} TNR={self.tnr():.4f} TSS={self.tss():.4f}\n"
        )


def evaluate_longitude_buckets(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    srs_lookup: Dict[LocationKey, int],
    threshold: float = 0.5,
) -> Tuple[BucketStats, BucketStats, int]:
    lon_in_30 = BucketStats(name="Longitude ≤ 30°")
    lon_out_30 = BucketStats(name="30° < Longitude ≤ 60°")
    skipped_missing_srs = 0

    model.eval()
    with torch.no_grad():
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

            # meta is expected to be a dict-like batch structure.
            ars = meta.get("ar")
            dates = meta.get("date")
            if ars is None or dates is None:
                raise KeyError("meta must contain keys 'ar' and 'date'")

            # Normalize to python lists for per-sample lookup
            if torch.is_tensor(ars):
                ars_list: Iterable = ars.detach().cpu().tolist()
            else:
                ars_list = list(ars)

            if torch.is_tensor(dates):
                # unlikely, but handle if date encoded as ints
                dates_list: Iterable = dates.detach().cpu().tolist()
                dates_list = [str(d) for d in dates_list]
            else:
                dates_list = list(dates)

            for i, (ar, date) in enumerate(zip(ars_list, dates_list)):
                try:
                    ar_i = int(ar)
                except Exception:
                    skipped_missing_srs += 1
                    continue
                date_s = str(date)
                key = (ar_i, date_s)
                lon = srs_lookup.get(key)
                if lon is None:
                    skipped_missing_srs += 1
                    continue

                abs_lon = abs(int(lon))
                if abs_lon <= 30:
                    lon_in_30.update(preds[i : i + 1], labels_t[i : i + 1])
                elif 30 < abs_lon <= 60:
                    lon_out_30.update(preds[i : i + 1], labels_t[i : i + 1])
                else:
                    # Should not exist per spec; skip if it does.
                    continue

    return lon_in_30, lon_out_30, skipped_missing_srs


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
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "SRS"),
        help="Root directory containing YYYY_SRS folders",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed decision threshold")
    args = parser.parse_args()

    if args.exp_dir is None and args.cfg_json is None:
        raise ValueError("Provide --exp-dir or --cfg-json")

    if args.cfg_json is not None:
        with open(args.cfg_json, "r") as f:
            cfg = json.load(f)
    else:
        cfg = _load_cfg_from_exp_dir(args.exp_dir)

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

    lon_in_30, lon_out_30, skipped = evaluate_longitude_buckets(
        model=model,
        test_loader=test_loader,
        device=device,
        srs_lookup=srs_lookup,
        threshold=float(args.threshold),
    )

    print(lon_in_30.report())
    print(lon_out_30.report())
    print(f"skipped_missing_srs={skipped}")


if __name__ == "__main__":
    main()
