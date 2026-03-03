#!/usr/bin/env python3
"""Create train/val/test splits that are **AR-exclusive**.

This is inspired by the workflow in `inaf-oact-ai/solar-flare-forecaster`:
- Parse per-sample metadata
- Group samples by active region (AR)
- Assign whole AR groups to splits so no AR appears in multiple splits
- Try to match requested split fractions by total *sample count*

For this repo, we parse the official AR-flares labels file lines like:
  1069_hmi.M_720s.20100505_000000_TAI.1.magnetogram_224.png,M1.2

We extract:
- ar: "1069" (prefix before first '_')
- timestamp: "20100505_000000" (first YYYYMMDD_HHMMSS occurrence)
- flare_type: NONE/C/M/X (from label)

Outputs:
- outdir/train_ars.txt, outdir/val_ars.txt, outdir/test_ars.txt
- outdir/summary.json with counts

Note: This script does NOT write WebDataset shards; it only writes split lists.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


_TS_RE = re.compile(r"(\d{8}_\d{6})")


def _parse_label_to_flare_type(raw: str) -> str:
    s = str(raw).strip().upper()
    if s == "0" or s == "NONE":
        return "NONE"
    # common formats: C4.7, M1.2, X2.0
    if s and s[0] in {"C", "M", "X"}:
        return s[0]
    return s


def _parse_ar_from_filename(fname: str) -> str:
    # official files are like: 1069_hmi.M_720s.20100505_000000_TAI.1.magnetogram_224.png
    base = Path(fname).name
    ar = base.split("_", 1)[0]
    return ar.replace("AR", "")


def _parse_ts_from_filename(fname: str) -> str | None:
    m = _TS_RE.search(fname)
    return m.group(1) if m else None


@dataclass(frozen=True)
class Sample:
    ar: str
    flare_type: str
    ts: str | None


def iter_samples_from_labels(labels_path: Path) -> Iterable[Sample]:
    with labels_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            fname, raw_label = line.split(",", 1)
            ar = _parse_ar_from_filename(fname)
            flare_type = _parse_label_to_flare_type(raw_label)
            ts = _parse_ts_from_filename(fname)
            yield Sample(ar=ar, flare_type=flare_type, ts=ts)


def _compute_targets(total: int, fractions: Tuple[float, float, float]) -> Dict[str, int]:
    train_f, val_f, test_f = fractions
    train_n = int(round(total * train_f))
    val_n = int(round(total * val_f))
    test_n = total - train_n - val_n
    # guard against rounding causing negatives
    if test_n < 0:
        deficit = -test_n
        take_from_val = min(deficit, val_n)
        val_n -= take_from_val
        deficit -= take_from_val
        train_n -= deficit
        test_n = 0
    return {"train": train_n, "val": val_n, "test": test_n}


def _initial_assignment(
    ar_sizes: Dict[str, int],
    targets: Dict[str, int],
    seed: int,
    pack_order: str,
) -> Dict[str, str]:
    rng = random.Random(seed)
    ars = list(ar_sizes.keys())
    rng.shuffle(ars)

    if pack_order == "size-desc":
        # sort by size but keep seed-sensitivity for ties
        ars.sort(key=lambda ar: (ar_sizes[ar], rng.random()), reverse=True)

    splits = ("train", "val", "test")
    current = {s: 0 for s in splits}
    assigned: Dict[str, str] = {}

    def cost(split: str, size: int) -> float:
        # primary: absolute deviation from target after adding this AR
        after = current[split] + size
        base = abs(after - targets[split])
        # tiny jitter to break ties reproducibly
        return base + 1e-6 * rng.random()

    for ar in ars:
        size = ar_sizes[ar]
        under = [s for s in splits if current[s] < targets[s]]
        candidates = under if under else list(splits)
        chosen = min(candidates, key=lambda s: cost(s, size))
        assigned[ar] = chosen
        current[chosen] += size

    return assigned


def _rebalance(
    ar_sizes: Dict[str, int],
    assignment: Dict[str, str],
    targets: Dict[str, int],
    max_iters: int = 200,
) -> Dict[str, str]:
    splits = ("train", "val", "test")

    def counts(asg: Dict[str, str]) -> Dict[str, int]:
        c = {s: 0 for s in splits}
        for ar, sp in asg.items():
            c[sp] += ar_sizes[ar]
        return c

    def total_l1(cnts: Dict[str, int]) -> int:
        return int(sum(abs(cnts[s] - targets[s]) for s in splits))

    asg = dict(assignment)
    for _ in range(max_iters):
        cnts = counts(asg)
        l1 = total_l1(cnts)

        most_under = min(splits, key=lambda s: cnts[s] - targets[s])
        most_over = max(splits, key=lambda s: cnts[s] - targets[s])

        under_def = targets[most_under] - cnts[most_under]
        over_excess = cnts[most_over] - targets[most_over]
        if under_def <= 0 or over_excess <= 0:
            break

        cands = [ar for ar, sp in asg.items() if sp == most_over]
        if not cands:
            break
        cands.sort(key=lambda ar: ar_sizes[ar])

        moved = False
        for ar in cands:
            asg[ar] = most_under
            new_cnts = counts(asg)
            if total_l1(new_cnts) < l1:
                moved = True
                break
            asg[ar] = most_over

        if not moved:
            break

    return asg


def make_splits(labels_path: Path, seed: int, fractions: Tuple[float, float, float], pack_order: str) -> Tuple[Dict[str, List[str]], dict]:
    # Group samples by AR
    ar_sizes: Dict[str, int] = defaultdict(int)
    ar_label_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    total = 0
    for s in iter_samples_from_labels(labels_path):
        ar_sizes[s.ar] += 1
        ar_label_counts[s.ar][s.flare_type] += 1
        total += 1

    targets = _compute_targets(total, fractions)
    assignment0 = _initial_assignment(ar_sizes, targets, seed=seed, pack_order=pack_order)
    assignment = _rebalance(ar_sizes, assignment0, targets)

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for ar, sp in assignment.items():
        splits[sp].append(ar)

    for sp in splits:
        splits[sp].sort(key=lambda x: int(x) if x.isdigit() else x)

    # Build a summary
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_labels: Dict[str, Dict[str, int]] = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}
    for ar, sp in assignment.items():
        split_counts[sp] += ar_sizes[ar]
        for lab, cnt in ar_label_counts[ar].items():
            split_labels[sp][lab] += cnt

    summary = {
        "labels_path": str(labels_path),
        "seed": seed,
        "fractions": {"train": fractions[0], "val": fractions[1], "test": fractions[2]},
        "targets": targets,
        "achieved": split_counts,
        "num_ars": {k: len(v) for k, v in splits.items()},
        "label_counts": {sp: dict(sorted(split_labels[sp].items(), key=lambda kv: (-kv[1], kv[0]))) for sp in split_labels},
    }

    return splits, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Create AR-exclusive train/val/test split lists from AR-flares labels.")
    ap.add_argument("--labels", type=Path, required=True, help="Path to C1.0_24hr_224_png_Labels.txt")
    ap.add_argument("--outdir", type=Path, required=True, help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fractions", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"), default=[0.7, 0.1, 0.2])
    ap.add_argument("--pack-order", choices=["random", "size-desc"], default="random")
    args = ap.parse_args()

    fr = tuple(float(x) for x in args.fractions)
    if abs(sum(fr) - 1.0) > 1e-6:
        raise SystemExit("--fractions must sum to 1.0")

    splits, summary = make_splits(args.labels, seed=args.seed, fractions=fr, pack_order=args.pack_order)

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "train_ars.txt").write_text("\n".join(splits["train"]) + "\n", encoding="utf-8")
    (args.outdir / "val_ars.txt").write_text("\n".join(splits["val"]) + "\n", encoding="utf-8")
    (args.outdir / "test_ars.txt").write_text("\n".join(splits["test"]) + "\n", encoding="utf-8")
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote: {args.outdir}/train_ars.txt")
    print(f"Wrote: {args.outdir}/val_ars.txt")
    print(f"Wrote: {args.outdir}/test_ars.txt")


if __name__ == "__main__":
    main()
