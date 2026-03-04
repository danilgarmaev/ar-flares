#!/usr/bin/env python3
"""Downsample a JSON metadata dataset with AR-stratification.

This is a repo-local utility for creating derived *static* datasets from a JSON
metadata file of the form:

{
  "data": [
    {
      "filepath": "...",
      "sname": "...",
      "label": "C" | "M" | "X" | "NONE",
      "id": 1,
      "ar": 1069,
      "timestamp": "2010-05-05 00:00:00"
    },
    ...
  ]
}

Capabilities (intended for the experiments discussed in this repo):
- Downsample label=="NONE" and/or label=="C" while keeping M/X intact.
- Perform downsampling stratified by AR to avoid dropping small AR groups.

Typical usages:
- C+ downsampling: reduce NONE only (keep all C/M/X)
- M+ downsampling: optionally reduce NONE and/or C, depending on chosen protocol

This script deliberately does not infer C+/M+ at runtime; it operates on the
explicit "label" field in the JSON.
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

LABELS = ("NONE", "C", "M", "X")


@dataclass(frozen=True)
class Targets:
    target_total: Optional[int]
    target_none: Optional[int]
    target_c: Optional[int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Downsample NONE and/or C stratified by AR")
    p.add_argument("--input", required=True, help="Path to input JSON")
    p.add_argument("--output", required=True, help="Path to output JSON")

    p.add_argument("--seed", type=int, default=None, help="Random seed")

    # Targets
    p.add_argument("--target-total", type=int, default=None, help="Target total records")
    p.add_argument("--target-none", type=int, default=None, help="Target NONE records")
    p.add_argument("--target-c", type=int, default=None, help="Target C records")

    p.add_argument(
        "--downsample-labels",
        nargs="+",
        choices=["NONE", "C"],
        default=["NONE"],
        help="Which labels are allowed to be downsampled. Default: NONE",
    )

    p.add_argument("--removed", default=None, help="Optional path to store removed entries")
    p.add_argument("--dry-run", action="store_true", help="Compute and print summary only")
    return p.parse_args()


def load_payload(path: str) -> dict:
    with open(path, "r") as f:
        payload = json.load(f)
    if "data" not in payload or not isinstance(payload["data"], list):
        raise ValueError("Input JSON must contain a top-level 'data' list")
    return payload


def save_payload(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def label_counts(records: Sequence[dict]) -> Counter:
    c = Counter()
    for r in records:
        c[r.get("label")] += 1
    return c


def weighted_sample_without_replacement(items: Sequence, weights: Sequence[float], k: int, rng: random.Random) -> List:
    """Efraimidis–Spirakis weighted sampling without replacement."""
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)

    keys = []
    for item, w in zip(items, weights):
        w = max(0.0, float(w))
        if w == 0.0:
            key = float("-inf")
        else:
            u = max(rng.random(), 1e-12)
            key = math.log(u) / w
        keys.append((key, item))

    keys.sort(reverse=True, key=lambda x: x[0])
    return [item for _, item in keys[:k]]


def group_indices_by_ar(records: Sequence[dict], allowed_labels: Iterable[str]) -> Dict[str, Dict[int, List[int]]]:
    """Return dict[label][ar] -> indices."""
    allowed = set(allowed_labels)
    out: Dict[str, Dict[int, List[int]]] = {lbl: defaultdict(list) for lbl in allowed}
    for idx, rec in enumerate(records):
        lbl = rec.get("label")
        if lbl in allowed:
            out[lbl][rec.get("ar")].append(idx)
    return out


def compute_keep_plan_per_label(
    groups: Dict[int, List[int]],
    keep_total: int,
    rng: random.Random,
) -> Dict[int, int]:
    """Compute how many to keep per AR for a given label group dict."""
    ars = list(groups.keys())
    sizes = {ar: len(groups[ar]) for ar in ars}
    g = len(ars)
    total = sum(sizes.values())

    keep_total = max(0, min(int(keep_total), int(total)))
    keep = {ar: 0 for ar in ars}

    if keep_total == 0 or g == 0:
        return keep

    if keep_total >= g:
        # keep at least 1 per group
        for ar in ars:
            if sizes[ar] > 0:
                keep[ar] = 1
        remaining = keep_total - sum(keep.values())
        if remaining > 0:
            weights = {ar: max(0, sizes[ar] - keep[ar]) for ar in ars}
            wsum = sum(weights.values())
            if wsum > 0:
                ideal = {ar: (weights[ar] / wsum) * remaining for ar in ars}
                floors = {ar: int(math.floor(ideal[ar])) for ar in ars}
                for ar in ars:
                    keep[ar] += floors[ar]
                leftover = remaining - sum(floors.values())
                if leftover > 0:
                    remainders = {ar: ideal[ar] - floors[ar] for ar in ars}
                    rsum = sum(remainders.values())
                    probs = [remainders[ar] if rsum > 0 else weights[ar] for ar in ars]
                    chosen = weighted_sample_without_replacement(ars, probs, leftover, rng)
                    for ar in chosen:
                        keep[ar] += 1
        for ar in ars:
            keep[ar] = min(keep[ar], sizes[ar])

    else:
        # cannot keep 1 per group; choose groups proportional to size
        probs = [sizes[ar] for ar in ars]
        chosen = set(weighted_sample_without_replacement(ars, probs, keep_total, rng))
        for ar in chosen:
            keep[ar] = 1

    return keep


def choose_keep_indices(groups: Dict[int, List[int]], keep_plan: Dict[int, int], rng: random.Random) -> Tuple[List[int], List[int]]:
    keep_idxs: List[int] = []
    remove_idxs: List[int] = []
    for ar, idxs in groups.items():
        k = int(keep_plan.get(ar, 0))
        if k >= len(idxs):
            keep_idxs.extend(idxs)
        else:
            chosen = rng.sample(idxs, k)
            keep_idxs.extend(chosen)
            keep_set = set(chosen)
            remove_idxs.extend([i for i in idxs if i not in keep_set])
    return keep_idxs, remove_idxs


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    payload = load_payload(args.input)
    records: List[dict] = payload["data"]

    counts_in = label_counts(records)

    allowed_downsample = set(args.downsample_labels)

    if args.target_total is None and args.target_none is None and args.target_c is None:
        raise ValueError("At least one of --target-total/--target-none/--target-c must be provided")

    # Baseline: keep all non-downsampled labels intact
    fixed_keep = [
        i
        for i, r in enumerate(records)
        if r.get("label") not in allowed_downsample
    ]

    # Determine per-label targets (if target_total is provided, we interpret remaining budget
    # after fixed_keep as coming from allowed_downsample labels, proportionally by current sizes
    # unless explicit per-label targets are provided).
    targets = Targets(args.target_total, args.target_none, args.target_c)

    current_allowed_counts = {lbl: counts_in.get(lbl, 0) for lbl in allowed_downsample}
    fixed_keep_count = len(fixed_keep)

    explicit_targets: Dict[str, Optional[int]] = {
        "NONE": targets.target_none if "NONE" in allowed_downsample else None,
        "C": targets.target_c if "C" in allowed_downsample else None,
    }

    # If target_total is used without explicit per-label targets, allocate remaining budget
    # to allowed labels proportionally.
    allocated_targets: Dict[str, int] = {lbl: current_allowed_counts.get(lbl, 0) for lbl in allowed_downsample}

    if targets.target_total is not None:
        total_target = int(targets.target_total)
        if total_target < fixed_keep_count:
            raise ValueError(
                f"--target-total={total_target} is smaller than fixed (non-downsampled) count ({fixed_keep_count})."
            )

        remaining_budget = total_target - fixed_keep_count

        # If any explicit targets provided, respect them first.
        for lbl, t in explicit_targets.items():
            if t is not None:
                allocated_targets[lbl] = max(0, min(int(t), current_allowed_counts.get(lbl, 0)))

        explicitly_set = {lbl for lbl, t in explicit_targets.items() if t is not None}
        remaining_labels = [lbl for lbl in allowed_downsample if lbl not in explicitly_set]

        remaining_budget_after_explicit = remaining_budget - sum(allocated_targets.get(lbl, 0) for lbl in explicitly_set)
        if remaining_budget_after_explicit < 0:
            raise ValueError("Explicit per-label targets exceed --target-total budget")

        if remaining_labels:
            denom = sum(current_allowed_counts.get(lbl, 0) for lbl in remaining_labels)
            if denom == 0:
                for lbl in remaining_labels:
                    allocated_targets[lbl] = 0
            else:
                # proportional allocation + rounding
                ideal = {lbl: (current_allowed_counts.get(lbl, 0) / denom) * remaining_budget_after_explicit for lbl in remaining_labels}
                floors = {lbl: int(math.floor(v)) for lbl, v in ideal.items()}
                for lbl in remaining_labels:
                    allocated_targets[lbl] = max(0, min(floors[lbl], current_allowed_counts.get(lbl, 0)))

                leftover = remaining_budget_after_explicit - sum(floors.values())
                if leftover > 0:
                    remainders = {lbl: ideal[lbl] - floors[lbl] for lbl in remaining_labels}
                    rsum = sum(remainders.values())
                    probs = [remainders[lbl] if rsum > 0 else current_allowed_counts.get(lbl, 0) for lbl in remaining_labels]
                    chosen = weighted_sample_without_replacement(remaining_labels, probs, leftover, rng)
                    for lbl in chosen:
                        allocated_targets[lbl] = min(allocated_targets[lbl] + 1, current_allowed_counts.get(lbl, 0))

    else:
        # No target_total; use per-label targets independently.
        for lbl, t in explicit_targets.items():
            if t is not None and lbl in allowed_downsample:
                allocated_targets[lbl] = max(0, min(int(t), current_allowed_counts.get(lbl, 0)))

    # Build keep/remove indices for each downsampled label with AR stratification.
    by_label_ar = group_indices_by_ar(records, allowed_downsample)

    keep_idxs_all = set(fixed_keep)
    remove_idxs_all: List[int] = []

    for lbl in sorted(allowed_downsample):
        groups = by_label_ar.get(lbl, {})
        keep_total = int(allocated_targets.get(lbl, 0))
        keep_plan = compute_keep_plan_per_label(groups, keep_total, rng)
        keep_lbl, remove_lbl = choose_keep_indices(groups, keep_plan, rng)
        keep_idxs_all.update(keep_lbl)
        remove_idxs_all.extend(remove_lbl)

    out_records = [r for i, r in enumerate(records) if i in keep_idxs_all]
    removed_records = [r for i, r in enumerate(records) if i not in keep_idxs_all]

    counts_out = label_counts(out_records)

    print("=== Summary ===")
    print(f"Input counts:  {dict(counts_in)}")
    print(f"Output counts: {dict(counts_out)}")
    print(f"Removed total: {len(removed_records)}")
    if targets.target_total is not None:
        print(f"Target total:  {targets.target_total}")
    if "NONE" in allowed_downsample:
        print(f"Target NONE:   {allocated_targets.get('NONE', 0)}")
    if "C" in allowed_downsample:
        print(f"Target C:      {allocated_targets.get('C', 0)}")

    if args.dry_run:
        print("[DRY-RUN] No files written.")
        return

    out_payload = dict(payload)
    out_payload["data"] = out_records
    save_payload(args.output, out_payload)
    print(f"[OK] Wrote output dataset to: {args.output}")

    if args.removed:
        save_payload(args.removed, {"data": removed_records})
        print(f"[OK] Wrote removed entries to: {args.removed}")


if __name__ == "__main__":
    main()
