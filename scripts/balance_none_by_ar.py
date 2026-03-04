#!/usr/bin/env python3
"""Downsample NONE entries stratified by AR.

This is a repo-local utility for creating downsampled *static* datasets from a
JSON metadata file of the form:

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

Only label=="NONE" entries are removed. C/M/X are always kept.
Removal is stratified by AR to avoid dropping small AR groups entirely unless
mathematically unavoidable.

Usage examples:
  python scripts/balance_none_by_ar.py --input train.json --output train_ds.json --target-none 12000 --seed 42
  python scripts/balance_none_by_ar.py --input train.json --output train_ds.json --target-total 50000 --seed 42
"""

import argparse
import json
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

LABEL_NONE = "NONE"


def parse_args():
    p = argparse.ArgumentParser(description="Downsample 'NONE' entries stratified by AR.")
    p.add_argument("--input", required=True, help="Path to input JSON file")
    p.add_argument("--output", required=True, help="Path to output JSON file")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--target-total", type=int, help="Desired total number of entries in output")
    grp.add_argument("--target-none", type=int, help="Desired number of NONE entries in output")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--removed", default=None, help="Optional path to save removed entries as JSON")
    p.add_argument("--dry-run", action="store_true", help="Compute plan and print summary; do not write files")
    return p.parse_args()


def load_data(path: str) -> List[dict]:
    with open(path, "r") as f:
        payload = json.load(f)
    if "data" not in payload or not isinstance(payload["data"], list):
        raise ValueError("Input JSON must contain a top-level 'data' list.")
    return payload["data"]


def save_data(path: str, data: List[dict]):
    with open(path, "w") as f:
        json.dump({"data": data}, f, ensure_ascii=False, indent=2)


def group_none_by_ar(records: List[dict]) -> Dict[int, List[int]]:
    """Return dict: ar -> list(indices in records) for NONE entries in that AR."""
    groups = defaultdict(list)
    for idx, rec in enumerate(records):
        if rec.get("label") == LABEL_NONE:
            ar = rec.get("ar")
            groups[ar].append(idx)
    return groups


def compute_removal_plan(
    none_groups: Dict[int, List[int]],
    keep_total_none: int,
    rng: random.Random,
) -> Tuple[Dict[int, int], int]:
    """Compute how many NONE entries to keep per AR.

    Strategy:
      1) If keep_total_none >= #groups: guarantee at least 1 kept per group,
         then distribute remaining proportionally to group sizes.
      2) If keep_total_none < #groups: choose keep_total_none groups to keep (1 each),
         sampled with probability proportional to group size.

    Returns: (keep_per_ar, actually_kept)
    """
    ars = list(none_groups.keys())
    sizes = {ar: len(none_groups[ar]) for ar in ars}
    g = len(ars)
    total_none = sum(sizes.values())

    keep_total_none = max(0, min(int(keep_total_none), int(total_none)))
    keep_per_ar = {ar: 0 for ar in ars}

    if keep_total_none == 0 or g == 0:
        return keep_per_ar, 0

    if keep_total_none >= g:
        for ar in ars:
            if sizes[ar] > 0:
                keep_per_ar[ar] = 1
        base_allocate = sum(keep_per_ar.values())
        remaining = keep_total_none - base_allocate

        if remaining > 0:
            weights = {ar: max(0, sizes[ar] - keep_per_ar[ar]) for ar in ars}
            weight_sum = sum(weights.values())
            if weight_sum > 0:
                ideal = {ar: (weights[ar] / weight_sum) * remaining for ar in ars}
                floors = {ar: int(math.floor(ideal[ar])) for ar in ars}
                allocated = sum(floors.values())
                for ar in ars:
                    keep_per_ar[ar] += floors[ar]

                leftover = remaining - allocated
                if leftover > 0:
                    remainders = {ar: ideal[ar] - floors[ar] for ar in ars}
                    rem_sum = sum(remainders.values())
                    probs = [remainders[ar] if rem_sum > 0 else weights[ar] for ar in ars]
                    chosen = weighted_sample_without_replacement(ars, probs, leftover, rng)
                    for ar in chosen:
                        keep_per_ar[ar] += 1

        for ar in ars:
            keep_per_ar[ar] = min(keep_per_ar[ar], sizes[ar])

    else:
        probs = [sizes[ar] for ar in ars]
        chosen_groups = set(weighted_sample_without_replacement(ars, probs, keep_total_none, rng))
        for ar in chosen_groups:
            keep_per_ar[ar] = 1

    actually_kept = sum(keep_per_ar.values())
    return keep_per_ar, actually_kept


def weighted_sample_without_replacement(items: List[int], weights: List[float], k: int, rng: random.Random) -> List[int]:
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


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    records = load_data(args.input)

    none_indices = [i for i, r in enumerate(records) if r.get("label") == LABEL_NONE]
    non_none_indices = [i for i, r in enumerate(records) if r.get("label") != LABEL_NONE]

    n_total = len(records)
    n_none = len(none_indices)
    n_non_none = len(non_none_indices)

    if args.target_none is not None:
        target_none = int(args.target_none)
        if target_none < 0:
            raise ValueError("--target-none must be non-negative")
        if target_none > n_none:
            print(f"[INFO] Requested target-none={target_none} > current NONE={n_none}. No removals on NONEs.")
            target_none = n_none
        target_total = n_non_none + target_none
    else:
        target_total = int(args.target_total)
        if target_total < n_non_none:
            raise ValueError(
                f"--target-total={target_total} is smaller than non-NONE count ({n_non_none}). "
                "Cannot remove non-NONE entries; increase target-total."
            )
        desired_remove = n_total - target_total
        if desired_remove <= 0:
            print(f"[INFO] Dataset already at or below target total ({n_total} <= {target_total}). No removals.")
            desired_remove = 0
        remove_none = min(desired_remove, n_none)
        target_none = n_none - remove_none

    none_groups = group_none_by_ar(records)
    keep_plan, _ = compute_removal_plan(none_groups, target_none, rng)

    keep_none_indices: List[int] = []
    remove_none_indices: List[int] = []

    for ar, idxs in none_groups.items():
        k_keep = int(keep_plan.get(ar, 0))
        if k_keep >= len(idxs):
            chosen_keep = list(idxs)
            chosen_remove: List[int] = []
        else:
            chosen_keep = rng.sample(idxs, k_keep)
            chosen_remove = [i for i in idxs if i not in chosen_keep]
        keep_none_indices.extend(chosen_keep)
        remove_none_indices.extend(chosen_remove)

    keep_set = set(non_none_indices) | set(keep_none_indices)
    out_records = [rec for i, rec in enumerate(records) if i in keep_set]
    removed_records = [rec for i, rec in enumerate(records) if i not in keep_set]

    out_none = sum(1 for r in out_records if r.get("label") == LABEL_NONE)

    print("=== Summary ===")
    print(f"Input:  total={n_total}, NONE={n_none}, non-NONE={n_non_none}")
    print(f"Target: total={target_total}, NONE={target_none}")
    print(f"Output: total={len(out_records)}, NONE={out_none}")
    print(f"Removed NONE: {len(remove_none_indices)}")

    if args.dry_run:
        print("[DRY-RUN] No files written.")
        return

    save_data(args.output, out_records)
    print(f"[OK] Wrote output dataset to: {args.output}")

    if args.removed:
        save_data(args.removed, removed_records)
        print(f"[OK] Wrote removed entries to: {args.removed}")


if __name__ == "__main__":
    main()
