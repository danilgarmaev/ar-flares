#!/usr/bin/env python3
"""Check AR overlap across WebDataset splits (train/val/test).

Your WebDataset shards store per-sample JSON sidecars like:
  {"label": 0/1, "reg_label": "C4.7"|"0", "ar": "1064"}

This script scans shards in `data/wds_out/{train,val,test}` (or a user-provided
base dir) and reports whether any AR appears in multiple splits.

This mirrors the idea of `macros/sanity_checks_video_metadata.py` in
`inaf-oact-ai/solar-flare-forecaster`, but adapted to this repo's shard format.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Set


def iter_json_members(tar_path: Path) -> Iterable[dict]:
    with tarfile.open(tar_path, "r") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            if not m.name.endswith(".json"):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            try:
                data = json.loads(f.read().decode("utf-8", errors="ignore"))
            except Exception:
                continue
            if isinstance(data, dict):
                yield data


def collect_ars(split_dir: Path, max_shards: int | None) -> Set[str]:
    ars: Set[str] = set()
    tars = sorted(split_dir.glob("*.tar"))
    if max_shards is not None:
        tars = tars[: int(max_shards)]

    for tp in tars:
        for rec in iter_json_members(tp):
            ar = rec.get("ar")
            if ar is None:
                continue
            ars.add(str(ar))
    return ars


def main() -> None:
    ap = argparse.ArgumentParser(description="Check AR overlap across WDS train/val/test splits.")
    ap.add_argument("--wds-base", type=Path, default=Path("data/wds_out"), help="Base directory containing train/val/test")
    ap.add_argument("--max-shards", type=int, default=None, help="Optional cap to scan only first N shards per split")
    args = ap.parse_args()

    base = args.wds_base
    split_dirs = {
        "train": base / "train",
        "val": base / "val",
        "test": base / "test",
    }

    for name, d in split_dirs.items():
        if not d.is_dir():
            raise SystemExit(f"Missing split dir: {d}")

    ars_by_split: Dict[str, Set[str]] = {}
    for name, d in split_dirs.items():
        ars_by_split[name] = collect_ars(d, max_shards=args.max_shards)

    inter_tv = ars_by_split["train"] & ars_by_split["val"]
    inter_tt = ars_by_split["train"] & ars_by_split["test"]
    inter_vt = ars_by_split["val"] & ars_by_split["test"]

    print("=== AR overlap report ===")
    for sp in ("train", "val", "test"):
        print(f"{sp:>5s}: {len(ars_by_split[sp])} unique ARs")

    any_bad = False
    for label, inter in [("train ∩ val", inter_tv), ("train ∩ test", inter_tt), ("val ∩ test", inter_vt)]:
        if inter:
            any_bad = True
            example = sorted(list(inter))[:30]
            print(f"❌ Overlap in {label}: {len(inter)} ARs (e.g. {example})")
        else:
            print(f"✅ No overlap in {label}")

    if any_bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
