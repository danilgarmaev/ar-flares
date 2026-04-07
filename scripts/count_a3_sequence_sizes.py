from __future__ import annotations

import os
import re
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LABELS = DATA / "C1.0_24hr_224_png_Labels.txt"
SPLITS = {
    "Train": DATA / "wds_out" / "train",
    "Validation": DATA / "wds_out" / "val",
    "Test": DATA / "wds_out" / "test",
}
CADENCES = [60, 120, 180, 240, 300]
SEQ_T = 16
TARGET_NEG_NONE = 100000
NAME_RE = re.compile(r"^(\d+)_")


def load_label_info() -> dict[str, tuple[int, int, str]]:
    out: dict[str, tuple[int, int, str]] = {}
    with LABELS.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fname, flare = line.split(",")
            flare = flare.strip().upper()
            label_c = 0 if flare in {"0", "NONE"} else 1
            label_m = 1 if flare.startswith(("M", "X")) else 0
            if flare in {"0", "NONE"}:
                subtype = "none"
            elif flare.startswith("C"):
                subtype = "c"
            else:
                subtype = "other"
            out[fname] = (label_c, label_m, subtype)
    return out


def load_entries(
    split_dir: Path,
    label_info: dict[str, tuple[int, int, str]],
    *,
    show_progress: bool = False,
) -> list[tuple[int, int, int, str]]:
    entries: list[tuple[int, int, int, str]] = []
    tars = sorted(split_dir.glob("*.tar"))
    for idx, tar_path in enumerate(tars, 1):
        with tarfile.open(tar_path, "r") as tf:
            names = sorted(m.name for m in tf.getmembers() if m.isfile() and m.name.endswith(".png"))
        for name in names:
            base = os.path.basename(name)
            m = NAME_RE.match(base)
            if not m:
                continue
            info = label_info.get(base)
            if info is None:
                continue
            ar = int(m.group(1))
            label_c, label_m, subtype = info
            entries.append((ar, label_c, label_m, subtype))
        if show_progress and (idx % 50 == 0 or idx == len(tars)):
            print(f"loaded {split_dir.name}: {idx}/{len(tars)} tar files", flush=True)
    return entries


def count_split(entries: list[tuple[int, int, int, str]], stride: int, label_mode: str) -> Counter:
    offsets = list(range(-(SEQ_T - 1) * stride, 1, stride))
    counts: Counter = Counter()
    n = len(entries)
    for i in range(0, n, stride):
        ar0, label_c, label_m, subtype = entries[i]
        ok = True
        for off in offsets:
            j = i + off
            if j < 0 or j >= n or entries[j][0] != ar0:
                ok = False
                break
        if not ok:
            continue
        label = label_c if label_mode == "C" else label_m
        counts["raw"] += 1
        if label == 1:
            counts["pos"] += 1
        else:
            counts["neg_total"] += 1
            counts[f"neg_{subtype}"] += 1
    return counts


def summarize_counts(*, show_progress: bool = False) -> list[dict[str, int | str]]:
    label_info = load_label_info()
    all_entries = {}
    for name, path in SPLITS.items():
        all_entries[name] = load_entries(path, label_info, show_progress=show_progress)
        if show_progress:
            print(f"{name}: loaded {len(all_entries[name]):,} frame entries", flush=True)
    rows: list[dict[str, int | str]] = []
    for cadence in CADENCES:
        stride = cadence // 12
        for label in ["C", "M"]:
            tr = count_split(all_entries["Train"], stride, label)
            va = count_split(all_entries["Validation"], stride, label)
            te = count_split(all_entries["Test"], stride, label)
            eff = tr["pos"] + min(tr["neg_none"], TARGET_NEG_NONE) + tr["neg_c"] + tr["neg_other"]
            rows.append(
                {
                    "cadence_min": cadence,
                    "label": label,
                    "train_raw_before_aug": tr["raw"],
                    "train_after_aug": tr["raw"],
                    "train_pos": tr["pos"],
                    "train_neg_total": tr["neg_total"],
                    "train_neg_none": tr["neg_none"],
                    "train_neg_c": tr["neg_c"],
                    "train_neg_other": tr["neg_other"],
                    "train_effective_post_subsample": eff,
                    "val_raw": va["raw"],
                    "val_pos": va["pos"],
                    "val_neg": va["neg_total"],
                    "test_raw": te["raw"],
                    "test_pos": te["pos"],
                    "test_neg": te["neg_total"],
                }
            )
    return rows


def main() -> None:
    rows = summarize_counts(show_progress=True)
    print(
        "\t".join(
            [
                "cadence_min",
                "label",
                "train_raw_before_aug",
                "train_after_aug",
                "train_pos",
                "train_neg_total",
                "train_neg_none",
                "train_neg_c",
                "train_neg_other",
                "train_effective_post_subsample",
                "val_raw",
                "val_pos",
                "val_neg",
                "test_raw",
                "test_pos",
                "test_neg",
            ]
        )
    )
    for row in rows:
        print(
            "\t".join(
                str(row[k])
                for k in [
                    "cadence_min",
                    "label",
                    "train_raw_before_aug",
                    "train_after_aug",
                    "train_pos",
                    "train_neg_total",
                    "train_neg_none",
                    "train_neg_c",
                    "train_neg_other",
                    "train_effective_post_subsample",
                    "val_raw",
                    "val_pos",
                    "val_neg",
                    "test_raw",
                    "test_pos",
                    "test_neg",
                ]
            )
        )


if __name__ == "__main__":
    main()
