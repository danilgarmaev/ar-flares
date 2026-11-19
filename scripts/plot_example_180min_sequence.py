#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WDS = ROOT / "data" / "wds_out" / "train"
FIGURES = ROOT / "figures"

AR = 1339
START = datetime(2011, 11, 4, 6, 36, 0)
CADENCE_MIN = 180
N_FRAMES = 16
SEARCH_SHARDS = [
    WDS / "shard-000015.tar",
    WDS / "shard-000016.tar",
    WDS / "shard-000017.tar",
]


def filename_for(ts: datetime) -> str:
    return f"{AR}_hmi.M_720s.{ts:%Y%m%d_%H%M%S}_TAI.1.magnetogram_224.png"


def load_targets(targets: set[str]) -> dict[str, Image.Image]:
    found: dict[str, Image.Image] = {}
    for tar_path in SEARCH_SHARDS:
        if len(found) == len(targets):
            break
        with tarfile.open(tar_path, "r") as tf:
            members = {Path(m.name).name: m for m in tf.getmembers() if m.isfile() and m.name.endswith(".png")}
            needed = targets.intersection(members).difference(found)
            for name in needed:
                fh = tf.extractfile(members[name])
                if fh is None:
                    continue
                found[name] = Image.open(io.BytesIO(fh.read())).convert("L")
    missing = sorted(targets.difference(found))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} target frames, first missing: {missing[0]}")
    return found


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    times = [START + timedelta(minutes=CADENCE_MIN * i) for i in range(N_FRAMES)]
    names = [filename_for(ts) for ts in times]
    images = load_targets(set(names))

    fig, axes = plt.subplots(4, 4, figsize=(8.6, 8.6))
    for idx, (ax, ts, name) in enumerate(zip(axes.ravel(), times, names)):
        ax.imshow(images[name], cmap="gray", vmin=10, vmax=245)
        ax.set_title(f"{ts:%Y-%m-%d %H:%M} TAI", fontsize=8, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.subplots_adjust(wspace=0.04, hspace=0.20, left=0.04, right=0.96, top=0.96, bottom=0.04)

    out_png = FIGURES / "example_ar1339_180min_sequence.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
