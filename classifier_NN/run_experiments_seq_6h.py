import argparse

from .config import get_default_cfg
from .train import main


def _build_cfg(name: str, overrides: dict, common_overrides: dict | None = None) -> dict:
    cfg = get_default_cfg()
    if common_overrides:
        cfg.update(common_overrides)
    cfg.update(overrides)
    cfg["model_name"] = name
    cfg["run_id"] = name
    return cfg


COMMON_6H_OVERRIDES = {
    "use_seq": True,
    "use_aug": False,
    "use_flow": False,
    "two_stream": False,
    "use_diff": False,
    "use_diff_attention": False,
    "image_size": 224,
    "backbone": "r3d_18",
    "pretrained": True,
    "pretrained_3d": False,
    "freeze_backbone": False,
    "balance_classes": False,
    "mixup": 0.0,
    "cutmix": 0.0,
    "use_focal": False,
    "loss_type": "ce",
    "optimizer": "adamw",
    "lr": 1e-4,
    "epochs": 10,
    "batch_size": 4,
    "num_workers": 2,
    "persistent_workers": False,
    "pin_memory": True,
    "redirect_log": True,
    "val_max_batches": None,
}


EXPERIMENTS = [
    {
        "name": "seq6h_r3d18_N16_k2",
        "seq_offsets": list(range(-30, 1, 2)),  # [-30,-28,...,0] (16 frames)
        "seq_stride": 2,
    },
    {
        "name": "seq6h_r3d18_N8_k4",
        "seq_offsets": list(range(-28, 1, 4)),  # [-28,-24,...,0] (8 frames)
        "seq_stride": 4,
    },
    {
        "name": "seq6h_r3d18_N4_k10",
        "seq_offsets": [-30, -20, -10, 0],
        "seq_stride": 10,
    },
]


def run_all(*, smoke: bool, epochs: int | None):
    common = dict(COMMON_6H_OVERRIDES)
    if epochs is not None:
        common["epochs"] = int(epochs)

    if smoke:
        common.update(
            {
                "epochs": 1,
                "batch_size": 2,
                "num_workers": 0,
                "persistent_workers": False,
                "redirect_log": False,
                "val_max_batches": 5,
                "max_train_shards": 2,
                "max_val_shards": 1,
                "max_test_shards": 1,
            }
        )

    for exp in EXPERIMENTS:
        offsets = exp["seq_offsets"]
        overrides = {
            "seq_offsets": offsets,
            "seq_T": len(offsets),
            "seq_stride": exp["seq_stride"],
            "notes": (
                "~6h temporal coverage (12-min cadence). "
                f"N={len(offsets)} frames, k={exp['seq_stride']} steps (k*12min). "
                "Strict skip of insufficient-history sequences; no augmentation."
            ),
        }

        name = exp["name"]
        print("\n" + "=" * 80)
        print(f"Launching: {name}")
        print(f"  seq_T={len(offsets)}")
        print(f"  k(seq_stride)={exp['seq_stride']}")
        print(f"  offsets={offsets}")
        print("=" * 80 + "\n")

        cfg = _build_cfg(name, overrides, common)
        main(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run 3 fast ~6h sequence experiments (r3d_18).")
    ap.add_argument("--smoke", action="store_true", help="Run tiny shard/1-epoch smoke test")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs for all runs")
    args = ap.parse_args()

    run_all(smoke=bool(args.smoke), epochs=args.epochs)
