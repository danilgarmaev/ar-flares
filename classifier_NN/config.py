"""Configuration settings for AR-flares classifier training.

Notes on design
---------------
This project historically ran in multiple environments (Compute Canada, local
workstations, hosted notebooks). Hard-coding paths based on `/scratch` being
present is brittle because many systems have `/scratch` even when the dataset
is in the repo checkout.

To keep things reproducible and portable, we resolve paths in this order:
  1) explicit environment variables (recommended)
  2) known common locations if they exist
  3) repo-relative `data/` and `results/`
"""

from __future__ import annotations

import os
import getpass
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _first_existing_dir(*candidates: str | os.PathLike | None) -> str | None:
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        if p.is_dir():
            return str(p)
    return None


def _first_writable_parent(*candidates: str | os.PathLike | None) -> str | None:
    """Return the first candidate whose parent is writable.

    This lets us pick a sensible results directory even when it doesn't exist yet.
    """
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        # If it's a directory-like path (no suffix), check it directly;
        # if it's a file-like path, check its parent directory.
        parent = p if p.suffix == "" else p.parent
        if parent.exists() and os.access(str(parent), os.W_OK):
            return str(p)
    return None

# ===================== CONFIG =====================
# Keep this for metadata/logging only; do not gate path resolution on it.
IS_COMPUTE_CANADA = os.path.exists('/scratch')

# Optional environment overrides (recommended on clusters)
ENV_WDS_BASE = os.environ.get("AR_FLARES_WDS_BASE")
ENV_WDS_FLOW_BASE = os.environ.get("AR_FLARES_WDS_FLOW_BASE")
ENV_RESULTS_BASE = os.environ.get("AR_FLARES_RESULTS_BASE")
ENV_INTENSITY_LABELS_ROOT = os.environ.get("AR_FLARES_INTENSITY_LABELS_ROOT")

# Common candidate locations
_REPO_WDS = str(_PROJECT_ROOT / "data" / "wds_out")
_REPO_WDS_FLOW = str(_PROJECT_ROOT / "data" / "wds_flow")
_REPO_RESULTS = str(_PROJECT_ROOT / "results")

_USER = os.environ.get("USER") or getpass.getuser() or Path.home().name
_SCRATCH_USER = str(Path("/scratch") / _USER)

_SCRATCH_WDS = str(Path(_SCRATCH_USER) / "ar-flares" / "data" / "wds_out")
_SCRATCH_WDS_LEGACY = str(Path(_SCRATCH_USER) / "ar-flares-legacy" / "data" / "wds_out")
_SCRATCH_WDS_FLOW = str(Path(_SCRATCH_USER) / "ar-flares" / "data" / "wds_flow")
_SCRATCH_WDS_FLOW_LEGACY = str(Path(_SCRATCH_USER) / "ar-flares-legacy" / "data" / "wds_flow")
_SCRATCH_RESULTS = str(Path(_SCRATCH_USER) / "ar-flares" / "results")

_TEAMSPACE_WDS = "/teamspace/studios/this_studio/AR-flares/data/wds_out"
_TEAMSPACE_WDS_FLOW = "/teamspace/studios/this_studio/AR-flares/data/wds_flow"
_TEAMSPACE_RESULTS = "/teamspace/studios/this_studio/AR-flares/results"

CFG = {
    # paths
    "wds_base": _first_existing_dir(
        ENV_WDS_BASE,
        _SCRATCH_WDS,
        _SCRATCH_WDS_LEGACY,
        _REPO_WDS,
        _TEAMSPACE_WDS,
    )
    or _REPO_WDS,
    "wds_flow_base": _first_existing_dir(
        ENV_WDS_FLOW_BASE,
        _SCRATCH_WDS_FLOW,
        _SCRATCH_WDS_FLOW_LEGACY,
        _REPO_WDS_FLOW,
        _TEAMSPACE_WDS_FLOW,
    )
    or _REPO_WDS_FLOW,
    "results_base": _first_writable_parent(
        ENV_RESULTS_BASE,
        _SCRATCH_RESULTS,
        _REPO_RESULTS,
        _TEAMSPACE_RESULTS,
    )
    or _REPO_RESULTS,

    "use_flow": False,   # set False to ignore optical flow, True to use it
    "use_diff": False,  # use pixel-intensity difference instead of optical flow
    "use_diff_attention": False, # 🔥 Physics-Informed Attention (Image_t - Image_{t-1})
    "min_flare_class": "C", # relabel threshold, "C" or "M"
    "two_stream": False,
    "use_lora": False,
    "use_seq": True,    # use temporal sequences (3D CNN / Video Transformer)

    # sequence settings (only if use_seq=True)
    "seq_T": 3,                      # number of frames per sequence (e.g. last 3 hours)
    # IMPORTANT: most of the codebase expects `seq_stride` (not `seq_stride_steps`).
    # Keep both for backwards compatibility.
    "seq_stride": 8,                 # spacing between frames (k * 12min)
    "seq_stride_steps": 8,           # deprecated alias of seq_stride
    "seq_offsets": [-16, -8, 0],     # past-only temporal offsets
    "seq_aggregate": "mean",         # "mean" or "attn" for temporal aggregation

    # data loader
    "image_size": 224,           # image resolution (112x112 or 224x224) - convnext_base supports 112 fine-tuning
    # Spatial resolution ablation for single-frame models.
    # If >1: downsample by factor using nearest-neighbor, then upsample back to image_size with bilinear.
    "spatial_downsample_factor": 1,
    "balance_classes": True,     # subsample negatives to match positive count (Train only)
    "balance_mode": "prob",     # 'prob' = per-epoch random negatives, 'fixed' = deterministic subset
    "neg_keep_prob": 0.25,       # probability to keep a negative when balance_mode='prob'

    # regularization
    "drop_rate": 0.2,            # Head dropout rate
    "drop_path_rate": 0.1,       # Stochastic depth rate (for ConvNeXt/ViT)
    "redirect_log": True,       # if True redirect stdout to log.txt; if False keep progress bar in terminal
    "batch_size": 64,
    "num_workers": 2,
    "pin_memory": True,
    "persistent_workers": False,
    "prefetch_factor": 2,

    # augmentation controls (used by datasets.py)
    # Keep defaults identical to historical behavior.
    "vertical_flip_prob": 0.5,

    # model / training
    # backbone options:
    #  - "convnext_base", "resnet18", etc. for single-frame 2D models
    #  - "simple3dcnn" for 3D CNN over sequences
    #  - "video_transformer" for temporal transformer over per-frame features
    "backbone": "simple3dcnn",  # default to 3D CNN for sequence experiments
    "pretrained": True,
    "freeze_backbone": False,   # allow full fine-tuning of convnext_base
    "lr": 1e-4,
    "epochs": 5,
    "seed": 42,

    # early stopping (validation TSS)
    # Set to an int (e.g., 5) to enable early stopping; None disables.
    "early_stopping_patience": None,
    "early_stopping_min_delta": 0.0,

    # loss
    "use_focal": False,
    "focal_gamma": 2.0,

    # evaluation
    "save_pr_curve": True,

    # checkpointing
    # Save last epoch weights (overwritten each epoch)
    "save_last_checkpoint": True,
    # Save resumable checkpoint (model+optimizer+scheduler+scaler+epoch+cfg) as last_full.pt
    "save_last_full_checkpoint": True,
    # Save best-F1 checkpoint (disabled by default to save space)
    "save_best_f1": False,

    # model selection for test evaluation
    # Options: "tss" (default), "f1", "val_loss"
    "model_selection": "tss",

    # validation controls
    # If set, validation only runs for this many batches (useful for very large
    # datasets and expensive sequence models). Leave as None for full validation.
    "val_max_batches": None,

    # multi-scale fusion settings
    "k_crops": 8,
    "crop_size": 64,
    "crop_stride": 32,
    "vit_name": "vit_base_patch16_224",
    "local_dim": 192,
    "cross_heads": 4,

    # flow encoder options
    "flow_encoder": "SmallFlowCNN",  # SmallFlowCNN, MediumFlowCNN, FlowResNetTiny

    # Optional metadata for experiment tracking
    "run_id": None,
    "notes": "",
}

# Automatically build a readable model name for results
CFG["model_name"] = (
    "two_stream_" + CFG["backbone"]
    if CFG["use_flow"] else
    "single_stream_" + CFG["backbone"]
)

# Append traceability flags
if CFG.get("use_diff", False):
    CFG["model_name"] += "_diff"
if CFG.get("min_flare_class", "C") == "M":
    CFG["model_name"] += "_MxOnly"

SPLIT_DIRS = {
    "Train": os.path.join(CFG["wds_base"], "train"),
    "Validation": os.path.join(CFG["wds_base"], "val"),
    "Test": os.path.join(CFG["wds_base"], "test"),
}

SPLIT_FLOW_DIRS = {
    "Train": os.path.join(CFG["wds_flow_base"], "train"),
    "Validation": os.path.join(CFG["wds_flow_base"], "val"),
    "Test": os.path.join(CFG["wds_flow_base"], "test"),
}

# Label map path
intensity_labels_path = (
    ENV_INTENSITY_LABELS_ROOT
    or _first_existing_dir(
        str(Path(_SCRATCH_USER) / "AR-flares" / "data"),
        str(Path(_SCRATCH_USER) / "ar-flares" / "data"),
        str(_PROJECT_ROOT / "data"),
        "/teamspace/studios/this_studio/AR-flares/data",
    )
    or str(_PROJECT_ROOT / "data")
)


def get_default_cfg():
    """Return a fresh copy of the default CFG dict.

    This helps avoid surprising cross-experiment interactions when scripts
    mutate configuration in-place. Callers should take a copy via this
    function and then apply overrides locally instead of modifying CFG
    directly.
    """
    import copy

    return copy.deepcopy(CFG)


def apply_physics_attention_overrides(cfg: dict) -> dict:
    """Return a copy of cfg with physics-informed attention settings applied.

    This keeps all physics-attention specific knobs in one place for
    reproducibility.
    """
    new_cfg = dict(cfg)
    new_cfg.update({
        "use_diff_attention": True,
        "use_diff": True,
        "use_flow": False,
        "two_stream": False,
        "use_seq": False,
    })
    return new_cfg
