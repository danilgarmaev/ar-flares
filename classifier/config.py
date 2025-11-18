import os

# ===================== ENVIRONMENT =====================
IS_COMPUTE_CANADA = os.path.exists('/scratch')

# ===================== CONFIG =====================
CFG = {
    # Paths
    "wds_base": '/scratch/dgarmaev/AR-flares/wds_out' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/wds_out',
    "wds_flow_base": '/scratch/dgarmaev/AR-flares/wds_flow' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/wds_flow',
    "results_base": '/scratch/dgarmaev/AR-flares/results' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/results',
    "data_base": '/scratch/dgarmaev/AR-flares/data' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/data',

    # Data
    "use_flow": False,
    "use_diff": False,
    "min_flare_class": "C",  # "C" or "M"
    "two_stream": False,
    
    # DataLoader
    "batch_size": 64,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": False,
    "prefetch_factor": 2,

    # Model
    "backbone": "vit_base_patch16_224",
    "pretrained": True,
    "freeze_backbone": True,
    "use_lora": False,
    
    # Training
    "lr": 1e-4,
    "epochs": 5,
    "seed": 42,

    # Loss
    "use_focal": False,
    "focal_gamma": 2.0,

    # Evaluation
    "save_pr_curve": True,
}

# Build readable model name
CFG["model_name"] = (
    "two_stream_" + CFG["backbone"]
    if CFG["two_stream"] else
    "single_stream_" + CFG["backbone"]
)

if CFG.get("use_diff", False):
    CFG["model_name"] += "_diff"
if CFG.get("min_flare_class", "C") == "M":
    CFG["model_name"] += "_MxOnly"

# Split directories
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

# ===================== LABEL MAPPING =====================
def load_flare_label_map(label_file: str, min_class="C"):
    """Load remapped labels for ≥M threshold."""
    flare_map = {}
    min_class = min_class.upper()
    if min_class != "M":
        return None

    with open(label_file, "r") as f:
        for line in f:
            fname, flare = line.strip().split(",")
            flare = flare.strip().upper()
            label = 1 if flare.startswith(("M", "X")) else 0
            flare_map[fname] = label
    return flare_map

LABEL_MAP = None
if CFG["min_flare_class"].upper() == "M":
    LABEL_MAP = load_flare_label_map(
        os.path.join(CFG["data_base"], "C1.0_24hr_224_png_Labels.txt"),
        min_class="M"
    )