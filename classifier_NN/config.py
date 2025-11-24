"""Configuration settings for AR-flares classifier training."""
import os

# ===================== CONFIG =====================
IS_COMPUTE_CANADA = os.path.exists('/scratch')

CFG = {
    # paths
    "wds_base": '/scratch/dgarmaev/AR-flares/wds_out' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/data/wds_out',
    "wds_flow_base": '/scratch/dgarmaev/AR-flares/wds_flow' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/data/wds_flow',
    "results_base": '/scratch/dgarmaev/AR-flares/results' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/results',

    "use_flow": False,   # set False to ignore optical flow, True to use it
    "use_diff": False,  # use pixel-intensity difference instead of optical flow
    "use_diff_attention": False, # 🔥 Physics-Informed Attention (Image_t - Image_{t-1})
    "min_flare_class": "C", # relabel threshold, "C" or "M"
    "two_stream": False,
    "use_lora": False,
    "use_seq": False,    # use temporal sequences

    # sequence settings (only if use_seq=True)
    "seq_T": 3,                      # number of frames per sequence
    "seq_stride_steps": 8,           # spacing between frames (8 * 12min = 96min)
    "seq_offsets": [-16, -8, 0],     # past-only temporal offsets
    "seq_aggregate": "mean",         # "mean" or "attn" for temporal aggregation

    # data loader
    "image_size": 112,           # image resolution (112x112 or 224x224) - convnext_base supports 112 fine-tuning
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

    # model / training
    "backbone": "convnext_base",  # switched to convnext_base per user request
    "pretrained": True,
    "freeze_backbone": False,   # allow full fine-tuning of convnext_base
    "lr": 1e-4,
    "epochs": 5,

    # loss
    "use_focal": False,
    "focal_gamma": 2.0,

    # evaluation
    "save_pr_curve": True,

    # multi-scale fusion settings
    "k_crops": 8,
    "crop_size": 64,
    "crop_stride": 32,
    "vit_name": "vit_base_patch16_224",
    "local_dim": 192,
    "cross_heads": 4,

    # flow encoder options
    "flow_encoder": "SmallFlowCNN",  # SmallFlowCNN, MediumFlowCNN, FlowResNetTiny
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
    "/scratch/dgarmaev/AR-flares/data"
    if IS_COMPUTE_CANADA
    else "/teamspace/studios/this_studio/AR-flares/data"
)
