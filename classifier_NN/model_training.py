import os, io, glob, json, tarfile, random
from typing import Iterable, List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

import json
import timm
from timm.data import resolve_data_config, create_transform
from timm.optim import create_optimizer_v2


import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from peft import LoraConfig, get_peft_model


# ===================== CONFIG =====================
IS_COMPUTE_CANADA = os.path.exists('/scratch')
CFG = {
    # paths
    "wds_base": '/scratch/dgarmaev/AR-flares/wds_out' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/data/wds_out',
    "wds_flow_base": '/scratch/dgarmaev/AR-flares/wds_flow' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/data/wds_flow',
    "results_base": '/scratch/dgarmaev/AR-flares/results' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/results',

    "use_flow": False,   # set False to ignore optical flow, True to use it
    "use_diff": False,  # use pixel-intensity difference instead of optical flow
    "min_flare_class": "C", # realbel threshold, "C" or "M"
    "two_stream": False,
    "use_lora": False,

    # data loader
    "batch_size": 64,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": False,
    "prefetch_factor": 2,

    # model / training
    "backbone": "vit_base_patch16_224",
    "pretrained": True,
    "freeze_backbone": True,
    "lr": 1e-4,
    "epochs": 5,

    # loss
    "use_focal": False,
    "focal_gamma": 2.0,

    "save_pr_curve": True,
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

# ===================================================

# --- flare label remapping ---
def load_flare_label_map(label_file: str, min_class="C"):
    """
    Load remapped labels only if we need to tighten the threshold (≥M).
    For min_class='C', we skip remapping entirely since JSON already has binary labels.
    """
    flare_map = {}
    min_class = min_class.upper()
    if min_class != "M":
        return None  # no remapping needed

    with open(label_file, "r") as f:
        for line in f:
            fname, flare = line.strip().split(",")
            flare = flare.strip().upper()
            # Only >=M flares are positive
            label = 1 if flare.startswith(("M", "X")) else 0
            flare_map[fname] = label
    return flare_map


intensity_labels_path = (
    "/scratch/dgarmaev/AR-flares/data"
    if IS_COMPUTE_CANADA
    else "/teamspace/studios/this_studio/AR-flares/data"
)
LABEL_MAP = None
if CFG["min_flare_class"].upper() == "M":
    LABEL_MAP = load_flare_label_map(
        os.path.join(intensity_labels_path, "C1.0_24hr_224_png_Labels.txt"),
        min_class="M"
    )


# ---------------- WebDataset Iterable ----------------
from contextlib import nullcontext  # make sure this import is at the top

class TarShardDataset(IterableDataset):
    def __init__(self, shard_paths: List[str], flow_paths: List[str] = None,
                 use_flow: bool = False,
                 shuffle_shards=True, shuffle_samples=False,
                 shuffle_buffer_size=2048, seed=42):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.flow_paths = list(flow_paths) if flow_paths else None
        self.use_flow = use_flow
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

        if self.use_flow:
            assert self.flow_paths is not None, "Flow shards required when use_flow=True"
            assert len(self.shard_paths) == len(self.flow_paths), "Image vs flow shard count mismatch"

    # def _sample_iter_from_tar(self, img_tar: str, flow_tar: str = None):
    #     with tarfile.open(img_tar, "r") as tf_img, \
    #          (tarfile.open(flow_tar, "r") if (self.use_flow and flow_tar) else nullcontext()) as tf_flow:

    #         by_key: Dict[str, Dict[str, tarfile.TarInfo]] = {}
    #         for m in tf_img.getmembers():
    #             if not m.isfile(): continue
    #             if m.name.endswith(".png"):
    #                 by_key.setdefault(m.name[:-4], {})["png"] = m
    #             elif m.name.endswith(".json"):
    #                 by_key.setdefault(m.name[:-5], {})["json"] = m

    #         if self.use_flow:
    #             for m in tf_flow.getmembers():
    #                 if m.isfile() and m.name.endswith(".flow.npz"):
    #                     by_key.setdefault(m.name[:-9], {})["flow"] = m

    #         keys = list(by_key.keys())
    #         if self.shuffle_samples:
    #             rng = random.Random(self.seed ^ hash(img_tar))
    #             rng.shuffle(keys)

    #         for k in keys:
    #             parts = by_key[k]
    #             if "png" not in parts or "json" not in parts:
    #                 continue
    #             if self.use_flow and "flow" not in parts:
    #                 continue

    #             meta = json.loads(tf_img.extractfile(parts["json"]).read().decode("utf-8"))

    #             # -- label -- 
    #             if LABEL_MAP is not None:
    #                 # use remapped label for ≥M threshold
    #                 fname = os.path.basename(parts["png"].name)
    #                 label = LABEL_MAP.get(fname, 0)
    #             else:
    #                 # use default binary label from meta (already C-thresholded)
    #                 label = int(meta["label"])

    #             # image → grayscale
    #             png_bytes = tf_img.extractfile(parts["png"]).read()
    #             img = Image.open(io.BytesIO(png_bytes)).convert("L").resize((224,224))
    #             img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)

    #             # --- difference option ---
    #             diff_t = None
    #             if CFG.get("use_diff", False):
    #                 prev_k = keys[keys.index(k)-1] if keys.index(k) > 0 else None
    #                 if prev_k and "png" in by_key[prev_k]:
    #                     prev_img = Image.open(io.BytesIO(tf_img.extractfile(by_key[prev_k]["png"]).read())).convert("L").resize((224,224))
    #                     prev_t = torch.from_numpy(np.array(prev_img, np.float32)/255.).unsqueeze(0)
    #                     diff_t = (img_t - prev_t).clamp(-1,1)
    #                     diff_t = (diff_t - diff_t.min()) / (diff_t.max() - diff_t.min() + 1e-8)

    #             # --- flow option ---
    #             flow_t = None
    #             if CFG.get("use_flow", False):
    #                 flow_bytes = tf_flow.extractfile(parts["flow"]).read()
    #                 with np.load(io.BytesIO(flow_bytes)) as f:
    #                     u = f["u8"].astype(np.float32) * float(f["su"])
    #                     v = f["v8"].astype(np.float32) * float(f["sv"])
    #                 flow_t = torch.from_numpy(np.stack([u,v],axis=0))

    #             # --- combine depending on setup ---
    #             if CFG["use_flow"] and not CFG["use_diff"]:
    #                 if CFG.get("two_stream", False):
    #                     # case 4: output tuple for separate encoders
    #                     yield (img_t.repeat(3,1,1), flow_t), label, meta
    #                 else:
    #                     # case 2: same-stream, combine 1 img + 2 flow = 3 channels
    #                     yield torch.cat([img_t, flow_t], dim=0), label, meta
    #             elif CFG["use_diff"]:
    #                 # case 3: add diff channel to image
    #                 x = torch.cat([img_t, diff_t], dim=0) if diff_t is not None else img_t.repeat(2,1,1)
    #                 yield x, label, meta
    #             else:
    #                 if CFG["backbone"].lower().startswith(("can", "ms_fusion", "multiscale")):
    #                     yield img_t, label, meta     # 1×224×224
    #                 else:# case 1: plain image replicated to 3 channels
    #                     yield img_t.repeat(3,1,1), label, meta

    def _sample_iter_from_tar(self, img_tar: str, flow_tar: str = None):
        with tarfile.open(img_tar, "r") as tf_img, \
            (tarfile.open(flow_tar, "r") if (self.use_flow and flow_tar) else nullcontext()) as tf_flow:

            # Index files by sorted key
            entries = []
            for m in tf_img.getmembers():
                if m.isfile() and m.name.endswith(".png"):
                    key = m.name[:-4]
                    json_name = key + ".json"
                    try:
                        jm = tf_img.getmember(json_name)
                    except KeyError:
                        continue
                    entries.append((key, m, jm))
            # sort for temporal order (assumes shard naming is chronological)
            entries.sort(key=lambda x: x[0])

            # Build quick map key->index
            key2idx = {k:i for i,(k,_,_) in enumerate(entries)}

            # helper to load image -> (1,224,224)
            def load_img(member_png):
                png_bytes = tf_img.extractfile(member_png).read()
                img = Image.open(io.BytesIO(png_bytes)).convert("L").resize((224,224))
                return torch.from_numpy(np.array(img, dtype=np.float32)/255.0).unsqueeze(0)

            # optional: parse AR id from JSON to ensure same region
            def get_meta(jm):
                return json.loads(tf_img.extractfile(jm).read().decode("utf-8"))

            T = CFG.get("seq_T", 3)
            offsets = CFG.get("seq_offsets", [-16, -8, 0])  # in multiples of 12-min frames

            for i in range(len(entries)):
                key0, png0, jm0 = entries[i]
                meta0 = get_meta(jm0)

                # figure out absolute indices for offsets relative to 'i'
                idxes = []
                ok = True
                for off in offsets:
                    j = i + off
                    if j < 0 or j >= len(entries):
                        ok = False; break
                    # enforce same AR/region if meta has an identifier
                    keyj, pngj, jmj = entries[j]
                    metaj = get_meta(jmj)
                    if ("ar" in meta0 and "ar" in metaj) and (meta0["ar"] != metaj["ar"]):
                        ok = False; break
                    idxes.append(j)
                if not ok: 
                    continue

                # load frames
                frames = []
                for j in idxes:
                    _, pngj, _ = entries[j]
                    frames.append(load_img(pngj))   # (1,H,W)
                x_seq = torch.stack(frames, dim=0)  # (T,1,224,224)

                # label from reference frame (last index)
                ref_idx = idxes[-1]  # reference = last (time 0)
                _, _, refjm = entries[ref_idx]
                meta = get_meta(refjm)
                if LABEL_MAP is not None:
                    fname = os.path.basename(entries[ref_idx][1].name)
                    label = LABEL_MAP.get(fname, 0)
                else:
                    label = int(meta["label"])

                # flow / diff disabled when using sequences (keep simple for baseline)
                if CFG.get("use_flow", False) or CFG.get("use_diff", False):
                    # you can extend later; for now keep flow/diff off with sequences
                    pass

                yield x_seq, label, meta


    def _roundrobin(self, iters: List[Iterable]):
        active = [iter(x) for x in iters]
        while active:
            new_active = []
            for it in active:
                try:
                    yield next(it)
                    new_active.append(it)
                except StopIteration:
                    pass
            active = new_active

    def _buffered_shuffle(self, iterable: Iterable):
        rng = random.Random(self.seed)
        buf = []
        for x in iterable:
            buf.append(x)
            if len(buf) >= self.shuffle_buffer_size:
                rng.shuffle(buf)
                while buf:
                    yield buf.pop()
        rng.shuffle(buf)
        while buf:
            yield buf.pop()

    def _compute_diff(self, img_t: torch.Tensor, prev_img_t: torch.Tensor):
        """Compute pixel-wise normalized difference between two frames (same AR region)."""
        diff = img_t - prev_img_t
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        return diff

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            img_shards = self.shard_paths
            flow_shards = self.flow_paths if self.use_flow else [None] * len(img_shards)
        else:
            wid = worker_info.id
            num_workers = worker_info.num_workers
            img_shards = self.shard_paths[wid::num_workers]
            flow_shards = self.flow_paths[wid::num_workers] if self.use_flow else [None] * len(img_shards)

        img_shards = list(img_shards)
        flow_shards = list(flow_shards)
        if self.shuffle_shards:
            rng = random.Random(self.seed + (worker_info.id if worker_info else 0))
            paired = list(zip(img_shards, flow_shards))
            rng.shuffle(paired)
            img_shards, flow_shards = zip(*paired)

        gens = [self._sample_iter_from_tar(img, flow) for img, flow in zip(img_shards, flow_shards)]
        stream = self._roundrobin(gens)

        if self.shuffle_samples:
            stream = self._buffered_shuffle(stream)

        for sample in stream:
            yield sample


def list_shards(split_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(split_dir, "*.tar")))

def make_dataset(split_name: str, shuffle_shards=True, shuffle_samples=False, seed=42):
    img_shards = list_shards(SPLIT_DIRS[split_name])
    if CFG["use_flow"]:
        flow_shards = list_shards(SPLIT_FLOW_DIRS[split_name])
    else:
        flow_shards = None
    if not img_shards:
        raise FileNotFoundError(f"No image shards for {split_name}")
    return TarShardDataset(img_shards, flow_shards,
                           use_flow=CFG["use_flow"],
                           shuffle_shards=shuffle_shards,
                           shuffle_samples=shuffle_samples,
                           seed=seed)


# --------- shard scanners (fast, no images) ---------
def count_raw_labels_all_shards(split_dir: str) -> Dict[int, int]:
    ctr = Counter()
    for tar_path in glob.glob(os.path.join(split_dir, "*.tar")):
        with tarfile.open(tar_path, "r") as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".json"):
                    try:
                        meta = json.loads(tf.extractfile(m).read().decode("utf-8"))
                        ctr.update([int(meta["label"])])
                    except Exception:
                        continue
    return dict(ctr)

def count_labels_all_shards(split_dir: str) -> Dict[int, int]:
    """Count samples per class using LABEL_MAP if available, else use meta['label']."""
    ctr = Counter()
    for tar_path in glob.glob(os.path.join(split_dir, "*.tar")):
        with tarfile.open(tar_path, "r") as tf:
            for m in tf.getmembers():
                if not (m.isfile() and m.name.endswith(".png")):
                    continue
                fname = os.path.basename(m.name)
                if LABEL_MAP is not None:
                    label = LABEL_MAP.get(fname, 0)
                else:
                    # load label from corresponding JSON (already binary at C1.0)
                    json_name = m.name.replace(".png", ".json")
                    json_member = tf.getmember(json_name)
                    meta = json.loads(tf.extractfile(json_member).read().decode("utf-8"))
                    label = int(meta["label"])
                ctr.update([label])
    return dict(ctr)



def count_samples_all_shards(split_dir: str) -> int:
    total = 0
    for tar_path in glob.glob(os.path.join(split_dir, "*.tar")):
        with tarfile.open(tar_path, "r") as tf:
            total += sum(1 for m in tf.getmembers() if m.isfile() and m.name.endswith(".json"))
    return total

# ---------------- losses ----------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight)
    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        return -((1-pt)**self.gamma * logpt)

# ---- LoRA integration ----

def apply_lora(model, r=8, alpha=16, dropout=0.1):
    config = LoraConfig(
        task_type="FEATURE_EXTRACTION",
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["qkv", "fc1", "fc2"],
        bias="none"
    )
    return get_peft_model(model, config)

import torch.nn as nn

class PeftTimmWrapper(nn.Module):
    """Wrapper to make timm models compatible with PEFT/LoRA"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        # PEFT passes input_ids, but timm models expect raw tensors
        x = input_ids if input_ids is not None else pixel_values
        return self.model(x)

from peft import LoraConfig, get_peft_model, TaskType

def apply_lora_to_timm(model, r=8, alpha=16, dropout=0.1):
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["qkv", "fc1", "fc2"],  # ViT layers
        bias="none"
    )
    model = PeftTimmWrapper(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model



# ---------------- model builder ----------------
def _enable_head_grads(model: nn.Module):
    """Turn grads back on for the classifier head, regardless of attribute name."""
    enabled = 0
    # Prefer timm's get_classifier() if available
    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
        if isinstance(head, nn.Module):
            for p in head.parameters(): p.requires_grad = True; enabled += p.numel()

    # Also try common names
    for name in ["head", "fc", "classifier", "cls", "last_linear"]:
        m = getattr(model, name, None)
        if isinstance(m, nn.Module):
            for p in m.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    enabled += p.numel()
    if enabled == 0:
        # Fallback: enable the last Linear modules we can find
        last_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            for p in last_linear.parameters(): p.requires_grad = True

# Two-stream model 

# ---------------- Custom CAN backbone ----------------
class CANSmall(nn.Module):
    """
    Convolutional Attention Network for magnetograms (domain-specific CNN+attention).
    Designed for single-channel grayscale input (1×224×224).
    """
    def __init__(self, in_chans=1, num_classes=2, embed_dim=128, num_heads=4, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
        )
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):  # (B,1,H,W)
        h = self.conv(x)                      # (B,embed_dim,H',W')
        h = h.flatten(2).transpose(1, 2)      # (B,N,embed_dim)
        h, _ = self.attn(h, h, h)
        h = h.mean(1)                         # global attention pooling
        return self.head(h)


# ---------------- flow stream CNN ----------------
class SmallFlowCNN(nn.Module):
    def __init__(self, out_dim=128, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),  # 🔹 spatial dropout helps regularize
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)  # 🔹 standard dropout before fusion
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

# ---------------- alternative flow encoders ----------------
class MediumFlowCNN(nn.Module):
    def __init__(self, out_dim=256, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 5, 2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

class FlowResNetTiny(nn.Module):
    """A mini ResNet block for optical flow."""
    def __init__(self, out_dim=256):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=False, in_chans=2, num_classes=out_dim)
    def forward(self, x):  # (B,2,H,W)
        return self.backbone(x)


import torch.nn.functional as F

# ---- Gradient-based crop selector (Sobel) ----
def sobel_grad_mag(x1):  # x1: (B,1,H,W), returns (B,1,H,W)
    # 3x3 Sobel kernels
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x1.dtype, device=x1.device).view(1,1,3,3)/4.0
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=x1.dtype, device=x1.device).view(1,1,3,3)/4.0
    gx = F.conv2d(x1, kx, padding=1)
    gy = F.conv2d(x1, ky, padding=1)
    g = torch.sqrt(gx*gx + gy*gy + 1e-8)
    return g

def topk_crops_from_grad(x1, k=8, crop_size=64, stride=32):
    """
    x1: (B,1,224,224), returns list of length B with tensors (k,1,crop_size,crop_size).
    Picks top-K windows by mean gradient magnitude.
    """
    B,_,H,W = x1.shape
    g = sobel_grad_mag(x1)  # (B,1,H,W)

    # Unfold gradient to windows and score by mean
    unfold = F.unfold(g, kernel_size=crop_size, stride=stride)  # (B, crop_size*crop_size, L)
    scores = unfold.mean(dim=1)                                 # (B, L)

    # Get top-K indices per batch
    k = min(k, scores.shape[1])
    topk_vals, topk_idx = scores.topk(k, dim=1)                 # (B,k)

    # Unfold original image and gather crops
    x_unf = F.unfold(x1, kernel_size=crop_size, stride=stride)  # (B, crop_size*crop_size, L)
    crops_list = []
    for b in range(B):
        cols = x_unf[b,:,topk_idx[b]]                           # (crop_area, k)
        crops = cols.t().contiguous().view(k, 1, crop_size, crop_size)
        crops_list.append(crops)
    return crops_list  # length B

# ---- Local CNN for 64x64 crops -> token ----
class LocalCropCNN(nn.Module):
    def __init__(self, in_ch=1, embed_dim=192, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),   # 32x32
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),      # 16x16
            nn.Conv2d(64, 128,3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),     # 8x8
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, crops_b):  # (k,1,64,64) -> (k, embed_dim)
        h = self.net(crops_b)
        z = self.proj(h)
        return z

# ---- Cross-Attention block: query=global CLS, keys/values=local tokens ----
class CrossAttn(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=4, dropout=0.0):
        super().__init__()
        self.q = nn.Linear(dim_q, dim_q)
        self.k = nn.Linear(dim_kv, dim_q)
        self.v = nn.Linear(dim_kv, dim_q)
        self.attn = nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.proj = nn.Linear(dim_q, dim_q)

    def forward(self, q_tok, kv_tok):
        """
        q_tok: (B,1,Dq)  [global CLS token]
        kv_tok: (B,K,Dv) [local crop tokens]
        returns: (B,1,Dq)
        """
        Q = self.q(q_tok)
        K = self.k(kv_tok)
        V = self.v(kv_tok)
        out, _ = self.attn(Q, K, V)   # (B,1,Dq)
        return self.proj(out)

# ---- Multi-Scale Fusion Model ----
class MultiScaleFusionModel(nn.Module):
    """
    Global branch: ViT-Base over 224x224 (grayscale -> repeated to 3ch internally).
    Local branch: top-K high-gradient 64x64 crops -> LocalCropCNN -> tokens.
    Fusion: Cross-attention (CLS queries locals), then MLP head.
    """
    def __init__(self, num_classes=2, k_crops=8, crop_size=64, stride=32,
                 vit_name="vit_base_patch16_224", local_dim=192, num_heads=4,
                 pretrained=True, freeze_backbone=False):
        super().__init__()
        self.k_crops = k_crops
        self.crop_size = crop_size
        self.stride = stride

        # Global ViT encoder without classifier (features only)
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        if freeze_backbone:
            for p in self.vit.parameters(): p.requires_grad = False
        self.global_dim = self.vit.num_features  # 768 for ViT-B/16

        # Local crop encoder
        self.local = LocalCropCNN(in_ch=1, embed_dim=local_dim)

        # Cross-attention: map local tokens to global dim inside attention block
        self.cross = CrossAttn(dim_q=self.global_dim, dim_kv=local_dim, num_heads=num_heads, dropout=0.1)

        # Classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(self.global_dim),
            nn.Linear(self.global_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):  # x: (B,1,224,224) grayscale
        B = x.size(0)

        # ---- Global branch (repeat to 3ch for ViT) ----
        x3 = x.repeat(1,3,1,1)                            # (B,3,224,224)
        g = self.vit(x3)                                  # (B, Dg) pooled features
        g_cls = g.unsqueeze(1)                            # (B,1,Dg) as a "CLS" token

        # ---- Local crops ----
        crops_list = topk_crops_from_grad(x, k=self.k_crops, crop_size=self.crop_size, stride=self.stride)
        # Pack variable list into a single tensor of tokens
        local_tokens = []
        for b in range(B):
            tok_b = self.local(crops_list[b])             # (K, Dl)
            local_tokens.append(tok_b)
        # Pad to fixed K if needed (rare when k > available windows)
        K = max(t.shape[0] for t in local_tokens)
        Dl = local_tokens[0].shape[1]
        kv = x.new_zeros((B, K, Dl), dtype=local_tokens[0].dtype)  # (B,K,Dl)
        for b in range(B):
            kb = local_tokens[b].shape[0]
            kv[b,:kb,:] = local_tokens[b]

        # ---- Cross-attention: CLS queries locals ----
        fused = self.cross(g_cls, kv)                     # (B,1,Dg)
        fused = fused.squeeze(1)                          # (B,Dg)

        # ---- Classify ----
        logits = self.head(fused)
        return logits




# ---------------- two-stream wrapper ----------------
class TwoStreamModel(nn.Module):
    def __init__(self, img_backbone="deit_tiny_patch16_224", flow_encoder="SmallFlowCNN",
                 num_classes=2, pretrained=True, freeze_backbone=True, flow_dim=128):
        super().__init__()
        # 1 Image stream
        self.img_model = timm.create_model(img_backbone, pretrained=pretrained, num_classes=0)
        if freeze_backbone:
            for p in self.img_model.parameters():
                p.requires_grad = False
        img_feat_dim = self.img_model.num_features

        # 2 Flow stream
        if flow_encoder == "SmallFlowCNN":
            self.flow_model = SmallFlowCNN(out_dim=flow_dim)
        elif flow_encoder == "MediumFlowCNN":
            self.flow_model = MediumFlowCNN(out_dim=flow_dim)
        elif flow_encoder == "FlowResNetTiny":
            self.flow_model = FlowResNetTiny(out_dim=flow_dim)
        else:
            raise ValueError(f"Unknown flow encoder '{flow_encoder}'")

        # 3 Fusion head
        self.head = nn.Sequential(
            nn.Linear(img_feat_dim + flow_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Handle two-stream input (tuple) or concatenated tensor
        if isinstance(x, (list, tuple)):
            img, flow = x  # unpack the two inputs
        else:
            # fallback: derive img and flow from single tensor
            img = x[:, :1].repeat(1, 3, 1, 1)
            flow = x[:, 1:, :, :]
        
        img_emb = self.img_model(img)
        flow_emb = self.flow_model(flow)
        z = torch.cat([img_emb, flow_emb], dim=1)
        return self.head(z)

# ---------------- sequence wrapper ----------------
class TemporalWrapper(nn.Module):
    """
    Shared 2D backbone across T frames. Input: (B,T,1,224,224).
    ViT/ConvNeXt expects 3ch so we repeat channel.
    Aggregate temporally via mean or attention, then classify.
    """
    def __init__(self, backbone_name="vit_base_patch16_224", num_classes=2,
                 pretrained=True, freeze_backbone=False, aggregate="mean"):
        super().__init__()
        self.aggregate = aggregate
        # feature extractor
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
        self.feat_dim = self.backbone.num_features

        if aggregate == "attn":
            self.temporal_attn = nn.MultiheadAttention(self.feat_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(self.feat_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):  # x: (B,T,1,H,W)
        B,T,_,H,W = x.shape
        x = x.reshape(B*T, 1, H, W).repeat(1,3,1,1)  # -> (B*T,3,H,W)
        feats = self.backbone(x)                     # (B*T, D)
        feats = feats.view(B, T, self.feat_dim)      # (B,T,D)

        if self.aggregate == "mean":
            z = feats.mean(dim=1)                    # (B,D)
        else:
            # attention over time (CLS-free): self-attend and pool
            z,_ = self.temporal_attn(feats, feats, feats)  # (B,T,D)
            z = self.norm(z).mean(dim=1)

        return self.head(z)


# ---------------- model builder (two-stream only) ----------------
def build_model(num_classes=2):
    if CFG.get("use_flow") and CFG.get("two_stream", False):
        # case 4️⃣: two-stream model
        model = TwoStreamModel(
            img_backbone=CFG["backbone"],
            flow_encoder=CFG.get("flow_encoder", "SmallFlowCNN"),
            num_classes=num_classes,
            pretrained=CFG["pretrained"],
            freeze_backbone=CFG["freeze_backbone"]
        )
        if CFG.get("use_lora", False):
            print("Using LoRA fine-tuning")
            model = apply_lora_to_timm(model)

        print("Built Two-Stream model.")
        return model

    if CFG.get("use_seq"): 
        model = TemporalWrapper(
            backbone_name=CFG["backbone"],
            num_classes=num_classes,
            pretrained=CFG["pretrained"],
            freeze_backbone=CFG["freeze_backbone"],
            aggregate=CFG.get("seq_aggregate", "mean"),
        )
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Built TemporalWrapper over {CFG['backbone']} | trainable={n_train:,}")
        return model

    # NEW: Multi-Scale Fusion backbone
    if CFG["backbone"].lower() in ["ms_fusion", "multiscale_fusion"]:
        model = MultiScaleFusionModel(
            num_classes=num_classes,
            k_crops=CFG.get("k_crops", 8),
            crop_size=CFG.get("crop_size", 64),
            stride=CFG.get("crop_stride", 32),
            vit_name=CFG.get("vit_name", "vit_base_patch16_224"),
            local_dim=CFG.get("local_dim", 192),
            num_heads=CFG.get("cross_heads", 4),
            pretrained=CFG.get("pretrained", True),
            freeze_backbone=CFG.get("freeze_backbone", False),
        )
        print("Built MultiScaleFusionModel.")
        return model

    if CFG["backbone"].lower() in ["can_small", "can"]:
        model = CANSmall(in_chans=1, num_classes=num_classes)
        print("Built CANSmall (Convolutional Attention Network).")
        return model

    # otherwise: single-stream convnext/vit/etc.
    in_chans = 3 if CFG.get("use_flow") else (2 if CFG.get("use_diff") else 3)
    model = timm.create_model(
        CFG["backbone"],
        pretrained=CFG["pretrained"],
        num_classes=num_classes,
        in_chans=in_chans
    )

    # reinitialize the classification head for safety
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        nn.init.xavier_uniform_(model.head.weight)
        model.head.bias.data.zero_()
    elif hasattr(model, "head") and isinstance(model.head, nn.Sequential):
        for m in model.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()



    if CFG["freeze_backbone"]:
        for p in model.parameters(): p.requires_grad = False
        _enable_head_grads(model)

    if CFG.get("use_lora", False):
        print("Using LoRA fine-tuning")
        model = apply_lora_to_timm(model)


    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built single-stream model with in_chans={in_chans}, trainable={n_train:,}")
    return model



# # Single-stream model 
# def build_model(backbone: str, num_classes=2, pretrained=True, freeze_backbone=True):
#     """
#     Robust head reset across timm backbones; returns (model, eval_transform).
#     """
#     # Create model WITHOUT forcing num_classes here (so we can reset cleanly)
#     model = timm.create_model(backbone, pretrained=pretrained)

#     # Replace classifier using timm API if available
#     if hasattr(model, "reset_classifier"):
#         # most timm models support this
#         model.reset_classifier(num_classes=num_classes)
#     else:
#         # manual heads for rare cases
#         replaced = False
#         if hasattr(model, "head") and isinstance(model.head, nn.Linear):
#             in_features = model.head.in_features
#             model.head = nn.Linear(in_features, num_classes); replaced = True
#         elif hasattr(model, "classifier"):
#             if isinstance(model.classifier, nn.Sequential):
#                 # find last Linear
#                 li = None
#                 for i in reversed(range(len(model.classifier))):
#                     if isinstance(model.classifier[i], nn.Linear):
#                         li = i; break
#                 if li is None: raise RuntimeError("Couldn't locate classifier Linear layer.")
#                 in_features = model.classifier[li].in_features
#                 model.classifier[li] = nn.Linear(in_features, num_classes); replaced = True
#             elif isinstance(model.classifier, nn.Linear):
#                 in_features = model.classifier.in_features
#                 model.classifier = nn.Linear(in_features, num_classes); replaced = True
#         elif hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
#             in_features = model.fc.in_features
#             model.fc = nn.Linear(in_features, num_classes); replaced = True
#         if not replaced:
#             raise RuntimeError(f"Don't know how to replace head for backbone='{backbone}'")

#     # Build transform for inference/eval (same for train unless you add aug)
#     data_cfg = resolve_data_config({}, model=model)
#     eval_transform = create_transform(**data_cfg, is_training=False)

#     # Freeze backbone if requested, then re-enable head gradients
#     if freeze_backbone:
#         for p in model.parameters(): p.requires_grad = False
#         _enable_head_grads(model)

#     # Sanity check
#     n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     n_total = sum(p.numel() for p in model.parameters())
#     print(f"Trainable params: {n_train:,} / {n_total:,}")
#     if n_train == 0:
#         raise RuntimeError("No trainable parameters found. Check head replacement / freezing logic.")

#     return model, eval_transform

# ---------------- metrics helper ----------------
class MetricsCalculator:
    def __init__(self): self.reset()
    def reset(self): self.TP=self.TN=self.FP=self.FN=0
    def update(self, y_true, y_pred):
        if torch.is_tensor(y_true): y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred): y_pred = y_pred.cpu().numpy()
        neg_t, neg_p = 1 - y_true, 1 - y_pred
        self.TP += int(np.sum(y_true * y_pred))
        self.TN += int(np.sum(neg_t * neg_p))
        self.FP += int(np.sum(neg_t * y_pred))
        self.FN += int(np.sum(y_true * neg_p))
    def compute(self):
        TPR = self.TP / (self.TP + self.FN + 1e-7)
        TNR = self.TN / (self.TN + self.FP + 1e-7)
        TSS = TPR + TNR - 1
        HSS = 2 * (self.TP * self.TN - self.FN * self.FP) / (
              (self.TP + self.FN)*(self.FN + self.TN) + (self.TP + self.FP)*(self.FP + self.TN) + 1e-7)
        return {"TPR":TPR,"TNR":TNR,"TSS":TSS,"HSS":HSS,
                "TP":self.TP,"TN":self.TN,"FP":self.FP,"FN":self.FN}

# ============================ MAIN ============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a dedicated experiment folder
    exp_id = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{CFG['model_name']}"
    exp_dir = os.path.join(CFG["results_base"], exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    plot_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")

    # Optional: save current config for reproducibility
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(CFG, f, indent=2)

    
    # exact counts & steps
    
    # if LABEL_MAP is not None: 
    # full_counts = count_labels_all_shards(SPLIT_DIRS["Train"])
    # else: 
    #     full_counts = {0:1, 1:1}
    #     print("skipping class count (C-threshold labels already embedded).")
    full_counts = {0: 610108, 1: 149249}
    Nn, Nf = full_counts.get(0, 1), full_counts.get(1, 1)
    print(f"Class counts (Train FULL): {full_counts}")
    
    

    total_train = count_samples_all_shards(SPLIT_DIRS["Train"])
    steps_per_epoch = max(1, total_train // CFG["batch_size"])
    print(f"Train samples: {total_train:,} | steps/epoch: {steps_per_epoch:,}")

    # model & transform
    # # Single-stream model
    # model, _ = build_model(
    #     CFG["backbone"],
    #     num_classes=2,
    #     pretrained=CFG["pretrained"],
    #     freeze_backbone=CFG["freeze_backbone"],
    # )


    # Two-Stream model
    model = build_model(num_classes=2).to(device) 
    model = model.to(device)

    # dataloaders
    train_ds = make_dataset("Train", shuffle_shards=True,  shuffle_samples=True)
    val_ds   = make_dataset("Validation", shuffle_shards=False, shuffle_samples=False)
    test_ds  = make_dataset("Test", shuffle_shards=False,   shuffle_samples=False)

    common = dict(
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        pin_memory=CFG["pin_memory"],
        persistent_workers=CFG["persistent_workers"],
        prefetch_factor=CFG["prefetch_factor"],
    )
    dls = {
        "Train": DataLoader(train_ds, **common),
        "Validation": DataLoader(val_ds, **common),
        "Test": DataLoader(test_ds, **common),
    }

    # loss
    class_weights = torch.tensor([1.0, Nn/Nf], dtype=torch.float32, device=device)
    criterion = FocalLoss(gamma=CFG["focal_gamma"], weight=class_weights) if CFG["use_focal"] \
                else nn.CrossEntropyLoss(weight=class_weights)

    # optimizer on TRAINABLE params only
    head_params = [p for p in model.parameters() if p.requires_grad]
    assert len(head_params) > 0, "No trainable parameters found (head may still be frozen)."
    # optimizer = optim.Adam(head_params, lr=CFG["lr"], weight_decay=0.05)
    optimizer = create_optimizer_v2(model, opt='adamw', lr=1e-4,
                                weight_decay=0.05, layer_decay=0.75)
    scaler = torch.GradScaler('cuda', enabled=torch.cuda.is_available())
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4, epochs=CFG["epochs"], steps_per_epoch=steps_per_epoch)



    # save dirs
    # tag = f'{CFG["backbone"]}_lr{CFG["lr"]}_ep{CFG["epochs"]}{"_focal" if CFG["use_focal"] else ""}'
    tag = (
        f'{CFG["model_name"]}_lr{CFG["lr"]}_ep{CFG["epochs"]}'
        f'{"_focal" if CFG["use_focal"] else ""}'
    )

    # model_path   = os.path.join(CFG["results_base"], "models", f"{tag}.pt")
    # plot_dir     = os.path.join(CFG["results_base"], "plots")
    # metrics_dir  = os.path.join(CFG["results_base"], "metrics")
    # os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # os.makedirs(plot_dir, exist_ok=True); os.makedirs(metrics_dir, exist_ok=True)
    # roc_path = os.path.join(plot_dir, f"{tag}_roc.png")
    # pr_path  = os.path.join(plot_dir, f"{tag}_pr.png")
    # cm_path  = os.path.join(plot_dir, f"{tag}_cm.png")
    # metrics_path = os.path.join(metrics_dir, f"{tag}_metrics.txt")

    model_path   = os.path.join(exp_dir, f"{tag}.pt")
    roc_path     = os.path.join(plot_dir, "roc.png")
    pr_path      = os.path.join(plot_dir, "pr.png")
    cm_path      = os.path.join(plot_dir, "confusion_matrix.png")
    metrics_path = os.path.join(exp_dir, "metrics.json")
    log_path     = os.path.join(exp_dir, "log.txt")

    # Optional: redirect prints to a log file
    import sys
    sys.stdout = open(log_path, "w", buffering=1)  # real-time flush
    print(f"Starting experiment {exp_id}")


    # ------------------ Training ------------------
    print("Starting training...")
    for epoch in range(CFG["epochs"]):
        model.train(); running_loss = 0.0; seen = 0
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{CFG['epochs']} - Train", leave=True)
        for inputs, labels, _ in dls["Train"]:
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(device, non_blocking=True) for x in inputs)
            else:
                inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running_loss += loss.item(); seen += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}"); pbar.update(1)
            if seen >= steps_per_epoch: break
        pbar.close()
        avg_train_loss = running_loss / max(1, seen)

        # ---- Validation ----
        model.eval(); val_loss=0.0; correct=0; total=0; vbatches=0
        with torch.no_grad():
            for inputs, labels, _ in dls["Validation"]:
                if isinstance(inputs, (list, tuple)):
                    inputs = tuple(x.to(device, non_blocking=True) for x in inputs)
                else:
                    inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item(); vbatches += 1
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        avg_val_loss = val_loss / max(1, vbatches)
        val_acc = 100.0 * correct / max(1, total)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")
        scheduler.step()
   
    # ------------------ Save model ------------------
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
 
    # ------------------ Test eval ------------------
    model.eval()
    all_probs, all_labels, all_preds_class = [], [], []
    metrics_calc = MetricsCalculator()
    with torch.no_grad():
        for inputs, labels, _ in dls["Test"]:
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(device, non_blocking=True) for x in inputs)
            else:
                inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
            preds_class = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_preds_class.extend(preds_class)
            metrics_calc.update(labels, preds_class)
    al = np.array(all_labels); ap = np.array(all_probs)

    # ROC
    fpr, tpr, _ = roc_curve(al, ap); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}'); plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {CFG['backbone']}"); plt.legend(); plt.savefig(roc_path, dpi=200, bbox_inches='tight'); plt.close()

    # PR-AUC
    if CFG["save_pr_curve"]:
        prec, rec, _ = precision_recall_curve(al, ap)
        pr_auc = auc(rec, prec)
        plt.figure(); plt.plot(rec, prec, label=f'PR-AUC = {pr_auc:.3f}')
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve - {CFG['backbone']}"); plt.legend()
        plt.savefig(pr_path, dpi=200, bbox_inches='tight'); plt.close()
    else:
        pr_auc = float('nan')

    # threshold by TSS
    best_tss, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        pred = (ap >= t).astype(int)
        TP = int(((al==1) & (pred==1)).sum())
        TN = int(((al==0) & (pred==0)).sum())
        FP = int(((al==0) & (pred==1)).sum())
        FN = int(((al==1) & (pred==0)).sum())
        TPR = TP / (TP + FN + 1e-7); TNR = TN / (TN + FP + 1e-7)
        TSS = TPR + TNR - 1
        if TSS > best_tss: best_tss, best_t = TSS, t

    # CM at 0.5
    cm = confusion_matrix(al, all_preds_class)
    plt.figure(figsize=(8,6))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(values_format='d')
    plt.title(f"Confusion Matrix (thr=0.5) - {CFG['backbone']}")
    cm_path = os.path.join(os.path.dirname(roc_path), f"{os.path.splitext(os.path.basename(roc_path))[0].replace('_roc','')}_cm.png")
    plt.savefig(cm_path, dpi=200, bbox_inches='tight'); plt.close()

    # ---------- Save metrics ----------
    final = metrics_calc.compute()
    results = {
        "AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "TPR": float(final["TPR"]),
        "TNR": float(final["TNR"]),
        "TSS": float(final["TSS"]),
        "HSS": float(final["HSS"]),
        "Best_TSS": float(best_tss),
        "Best_threshold": float(best_t),
        "TP": int(final["TP"]),
        "TN": int(final["TN"]),
        "FP": int(final["FP"]),
        "FN": int(final["FN"]),
        "seed": CFG.get("seed", None),
        "backbone": CFG["backbone"],
        "use_flow": CFG["use_flow"],
        "epochs": CFG["epochs"],
        "date": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ---------- Save metrics ----------
    final = metrics_calc.compute()
    results = {
        "AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "TPR": float(final["TPR"]),
        "TNR": float(final["TNR"]),
        "TSS": float(final["TSS"]),
        "HSS": float(final["HSS"]),
        "Best_TSS": float(best_tss),
        "Best_threshold": float(best_t),
        "TP": int(final["TP"]),
        "TN": int(final["TN"]),
        "FP": int(final["FP"]),
        "FN": int(final["FN"]),
        "seed": CFG.get("seed", None),
        "backbone": CFG["backbone"],
        "use_flow": CFG["use_flow"],
        "epochs": CFG["epochs"],
        "date": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(
        f"ROC AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f} | "
        f"TSS={best_tss:.4f} @ thr={best_t:.3f}"
    )

    # ------------------ Optional: Obsidian summary file ------------------
    summary_path = os.path.join(exp_dir, f"{tag}_summary.md")
    with open(summary_path, "w") as f:
        f.write(f"# AR-Flares Experiment Summary\n\n")
        f.write(f"**Experiment tag:** `{tag}`\n")
        f.write(f"**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Backbone:** {CFG['backbone']}\n")
        f.write(f"**Use Flow:** {CFG['use_flow']}\n")
        f.write(f"**Freeze Backbone:** {CFG['freeze_backbone']}\n")
        f.write(f"**Learning Rate:** {CFG['lr']}\n")
        f.write(f"**Epochs:** {CFG['epochs']}\n")
        f.write(f"**Batch Size:** {CFG['batch_size']}\n")
        f.write(f"**Seed**: `{CFG.get('seed', 'N/A')}`\n\n")


        f.write("### Results\n")
        f.write(f"- **AUC:** {roc_auc:.4f}\n")
        f.write(f"- **PR-AUC:** {pr_auc:.4f}\n")
        f.write(f"- **TSS:** {best_tss:.4f}\n")
        f.write(f"- **HSS:** {final['HSS']:.4f}\n")
        f.write(f"- **Best threshold (TSS):** {best_t:.3f}\n\n")

        f.write("### Notes\n")
        f.write("- Model trained on: Train split\n")
        f.write("- Validated on: Validation split\n")
        f.write("- Tested on: Test split\n")
        f.write("- Next steps: _fill this in manually in Obsidian_\n\n")

        f.write("### File Paths\n")
        f.write(f"- Model: `{model_path}`\n")
        f.write(f"- Metrics: `{metrics_path}`\n")
        f.write(f"- ROC: `{roc_path}`\n")
        f.write(f"- PR: `{pr_path}`\n")
        f.write(f"- Confusion Matrix: `{cm_path}`\n")
        f.write(f"- Log: `{log_path}`\n")

    print(f"Summary written to {summary_path}")


# if __name__ == "__main__":
#     # List of backbones to test overnight
#     backbone_list = [
#         "convnext_tiny", 
#         "efficientnet_b0",
#         "resnet50",
#     ]

#     # optional: shorter epoch count for sanity check
#     CFG["epochs"] = 5
#     CFG["freeze_backbone"] = True   # keep consistent for fair comparison
#     CFG["use_flow"] = True         # or True, depending on what you test

#     for backbone in backbone_list:
#         print(f"\n{'='*80}\n🏁 Starting experiment with backbone: {backbone}\n{'='*80}")
#         CFG["backbone"] = backbone
#         CFG["model_name"] = (
#             "two_stream_" + backbone if CFG["use_flow"]
#             else "single_stream_" + backbone
#         )

#         try:
#             main()
#             torch.cuda.empty_cache()
#         except Exception as e:
#             print(f"Failed on {backbone}: {e}")

# if __name__ == "__main__":
#     CFG["epochs"] = 5
#     CFG["freeze_backbone"] = True

#     # ---------------- Single-stream baseline ----------------
#     CFG["use_flow"] = False
#     CFG["backbone"] = "convnext_tiny"
#     CFG["model_name"] = "single_stream_convnext_tiny"
#     print(f"\n{'='*90}")
#     print("Starting Single-Stream baseline: convnext_tiny")
#     print(f"{'='*90}")
#     try:
#         main()
#         torch.cuda.empty_cache()
#     except Exception as e:
#         print(f"Failed single-stream convnext_tiny: {e}")

#     # ---------------- Two-stream experiments ----------------
#     CFG["use_flow"] = True
#     img_backbone = "convnext_tiny"
#     flow_encoders = ["SmallFlowCNN", "FlowResNetTiny"]  # pick 2 flow variants

#     for flow_encoder in flow_encoders:
#         print(f"\n{'='*90}")
#         print(f"Starting Two-Stream: IMG={img_backbone} | FLOW={flow_encoder}")
#         print(f"{'='*90}")

#         CFG["backbone"] = img_backbone
#         CFG["model_name"] = f"two_stream_{img_backbone}_{flow_encoder}"

#         try:
#             main()
#             torch.cuda.empty_cache()
#         except Exception as e:
#             print(f"Failed on {img_backbone} + {flow_encoder}: {e}")

# if __name__ == "__main__":

#     CFG.update({
#     "backbone": "convnext_tiny",    # or "convnext_base" if GPU allows
#     "pretrained": True,
#     "freeze_backbone": False,
#     "lr": 3e-5,
#     "epochs": 10,
#     "batch_size": 64,
#     "use_flow": True,               # keep single-stream for clarity
#     "two_stream": True,
#     "flow_encoder": "SmallFlowCNN",
#     "use_focal": True,
#     "focal_gamma": 2.0, 
#     "model_name": "two_stream_convnext_tiny_smallflowcnn"
#     })

#     # Temporal 
#     CFG.update({
#     "use_seq": True,                 # turn on sequence mode
#     "seq_T": 3,                      # number of frames
#     "seq_stride_steps": 8,           # 8 * 12min = 96min between frames
#     "seq_offsets": [-16, -8, 0],     # past-only: (-192, -96, 0) minutes
#     "seq_aggregate": "mean",         # or "attn"
#     })

#     main()

if __name__ == "__main__":
    CFG.update({
        "backbone": "convnext_tiny",     # or "vit_base_patch16_224" for ViT variant
        "pretrained": True,
        "freeze_backbone": False,
        "lr": 3e-5,
        "epochs": 10,
        "batch_size": 64,

        # ---- turn OFF flow for temporal baseline ----
        "use_flow": True,
        "two_stream": False,

        "use_focal": True,
        "focal_gamma": 2.0,

        "model_name": "temporal_convnext_tiny_seq3_mean_with_flow",
    })

    # ---- Temporal setup ----
    CFG.update({
        "use_seq": True,                 # activate sequence mode
        "seq_T": 3,                      # 3 frames per sample
        "seq_stride_steps": 8,           # spacing (8 * 12 min = 96 min)
        "seq_offsets": [-16, -8, 0],     # past-only (no leakage)
        "seq_aggregate": "mean",         # mean or attn
    })

    main()



# Vesion that didn't work: 
# CFG.update({
    #     "backbone": "can_small",
    #     "pretrained": False,
    #     "freeze_backbone": False,
    #     "use_flow": False,
    #     "use_diff": False,
    #     "two_stream": False,
    #     "lr": 1e-3,
    #     "epochs": 20,
    #     "batch_size": 64
    # })

    # CFG["model_name"] = f"single_stream_{CFG['backbone']}"
    # print("\n=== Training CANSmall (from scratch) ===\n")

    # CFG.update({
    #     "backbone": "ms_fusion",
    #     "vit_name": "vit_base_patch16_224",
    #     "pretrained": True,
    #     "freeze_backbone": False,   # fine-tune ViT
    #     "k_crops": 8,
    #     "crop_size": 64,
    #     "crop_stride": 32,
    #     "local_dim": 192,
    #     "cross_heads": 4,
    #     "lr": 3e-5,                 # ViT friendly
    #     "epochs": 5,               # 20–30 if you can
    #     "batch_size": 64,
    #     "use_flow": False,          # start without flow; add later if you want
    #     "two_stream": False,
    # })

    # CFG["model_name"] = f"single_stream_{CFG['backbone']}"    
    # main()

# if __name__ == "__main__":
#     setups = [
#         {"use_flow": False, "use_diff": False, "two_stream": False, "desc": "Single"},
#         # {"use_flow": True,  "use_diff": False, "two_stream": False, "desc": "Single+Flow"},
#         # {"use_flow": False, "use_diff": True,  "two_stream": False, "desc": "Single+Diff"},
#         {"use_flow": True,  "use_diff": False, "two_stream": True,  "desc": "TwoStream"},
#     ]

#     for setup in setups:
#         CFG.update(setup)

#         CFG["backbone"] = "can_small"
#         CFG["pretrained"] = False
#         CFG["freeze_backbone"] = False
#         CFG["epochs"] = 5
            
#         # --- Proper run naming ---
#         if CFG.get("two_stream", False):
#             CFG["model_name"] = f"two_stream_{CFG['backbone']}"
#         elif CFG.get("use_diff", False):
#             CFG["model_name"] = f"single_stream_{CFG['backbone']}_diff"
#         elif CFG.get("use_flow", False):
#             CFG["model_name"] = f"single_stream_{CFG['backbone']}_flow"
#         else:
#             CFG["model_name"] = f"single_stream_{CFG['backbone']}"

#         print(f"\n=== Training {setup['desc']} ({CFG['model_name']}) ===\n")
#         main()
#         torch.cuda.empty_cache()
