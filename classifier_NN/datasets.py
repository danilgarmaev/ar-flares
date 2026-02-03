"""Dataset classes and data loading utilities for AR-flares classifier."""
import os
import io
import glob
import json
import tarfile
import random
from typing import Iterable, List, Dict
from contextlib import nullcontext
from collections import Counter

import torch
from torch.utils.data import IterableDataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from .config import CFG, SPLIT_DIRS, SPLIT_FLOW_DIRS, intensity_labels_path


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


# Initialize LABEL_MAP based on config
LABEL_MAP = None
if CFG["min_flare_class"].upper() == "M":
    LABEL_MAP = load_flare_label_map(
        os.path.join(intensity_labels_path, "C1.0_24hr_224_png_Labels.txt"),
        min_class="M"
    )

def _build_basic_transform(img_size: int):
    """Basic transform (Resize + ToTensor) parameterized by image size.

    Supports an optional spatial-resolution ablation for single-frame models:
      - Downsample by an integer factor using nearest-neighbor (no interpolation)
      - Upsample back to `img_size` using bilinear

    This reduces spatial information while keeping the network input size fixed.
    """
    factor = int(CFG.get("spatial_downsample_factor", 1) or 1)
    if factor <= 1:
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    ds = max(1, img_size // factor)
    return transforms.Compose([
        transforms.Resize((ds, ds), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

# Augmentation transform (Horizontal Flip + RandomAffine)
# Note: No vertical flips to preserve polarity physics
# AUG_TRANSFORM = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
#     transforms.ToTensor(),
# ])

# More robust augmentation 
class PolarityInversion:
    """Invert magnetic polarity: (I-128)*(-1)+128 on 0–255 8-bit images."""
    def __call__(self, img: Image.Image) -> Image.Image:
        # Work in uint8 space to respect 0–255 and 128 as zero-field reference
        arr = np.array(img, dtype=np.int16)
        arr = (arr - 128) * -1 + 128
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode=img.mode)


class AddIntegerNoise:
    """Add uniform integer noise in [-5, 5] and clamp to [0, 255]."""
    def __init__(self, max_abs_noise: int = 5):
        self.max_abs_noise = max_abs_noise

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img, dtype=np.int16)
        noise = np.random.randint(-self.max_abs_noise, self.max_abs_noise + 1, size=arr.shape, dtype=np.int16)
        arr = arr + noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode=img.mode)


def _build_aug_transform(img_size: int):
    """Augmentation transform parameterized by image size.

    Uses only physics-respecting operations (no vertical flip if disabled in cfg).

    NOTE:
      * This is a **per-image** transform; for sequences we now apply the same
        geometric ops (resize/flip/rotate/polarity) consistently across all
        frames in a sequence. See `_apply_seq_augment` in `TarShardDataset`.
    """
    aug_preset = str(CFG.get("aug_preset", "robust")).lower()
    factor = int(CFG.get("spatial_downsample_factor", 1) or 1)
    ds = max(1, img_size // factor) if factor > 1 else img_size

    # Apply spatial ablation before aug ops so augmentation doesn't re-inject
    # fine-grained detail. For factor==1 this is just an identity resize.
    if aug_preset in {"paper_vgg", "vgg_paper", "paper"}:
        # Paper-style augmentation (previously commented):
        # Resize -> RandomHorizontalFlip -> RandomAffine -> ToTensor
        # (no vertical flips / polarity inversion / noise)
        return transforms.Compose([
            transforms.Resize(
                (ds, ds),
                interpolation=(
                    transforms.InterpolationMode.NEAREST if factor > 1 else transforms.InterpolationMode.BILINEAR
                ),
            ),
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
        ])

    # Default: robust augmentation
    vflip_prob = float(CFG.get("vertical_flip_prob", 0.5))
    return transforms.Compose([
        transforms.Resize((ds, ds), interpolation=transforms.InterpolationMode.NEAREST if factor > 1 else transforms.InterpolationMode.BILINEAR),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=vflip_prob),
        transforms.RandomRotation(30),  # limit to ±30°
        PolarityInversion(),            # magnetic polarity inversion
        AddIntegerNoise(max_abs_noise=5),  # ±5 noise with clamping
        transforms.ToTensor(),
    ])


class TarShardDataset(IterableDataset):
    """WebDataset-style iterable dataset for tar archives containing images (and optional flow / sequences).

    Sampling / Balancing modes (Train split only):
      balance_classes=False: yield all samples (imbalanced).
      balance_mode='prob': per-epoch probabilistic negative downsampling.
          Each negative kept with probability neg_keep_prob (e.g. 0.25).
          Selection re-drawn every epoch -> fresh, stochastic exposure.
      balance_mode='fixed': deterministic, epoch-stable subset of negatives.
          Each negative deterministically mapped to a pseudo-random number in [0,1)
          via hashing its metadata; kept if value < neg_keep_prob. Stable across runs
          given identical shard ordering & metadata. Avoids variance in sample counts.

    Rationale: 'prob' maximizes diversity of seen negatives across epochs (good when there
    are many redundant negatives). 'fixed' provides reproducibility & consistent epoch length.

    Validation/Test are never balanced to preserve true distribution.
    """
    
    def __init__(self, shard_paths: List[str], flow_paths: List[str] = None,
                 split_name: str = "Train",
                 use_flow: bool = False,
                 shuffle_shards=True, shuffle_samples=False,
                 shuffle_buffer_size=2048, seed=42,
                 img_size: int | None = None,
                 use_aug: bool | None = None):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.flow_paths = list(flow_paths) if flow_paths else None
        self.split_name = split_name
        self.use_flow = use_flow
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        
        # Resolve transform config
        if img_size is None:
            img_size = CFG.get("image_size", 224)
        if use_aug is None:
            use_aug = CFG.get("use_aug", False)
        self.use_aug = (self.split_name == "Train" and use_aug)

        # Base (non-aug) transform is always available, and we keep the
        # original aug transform for single-frame mode. For sequences we
        # apply consistent aug per sequence via `_apply_seq_augment`.
        self.base_transform = _build_basic_transform(img_size)
        self.aug_transform = _build_aug_transform(img_size) if self.use_aug else self.base_transform

        # Backwards-compat attribute used in single-frame branch
        # (see _sample_iter_from_tar, SINGLE-FRAME MODE).
        # For Train, this includes augmentation; for Val/Test, it's just
        # the basic resize+ToTensor transform.
        self.transform = self.aug_transform

        if self.use_flow:
            assert self.flow_paths is not None, "Flow shards required when use_flow=True"
            assert len(self.shard_paths) == len(self.flow_paths), "Image vs flow shard count mismatch"

    def _sample_iter_from_tar(self, img_tar: str, flow_tar: str = None):
        # Check if sequence mode or diff attention is enabled
        use_seq = CFG.get("use_seq", False)
        use_diff_attn = CFG.get("use_diff_attention", False)
        
        if use_seq or use_diff_attn:
            # SEQUENCE / DIFF MODE: Load temporal sequences of frames
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
                entries.sort(key=lambda x: x[0])

                def load_img_raw(member_png):
                    """Load raw PIL image (L mode) without tensor/augmentations.

                    Sequence-level augmentations are applied later so that all
                    frames in a sequence share the same geometric transform.
                    """
                    png_bytes = tf_img.extractfile(member_png).read()
                    img = Image.open(io.BytesIO(png_bytes)).convert("L")
                    return img

                def _apply_seq_augment(pil_imgs: List[Image.Image]) -> List[torch.Tensor]:
                    """Apply **consistent** augmentations to a sequence of frames.

                    Strategy:
                      * Sample horizontal flip, vertical flip, rotation angle,
                        and polarity inversion **once** per sequence.
                      * Apply those to every frame in `pil_imgs`.
                      * AddIntegerNoise is left per-frame (noise is fine to
                        vary temporally), but you can change that if desired.
                    """
                    if not self.use_aug:
                        # Just resize + ToTensor via base_transform
                        return [self.base_transform(img) for img in pil_imgs]

                    # Sample geometric decisions once per sequence
                    hflip = random.random() < 0.5
                    vflip_prob = float(CFG.get("vertical_flip_prob", 0.5))
                    vflip = random.random() < vflip_prob
                    angle = random.uniform(-30.0, 30.0)
                    polarity = True  # always apply in aug mode

                    out_tensors: List[torch.Tensor] = []
                    for img in pil_imgs:
                        # Resize first
                        img_t = transforms.Resize(
                            (self.base_transform.transforms[0].size[0],
                             self.base_transform.transforms[0].size[1]),
                            interpolation=transforms.InterpolationMode.BILINEAR,
                        )(img)

                        # Apply shared geometric transforms
                        if hflip:
                            img_t = transforms.functional.hflip(img_t)
                        if vflip:
                            img_t = transforms.functional.vflip(img_t)
                        img_t = transforms.functional.rotate(img_t, angle)

                        # Polarity inversion (shared decision)
                        if polarity:
                            img_t = PolarityInversion()(img_t)

                        # Per-frame integer noise (can be made shared if desired)
                        img_t = AddIntegerNoise(max_abs_noise=5)(img_t)

                        # To tensor
                        img_t = transforms.ToTensor()(img_t)
                        out_tensors.append(img_t)

                    return out_tensors

                def get_meta(jm):
                    mm = json.loads(tf_img.extractfile(jm).read().decode("utf-8"))
                    # Include the JSON member name / base key so callers can
                    # derive the original filename (useful for evaluation scripts)
                    try:
                        mm.setdefault("basename", jm.name)
                    except Exception:
                        pass
                    return mm

                if use_diff_attn:
                    # For Diff Attention, we need [t-1, t]
                    offsets = [-1, 0]
                else:
                    T = CFG.get("seq_T", 3)
                    offsets = CFG.get("seq_offsets", [-16, -8, 0])

                # Optional stride over sequence start indices to avoid
                # generating highly-overlapping windows (which can blow up
                # the effective number of training samples and steps/epoch
                # for long sequences like T=16 or T=32).
                seq_stride = CFG.get("seq_stride", CFG.get("seq_stride_steps", 1))

                # For sequence/diff modes we cannot shuffle `entries` (it would
                # destroy temporal ordering required by offsets). Instead we
                # shuffle the list of start indices. This gives per-shard
                # sample shuffling without buffering huge tensors in memory.
                start_indices = list(range(0, len(entries), max(1, seq_stride)))
                if self.shuffle_samples:
                    rng = random.Random(self.seed ^ hash(img_tar))
                    rng.shuffle(start_indices)

                for i in start_indices:
                    key0, png0, jm0 = entries[i]
                    meta0 = get_meta(jm0)

                    idxes = []
                    ok = True
                    for off in offsets:
                        j = i + off
                        if j < 0 or j >= len(entries):
                            ok = False
                            break
                        keyj, pngj, jmj = entries[j]
                        metaj = get_meta(jmj)
                        # Ensure same Active Region (AR)
                        if ("ar" in meta0 and "ar" in metaj) and (meta0["ar"] != metaj["ar"]):
                            ok = False
                            break
                        idxes.append(j)
                    if not ok:
                        continue

                    # Load raw frames first, then apply sequence-level
                    # augmentations/resize so that all frames share the same
                    # geometric ops.
                    pil_frames: List[Image.Image] = []
                    for j in idxes:
                        _, pngj, _ = entries[j]
                        pil_frames.append(load_img_raw(pngj))

                    frames = _apply_seq_augment(pil_frames)
                    
                    if use_diff_attn:
                        # frames[0] is t-1, frames[1] is t
                        img_prev = frames[0]
                        img_curr = frames[1]
                        diff = img_curr - img_prev # Difference image
                        
                        # Yield (Image, Diff) tuple
                        # Image is (1,H,W), Diff is (1,H,W)
                        # Replicate Image to 3ch for backbone if needed, but let model handle it
                        # We yield (img_curr, diff)
                        # Note: Standard models expect 3ch. We'll handle this in the model or here.
                        # Let's yield tensors.
                        x_out = (img_curr.repeat(3, 1, 1), diff) # (3,H,W), (1,H,W)
                    else:
                        x_seq = torch.stack(frames, dim=0)  # (T,1,H,W)
                        x_out = x_seq

                    ref_idx = idxes[-1]
                    _, _, refjm = entries[ref_idx]
                    meta = get_meta(refjm)
                    # Also include the base key for the reference frame (without extension)
                    try:
                        meta.setdefault("basename", entries[ref_idx][0])
                    except Exception:
                        pass
                    if LABEL_MAP is not None:
                        fname = os.path.basename(entries[ref_idx][1].name)
                        label = LABEL_MAP.get(fname, 0)
                    else:
                        label = int(meta["label"])

                    yield x_out, label, meta
        else:
            # SINGLE-FRAME MODE: Load individual images
            with tarfile.open(img_tar, "r") as tf_img, \
                (tarfile.open(flow_tar, "r") if (self.use_flow and flow_tar) else nullcontext()) as tf_flow:

                by_key: Dict[str, Dict[str, tarfile.TarInfo]] = {}
                for m in tf_img.getmembers():
                    if not m.isfile():
                        continue
                    if m.name.endswith(".png"):
                        by_key.setdefault(m.name[:-4], {})["png"] = m
                    elif m.name.endswith(".json"):
                        by_key.setdefault(m.name[:-5], {})["json"] = m

                if self.use_flow:
                    for m in tf_flow.getmembers():
                        if m.isfile() and m.name.endswith(".flow.npz"):
                            by_key.setdefault(m.name[:-9], {})["flow"] = m

                keys = list(by_key.keys())
                if self.shuffle_samples:
                    rng = random.Random(self.seed ^ hash(img_tar))
                    rng.shuffle(keys)

                for k in keys:
                    parts = by_key[k]
                    if "png" not in parts or "json" not in parts:
                        continue
                    if self.use_flow and "flow" not in parts:
                        continue

                    meta = json.loads(tf_img.extractfile(parts["json"]).read().decode("utf-8"))
                    # Attach the base key (filename without extension) so downstream
                    # code can recover timestamp or other info encoded in the name.
                    try:
                        meta.setdefault("basename", k)
                    except Exception:
                        pass

                    # Label
                    if LABEL_MAP is not None:
                        fname = os.path.basename(parts["png"].name)
                        label = LABEL_MAP.get(fname, 0)
                    else:
                        label = int(meta["label"])

                    # Load image
                    png_bytes = tf_img.extractfile(parts["png"]).read()
                    img = Image.open(io.BytesIO(png_bytes)).convert("L")
                    img_t = self.transform(img)  # (1, H, W)

                    # Flow/diff options (simplified for now)
                    if CFG.get("use_flow", False):
                        flow_bytes = tf_flow.extractfile(parts["flow"]).read()
                        with np.load(io.BytesIO(flow_bytes)) as f:
                            u = f["u8"].astype(np.float32) * float(f["su"])
                            v = f["v8"].astype(np.float32) * float(f["sv"])
                        flow_t = torch.from_numpy(np.stack([u, v], axis=0))
                        if CFG.get("two_stream", False):
                            yield (img_t.repeat(3, 1, 1), flow_t), label, meta
                        else:
                            yield torch.cat([img_t, flow_t], dim=0), label, meta
                    elif CFG.get("use_diff", False):
                        # Diff mode not fully implemented in refactor yet
                        yield img_t.repeat(2, 1, 1), label, meta
                    else:
                        # Standard single-frame: replicate grayscale to 3 channels for RGB models
                        if CFG["backbone"].lower().startswith(("can", "ms_fusion", "multiscale")):
                            yield img_t, label, meta  # Keep single channel for CAN
                        else:
                            yield img_t.repeat(3, 1, 1), label, meta  # (3, H, W)

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

        # IMPORTANT:
        # Do NOT open all shards concurrently.
        # Round-robin over per-shard generators keeps many tarfiles open at once
        # (one per shard per worker), which can hit the OS file-descriptor limit
        # and cause DataLoader workers to exit unexpectedly.
        img_shards = list(img_shards)
        flow_shards = list(flow_shards)
        paired = list(zip(img_shards, flow_shards))
        if self.shuffle_shards:
            rng = random.Random(self.seed + (worker_info.id if worker_info else 0))
            rng.shuffle(paired)

        def _shard_stream():
            for img, flow in paired:
                yield from self._sample_iter_from_tar(img, flow)

        stream = _shard_stream()

        # In sequence/diff modes, `shuffle_samples=True` is handled inside
        # `_sample_iter_from_tar` by shuffling start indices. Avoid buffering
        # full tensors here (can be massive for T=16/32 and OOM workers).
        if self.shuffle_samples and not (CFG.get("use_seq", False) or CFG.get("use_diff_attention", False)):
            stream = self._buffered_shuffle(stream)

        # Class balancing: subsample negatives to match positive count
        balance_classes = CFG.get("balance_classes", False) and self.split_name == "Train"
        balance_mode = CFG.get("balance_mode", "prob")
        neg_keep_prob = CFG.get("neg_keep_prob", 0.25)

        if balance_classes and balance_mode == "prob":
            rng = random.Random(self.seed + (worker_info.id if worker_info else 0) + 1000)
            for sample in stream:
                x, label, meta = sample
                if label == 0:
                    if rng.random() < neg_keep_prob:
                        yield sample
                else:
                    yield sample
        elif balance_classes and balance_mode == "fixed":
            # Deterministic selection: hash meta JSON (sorted keys) to get a stable float in [0,1)
            for sample in stream:
                x, label, meta = sample
                if label == 0:
                    try:
                        h = hash(json.dumps(meta, sort_keys=True)) & 0xffffffff
                        keep_val = (h % 10_000_000) / 10_000_000.0
                    except Exception:
                        # Fallback: always keep if hashing fails
                        keep_val = 0.0
                    if keep_val < neg_keep_prob:
                        yield sample
                else:
                    yield sample
        else:
            for sample in stream:
                yield sample


def list_shards(split_dir: str) -> List[str]:
    """List all tar shards in a directory."""
    return sorted(glob.glob(os.path.join(split_dir, "*.tar")))


def make_dataset(split_name: str, shuffle_shards=True, shuffle_samples=False, seed=42,
                 img_size: int | None = None, use_aug: bool | None = None):
    """Create a TarShardDataset for the given split.

    img_size and use_aug can be overridden explicitly, otherwise values fall back
    to the global CFG. This keeps backwards compatibility while allowing
    cfg-driven datasets via create_dataloaders.
    """
    img_shards = list_shards(SPLIT_DIRS[split_name])
    # Optional shard limiting for smoke tests / lightweight runs
    max_train_shards = CFG.get("max_train_shards", None)
    max_val_shards = CFG.get("max_val_shards", None)
    max_test_shards = CFG.get("max_test_shards", None)
    if split_name == "Train" and max_train_shards is not None:
        img_shards = img_shards[: max(1, int(max_train_shards))]
    elif split_name == "Validation" and max_val_shards is not None:
        img_shards = img_shards[: max(1, int(max_val_shards))]
    elif split_name == "Test" and max_test_shards is not None:
        img_shards = img_shards[: max(1, int(max_test_shards))]
    if CFG["use_flow"]:
        flow_shards = list_shards(SPLIT_FLOW_DIRS[split_name])
        # Mirror shard limiting for flow shards if enabled
        if split_name == "Train" and max_train_shards is not None:
            flow_shards = flow_shards[: max(1, int(max_train_shards))]
        elif split_name == "Validation" and max_val_shards is not None:
            flow_shards = flow_shards[: max(1, int(max_val_shards))]
        elif split_name == "Test" and max_test_shards is not None:
            flow_shards = flow_shards[: max(1, int(max_test_shards))]
    else:
        flow_shards = None
    if not img_shards:
        raise FileNotFoundError(f"No image shards for {split_name}")
    return TarShardDataset(
        img_shards,
        flow_shards,
        split_name=split_name,
        use_flow=CFG["use_flow"],
        shuffle_shards=shuffle_shards,
        shuffle_samples=shuffle_samples,
        seed=seed,
        img_size=img_size,
        use_aug=use_aug,
    )


# --------- shard scanners (fast, no images) ---------
def count_raw_labels_all_shards(split_dir: str) -> Dict[int, int]:
    """Count raw labels from JSON metadata in shards."""
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


def count_samples_all_shards(split_dir: str, estimate: bool = True) -> int:
    """
    Count total samples across all shards.
    If estimate=True (default), samples from first 10 shards and extrapolates.
    This is much faster for large datasets with many tar files.
    """
    tar_paths = sorted(glob.glob(os.path.join(split_dir, "*.tar")))
    
    if not tar_paths:
        return 0
    
    if estimate and len(tar_paths) > 10:
        # Sample first 10 shards and extrapolate
        sample_count = 0
        for tar_path in tar_paths[:10]:
            with tarfile.open(tar_path, "r") as tf:
                sample_count += sum(1 for m in tf.getmembers() if m.isfile() and m.name.endswith(".json"))
        
        avg_per_shard = sample_count / 10
        total = int(avg_per_shard * len(tar_paths))
        print(f"  Estimated from {len(tar_paths)} shards: ~{total:,} samples")
        return total
    else:
        # Count all shards (for smaller datasets or when exact count needed)
        total = 0
        for tar_path in tar_paths:
            with tarfile.open(tar_path, "r") as tf:
                total += sum(1 for m in tf.getmembers() if m.isfile() and m.name.endswith(".json"))
        return total


def create_dataloaders():
    """Create train, validation, and test dataloaders.

    Uses image_size and use_aug from CFG; callers that want per-run overrides
    should update cfg before calling train, which will be reflected via CFG.
    """
    img_size = CFG.get("image_size", 224)
    use_aug = CFG.get("use_aug", False)

    seed = int(CFG.get("seed", 42))
    train_ds = make_dataset(
        "Train",
        shuffle_shards=True,
        shuffle_samples=True,
        seed=seed,
        img_size=img_size,
        use_aug=use_aug,
    )
    val_ds = make_dataset(
        "Validation",
        shuffle_shards=False,
        shuffle_samples=False,
        seed=seed,
        img_size=img_size,
        use_aug=False,
    )
    test_ds = make_dataset(
        "Test",
        shuffle_shards=False,
        shuffle_samples=False,
        seed=seed,
        img_size=img_size,
        use_aug=False,
    )

    common = dict(
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        pin_memory=CFG["pin_memory"],
    )
    
    # prefetch_factor and persistent_workers only work with num_workers > 0
    if CFG["num_workers"] > 0:
        common["persistent_workers"] = CFG["persistent_workers"]
        common["prefetch_factor"] = CFG["prefetch_factor"]
    
    return {
        "Train": DataLoader(train_ds, drop_last=True, **common),
        "Validation": DataLoader(val_ds, **common),
        "Test": DataLoader(test_ds, **common),
    }
