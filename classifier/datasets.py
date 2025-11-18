import os, io, glob, json, tarfile, random
from typing import Iterable, List, Dict
from contextlib import nullcontext
from collections import Counter

import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from PIL import Image

from config import CFG, SPLIT_DIRS, SPLIT_FLOW_DIRS, LABEL_MAP


class TarShardDataset(IterableDataset):
    def __init__(self, shard_paths: List[str], flow_paths: List[str] = None,
                 use_flow: bool = False, shuffle_shards=True, shuffle_samples=False,
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
            assert len(self.shard_paths) == len(self.flow_paths), "Shard count mismatch"

    def _sample_iter_from_tar(self, img_tar: str, flow_tar: str = None):
        with tarfile.open(img_tar, "r") as tf_img, \
             (tarfile.open(flow_tar, "r") if (self.use_flow and flow_tar) else nullcontext()) as tf_flow:

            by_key: Dict[str, Dict[str, tarfile.TarInfo]] = {}
            
            # Index image tar
            for m in tf_img.getmembers():
                if not m.isfile(): continue
                if m.name.endswith(".png"):
                    by_key.setdefault(m.name[:-4], {})["png"] = m
                elif m.name.endswith(".json"):
                    by_key.setdefault(m.name[:-5], {})["json"] = m

            # Index flow tar
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

                # Label
                if LABEL_MAP is not None:
                    fname = os.path.basename(parts["png"].name)
                    label = LABEL_MAP.get(fname, 0)
                else:
                    label = int(meta["label"])

                # Image
                png_bytes = tf_img.extractfile(parts["png"]).read()
                img = Image.open(io.BytesIO(png_bytes)).convert("L").resize((224,224))
                img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)

                # Difference
                diff_t = None
                if CFG.get("use_diff", False):
                    prev_k = keys[keys.index(k)-1] if keys.index(k) > 0 else None
                    if prev_k and "png" in by_key[prev_k]:
                        prev_img = Image.open(io.BytesIO(tf_img.extractfile(by_key[prev_k]["png"]).read())).convert("L").resize((224,224))
                        prev_t = torch.from_numpy(np.array(prev_img, np.float32)/255.).unsqueeze(0)
                        diff_t = (img_t - prev_t).clamp(-1,1)
                        diff_t = (diff_t - diff_t.min()) / (diff_t.max() - diff_t.min() + 1e-8)

                # Flow
                flow_t = None
                if CFG.get("use_flow", False):
                    flow_bytes = tf_flow.extractfile(parts["flow"]).read()
                    with np.load(io.BytesIO(flow_bytes)) as f:
                        u = f["u8"].astype(np.float32) * float(f["su"])
                        v = f["v8"].astype(np.float32) * float(f["sv"])
                    flow_t = torch.from_numpy(np.stack([u,v],axis=0))

                # Combine
                if CFG["use_flow"] and not CFG["use_diff"]:
                    if CFG.get("two_stream", False):
                        yield (img_t.repeat(3,1,1), flow_t), label, meta
                    else:
                        yield torch.cat([img_t, flow_t], dim=0), label, meta
                elif CFG["use_diff"]:
                    x = torch.cat([img_t, diff_t], dim=0) if diff_t is not None else img_t.repeat(2,1,1)
                    yield x, label, meta
                else:
                    if CFG["backbone"].lower().startswith(("can", "ms_fusion", "multiscale")):
                        yield img_t, label, meta
                    else:
                        yield img_t.repeat(3,1,1), label, meta

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


# ===================== UTILITIES =====================
def list_shards(split_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(split_dir, "*.tar")))


def make_dataset(split_name: str, shuffle_shards=True, shuffle_samples=False, seed=42):
    img_shards = list_shards(SPLIT_DIRS[split_name])
    flow_shards = list_shards(SPLIT_FLOW_DIRS[split_name]) if CFG["use_flow"] else None
    
    if not img_shards:
        raise FileNotFoundError(f"No image shards for {split_name}")
    
    return TarShardDataset(
        img_shards, flow_shards,
        use_flow=CFG["use_flow"],
        shuffle_shards=shuffle_shards,
        shuffle_samples=shuffle_samples,
        seed=seed
    )


def count_labels_all_shards(split_dir: str) -> Dict[int, int]:
    """Count samples per class using LABEL_MAP if available."""
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


def make_dataloaders():
    """Create train/val/test dataloaders."""
    train_ds = make_dataset("Train", shuffle_shards=True, shuffle_samples=True)
    val_ds = make_dataset("Validation", shuffle_shards=False, shuffle_samples=False)
    test_ds = make_dataset("Test", shuffle_shards=False, shuffle_samples=False)

    common = dict(
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        pin_memory=CFG["pin_memory"],
        persistent_workers=CFG["persistent_workers"],
        prefetch_factor=CFG["prefetch_factor"],
    )

    return {
        "Train": DataLoader(train_ds, **common),
        "Validation": DataLoader(val_ds, **common),
        "Test": DataLoader(test_ds, **common),
    }