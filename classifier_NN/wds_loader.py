import os, io, tarfile, json, glob, random
from typing import Iterable, List, Tuple, Dict, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import Counter

# ------------------ CONFIG ------------------
IS_COMPUTE_CANADA = os.path.exists('/scratch')
# Where you put the shards created earlier:
#   expected structure:
#   WDS_BASE/train/shard-000000.tar
#   WDS_BASE/val/shard-000000.tar
#   WDS_BASE/test/shard-000000.tar
WDS_BASE = '/scratch/dgarmaev/AR-flares/wds_out' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/wds_out'

SPLIT_DIRS = {
    "Train": os.path.join(WDS_BASE, "train"),
    "Validation": os.path.join(WDS_BASE, "val"),
    "Test": os.path.join(WDS_BASE, "test"),
}

BATCH_SIZE = 64
NUM_WORKERS = 4  # bump if I/O is good on Lightning
PIN_MEMORY = True

# Your transforms (images are already 224x224 PNGs; keep Resize if unsure)
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Optionally normalize:
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# --------------------------------------------


class TarShardDataset(IterableDataset):
    """
    Streams samples from a list of .tar shards that contain pairs:
      - {basename}.png
      - {basename}.json with {"label": 0/1, "reg_label": "...", "ar": "...."}
    """
    def __init__(self,
                 shard_paths: List[str],
                 transform=None,
                 shuffle_shards: bool = True,
                 shuffle_samples: bool = False,
                 shuffle_buffer_size: int = 2048,
                 seed: int = 42):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.transform = transform
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def _sample_iter_from_tar(self, tar_path: str):
        with tarfile.open(tar_path, "r") as tf:
            # Gather png/json members by base key
            by_key: Dict[str, Dict[str, tarfile.TarInfo]] = {}
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                if m.name.endswith(".png"):
                    key = m.name[:-4]
                    by_key.setdefault(key, {})["png"] = m
                elif m.name.endswith(".json"):
                    key = m.name[:-5]
                    by_key.setdefault(key, {})["json"] = m

            # Optionally permute keys for intra-shard shuffle
            keys = list(by_key.keys())
            if self.shuffle_samples:
                rng = random.Random(self.seed ^ hash(tar_path))
                rng.shuffle(keys)

            for k in keys:
                parts = by_key[k]
                if "png" not in parts or "json" not in parts:
                    continue
                # Read bytes
                png_bytes = tf.extractfile(parts["png"]).read()
                meta = json.loads(tf.extractfile(parts["json"]).read().decode("utf-8"))
                # Decode image
                img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                label = int(meta["label"])
                yield img, label, meta

    def _roundrobin(self, iters: List[Iterable]):
        # simple round-robin over shard iterators
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
        # Reservoir-like shuffle buffer for IterableDataset
        rng = random.Random(self.seed)
        buf = []
        for x in iterable:
            buf.append(x)
            if len(buf) >= self.shuffle_buffer_size:
                rng.shuffle(buf)
                while buf:
                    yield buf.pop()
        # Flush remaining
        rng.shuffle(buf)
        while buf:
            yield buf.pop()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # Split shards across workers
        if worker_info is None:
            shards = self.shard_paths
        else:
            num_workers = worker_info.num_workers
            wid = worker_info.id
            shards = self.shard_paths[wid::num_workers]

        shards = list(shards)
        if self.shuffle_shards:
            rng = random.Random(self.seed + (worker_info.id if worker_info else 0))
            rng.shuffle(shards)

        # Build per-shard generators
        gens = [self._sample_iter_from_tar(sp) for sp in shards]
        stream = self._roundrobin(gens)

        # Optional buffered sample-level shuffling across shards
        if self.shuffle_samples:
            stream = self._buffered_shuffle(stream)

        for sample in stream:
            yield sample


def list_shards(split_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(split_dir, "*.tar")))


def make_dataset(split_name: str,
                 transform=None,
                 shuffle_shards=True,
                 shuffle_samples=False,
                 seed=42) -> TarShardDataset:
    shard_dir = SPLIT_DIRS[split_name]
    shards = list_shards(shard_dir)
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {shard_dir}")
    return TarShardDataset(
        shards,
        transform=transform or DEFAULT_TRANSFORM,
        shuffle_shards=shuffle_shards,
        shuffle_samples=shuffle_samples,
        seed=seed
    )


def make_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_ds = make_dataset("Train", shuffle_shards=True, shuffle_samples=True)
    val_ds   = make_dataset("Validation", shuffle_shards=False, shuffle_samples=False)
    test_ds  = make_dataset("Test", shuffle_shards=False, shuffle_samples=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=PIN_MEMORY)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=PIN_MEMORY)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers,
                          pin_memory=PIN_MEMORY)
    return {"Train": train_dl, "Validation": val_dl, "Test": test_dl}


# ---------- quick sanity tools ----------
# def peek_n(dataloader: DataLoader, n=5):
#     it = iter(dataloader)
#     imgs, labels, metas = [], [], []
#     seen = 0
#     while seen < n:
#         batch = next(it)
#         # batch = (images, labels, meta) with collate of dicts disabled -> meta list of dicts
#         images, y, meta = batch
#         b = images.shape[0]
#         take = min(b, n - seen)
#         imgs.append(images[:take])
#         labels.append(y[:take])
#         # meta is a list of dicts; slice it
#         metas.extend(meta[:take])
#         seen += take
#     imgs = torch.cat(imgs, dim=0)
#     labels = torch.cat(labels, dim=0)
#     return imgs, labels, metas

def peek_n(dataloader: DataLoader, n=5):
    it = iter(dataloader)
    imgs, labels, metas = [], [], []
    seen = 0
    while seen < n:
        images, y, meta = next(it)   # meta may be a dict or a list of dicts
        b = images.shape[0]
        take = min(b, n - seen)
        imgs.append(images[:take])
        labels.append(y[:take])

        if isinstance(meta, dict):  
            # batch size = 1, just append the single dict
            metas.append(meta)
        elif isinstance(meta, list):  
            # batch size > 1, take the first "take" dicts
            metas.extend(meta[:take])
        else:
            raise TypeError(f"Unexpected meta type: {type(meta)}")

        seen += take

    imgs = torch.cat(imgs, dim=0)
    labels = torch.cat(labels, dim=0)
    return imgs, labels, metas



def count_labels(dataloader: DataLoader, max_batches: Optional[int] = None) -> Dict[int, int]:
    ctr = Counter()
    for i, (_, y, _) in enumerate(dataloader):
        ctr.update(y.tolist())
        if max_batches is not None and i + 1 >= max_batches:
            break
    return dict(ctr)


# --------------- main (demo) ---------------
if __name__ == "__main__":
    dls = make_dataloaders()
    # peek train
    x, y, m = peek_n(dls["Train"], n=8)
    print("Peek:", x.shape, y.tolist(), [mm.get("ar") for mm in m])
    # count label distribution (first 200 batches to keep it quick)
    dist_train = count_labels(dls["Train"], max_batches=200)
    dist_val   = count_labels(dls["Validation"], max_batches=200)
    dist_test  = count_labels(dls["Test"], max_batches=200)
    print("Label dist (train, ~subset):", dist_train)
    print("Label dist (val, ~subset):  ", dist_val)
    print("Label dist (test, ~subset): ", dist_test)
