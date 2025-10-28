# import tarfile, io, numpy as np

# tar_path = "/teamspace/studios/this_studio/AR-flares/wds_flow/train/shard-000000.tar"

# with tarfile.open(tar_path, "r") as tf:
#     for m in tf.getmembers():
#         if m.name.endswith(".flow.npz"):
#             print("Checking:", m.name)
#             # Read the npz into memory
#             f = tf.extractfile(m)
#             data = np.load(io.BytesIO(f.read()))
#             print("Keys in npz:", list(data.keys()))
#             for k in data.keys():
#                 arr = data[k]
#                 print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")
#             break  # just check the first one

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

# Base paths: image+json shards in WDS_BASE, optical flow shards in WDS_FLOW_BASE
WDS_BASE = '/scratch/dgarmaev/AR-flares/wds_out' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/wds_out'
WDS_FLOW_BASE = '/scratch/dgarmaev/AR-flares/wds_flow' if IS_COMPUTE_CANADA else '/teamspace/studios/this_studio/AR-flares/wds_flow'

SPLIT_DIRS = {
    "Train": os.path.join(WDS_BASE, "train"),
    "Validation": os.path.join(WDS_BASE, "val"),
    "Test": os.path.join(WDS_BASE, "test"),
}
SPLIT_FLOW_DIRS = {
    "Train": os.path.join(WDS_FLOW_BASE, "train"),
    "Validation": os.path.join(WDS_FLOW_BASE, "val"),
    "Test": os.path.join(WDS_FLOW_BASE, "test"),
}

BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True

RESIZE_TO = (224, 224)
RESIZE = transforms.Resize(RESIZE_TO)

# --------------------------------------------


class TarShardDataset(IterableDataset):
    """
    Streams samples from a list of paired .tar shards:
      - {basename}.png (original image)
      - {basename}.json (metadata with label)
      - {basename}.flow.npz (optical flow with keys "u","v")
    """
    def __init__(self,
                 shard_paths: List[str],
                 flow_paths: List[str],
                 shuffle_shards: bool = True,
                 shuffle_samples: bool = False,
                 shuffle_buffer_size: int = 2048,
                 seed: int = 42):
        super().__init__()
        assert len(shard_paths) == len(flow_paths), "Image and flow shard counts must match"
        self.shard_paths = list(shard_paths)
        self.flow_paths = list(flow_paths)
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def _sample_iter_from_tar(self, img_tar: str, flow_tar: str):
        with tarfile.open(img_tar, "r") as tf_img, tarfile.open(flow_tar, "r") as tf_flow:
            # Build maps: basename -> members
            by_key: Dict[str, Dict[str, tarfile.TarInfo]] = {}

            for m in tf_img.getmembers():
                if not m.isfile(): continue
                if m.name.endswith(".png"):
                    key = m.name[:-4]
                    by_key.setdefault(key, {})["png"] = m
                elif m.name.endswith(".json"):
                    key = m.name[:-5]
                    by_key.setdefault(key, {})["json"] = m

            for m in tf_flow.getmembers():
                if not m.isfile(): continue
                if m.name.endswith(".flow.npz"):
                    key = m.name[:-9]  # strip ".flow.npz"
                    by_key.setdefault(key, {})["flow"] = m

            keys = list(by_key.keys())
            if self.shuffle_samples:
                rng = random.Random(self.seed ^ hash(img_tar))
                rng.shuffle(keys)

            for k in keys:
                parts = by_key[k]
                if not all(x in parts for x in ["png", "json", "flow"]):
                    continue

                # Load metadata
                meta = json.loads(tf_img.extractfile(parts["json"]).read().decode("utf-8"))
                label = int(meta["label"])

                # Load image (grayscale)
                png_bytes = tf_img.extractfile(parts["png"]).read()
                img = Image.open(io.BytesIO(png_bytes)).convert("L")
                img = RESIZE(img)
                img_t = transforms.ToTensor()(img)  # (1,H,W)

                # Load flow
                flow_bytes = tf_flow.extractfile(parts["flow"]).read()
                with np.load(io.BytesIO(flow_bytes)) as f:
                    u = f["u8"].astype(np.float32) * float(f["su"])
                    v = f["v8"].astype(np.float32) * float(f["sv"])
                u_t = torch.from_numpy(u).unsqueeze(0)
                v_t = torch.from_numpy(v).unsqueeze(0)

                # Resize flow to 224x224
                u_t = RESIZE(u_t)
                v_t = RESIZE(v_t)

                # Final stacked tensor (3,224,224)
                x = torch.cat([img_t, u_t, v_t], dim=0)

                yield x, label, meta

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
            flow_shards = self.flow_paths
        else:
            num_workers = worker_info.num_workers
            wid = worker_info.id
            img_shards = self.shard_paths[wid::num_workers]
            flow_shards = self.flow_paths[wid::num_workers]

        shard_pairs = list(zip(img_shards, flow_shards))
        if self.shuffle_shards:
            rng = random.Random(self.seed + (worker_info.id if worker_info else 0))
            rng.shuffle(shard_pairs)

        gens = [self._sample_iter_from_tar(img_sp, flow_sp) for img_sp, flow_sp in shard_pairs]
        stream = self._roundrobin(gens)
        if self.shuffle_samples:
            stream = self._buffered_shuffle(stream)

        for sample in stream:
            yield sample


def list_shards(split_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(split_dir, "*.tar")))


def make_dataset(split_name: str,
                 shuffle_shards=True,
                 shuffle_samples=False,
                 seed=42) -> TarShardDataset:
    img_shards = list_shards(SPLIT_DIRS[split_name])
    flow_shards = list_shards(SPLIT_FLOW_DIRS[split_name])
    if not img_shards or not flow_shards:
        raise FileNotFoundError(f"Missing shards for {split_name}")
    return TarShardDataset(
        img_shards,
        flow_shards,
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


def peek_n(dataloader: DataLoader, n=5):
    it = iter(dataloader)
    imgs, labels, metas = [], [], []
    seen = 0
    while seen < n:
        images, y, meta = next(it)
        b = images.shape[0]
        take = min(b, n - seen)
        imgs.append(images[:take])
        labels.append(y[:take])

        if isinstance(meta, dict):
            metas.append(meta)
        elif isinstance(meta, list):
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
    x, y, m = peek_n(dls["Train"], n=8)
    print("Peek:", x.shape, y.tolist(), [mm.get("ar") for mm in m])
    print("Label dist (train):", count_labels(dls["Train"], max_batches=200))
    print("Label dist (val):  ", count_labels(dls["Validation"], max_batches=200))
    print("Label dist (test): ", count_labels(dls["Test"], max_batches=200))
