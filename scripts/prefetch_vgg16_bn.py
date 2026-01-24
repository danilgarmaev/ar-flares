"""Prefetch timm VGG16-BN pretrained weights on a LOGIN node.

Why
----
Compute nodes on Narval typically have no outbound internet. `timm` will try to
fetch pretrained weights (often hosted on Hugging Face) unless they are already
present in the local cache.

This script primes the cache so SLURM jobs can run with `--pretrained` while
remaining offline.

Usage (run on login node):
  source ~/envs/ar-flares/bin/activate
  export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf
  export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch
  python scripts/prefetch_vgg16_bn.py

Then submit the full job with:
  sbatch --export=ALL,PRETRAINED=1 scripts/a1_vgg_full_a100.slurm
"""

from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    hf_home = os.environ.get("HF_HOME")
    torch_home = os.environ.get("TORCH_HOME")

    if not hf_home or not torch_home:
        raise SystemExit(
            "Please set HF_HOME and TORCH_HOME to a scratch cache dir before running.\n"
            "Example:\n"
            "  export HF_HOME=/home/dgarmaev/scratch/ar-flares/.cache/hf\n"
            "  export TORCH_HOME=/home/dgarmaev/scratch/ar-flares/.cache/torch\n"
        )

    Path(hf_home).mkdir(parents=True, exist_ok=True)
    Path(torch_home).mkdir(parents=True, exist_ok=True)

    print(f"HF_HOME={hf_home}")
    print(f"TORCH_HOME={torch_home}")

    import timm  # noqa: PLC0415

    print("Downloading timm vgg16_bn pretrained weights...")
    _ = timm.create_model("vgg16_bn", pretrained=True)
    print("OK: weights should now be cached for offline compute jobs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
