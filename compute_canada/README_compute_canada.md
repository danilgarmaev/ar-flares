## Compute Canada quickstart (Narval / Cedar / etc.)

This repo is designed to run on Compute Canada style clusters (no GUI, SLURM, scratch storage).

### 1) Where training looks for data/results

`classifier_NN/config.py` resolves paths in this order (most explicit wins):

1) Environment variables (recommended on clusters)
	- `AR_FLARES_WDS_BASE` (expected to contain `train/`, `val/`, `test/`)
	- `AR_FLARES_WDS_FLOW_BASE` (optional; also contains `train/`, `val/`, `test/`)
	- `AR_FLARES_RESULTS_BASE`
	- `AR_FLARES_INTENSITY_LABELS_ROOT`
2) Common locations that already exist (including `/scratch/$USER/...`)
3) Repo-relative fallbacks under `data/` and `results/`

Practical default on Compute Canada is to keep large data + results on scratch, e.g.:

```bash
export AR_FLARES_WDS_BASE=/scratch/$USER/ar-flares/data/wds_out
export AR_FLARES_WDS_FLOW_BASE=/scratch/$USER/ar-flares/data/wds_flow   # optional
export AR_FLARES_RESULTS_BASE=/scratch/$USER/ar-flares/results
```

### 2) Environment setup (recommended: modules + venv)

On Compute Canada you typically do this on the **login node**, then submit jobs.

```bash
cd /home/$USER/ar-flares

module load python/3.11
module load cuda/12.2

# Create venv somewhere persistent (HOME) or fast (SCRATCH).
python -m venv /scratch/$USER/venvs/ar-flares
source /scratch/$USER/venvs/ar-flares/bin/activate

python -m pip install --upgrade pip setuptools wheel

# Editable install so `python -m classifier_NN.train` works reliably:
python -m pip install -e .
```

### 3) Internet / offline installs

Compute nodes often have **no outbound internet**. Login nodes sometimes do, sometimes don’t (depends on cluster / policy).

If there is **no internet**, use one of these patterns:

- **Wheelhouse (recommended)**: build/download wheels on a machine with internet, copy to the cluster, then:

```bash
python -m pip install --no-index --find-links /scratch/$USER/wheels -r requirements.txt
python -m pip install --no-index --find-links /scratch/$USER/wheels -e .
```

- **Local mirror**: if your institution provides an internal PyPI mirror, configure `pip.conf` to point to it.

Notes for PyTorch:
- Installing CUDA-enabled PyTorch with pip often needs the PyTorch index URL.
- On clusters, the most reliable approach is either (a) a provided PyTorch module/stack (if available), or (b) a prebuilt wheelhouse you bring in.

### 4) Smoke test (CPU) before using GPUs

```bash
cd /home/$USER/ar-flares
source /scratch/$USER/venvs/ar-flares/bin/activate
python -m classifier_NN.smoke_test
```

### 5) Start training (single GPU)

Preferred (module invocation; required because `classifier_NN/train.py` uses relative imports):

```bash
python -m classifier_NN.train
```

Or run an experiment runner:

```bash
python -m classifier_NN.run_experiments
```

### 6) SLURM

See submission scripts in `compute_canada/slurm/`.






