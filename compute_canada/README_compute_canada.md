## Compute Canada quickstart (Narval / Cedar / etc.)

This repo is designed to run on Compute Canada style clusters (no GUI, SLURM, scratch storage).

### 1) Where training looks for data/results

`classifier_NN/config.py` automatically switches to Compute Canada paths when `/scratch` exists:

- `wds_base`: `/scratch/<user>/AR-flares/wds_out/{train,val,test}`
- `wds_flow_base`: `/scratch/<user>/AR-flares/wds_flow/{train,val,test}` (optional)
- `results_base`: `/scratch/<user>/AR-flares/results`

If your username is not `dgarmaev`, edit those paths in `classifier_NN/config.py` (or better: set them via env vars in a later refactor).

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






