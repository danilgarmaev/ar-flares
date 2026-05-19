# Collaborator Setup

This repo keeps code and lightweight metadata in Git, but not the real dataset, full logs, or large training outputs.

## Local Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Dataset Paths

Point the training code at your local dataset copy with environment variables:

```bash
export AR_FLARES_WDS_BASE=/path/to/ar-flares/data/wds_out
export AR_FLARES_WDS_FLOW_BASE=/path/to/ar-flares/data/wds_flow   # optional
export AR_FLARES_RESULTS_BASE=/path/to/ar-flares/results
export AR_FLARES_INTENSITY_LABELS_ROOT=/path/to/ar-flares/data
```

Expected layout under `AR_FLARES_WDS_BASE`:

```text
wds_out/
  train/
  val/
  test/
```

## Safe Local Checks

CPU-only smoke test with synthetic input:

```bash
python -m classifier_NN.smoke_test
```

If your local shards are present, the smoke test will also try one dataloader batch automatically.

## Training Entry Points

Single training run:

```bash
python -m classifier_NN.train
```

Stable experiment entrypoint:

```bash
python -m classifier_NN.run_experiments
```

Direct experiment modules:

```bash
python -m classifier_NN.experiments.run_experiments_A1 --single --backbone convnext_tiny --resolution 112 --seed 0
python -m classifier_NN.experiments.run_experiments_16frames_cadence --single --backbone r2plus1d_18 --interval-min 72 --seed 0
```

## Local-Only Paths

These should stay out of GitHub:

- `data/`
- `results/`
- `logs/`
- local caches under `.cache/`
- large checkpoints and raw SLURM logs
