# AR-Flares

AR-Flares is the active PyTorch training codebase for solar flare classification experiments in this repo.

## Supported Entry Points

```bash
python -m classifier_NN.smoke_test
python -m classifier_NN.train
python -m classifier_NN.run_experiments
```

Direct experiment modules also live under `classifier_NN.experiments`.

## Data Policy

Raw datasets, WebDataset shards, large logs, checkpoints, and full training outputs are local-only and are not meant to be committed to GitHub.

Collaborators should point the code at their own local dataset copy with:

```bash
export AR_FLARES_WDS_BASE=/path/to/wds_out
export AR_FLARES_WDS_FLOW_BASE=/path/to/wds_flow   # optional
export AR_FLARES_RESULTS_BASE=/path/to/results
export AR_FLARES_INTENSITY_LABELS_ROOT=/path/to/data
```

## Collaborator Setup

Local setup and smoke-test instructions live in [docs/COLLABORATOR_SETUP.md](/home/dgarmaev/scratch/ar-flares/docs/COLLABORATOR_SETUP.md).

The default smoke test is data-free, but if local shards are present it will also validate one real dataloader batch.

## Repo Layout

- `classifier_NN/`: active training pipeline, models, datasets, and experiment runners
- `scripts/`: analysis, plotting, and SLURM helper scripts
- `compute_canada/`: cluster-specific setup and submission notes
- `archive/`: historical code and research material kept for reference

## Cluster Notes

Compute Canada / SLURM-specific instructions are in [compute_canada/README_compute_canada.md](/home/dgarmaev/scratch/ar-flares/compute_canada/README_compute_canada.md).

## Normalization Experiments

The active model stack now exposes configurable normalization choices for custom 2D CNN blocks, 3D CNN blocks, and token-feature layers. Existing defaults preserve current behavior, while future custom normalization modules can be registered without rewriting each model class.
