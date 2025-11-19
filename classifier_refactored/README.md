# AR-Flares Refactored Pipeline

This directory contains the clean, modular training pipeline for AR flare classification. It supersedes exploratory scripts in `classifier_NN/`.

## Modules
- `config.py` – Hyperparameters & paths (image size, backbone, sampling mode).
- `datasets.py` – Tar shard iterable dataset with optional flow/sequence support and class balancing.
- `models.py` – Backbone selection (ConvNeXt, ViT, CAN, multi-scale fusion, temporal wrapper, two-stream).
- `losses.py` – Focal or standard cross entropy with optional class weights.
- `metrics.py` – Evaluation metrics & plots (ROC, PR, confusion matrix, Precision/Recall/F1, TSS, HSS).
- `train.py` – End-to-end training loop with per-epoch JSONL logging & curve plots.

## Sampling Modes
Configured via `CFG['balance_classes']` and `CFG['balance_mode']`:
- `balance_classes=False` – Use full imbalanced distribution.
- `balance_mode='prob'` – Per-epoch probabilistic negative downsampling; each negative kept with probability `neg_keep_prob`. Maximizes diversity across epochs.
- `balance_mode='fixed'` – Deterministic subset of negatives selected by hashing metadata; stable across epochs for reproducibility & consistent epoch length.

Validation/Test are never balanced to preserve real-world distribution.

## Quick Start
```bash
cd classifier_refactored
python train.py
```

## Example Config Change
```python
CFG.update({
  "backbone": "convnext_base",
  "image_size": 112,
  "balance_classes": True,
  "balance_mode": "prob",
  "neg_keep_prob": 0.25,
})
```

## Outputs
Results stored under `results/<timestamp>_<model_name>/` with:
- `log.txt` – Streaming log
- `epoch_metrics.jsonl` – Per-epoch metrics
- `training_curves.png` – Loss & accuracy curves
- `roc.png`, `pr.png`, `confusion_matrix.png` – Evaluation plots
- `metrics.json` – Final test metrics

## Roadmap
- Expand sequence aggregation (attention improvements)
- Add AMP autocast for CPU fallback gracefully
- Implement experiment registry YAML

See root `README.md` for broader project context.
